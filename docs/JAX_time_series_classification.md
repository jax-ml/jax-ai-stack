---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: jax-env
  language: python
  name: python3
---

# Time series classification with JAX

In this tutorial, we're going to perform time series classification with a Convolutional Neural Network.
We will use the FordA dataset from the [UCR archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/),
which contains measurements of engine noise captured by a motor sensor.

We need to assess if an engine is malfunctioning based on the recorded noises it generates.
Each sample comprises of noise measurements across time, together with a "yes/no" label,
so this is a binary classification problem.

Although convolution models are mainly associated with image processing, they are also useful
for time series data because they can extract temporal structures.

+++

## Tools overview and setup

Here's a list of key packages that belong to the JAX AI stack required for this tutorial:

- [JAX](https://github.com/jax-ml/jax) for array computations.
- [Flax](https://github.com/google/flax) for constructing neural networks.
- [Optax](https://github.com/google-deepmind/optax) for gradient processing and optimization.
- [Grain](https://github.com/google/grain/) to define data sources.
- [tqdm](https://tqdm.github.io/) for a progress bar to monitor the training progress.

We'll start by installing and importing these packages.

```{code-cell} ipython3
# Required packages
# !pip install -U jax flax optax
# !pip install -U grain tqdm requests matplotlib
```

```{code-cell} ipython3
import jax
import jax.numpy as jnp
from flax import nnx
import optax

import numpy as np
import matplotlib.pyplot as plt
import grain.python as grain
import tqdm
```

## Load the dataset

We load dataset files into NumPy arrays, add singleton dimension to take convolution features
into account, and change `-1` label to `0` (so that the expected values are `0` and `1`):

```{code-cell} ipython3
def prepare_ucr_dataset() -> tuple:
    root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

    train_data = np.loadtxt(root_url + "FordA_TRAIN.tsv", delimiter="\t")
    x_train, y_train = train_data[:, 1:], train_data[:, 0].astype(int)

    test_data = np.loadtxt(root_url + "FordA_TEST.tsv", delimiter="\t")
    x_test, y_test = test_data[:, 1:], test_data[:, 0].astype(int)

    x_train = x_train.reshape((*x_train.shape, 1))
    x_test = x_test.reshape((*x_test.shape, 1))

    rng = np.random.RandomState(113)
    indices = rng.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]

    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    return (x_train, y_train), (x_test, y_test)
```

```{code-cell} ipython3
(x_train, y_train), (x_test, y_test) = prepare_ucr_dataset()
```

Let's visualize example samples from each class.

```{code-cell} ipython3
classes = np.unique(np.concatenate((y_train, y_test), axis=0))
for c in classes:
    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label="class " + str(c))
plt.legend()
plt.show()
```

### Create a Data Loader using Grain

For handling input data we're going to use Grain, a pure Python package developed for JAX and
Flax models.

Grain follows the source-sampler-loader paradigm. Grain supports custom setups where data sources
might come in different forms, but they all need to implement the `grain.RandomAccessDataSource`
interface. See [PyGrain Data Sources](https://github.com/google/grain/blob/main/docs/data_sources.md)
for more details.

Our dataset is comprised of relatively small NumPy arrays so our `DataSource` is uncomplicated:

```{code-cell} ipython3
class DataSource(grain.RandomAccessDataSource):
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __getitem__(self, idx):
        return {"measurement": self._x[idx], "label": self._y[idx]}

    def __len__(self):
        return len(self._x)
```

```{code-cell} ipython3
train_source = DataSource(x_train, y_train)
test_source = DataSource(x_test, y_test)
```

Samplers determine the order in which records are processed, and we'll use the
[`IndexSmapler`](https://github.com/google/grain/blob/main/docs/data_loader/samplers.md#index-sampler)
recommended by Grain.

Finally, we'll create `DataLoader`s that handle orchestration of loading.
We'll leverage Grain's multiprocessing capabilities to scale processing up to 4 workers.

```{code-cell} ipython3
seed = 12
train_batch_size = 128
test_batch_size = 2 * train_batch_size

train_sampler = grain.IndexSampler(
    len(train_source),
    shuffle=True,
    seed=seed,
    shard_options=grain.NoSharding(),  # No sharding since this is a single-device setup
    num_epochs=1,                      # Iterate over the dataset for one epoch
)

test_sampler = grain.IndexSampler(
    len(test_source),
    shuffle=False,
    seed=seed,
    shard_options=grain.NoSharding(),  # No sharding since this is a single-device setup
    num_epochs=1,                      # Iterate over the dataset for one epoch
)


train_loader = grain.DataLoader(
    data_source=train_source,
    sampler=train_sampler,  # Sampler to determine how to access the data
    worker_count=4,         # Number of child processes launched to parallelize the transformations among
    worker_buffer_size=2,   # Count of output batches to produce in advance per worker
    operations=[
        grain.Batch(train_batch_size, drop_remainder=True),
    ]
)

test_loader = grain.DataLoader(
    data_source=test_source,
    sampler=test_sampler,  # Sampler to determine how to access the data
    worker_count=4,        # Number of child processes launched to parallelize the transformations among
    worker_buffer_size=2,  # Count of output batches to produce in advance per worker
    operations=[
        grain.Batch(test_batch_size),
    ]
)
```

## Define the Model

Let's now construct the Convolutional Neural Network with Flax by subclassing `nnx.Module`.
You can learn more about the [Flax NNX module system in the Flax documentation](https://flax.readthedocs.io/en/latest/nnx_basics.html#the-flax-nnx-module-system).

Let's have three convolution and dense layers, and use ReLU activation function for middle
layers and softmax in the final layer for binary classification output.

```{code-cell} ipython3
class MyModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.conv_1 = nnx.Conv(
            in_features=1, out_features=64, kernel_size=3, padding="SAME", rngs=rngs
        )
        self.layer_norm_1 = nnx.LayerNorm(num_features=64, epsilon=0.001, rngs=rngs)

        self.conv_2 = nnx.Conv(
            in_features=64, out_features=64, kernel_size=3, padding="SAME", rngs=rngs
        )
        self.layer_norm_2 = nnx.LayerNorm(num_features=64, epsilon=0.001, rngs=rngs)

        self.conv_3 = nnx.Conv(
            in_features=64, out_features=64, kernel_size=3, padding="SAME", rngs=rngs
        )
        self.layer_norm_3 = nnx.LayerNorm(num_features=64, epsilon=0.001, rngs=rngs)

        self.dense_1 = nnx.Linear(in_features=64, out_features=2, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = self.conv_1(x)
        x = self.layer_norm_1(x)
        x = jax.nn.relu(x)

        x = self.conv_2(x)
        x = self.layer_norm_2(x)
        x = jax.nn.relu(x)

        x = self.conv_3(x)
        x = self.layer_norm_3(x)
        x = jax.nn.relu(x)

        x = jnp.mean(x, axis=(1,))  # global average pooling
        x = self.dense_1(x)
        x = jax.nn.softmax(x)
        return x
```

```{code-cell} ipython3
model = MyModel(rngs=nnx.Rngs(0))
nnx.display(model)
```

## Train the Model

To train our Flax model we need to construct an `nnx.Optimizer` object with our model and
a selected optimization algorithm. The optimizer object manages the model’s parameters and
applies gradients during training.

We're going to use [Adam optimizer](https://optax.readthedocs.io/en/latest/api/optimizers.html#adam),
a popular choice for Deep Learning models. We'll use it through
[Optax](https://optax.readthedocs.io/en/latest/index.html), an optimization library developed for JAX.

```{code-cell} ipython3
num_epochs = 300
learning_rate = 0.0005
momentum = 0.9

optimizer = nnx.Optimizer(model, optax.adam(learning_rate, momentum))
```

We'll define a loss and logits computation function using Optax's
[`losses.softmax_cross_entropy_with_integer_labels`](https://optax.readthedocs.io/en/latest/api/losses.html#optax.losses.softmax_cross_entropy_with_integer_labels).

```{code-cell} ipython3
def compute_losses_and_logits(model: nnx.Module, batch_tokens: jax.Array, labels: jax.Array):
    logits = model(batch_tokens)

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()
    return loss, logits
```

We'll now define the training and evaluation step functions. The loss and logits from both
functions will be used for calculating accuracy metrics.

For training, we'll use `nnx.value_and_grad` to compute the gradients, and then update
the model’s parameters using our optimizer.

Notice the use of [`nnx.jit`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.jit). This sets up the functions for just-in-time (JIT) compilation with [XLA](https://openxla.org/xla)
for performant execution across different hardware accelerators like GPUs and TPUs.

```{code-cell} ipython3
@nnx.jit
def train_step(
    model: nnx.Module, optimizer: nnx.Optimizer, batch: dict[str, jax.Array]
):
    batch_tokens = jnp.array(batch["measurement"])
    labels = jnp.array(batch["label"], dtype=jnp.int32)

    grad_fn = nnx.value_and_grad(compute_losses_and_logits, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch_tokens, labels)

    optimizer.update(grads)  # In-place updates.

    return loss

@nnx.jit
def eval_step(
    model: nnx.Module, batch: dict[str, jax.Array], eval_metrics: nnx.MultiMetric
):
    batch_tokens = jnp.array(batch["measurement"])
    labels = jnp.array(batch["label"], dtype=jnp.int32)
    loss, logits = compute_losses_and_logits(model, batch_tokens, labels)

    eval_metrics.update(
        loss=loss,
        logits=logits,
        labels=labels,
    )
```

```{code-cell} ipython3
eval_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average('loss'),
    accuracy=nnx.metrics.Accuracy(),
)

train_metrics_history = {
    "train_loss": [],
}

eval_metrics_history = {
    "test_loss": [],
    "test_accuracy": [],
}
```

We can now train the CNN model. We'll evaluate the model’s performance on the test set
after each epoch, and print the metrics: total loss and accuracy.

```{code-cell} ipython3
bar_format = "{desc}[{n_fmt}/{total_fmt}]{postfix} [{elapsed}<{remaining}]"
train_total_steps = len(x_train) // train_batch_size

def train_one_epoch(epoch: int):
    model.train()
    with tqdm.tqdm(
        desc=f"[train] epoch: {epoch}/{num_epochs}, ",
        total=train_total_steps,
        bar_format=bar_format,
        miniters=10,
        leave=True,
    ) as pbar:
        for batch in train_loader:
            loss = train_step(model, optimizer, batch)
            train_metrics_history["train_loss"].append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)

def evaluate_model(epoch: int):
    # Compute the metrics on the train and val sets after each training epoch.
    model.eval()

    eval_metrics.reset()  # Reset the eval metrics
    for test_batch in test_loader:
        eval_step(model, test_batch, eval_metrics)

    for metric, value in eval_metrics.compute().items():
        eval_metrics_history[f'test_{metric}'].append(value)

    if epoch % 10 == 0:
        print(f"[test] epoch: {epoch + 1}/{num_epochs}")
        print(f"- total loss: {eval_metrics_history['test_loss'][-1]:0.4f}")
        print(f"- Accuracy: {eval_metrics_history['test_accuracy'][-1]:0.4f}")
```

```{code-cell} ipython3
%%time
for epoch in range(num_epochs):
    train_one_epoch(epoch)
    evaluate_model(epoch)
```

Finally, let's visualize the loss and accuracy with Matplotlib.

```{code-cell} ipython3
plt.plot(train_metrics_history["train_loss"], label="Loss value during the training")
plt.legend()
```

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].set_title("Loss value on test set")
axs[0].plot(eval_metrics_history["test_loss"])
axs[1].set_title("Accuracy on test set")
axs[1].plot(eval_metrics_history["test_accuracy"])
```

Our model reached almost 90% accuracy on the test set after 300 epochs, but it's worth noting
that the loss function isn't completely flat yet. We could continue until the curve flattens,
but we also need to pay attention to validation accuracy so as to spot when the model starts
overfitting.

For model early stopping and selecting best model, you can check out [Orbax](https://github.com/google/orbax),
a library which provides checkpointing and persistence utilities.
