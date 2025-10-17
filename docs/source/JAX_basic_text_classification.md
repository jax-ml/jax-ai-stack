---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Basic text classification with 1D CNN

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jax-ml/jax-ai-stack/blob/main/docs/source/JAX_basic_text_classification.ipynb)

In this tutorial we learn how to perform text classification from raw text data and train a basic 1D Convolutional Neural Network to perform sentiment analysis using JAX. This tutorial is originally inspired by ["Text classification from scratch with Keras"](https://keras.io/examples/nlp/text_classification_from_scratch/#build-a-model).

We will use the IMDB movie review dataset to classify the review to "positive" and "negative" classes. We implement from scratch a simple model using Flax, train it and compute metrics on the test set.

+++

## Setup

We will be using the following packages in this tutorial:
- [Tiktoken](https://github.com/openai/tiktoken) to tokenize the raw text
- [Grain](https://github.com/google/grain) for efficient data loading and batching
- [tqdm](https://tqdm.github.io/) for a progress bar to monitor the training progress.

```{code-cell} ipython3
!pip install grain tiktoken tqdm
```

### Load the data: IMDB movie review sentiment classification

Let us download the dataset and briefly inspect the structure. We will be using only two classes: "positive" and "negative" for the sentiment analysis.

```{code-cell} ipython3
!rm -rf /tmp/data/imdb
!mkdir -p /tmp/data/imdb
!wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz -O /tmp/data/imdb/aclImdb_v1.tar.gz
!cd /tmp/data/imdb/ && tar -xf aclImdb_v1.tar.gz
```

```{code-cell} ipython3
!echo "Number of positive samples in train set:"
!ls /tmp/data/imdb/aclImdb/train/pos | wc -l
!echo "Number of negative samples in train set:"
!ls /tmp/data/imdb/aclImdb/train/neg | wc -l
!echo "Number of positive samples in test set:"
!ls /tmp/data/imdb/aclImdb/test/pos | wc -l
!echo "Number of negative samples in test set:"
!ls /tmp/data/imdb/aclImdb/test/neg | wc -l
!echo "First 10 files with positive samples in train/test sets:"
!ls /tmp/data/imdb/aclImdb/train/pos | head
!ls /tmp/data/imdb/aclImdb/test/pos | head
!echo "Display a single positive sample:"
!cat /tmp/data/imdb/aclImdb/train/pos/6248_7.txt
```

Next, we will:
- create the dataset Python class to read samples from the disk
- use [Tiktoken](https://github.com/openai/tiktoken) to encode raw text into tokens and
- use [Grain](https://github.com/google/grain) for efficient data loading and batching.

```{code-cell} ipython3
from pathlib import Path


class SentimentAnalysisDataset:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        assert self.path.exists()

        pos_texts = list((self.path / "pos").glob("*.txt"))
        neg_texts = list((self.path / "neg").glob("*.txt"))
        self.text_files = pos_texts + neg_texts
        assert len(self.text_files) > 0
        # Label 0 for Positive comments
        # Label 1 for Negative comments
        self.labels = [0] * len(pos_texts) + [1] * len(neg_texts)

    def __len__(self) -> int:
        return len(self.text_files)

    def read_text_file(self, path: str | Path) -> str:
        with open(path, "r") as handler:
            lines = handler.readlines()
        return "\n".join(lines)

    def __getitem__(self, index: int) -> tuple[str, int]:
        label = self.labels[index]
        text = self.read_text_file(self.text_files[index])
        return {"text": text, "label": label}


root_path = Path("/tmp/data/imdb/aclImdb/")
train_dataset = SentimentAnalysisDataset(root_path / "train")
test_dataset = SentimentAnalysisDataset(root_path / "test")

print("- Number of samples in train and test sets:", len(train_dataset), len(test_dataset))
print("- First train sample:", train_dataset[0])
print("- First test sample:", test_dataset[0])
```

Now, we can create a string-to-tokens preprocessing transformation and set up data loaders. We are going to use the GPT-2 tokenizer via [Tiktoken](https://github.com/openai/tiktoken).

```{code-cell} ipython3
import numpy as np

import tiktoken
import grain.python as grain


seed = 12
train_batch_size = 128
test_batch_size = 2 * train_batch_size
tokenizer = tiktoken.get_encoding("gpt2")
# max length of tokenized text
max_length = 500
vocab_size = tokenizer.n_vocab


class TextPreprocessing(grain.MapTransform):
    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def map(self, data):
        text = data["text"]
        encoded = self.tokenizer.encode(text)
        # Cut to max length
        encoded = encoded[:self.max_length]
        # Pad with zeros if needed
        encoded = np.array(encoded + [0] * (self.max_length - len(encoded)))
        return {
            "text": encoded,
            "label": data["label"],
        }


train_sampler = grain.IndexSampler(
    len(train_dataset),
    shuffle=True,
    seed=seed,
    shard_options=grain.NoSharding(),  # No sharding since this is a single-device setup
    num_epochs=1,                      # Iterate over the dataset for one epoch
)

test_sampler = grain.IndexSampler(
    len(test_dataset),
    shuffle=False,
    seed=seed,
    shard_options=grain.NoSharding(),  # No sharding since this is a single-device setup
    num_epochs=1,                      # Iterate over the dataset for one epoch
)


train_loader = grain.DataLoader(
    data_source=train_dataset,
    sampler=train_sampler,                 # Sampler to determine how to access the data
    worker_count=4,                        # Number of child processes launched to parallelize the transformations among
    worker_buffer_size=2,                  # Count of output batches to produce in advance per worker
    operations=[
        TextPreprocessing(tokenizer, max_length=max_length),
        grain.Batch(train_batch_size, drop_remainder=True),
    ]
)

test_loader = grain.DataLoader(
    data_source=test_dataset,
    sampler=test_sampler,                  # Sampler to determine how to access the data
    worker_count=4,                        # Number of child processes launched to parallelize the transformations among
    worker_buffer_size=2,                  # Count of output batches to produce in advance per worker
    operations=[
        TextPreprocessing(tokenizer, max_length=max_length),
        grain.Batch(test_batch_size),
    ]
)
```

```{code-cell} ipython3
train_batch = next(iter(train_loader))
```

```{code-cell} ipython3
print("Train encoded text batch info:", type(train_batch["text"]), train_batch["text"].shape, train_batch["text"].dtype)
print("Train labels batch info:", type(train_batch["label"]), train_batch["label"].shape, train_batch["label"].dtype)
```

Let's check few samples of the training batch. We expect to see integer tokens for the input text and integer value for the labels:

```{code-cell} ipython3
print("Train batch data:", train_batch["text"][:2, :12], train_batch["label"][:2])
```

## Model for text classification

We choose a simple 1D convnet to classify the text. The first layer of the model transforms input tokens into float features using an embedding layer (`nnx.Embed`), then they are encoded further with convolutions. Finally, we classify encoded features using fully-connected layers.

```{code-cell} ipython3
from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx


class TextConvNet(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 2,
        embed_dim: int = 256,
        hidden_dim: int = 320,
        dropout_rate: float = 0.5,
        conv_ksize: int = 12,
        activation_layer: Callable = nnx.relu,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.activation_layer = activation_layer
        self.token_embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=embed_dim,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.conv1 = nnx.Conv(
            in_features=embed_dim,
            out_features=hidden_dim,
            kernel_size=conv_ksize,
            strides=conv_ksize // 2,
            rngs=rngs,
        )
        self.lnorm1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=hidden_dim,
            out_features=hidden_dim,
            kernel_size=conv_ksize,
            strides=conv_ksize // 2,
            rngs=rngs,
        )
        self.lnorm2 = nnx.LayerNorm(hidden_dim, rngs=rngs)

        self.fc1 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, num_classes, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x.shape: (N, max_length)
        x = self.token_embedding(x)
        x = self.dropout(x)   # x.shape: (N, max_length, embed_dim)

        x = self.conv1(x)
        x = self.lnorm1(x)
        x = self.activation_layer(x)
        x = self.conv2(x)
        x = self.lnorm2(x)
        x = self.activation_layer(x)  # x.shape: (N, K, hidden_dim)

        x = nnx.max_pool(x, window_shape=(x.shape[1], ))  # x.shape: (N, 1, hidden_dim)
        x = x.reshape((-1, x.shape[-1]))  # x.shape: (N, hidden_dim)

        x = self.fc1(x)  # x.shape: (N, hidden_dim)
        x = self.activation_layer(x)
        x = self.dropout(x)
        x = self.fc2(x)  # x.shape: (N, 2)

        return x


# Let's check the model on a dummy input
x = jnp.ones((4, max_length), dtype="int32")
module = TextConvNet(vocab_size)
y = module(x)
print("Prediction shape (N, num_classes): ", y.shape)
```

```{code-cell} ipython3
model = TextConvNet(
    vocab_size,
    num_classes=2,
    embed_dim=128,
    hidden_dim=128,
    conv_ksize=7,
    activation_layer=nnx.relu,
)
```

## Train the model

We can now train the model using training data loader and compute metrics: accuracy and loss on test data loader.
Below we set up the optimizer and define the loss function as Cross-Entropy.
Next, we define the train step where we compute the loss value and update the model parameters.
In the eval step we use the model to compute the metrics: accuracy and loss value.

```{code-cell} ipython3
import optax


num_epochs = 10
learning_rate = 0.0005
momentum = 0.9

optimizer = nnx.Optimizer(model, optax.adam(learning_rate, momentum))
```

```{code-cell} ipython3
def compute_losses_and_logits(model: nnx.Module, batch_tokens: jax.Array, labels: jax.Array):
    logits = model(batch_tokens)

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()
    return loss, logits
```

```{code-cell} ipython3
@nnx.jit
def train_step(
    model: nnx.Module, optimizer: nnx.Optimizer, batch: dict[str, jax.Array]
):
    # Convert numpy arrays to jax.Array on GPU
    batch_tokens = jnp.array(batch["text"])
    labels = jnp.array(batch["label"], dtype=jnp.int32)

    grad_fn = nnx.value_and_grad(compute_losses_and_logits, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch_tokens, labels)

    optimizer.update(grads)  # In-place updates.

    return loss


@nnx.jit
def eval_step(
    model: nnx.Module, batch: dict[str, jax.Array], eval_metrics: nnx.MultiMetric
):
    # Convert numpy arrays to jax.Array on GPU
    batch_tokens = jnp.array(batch["text"])
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

```{code-cell} ipython3
import tqdm


bar_format = "{desc}[{n_fmt}/{total_fmt}]{postfix} [{elapsed}<{remaining}]"
train_total_steps = len(train_dataset) // train_batch_size


def train_one_epoch(epoch):
    model.train()  # Set model to the training mode: e.g. update batch statistics
    with tqdm.tqdm(
        desc=f"[train] epoch: {epoch}/{num_epochs}, ",
        total=train_total_steps,
        bar_format=bar_format,
        leave=True,
    ) as pbar:
        for batch in train_loader:
            loss = train_step(model, optimizer, batch)
            train_metrics_history["train_loss"].append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)


def evaluate_model(epoch):
    # Compute the metrics on the train and val sets after each training epoch.
    model.eval()  # Set model to evaluation model: e.g. use stored batch statistics

    eval_metrics.reset()  # Reset the eval metrics
    for test_batch in test_loader:
        eval_step(model, test_batch, eval_metrics)

    for metric, value in eval_metrics.compute().items():
        eval_metrics_history[f'test_{metric}'].append(value)

    print(f"[test] epoch: {epoch + 1}/{num_epochs}")
    print(f"- total loss: {eval_metrics_history['test_loss'][-1]:0.4f}")
    print(f"- Accuracy: {eval_metrics_history['test_accuracy'][-1]:0.4f}")
```

Now, we can start the training.

```{code-cell} ipython3
%%time

for epoch in range(num_epochs):
    train_one_epoch(epoch)
    evaluate_model(epoch)
```

Let's visualize the collected metrics:

```{code-cell} ipython3
import matplotlib.pyplot as plt


plt.plot(train_metrics_history["train_loss"], label="Loss value during the training")
plt.legend()
```

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(10, 10))
axs[0].set_title("Loss value on test set")
axs[0].plot(eval_metrics_history["test_loss"])
axs[1].set_title("Accuracy on test set")
axs[1].plot(eval_metrics_history["test_accuracy"])
```

We can observe that the model starts overfitting after the 5-th epoch and the best accuracy it could achieve is around 0.87. Let us also check few model's predictions on the test data:

```{code-cell} ipython3
data = test_dataset[10]
```

```{code-cell} ipython3
text_processing = TextPreprocessing(tokenizer, max_length=max_length)
processed_data = text_processing.map(data)
model.eval()
preds = model(processed_data["text"][None, :])
pred_label = preds.argmax(axis=-1).item()
confidence = nnx.softmax(preds, axis=-1)

print("- Text:\n", data["text"])
print("")
print(f"- Expected review sentiment: {'positive' if data['label'] == 0 else 'negative'}")
print(f"- Predicted review sentiment: {'positive' if pred_label == 0 else 'negative'}, confidence: {confidence[0, pred_label]:.3f}")
```

```{code-cell} ipython3
data = test_dataset[20]
```

```{code-cell} ipython3
text_processing = TextPreprocessing(tokenizer, max_length=max_length)
processed_data = text_processing.map(data)
model.eval()
preds = model(processed_data["text"][None, :])
pred_label = preds.argmax(axis=-1).item()
confidence = nnx.softmax(preds, axis=-1)

print("- Text:\n", data["text"])
print("")
print(f"- Expected review sentiment: {'positive' if data['label'] == 0 else 'negative'}")
print(f"- Predicted review sentiment: {'positive' if pred_label == 0 else 'negative'}, confidence: {confidence[0, pred_label]:.3f}")
```

## Further reading

In this tutorial we implemented from scratch a simple convolutional neural network and trained it on a text classification dataset. Trained model shows 87% classification accuracy due to its convolutional nature. Next steps to improve the metrics could be to change the model to a transformer-based architecture.

- Model checkpointing and exporting using [Orbax](https://orbax.readthedocs.io/en/latest/).
- Other NLP tutorials in [jax-ai-stack](https://jax-ai-stack.readthedocs.io/en/latest/getting_started.html).
