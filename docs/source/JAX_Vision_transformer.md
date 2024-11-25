---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Vision Transformer with JAX & FLAX


In this tutorial we implement from scratch Vision Transformer (ViT) model based on the paper by Dosovitskiy et al: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). We will train this model on [Food 101](https://huggingface.co/datasets/ethz/food101) dataset.
This tutorial is originally inspired by [HuggingFace Image classification tutorial](https://huggingface.co/docs/transformers/tasks/image_classification).

+++

## Requirements installation

We will need to install the following Python packages:
- HuggingFace [Datasets](https://huggingface.co/docs/datasets/) will be used for dataset provision
- [TorchVision](https://pytorch.org/vision) will be used for image augmentations
- [grain](https://github.com/google/grain/) will be be used for efficient data loading
- [tqdm](https://tqdm.github.io/) for a progress bar to monitor the training progress.

```{code-cell} ipython3
# !pip install -U datasets grain torchvision tqdm
# !pip install -U flax optax
```

```{code-cell} ipython3
import jax
import flax
import optax
print("Jax version:", jax.__version__)
print("Flax version:", flax.__version__)
print("Optax version:", optax.__version__)
```

## Vision Transformer implementation

Vision Transformer (ViT) architecture contains the following block:
- Patch and position embedding
- Transformer Encoder
- Classification head

![ViT-architecture](https://github.com/google-research/vision_transformer/raw/main/vit_figure.png)

Original JAX implementation can be found [here](https://github.com/google-research/vision_transformer/).

```{code-cell} ipython3
import jax.numpy as jnp
from flax import nnx


class VisionTransformer(nnx.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        img_size: int = 224,
        patch_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        hidden_size: int = 768,
        dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        # Patch and position embedding
        n_patches = (img_size // patch_size) ** 2
        self.patch_embeddings = nnx.Conv(
            in_channels,
            hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=True,
            rngs=rngs,
        )

        initializer = jax.nn.initializers.truncated_normal(stddev=0.02)
        self.position_embeddings = nnx.Param(
            initializer(rngs.params(), (1, n_patches + 1, hidden_size), jnp.float32)
        )
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        self.cls_token = nnx.Param(jnp.zeros((1, 1, hidden_size)))

        # Transformer Encoder blocks
        self.encoder = nnx.Sequential(*[
            TransformerEncoder(hidden_size, mlp_dim, num_heads, dropout_rate, rngs=rngs)
            for i in range(num_layers)
        ])
        self.lnorm = nnx.LayerNorm(hidden_size, rngs=rngs)

        # Classification head
        self.classifier = nnx.Linear(hidden_size, num_classes, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Patch and position embedding
        patches = self.patch_embeddings(x)
        batch_size = patches.shape[0]
        patches = patches.reshape(batch_size, -1, patches.shape[-1])

        cls_token = jnp.tile(self.cls_token, [batch_size, 1, 1])
        x = jnp.concat([cls_token, patches], axis=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        # Encoder blocks
        x = self.encoder(embeddings)
        x = self.lnorm(x)

        # fetch the first token
        x = x[:, 0]

        # Classification
        return self.classifier(x)


class TransformerEncoder(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:

        self.norm1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            dropout_rate=dropout_rate,
            broadcast_dropout=False,
            decode=False,
            deterministic=False,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(hidden_size, rngs=rngs)

        self.mlp = nnx.Sequential(
            nnx.Linear(hidden_size, mlp_dim, rngs=rngs),
            nnx.gelu,
            nnx.Dropout(dropout_rate, rngs=rngs),
            nnx.Linear(mlp_dim, hidden_size, rngs=rngs),
            nnx.Dropout(dropout_rate, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# We use a configuration to make smaller model to reduce the training time
x = jnp.ones((4, 120, 120, 3))
model = VisionTransformer(num_classes=10, num_layers=4, num_heads=4, img_size=120, patch_size=8)
y = model(x)
print("Predictions shape: ", y.shape)
```

In the following sections we set up a image classification dataset and train this model.

+++

## Food 101 dataset

In the this tutorial we use [Food 101](https://huggingface.co/datasets/ethz/food101) dataset which consists of 101 food categories, with 101,000 images. For each class, 250 manually reviewed test images are provided as well as 750 training images. On purpose, the training images were not cleaned, and thus still contain some amount of noise. This comes mostly in the form of intense colors and sometimes wrong labels. All images were rescaled to have a maximum side length of 512 pixels.

We will download the data using [HuggingFace Datasets](https://huggingface.co/docs/datasets/) and select 10 classes to reduce the dataset size and the model training time. We will be using [TorchVision](https://pytorch.org/vision) to transform input images and [`grain`](https://github.com/google/grain/) for efficient data loading.

```{code-cell} ipython3
from datasets import load_dataset

# Select first 10 classes to reduce the dataset size and the training time.
train_size = 10 * 750
val_size = 10 * 250

train_dataset = load_dataset("food101", split=f"train[:{train_size}]")
val_dataset = load_dataset("food101", split=f"validation[:{val_size}]")

# Let's create labels mapping where we map current labels between 0 and 9
labels_mapping = {}
index = 0
for i in range(0, len(val_dataset), 250):
    label = val_dataset[i]["label"]
    if label not in labels_mapping:
        labels_mapping[label] = index
        index += 1


print("Training dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))
```

```{code-cell} ipython3
import matplotlib.pyplot as plt


def display_datapoints(*datapoints, tag="", names_map=None):
    num_samples = len(datapoints)

    fig, axs = plt.subplots(1, num_samples, figsize=(20, 10))
    for i, datapoint in enumerate(datapoints):
        if isinstance(datapoint, dict):
            img, label = datapoint["image"], datapoint["label"]
        else:
            img, label = datapoint

        if hasattr(img, "dtype") and img.dtype in (np.float32, ):
            img = ((img - img.min()) / (img.max() - img.min()) * 255.0).astype(np.uint8)

        label_str = f" ({names_map[label]})" if names_map is not None else ""
        axs[i].set_title(f"{tag}Label: {label}{label_str}")
        axs[i].imshow(img)
```

Let's display few samples of the dataset:

```{code-cell} ipython3
display_datapoints(
    train_dataset[0], train_dataset[1000], train_dataset[2000], train_dataset[3000],
    tag="(Training) ",
    names_map=train_dataset.features["label"].names
)

display_datapoints(
    val_dataset[0], val_dataset[1000], val_dataset[2000], val_dataset[-1],
    tag="(Validation) ",
    names_map=val_dataset.features["label"].names
)
```

Let's define training and testing image preprocessing methods. Training image transformations will also contain random augmentations to prevent overfitting and make trained model more robust.

```{code-cell} ipython3
import numpy as np
from torchvision.transforms import v2 as T


img_size = 120


def to_np_array(pil_image):
  return np.asarray(pil_image.convert("RGB"))


def normalize(image):
  mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
  std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
  image = image.astype(np.float32) / 255.0
  return (image - mean) / std


tv_train_transforms = T.Compose([
    T.RandomResizedCrop((img_size, img_size), scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.Lambda(to_np_array),
    T.Lambda(normalize),
])


tv_test_transforms = T.Compose([
    T.Resize((img_size, img_size)),
    T.Lambda(to_np_array),
    T.Lambda(normalize),
])


def get_transform(fn):
    def wrapper(batch):
        batch["image"] = [
            fn(pil_image) for pil_image in batch["image"]
        ]
        # map label index between 0 - 9
        batch["label"] = [
            labels_mapping[label] for label in batch["label"]
        ]
        return batch
    return wrapper


train_transforms = get_transform(tv_train_transforms)
val_transforms = get_transform(tv_test_transforms)

train_dataset = train_dataset.with_transform(train_transforms)
val_dataset = val_dataset.with_transform(val_transforms)
```

```{code-cell} ipython3
import grain.python as grain


seed = 12
train_batch_size = 64
val_batch_size = 2 * train_batch_size


# Create an IndexSampler with no sharding for single-device computations
train_sampler = grain.IndexSampler(
    len(train_dataset),  # The total number of samples in the data source
    shuffle=True,            # Shuffle the data to randomize the order of samples
    seed=seed,               # Set a seed for reproducibility
    shard_options=grain.NoSharding(),  # No sharding since this is a single-device setup
    num_epochs=1,            # Iterate over the dataset for one epoch
)

val_sampler = grain.IndexSampler(
    len(val_dataset),  # The total number of samples in the data source
    shuffle=False,         # Do not shuffle the data
    seed=seed,             # Set a seed for reproducibility
    shard_options=grain.NoSharding(),  # No sharding since this is a single-device setup
    num_epochs=1,          # Iterate over the dataset for one epoch
)


train_loader = grain.DataLoader(
    data_source=train_dataset,
    sampler=train_sampler,                 # Sampler to determine how to access the data
    worker_count=4,                        # Number of child processes launched to parallelize the transformations among
    worker_buffer_size=2,                  # Count of output batches to produce in advance per worker
    operations=[
        grain.Batch(train_batch_size, drop_remainder=True),
    ]
)

# Validation dataset loader
val_loader = grain.DataLoader(
    data_source=val_dataset,
    sampler=val_sampler,                   # Sampler to determine how to access the data
    worker_count=4,                        # Number of child processes launched to parallelize the transformations among
    worker_buffer_size=2,
    operations=[
        grain.Batch(val_batch_size),
    ]
)
```

Let's visualize training and validation batches

```{code-cell} ipython3
train_batch = next(iter(train_loader))
val_batch = next(iter(val_loader))
```

```{code-cell} ipython3
print("Training batch info:", train_batch["image"].shape, train_batch["image"].dtype, train_batch["label"].shape, train_batch["label"].dtype)
print("Validation batch info:", val_batch["image"].shape, val_batch["image"].dtype, val_batch["label"].shape, val_batch["label"].dtype)
```

```{code-cell} ipython3
display_datapoints(
    *[(train_batch["image"][i], train_batch["label"][i]) for i in range(5)],
    tag="(Training) ",
    names_map=train_dataset.features["label"].names
)
```

```{code-cell} ipython3
display_datapoints(
    *[(val_batch["image"][i], val_batch["label"][i]) for i in range(5)],
    tag="(Validation) ",
    names_map=val_dataset.features["label"].names
)
```

## Model training

We defined training and validation datasets and the model. In this section we will train the model and define the loss function and the optimizer to perform the parameters optimization.

```{code-cell} ipython3
import optax

num_epochs = 50
learning_rate = 0.005
momentum = 0.8
total_steps = len(train_dataset) // train_batch_size

lr_schedule = optax.linear_schedule(learning_rate, 0.0, num_epochs * total_steps)

iterate_subsample = np.linspace(0, num_epochs * total_steps, 100)
plt.plot(
    np.linspace(0, num_epochs, len(iterate_subsample)),
    [lr_schedule(i) for i in iterate_subsample],
    lw=3,
)
plt.title("Learning rate")
plt.xlabel("Epochs")
plt.ylabel("Learning rate")
plt.grid()
plt.xlim((0, num_epochs))
plt.show()


optimizer = nnx.Optimizer(model, optax.sgd(lr_schedule, momentum, nesterov=True))
```

```{code-cell} ipython3
def compute_losses_and_logits(model: nnx.Module, images: jax.Array, labels: jax.Array):
    logits = model(images)

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()
    return loss, logits
```

```{code-cell} ipython3
@nnx.jit
def train_step(
    model: nnx.Module, optimizer: nnx.Optimizer, batch: dict[str, np.ndarray]
):
    # Convert np.ndarray to jax.Array on GPU
    images = jnp.array(batch["image"])
    labels = jnp.array(batch["label"], dtype=jnp.int32)

    grad_fn = nnx.value_and_grad(compute_losses_and_logits, has_aux=True)
    (loss, logits), grads = grad_fn(model, images, labels)

    optimizer.update(grads)  # In-place updates.

    return loss


@nnx.jit
def eval_step(
    model: nnx.Module, batch: dict[str, np.ndarray], eval_metrics: nnx.MultiMetric
):
    # Convert np.ndarray to jax.Array on GPU
    images = jnp.array(batch["image"])
    labels = jnp.array(batch["label"], dtype=jnp.int32)
    loss, logits = compute_losses_and_logits(model, images, labels)

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
    "val_loss": [],
    "val_accuracy": [],
}
```

```{code-cell} ipython3
import tqdm


bar_format = "{desc}[{n_fmt}/{total_fmt}]{postfix} [{elapsed}<{remaining}]"


def train_one_epoch(epoch):
    model.train()  # Set model to the training mode: e.g. update batch statistics
    with tqdm.tqdm(
        desc=f"[train] epoch: {epoch}/{num_epochs}, ",
        total=total_steps,
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
    for val_batch in val_loader:
        eval_step(model, val_batch, eval_metrics)

    for metric, value in eval_metrics.compute().items():
        eval_metrics_history[f'val_{metric}'].append(value)

    print(f"[val] epoch: {epoch + 1}/{num_epochs}")
    print(f"- total loss: {eval_metrics_history['val_loss'][-1]:0.4f}")
    print(f"- Accuracy: {eval_metrics_history['val_accuracy'][-1]:0.4f}")
```

Let's train the model.

```{code-cell} ipython3
%%time

for epoch in range(num_epochs):
    train_one_epoch(epoch)
    evaluate_model(epoch)
```

Let's visualize collected metrics:

```{code-cell} ipython3
plt.plot(train_metrics_history["train_loss"], label="Loss value during the training")
plt.legend()
```

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(10, 10))
axs[0].set_title("Loss value on validation set")
axs[0].plot(eval_metrics_history["val_loss"])
axs[1].set_title("Accuracy on validation set")
axs[1].plot(eval_metrics_history["val_accuracy"])
```

Let's also check few model's predictions on the test data:

```{code-cell} ipython3
test_indices = [1, 250, 500, 750, 1000]

test_images = jnp.array([val_dataset[i]["image"] for i in test_indices])
expected_labels = [val_dataset[i]["label"] for i in test_indices]

model.eval()
preds = model(test_images)
```

```{code-cell} ipython3
num_samples = len(test_indices)
names_map = train_dataset.features["label"].names
inv_labels_mapping = {v: k for k, v in labels_mapping.items()}

probas = nnx.softmax(preds, axis=1)
pred_labels = probas.argmax(axis=1)


fig, axs = plt.subplots(1, num_samples, figsize=(20, 10))
for i in range(num_samples):
    img, expected_label = test_images[i], expected_labels[i]

    pred_label = pred_labels[i].item()
    proba = probas[i, pred_label].item()
    if img.dtype in (np.float32, ):
        img = ((img - img.min()) / (img.max() - img.min()) * 255.0).astype(np.uint8)

    expected_label_str = names_map[inv_labels_mapping[expected_label]]
    pred_label_str = names_map[inv_labels_mapping[pred_label]]
    axs[i].set_title(f"Expected: {expected_label_str} vs \nPredicted: {pred_label_str}, P={proba:.2f}")
    axs[i].imshow(img)
```

## Further reading

In this tutorial we implemented from scratch Vision Transformer model and trained it on a subset of Food 101 dataset. Trained model shows 67% classification accuracy. Next steps could be to finetune hyperparameters like the learning rate and train for more epochs.

- Model checkpointing and exporting using [Orbax](https://orbax.readthedocs.io/en/latest/).
- Optimizers and the learning rate scheduling using [Optax](https://optax.readthedocs.io/en/latest/).
- Other Computer Vision tutorials in [jax-ai-stack](https://jax-ai-stack.readthedocs.io/en/latest/tutorials.html).

```{code-cell} ipython3

```
