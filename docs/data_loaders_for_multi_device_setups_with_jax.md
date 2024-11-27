---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "PUFGZggH49zp"}

# Introduction to Data Loaders for Multi-Device Training with JAX

+++ {"id": "3ia4PKEV5Dr8"}

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jax-ml/jax-ai-stack/blob/main/docs/data_loaders_for_multi_device_setups_with_jax.ipynb)

This tutorial explores various data loading strategies for **JAX** in **multi-device distributed** environments, leveraging [**TPUs**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html#what-is-a-tpu). While JAX doesn't include a built-in data loader, it seamlessly integrates with popular data loading libraries, including:
*   [**PyTorch DataLoader**](https://github.com/pytorch/data)
*   [**TensorFlow Datasets (TFDS)**](https://github.com/tensorflow/datasets)
*   [**Grain**](https://github.com/google/grain)
*   [**Hugging Face**](https://huggingface.co/docs/datasets/en/use_with_jax#data-loading)

You'll see how to use each of these libraries to efficiently load data for a simple image classification task using the MNIST dataset.

Building on the [Data Loaders on GPU](https://jax-ai-stack.readthedocs.io/en/latest/data_loaders_on_gpu_with_jax.html) tutorial, this guide introduces optimizations for distributed training across multiple GPUs or TPUs. It focuses on data sharding with `Mesh` and `NamedSharding` to efficiently partition and synchronize data across devices. By leveraging multi-device setups, you'll maximize resource utilization for large datasets in distributed environments.

+++ {"id": "-rsMgVtO6asW"}

Import JAX API

```{code-cell}
:id: tDJNQ6V-Dg5g

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random, device_put
from jax.sharding import Mesh, PartitionSpec, NamedSharding
```

+++ {"id": "TsFdlkSZKp9S"}

### Checking TPU Availability for JAX

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: N3sqvaF3KJw1
outputId: ee3286d0-b75f-46c5-8548-b57e3d895dd7
---
jax.devices()
```

+++ {"id": "qyJ_WTghDnIc"}

### Setting Hyperparameters and Initializing Parameters

You'll define hyperparameters for your model and data loading, including layer sizes, learning rate, batch size, and the data directory. You'll also initialize the weights and biases for a fully-connected neural network.

```{code-cell}
:id: qLNOSloFDka_

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Function to initialize network parameters for all layers based on defined sizes
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

layer_sizes = [784, 512, 512, 10]  # Layers of the network
step_size = 0.01                   # Learning rate
num_epochs = 8                     # Number of training epochs
batch_size = 128                   # Batch size for training
n_targets = 10                     # Number of classes (digits 0-9)
num_pixels = 28 * 28               # Each MNIST image is 28x28 pixels
data_dir = '/tmp/mnist_dataset'    # Directory for storing the dataset

# Initialize network parameters using the defined layer sizes and a random seed
params = init_network_params(layer_sizes, random.PRNGKey(0))
```

+++ {"id": "rHLdqeI7D2WZ"}

### Model Prediction with Auto-Batching

In this section, you'll define the `predict` function for your neural network. This function computes the output of the network for a single input image.

To efficiently process multiple images simultaneously, you'll use [`vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap), which allows you to vectorize the `predict` function and apply it across a batch of inputs. This technique, called auto-batching, improves computational efficiency by leveraging hardware acceleration.

```{code-cell}
:id: bKIYPSkvD1QV

from jax.scipy.special import logsumexp

def relu(x):
  return jnp.maximum(0, x)

def predict(params, image):
  # per-example predictions
  activations = image
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = relu(outputs)

  final_w, final_b = params[-1]
  logits = jnp.dot(final_w, activations) + final_b
  return logits - logsumexp(logits)

# Make a batched version of the `predict` function
batched_predict = vmap(predict, in_axes=(None, 0))
```

+++ {"id": "AMWmxjVEpH2D"}

Multi-device setup using a Mesh of devices

```{code-cell}
:id: 4Jc5YLFnpE-_

# Get the number of available devices (GPUs/TPUs) for sharding
num_devices = len(jax.devices())

# Multi-device setup using a Mesh of devices
devices = jax.devices()
mesh = Mesh(devices, ('device',))

# Define the sharding specification - split the data along the first axis (batch)
sharding_spec = PartitionSpec('device')
```

+++ {"id": "rLqfeORsERek"}

### Utility and Loss Functions

You'll now define utility functions for:
- One-hot encoding: Converts class indices to binary vectors.
- Accuracy calculation: Measures the performance of the model on the dataset.
- Loss computation: Calculates the difference between predictions and targets.

To optimize performance:
- [`grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html#jax.grad) is used to compute gradients of the loss function with respect to network parameters.
- [`jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit) compiles the update function, enabling faster execution by leveraging JAX's [XLA](https://openxla.org/xla) compilation.

- [`device_put`](https://jax.readthedocs.io/en/latest/_autosummary/jax.device_put.html) to distribute the dataset across TPU cores.

```{code-cell}
:id: sA0a06raEQfS

import time

def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(params, images, targets):
  """Calculate the accuracy of predictions."""
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
  return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
  """Calculate the loss between predictions and targets."""
  preds = batched_predict(params, images)
  return -jnp.mean(preds * targets)

@jit
def update(params, x, y):
  """Update the network parameters using gradient descent."""
  grads = grad(loss)(params, x, y)
  return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]

def reshape_and_one_hot(x, y):
    """Reshape and one-hot encode the inputs."""
    x = jnp.reshape(x, (len(x), num_pixels))
    y = one_hot(y, n_targets)
    return x, y

def train_model(num_epochs, params, training_generator, data_loader_type='streamed'):
    """Train the model for a given number of epochs and device_put for TPU transfer."""
    for epoch in range(num_epochs):
        start_time = time.time()
        for x, y in training_generator() if data_loader_type == 'streamed' else training_generator:
            x, y = reshape_and_one_hot(x, y)
            x, y = device_put(x, NamedSharding(mesh, sharding_spec)), device_put(y, NamedSharding(mesh, sharding_spec))
            params = update(params, x, y)

        print(f"Epoch {epoch + 1} in {time.time() - start_time:.2f} sec: "
              f"Train Accuracy: {accuracy(params, train_images, train_labels):.4f},"
              f"Test Accuracy: {accuracy(params, test_images, test_labels):.4f}")
```

+++ {"id": "Hsionp5IYsQ9"}

## Loading Data with PyTorch DataLoader

This section shows how to load the MNIST dataset using PyTorch's DataLoader, convert the data to NumPy arrays, and apply transformations to flatten and cast images.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 33Wyf77WzNjA
outputId: a2378431-79f2-4dc4-aa1a-d98704657d26
---
!pip install torch torchvision
```

```{code-cell}
:id: kO5_WzwY59gE

import numpy as np
from jax.tree_util import tree_map
from torch.utils import data
from torchvision.datasets import MNIST
```

```{code-cell}
:id: 6f6qU8PCc143

def numpy_collate(batch):
  """Collate function to convert a batch of PyTorch data into NumPy arrays."""
  return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
    """Custom DataLoader to return NumPy arrays from a PyTorch Dataset."""
    def __init__(self, dataset, batch_size=1,
                  shuffle=False, sampler=None,
                  batch_sampler=None, num_workers=0,
                  pin_memory=False, drop_last=False,
                  timeout=0, worker_init_fn=None):
      super(self.__class__, self).__init__(dataset,
          batch_size=batch_size,
          shuffle=shuffle,
          sampler=sampler,
          batch_sampler=batch_sampler,
          num_workers=num_workers,
          collate_fn=numpy_collate,
          pin_memory=pin_memory,
          drop_last=drop_last,
          timeout=timeout,
          worker_init_fn=worker_init_fn)
class FlattenAndCast(object):
  """Transform class to flatten and cast images to float32."""
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))
```

+++ {"id": "ec-MHhv6hYsK"}

### Load Dataset with Transformations

Standardize the data by flattening the images, casting them to `float32`, and ensuring consistent data types.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: nSviwX9ohhUh
outputId: 0bb3bc04-11ac-4fb6-8854-76a3f5e725a5
---
mnist_dataset = MNIST(data_dir, download=True, transform=FlattenAndCast())
```

+++ {"id": "kbdsqvPZGrsa"}

### Full Training Dataset for Accuracy Checks

Convert the entire training dataset to JAX arrays.

```{code-cell}
:id: c9ZCJq_rzPck

train_images = jnp.array(mnist_dataset.data.numpy().reshape(len(mnist_dataset.data), -1), dtype=jnp.float32)
train_labels = one_hot(np.array(mnist_dataset.targets), n_targets)
```

+++ {"id": "WXUh0BwvG8Ko"}

### Get Full Test Dataset

Load and process the full test dataset.

```{code-cell}
:id: brlLG4SqGphm

mnist_dataset_test = MNIST(data_dir, download=True, train=False)
test_images = jnp.array(mnist_dataset_test.data.numpy().reshape(len(mnist_dataset_test.data), -1), dtype=jnp.float32)
test_labels = one_hot(np.array(mnist_dataset_test.targets), n_targets)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Oz-UVnCxG5E8
outputId: 0f44cb63-b12c-47a7-8bd5-ed773e2b2ec5
---
print('Train:', train_images.shape, train_labels.shape)
print('Test:', test_images.shape, test_labels.shape)
```

+++ {"id": "mfSnfJND6I8G"}

### Training Data Generator

Define a generator function using PyTorch's DataLoader for batch training.
Setting `num_workers > 0` enables multi-process data loading, which can accelerate data loading for larger datasets or intensive preprocessing tasks. Experiment with different values to find the optimal setting for your hardware and workload.

Note: When setting `num_workers > 0`, you may see the following `RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.`
This warning can be safely ignored since data loaders do not use JAX within the forked processes.

```{code-cell}
:id: Kxbl6bcx6crv

def pytorch_training_generator(mnist_dataset):
    return NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)
```

+++ {"id": "Xzt2x9S1HC3T"}

### Training Loop (PyTorch DataLoader)

The training loop uses the PyTorch DataLoader to iterate through batches and update model parameters.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: MUrJxpjvUyOm
outputId: 629a19b1-acba-418a-f04b-3b78d7909de1
---
train_model(num_epochs, params, pytorch_training_generator(mnist_dataset), data_loader_type='iterable')
```

+++ {"id": "ACy1PoSVa3zH"}

## Loading Data with TensorFlow Datasets (TFDS)

This section demonstrates how to load the MNIST dataset using TFDS, fetch the full dataset for evaluation, and define a training generator for batch processing. GPU usage is explicitly disabled for TensorFlow.

+++ {"id": "tcJRzpyOveWK"}

Ensure you have the latest versions of both TensorFlow and TensorFlow Datasets

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: _f55HPGAZu6P
outputId: 838c8f76-aa07-49d5-986d-3c88ed516b22
---
!pip install --upgrade tensorflow tensorflow-datasets
```

```{code-cell}
:id: sGaQAk1DHMUx

import tensorflow_datasets as tfds
```

+++ {"id": "F6OlzaDqwe4p"}

### Fetch Full Dataset for Evaluation

Load the dataset with `tfds.load`, convert it to NumPy arrays, and process it for evaluation.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 104
  referenced_widgets: [43d95e3e6b704cb5ae941541862e35fe, fca543b71352477db00545b3990d44fa,
    d3c971a3507249c9a22cad026e46d739, 6da776e94f7740b9aae06f298c1e03cd, b4aec5e3895e4a19912c74777e9ea835,
    ef4dc5b756d74129bd2d643d99a1ab2e, 30243b81748e497eb526b25404e95826, 3bb9b93e595d4a0ca973ded476c0a5d0,
    b770951ecace4b02ad1575fe9eb9e640, 79009c4ea2bf46b1a3a2c6558fa6ec2f, 5cb081d3a038482583350d018a768bd4]
id: 1hOamw_7C8Pb
outputId: 0e3805dc-1bfd-4222-9052-0b2111ea3091
---
# tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True)
mnist_data = tfds.as_numpy(mnist_data)
train_data, test_data = mnist_data['train'], mnist_data['test']

# Full train set
train_images, train_labels = train_data['image'], train_data['label']
train_images = jnp.reshape(train_images, (len(train_images), num_pixels))
train_labels = one_hot(train_labels, n_targets)

# Full test set
test_images, test_labels = test_data['image'], test_data['label']
test_images = jnp.reshape(test_images, (len(test_images), num_pixels))
test_labels = one_hot(test_labels, n_targets)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Td3PiLdmEf7z
outputId: 464da4f6-f028-4667-889d-a812382739b0
---
print('Train:', train_images.shape, train_labels.shape)
print('Test:', test_images.shape, test_labels.shape)
```

+++ {"id": "yy9PunCJdI-G"}

### Define the Training Generator

Create a generator function to yield batches of data for training.

```{code-cell}
:id: vX59u8CqEf4J

def training_generator():
  # as_supervised=True gives us the (image, label) as a tuple instead of a dict
  ds = tfds.load(name='mnist', split='train', as_supervised=True, data_dir=data_dir)
  # You can build up an arbitrary tf.data input pipeline
  ds = ds.batch(batch_size).prefetch(1)
  # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
  return tfds.as_numpy(ds)
```

+++ {"id": "EAWeUdnuFNBY"}

### Training Loop (TFDS)

Use the training generator in a custom training loop.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: AsFKboVRaV6r
outputId: 9cb33f79-1b17-439d-88d3-61cd984124f6
---
train_model(num_epochs, params, training_generator)
```

+++ {"id": "-ryVkrAITS9Z"}

## Loading Data with Grain

This section demonstrates how to load MNIST data using Grain, a data-loading library. You'll define a custom dataset class for Grain and set up a Grain DataLoader for efficient training.

+++ {"id": "waYhUMUGmhH-"}

Install Grain

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: L78o7eeyGvn5
outputId: 8f32bb0f-9a73-48a9-dbcd-4eb93ba3f606
---
!pip install grain
```

+++ {"id": "66bH3ZDJ7Iat"}

Import Required Libraries (import MNIST dataset from torchvision)

```{code-cell}
:id: mS62eVL9Ifmz

import numpy as np
import grain.python as pygrain
from torchvision.datasets import MNIST
```

+++ {"id": "0h6mwVrspPA-"}

### Define Dataset Class

Create a custom dataset class to load MNIST data for Grain.

```{code-cell}
:id: bnrhac5Hh7y1

class Dataset:
    def __init__(self, data_dir, train=True):
        self.data_dir = data_dir
        self.train = train
        self.load_data()

    def load_data(self):
        # Load the MNIST dataset using PyGrain
        self.dataset = MNIST(self.data_dir, download=True, train=self.train)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return np.ravel(np.array(img, dtype=np.float32)), label
```

+++ {"id": "53mf8bWEsyTr"}

### Initialize the Dataset

```{code-cell}
:id: pN3oF7-ostGE

mnist_dataset = Dataset(data_dir)
```

+++ {"id": "GqD-ycgBuwv9"}

### Get the full train and test dataset

```{code-cell}
:id: f1VnTuX3u_kL

train_images = jnp.array([mnist_dataset[i][0] for i in range(len(mnist_dataset))], dtype=jnp.float32)
train_labels = one_hot(np.array([mnist_dataset[i][1] for i in range(len(mnist_dataset))]), n_targets)

mnist_dataset_test = MNIST(data_dir, download=True, train=False)

# Convert test images to JAX arrays and encode test labels as one-hot vectors
test_images = jnp.array([np.ravel(np.array(mnist_dataset_test[i][0], dtype=np.float32)) for i in range(len(mnist_dataset_test))], dtype=jnp.float32)
test_labels = one_hot(np.array([mnist_dataset_test[i][1] for i in range(len(mnist_dataset_test))]), n_targets)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: a2NHlp9klrQL
outputId: cc9e0958-8484-4669-a2d1-abac36a3097f
---
print("Train:", train_images.shape, train_labels.shape)
print("Test:", test_images.shape, test_labels.shape)
```

+++ {"id": "1QPbXt7O0JN-"}

### Initialize PyGrain DataLoader

```{code-cell}
:id: 9RuFTcsCs2Ac

sampler = pygrain.SequentialSampler(
    num_records=len(mnist_dataset),
    shard_options=pygrain.ShardByJaxProcess())  # Shard across TPU cores

def pygrain_training_generator():
    return pygrain.DataLoader(
        data_source=mnist_dataset,
        sampler=sampler,
        operations=[pygrain.Batch(batch_size)],
    )
```

+++ {"id": "GvpJPHAbeuHW"}

### Training Loop (Grain)

Run the training loop using the Grain DataLoader.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: cjxJRtiTadEI
outputId: a620e9f7-7a01-4ba8-fe16-6f988401c7c1
---
train_model(num_epochs, params, pygrain_training_generator)
```

+++ {"id": "oixvOI816qUn"}

## Loading Data with Hugging Face

This section demonstrates loading MNIST data using the Hugging Face `datasets` library. You'll format the dataset for JAX compatibility, prepare flattened images and one-hot-encoded labels, and define a training generator.

+++ {"id": "o51P6lr86wz-"}

Install the Hugging Face `datasets` library.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 19ipxPhI6oSN
outputId: e0d52dfb-6c60-4539-a043-574d2533a744
---
!pip install datasets
```

```{code-cell}
:id: 8v1N59p76zn0

from datasets import load_dataset
```

+++ {"id": "8Gaj11tO7C86"}

Load the MNIST dataset from Hugging Face and format it as `numpy` arrays for quick access or `jax` to get JAX arrays.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 301
  referenced_widgets: [86617153e14143c6900da3535b74ef07, 8de57c9ecba14aa5b1f642af5c7e9094,
    515fe154b1b74ed981e877aef503aa99, 4e291a8b028847328ea1d9a650c20beb, 87a0c8cdc0ad423daba7082b985cbd2b,
    4764b5b806b94734b760cf6cc2fc224d, 5307bf3142804235bb688694c517d80c, 6a2fd6755667443abe7710ad607a79cc,
    91bc1755904e40db8d758db4d09754e3, 69c38d75960542fb83fa087cae761957, dc31cb349c9b4c3580b2b77cbad1325c,
    d451224a0ce540648b0c28d433d85803, 52f2f12dcffe4507ab92286fd3810db6, 6ab919475c80413e94afa66304b05338,
    305d05093c6e411cb438a0bbf122d574, aa11f21e68994a8d9ddead215f2f4920, 59a7233abf61461b8b3feeb31b2f544f,
    9d909399be9a4fa48bc3d781905c7f5a, 5b6172eb4e0541a3b07d4f82de77a303, bc3bec617b0040f487f80134537a3068,
    9fe417f8159244f8ac808f2844922cf3, c4748e35e8574bb286a527295df98c8e, f50572e8058c4864bb8143c364d191f9,
    436955f611674e27b4ddf3e040cc5ce9, 048231bf788c447091b8ef0174101f42, 97009f7e20d84c7c9d89f7497efc494c,
    84e2844437884f6c89683e6545a2262e, df3019cc6aa44a4cbcb62096444769a7, ce17fe81850c49cd924297d21ecda621,
    422117e32e0b4a95bed7925c99fd9f78, 56ab1fa0212a43a4a70838e440be0e9c, 1c5483472cea483bbf2a8fe2a9182ce0,
    00034cb6a66143d8a87922befb1da7a6, 368b51d79aed4184854f155e2951da81, eb9de18be48d4a0db1034a38a0287ea6,
    dbec1d9b196849a5ad79a5f083dbe64e, 66db6915d27b4fb49e1b44f70cb61654, 80f3e3a30dc24d3fa54bb72dc1c60182,
    c320096ba1e74c7bbbd9509cc11c22e9, a664dd9c446040e8b175bb91d1c051db, 66c7826ff9b4455db9f7e9717a432f73,
    74ec8cec0f3c4c04b76f5fb87ea2d9bb, ea4537aef1e247378de1935ad50ef76c, a9cffb2f5e194dfaba516bb4c8c47e3f,
    4f17b7ab6ae94ce3b122561bcd8d4427, 3c0bdc06fe07412bacc00daa6f1eec34, 1ba273ced1484bcf9855366ff0dc3645,
    7413d8bab616446ba6b820a3f874f6a0, 53c160c26c634b53a914be18ed91016c, ebc4ad2fae264e72a5307a0481a97ab3,
    83ab5e7617fb45898c259bc20f71e958, 21f1138e807e4946953e3074d72d9a27, 86d7357878634706b9e214103efa262a,
    3713a0e1880a43bc8b23225dbb8b4c45, f9f85ce1cbf34a7da27804ce7cc6444e]
id: a22kTvgk6_fJ
outputId: 53e1d208-5360-479b-c097-0c03c7fac3e8
---
mnist_dataset = load_dataset("mnist", cache_dir=data_dir).with_format("numpy")
```

+++ {"id": "tgI7dIaX7JzM"}

### Extract images and labels

Get image shape and flatten for model input.

```{code-cell}
:id: NHrKatD_7HbH

train_images = mnist_dataset["train"]["image"]
train_labels = mnist_dataset["train"]["label"]
test_images = mnist_dataset["test"]["image"]
test_labels = mnist_dataset["test"]["label"]

# Extract image shape
image_shape = train_images.shape[1:]
num_features = image_shape[0] * image_shape[1]

# Flatten the images
train_images = train_images.reshape(-1, num_features)
test_images = test_images.reshape(-1, num_features)

# One-hot encode the labels
train_labels = one_hot(train_labels, n_targets)
test_labels = one_hot(test_labels, n_targets)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: dITh435Z7Nwb
outputId: cd77ebf6-7d44-420f-f8d8-4357f915c956
---
print('Train:', train_images.shape, train_labels.shape)
print('Test:', test_images.shape, test_labels.shape)
```

+++ {"id": "kk_4zJlz7T1E"}

### Define Training Generator

Set up a generator to yield batches of images and labels for training.

```{code-cell}
:id: -zLJhogj7RL-

def hf_training_generator():
    """Yield batches for training."""
    for batch in mnist_dataset["train"].iter(batch_size):
        x, y = batch["image"], batch["label"]
        yield x, y
```

+++ {"id": "HIsGfkLI7dvZ"}

### Training Loop (Hugging Face Datasets)

Run the training loop using the Hugging Face training generator.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Ui6aLiZP7aLe
outputId: 48347baf-30f2-443d-b3bf-b12100d96b8f
---
train_model(num_epochs, params, hf_training_generator)
```

+++ {"id": "_JR0V1Aix9Id"}

## Summary

This notebook has introduced efficient methods for multi-device distributed data loading on TPUs with JAX. You explored how to leverage popular libraries like PyTorch DataLoader, TensorFlow Datasets, Grain, and Hugging Face Datasets to streamline the data loading process for machine learning tasks. Each library offers distinct advantages, allowing you to select the best approach for your specific project needs.

For more detailed strategies on distributed data loading with JAX, including global data pipelines and per-device processing, refer to the [Distributed Data Loading Guide](https://jax.readthedocs.io/en/latest/distributed_data_loading.html).
