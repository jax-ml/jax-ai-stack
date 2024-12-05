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

# Introduction to Data Loaders on CPU with JAX

+++ {"id": "3ia4PKEV5Dr8"}

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jax-ml/jax-ai-stack/blob/main/docs/source/data_loaders_on_cpu_with_jax.ipynb)

This tutorial explores different data loading strategies for using **JAX** on a single [**CPU**](https://jax.readthedocs.io/en/latest/glossary.html#term-CPU). While JAX doesn't include a built-in data loader, it seamlessly integrates with popular data loading libraries, including:

- [**PyTorch DataLoader**](https://github.com/pytorch/data)
- [**TensorFlow Datasets (TFDS)**](https://github.com/tensorflow/datasets)
- [**Grain**](https://github.com/google/grain)
- [**Hugging Face**](https://huggingface.co/docs/datasets/en/use_with_jax#data-loading)

In this tutorial, you'll learn how to efficiently load data using these libraries for a simple image classification task based on the MNIST dataset.

Compared to GPU or multi-device setups, CPU-based data loading is straightforward as it avoids challenges like GPU memory management and data synchronization across devices. This makes it ideal for smaller-scale tasks or scenarios where data resides exclusively on the CPU.

If you're looking for GPU-specific data loading advice, see [Data Loaders on GPU](https://jax-ai-stack.readthedocs.io/en/latest/data_loaders_on_gpu_with_jax.html).

If you're looking for a multi-device data loading strategy, see [Data Loaders on Multi-Device Setups](https://jax-ai-stack.readthedocs.io/en/latest/data_loaders_for_multi_device_setups_with_jax.html).

+++ {"id": "pEsb135zE-Jo"}

## Setting JAX to Use CPU Only

First, you'll restrict JAX to use only the CPU, even if a GPU is available. This ensures consistency and allows you to focus on CPU-based data loading.

```{code-cell}
:id: vqP6xyObC0_9

import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
```

+++ {"id": "-rsMgVtO6asW"}

Import JAX API

```{code-cell}
:id: tDJNQ6V-Dg5g

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
```

+++ {"id": "TsFdlkSZKp9S"}

### CPU Setup Verification

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: N3sqvaF3KJw1
outputId: 449c83d9-d050-4b15-9a8d-f71e340501f2
---
jax.devices()
```

+++ {"id": "qyJ_WTghDnIc"}

## Setting Hyperparameters and Initializing Parameters

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
step_size = 0.01                   # Learning rate for optimization
num_epochs = 8                     # Number of training epochs
batch_size = 128                   # Batch size for training
n_targets = 10                     # Number of classes (digits 0-9)
num_pixels = 28 * 28               # Input size (MNIST images are 28x28 pixels)
data_dir = '/tmp/mnist_dataset'    # Directory for storing the dataset

# Initialize network parameters using the defined layer sizes and a random seed
params = init_network_params(layer_sizes, random.PRNGKey(0))
```

+++ {"id": "6Ci_CqW7q6XM"}

## Model Prediction with Auto-Batching

In this section, you'll define the `predict` function for your neural network. This function computes the output of the network for a single input image.

To efficiently process multiple images simultaneously, you'll use [`vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap), which allows you to vectorize the `predict` function and apply it across a batch of inputs. This technique, called auto-batching, improves computational efficiency by leveraging hardware acceleration.

```{code-cell}
:id: bKIYPSkvD1QV

from jax.scipy.special import logsumexp

def relu(x):
  return jnp.maximum(0, x)

def predict(params, image):
  # per-example prediction
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

+++ {"id": "niTSr34_sDZi"}

## Utility and Loss Functions

You'll now define utility functions for:

- One-hot encoding: Converts class indices to binary vectors.
- Accuracy calculation: Measures the performance of the model on the dataset.
- Loss computation: Calculates the difference between predictions and targets.

To optimize performance:

- [`grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html#jax.grad) is used to compute gradients of the loss function with respect to network parameters.
- [`jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit) compiles the update function, enabling faster execution by leveraging JAX's [XLA](https://openxla.org/xla) compilation.

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
    """Train the model for a given number of epochs."""
    for epoch in range(num_epochs):
        start_time = time.time()
        for x, y in training_generator() if data_loader_type == 'streamed' else training_generator:
            x, y = reshape_and_one_hot(x, y)
            params = update(params, x, y)

        print(f"Epoch {epoch + 1} in {time.time() - start_time:.2f} sec: "
              f"Train Accuracy: {accuracy(params, train_images, train_labels):.4f}, "
              f"Test Accuracy: {accuracy(params, test_images, test_labels):.4f}")
```

+++ {"id": "Hsionp5IYsQ9"}

## Loading Data with PyTorch DataLoader

This section shows how to load the MNIST dataset using PyTorch's DataLoader, convert the data to NumPy arrays, and apply transformations to flatten and cast images.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: jmsfrWrHxIhC
outputId: 33dfeada-a763-4d26-f778-a27966e34d55
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
  """Convert a batch of PyTorch data to NumPy arrays."""
  return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
    """Custom DataLoader to return NumPy arrays from a PyTorch Dataset."""
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=numpy_collate, **kwargs)

class FlattenAndCast(object):
  """Transform class to flatten and cast images to float32."""
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))
```

+++ {"id": "mfSnfJND6I8G"}

### Load Dataset with Transformations

Standardize the data by flattening the images, casting them to `float32`, and ensuring consistent data types.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Kxbl6bcx6crv
outputId: 372bbf4c-3ad5-4fd8-cc5d-27b50f5e4f38
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
outputId: abbaa26d-491a-4e63-e8c9-d3c571f53a28
---
print('Train:', train_images.shape, train_labels.shape)
print('Test:', test_images.shape, test_labels.shape)
```

+++ {"id": "m3zfxqnMiCbm"}

### Training Data Generator

Define a generator function using PyTorch's DataLoader for batch training. Setting `num_workers > 0` enables multi-process data loading, which can accelerate data loading for larger datasets or intensive preprocessing tasks. Experiment with different values to find the optimal setting for your hardware and workload.

Note: When setting `num_workers > 0`, you may see the following `RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.` This warning can be safely ignored since data loaders do not use JAX within the forked processes.

```{code-cell}
:id: B-fES82EiL6Z

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
id: vtUjHsh-rJs8
outputId: 4766333e-4366-493b-995a-102778d1345a
---
train_model(num_epochs, params, pytorch_training_generator(mnist_dataset), data_loader_type='iterable')
```

+++ {"id": "Nm45ZTo6yrf5"}

## Loading Data with TensorFlow Datasets (TFDS)

This section demonstrates how to load the MNIST dataset using TFDS, fetch the full dataset for evaluation, and define a training generator for batch processing. GPU usage is explicitly disabled for TensorFlow.

```{code-cell}
:id: sGaQAk1DHMUx

import tensorflow_datasets as tfds
import tensorflow as tf

# Ensuring CPU-Only Execution, disable any GPU usage(if applicable) for TF
tf.config.set_visible_devices([], device_type='GPU')
```

+++ {"id": "3xdQY7H6wr3n"}

### Fetch Full Dataset for Evaluation

Load the dataset with `tfds.load`, convert it to NumPy arrays, and process it for evaluation.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 104
  referenced_widgets: [b8cdabf5c05848f38f03850cab08b56f, a8b76d5f93004c089676e5a2a9b3336c,
    119ac8428f9441e7a25eb0afef2fbb2a, 76a9815e5c2b4764a13409cebaf66821, 45ce8dd5c4b949afa957ec8ffb926060,
    05b7145fd62d4581b2123c7680f11cdd, b96267f014814ec5b96ad7e6165104b1, bce34bdbfbd64f1f8353a4e8515cee0b,
    93b8206f8c5841a692cdce985ae301d8, c95f592620c64da595cc787567b2c4db, 8a97071f862c4ec3b4b4140d2e34eda2]
id: 1hOamw_7C8Pb
outputId: ca166490-22db-4732-b29f-866b7593e489
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
outputId: 96403b0f-6079-43ce-df16-d4583f09906b
---
print('Train:', train_images.shape, train_labels.shape)
print('Test:', test_images.shape, test_labels.shape)
```

+++ {"id": "UWRSaalfdyDX"}

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
id: h2sO13XDGvq1
outputId: a150246e-ceb5-46ac-db71-2a8177a9d04d
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
outputId: 76d16565-0d9e-4f5f-c6b1-4cf4a683d0e7
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

# Convert training data to JAX arrays and encode labels as one-hot vectors
train_images = jnp.array([mnist_dataset[i][0] for i in range(len(mnist_dataset))], dtype=jnp.float32)
train_labels = one_hot(np.array([mnist_dataset[i][1] for i in range(len(mnist_dataset))]), n_targets)

# Load test dataset and process it
mnist_dataset_test = MNIST(data_dir, download=True, train=False)
test_images = jnp.array([np.ravel(np.array(mnist_dataset_test[i][0], dtype=np.float32)) for i in range(len(mnist_dataset_test))], dtype=jnp.float32)
test_labels = one_hot(np.array([mnist_dataset_test[i][1] for i in range(len(mnist_dataset_test))]), n_targets)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: a2NHlp9klrQL
outputId: 14be58c0-851e-4a44-dfcc-d02f0718dab5
---
print("Train:", train_images.shape, train_labels.shape)
print("Test:", test_images.shape, test_labels.shape)
```

+++ {"id": "fETnWRo2crhf"}

### Initialize PyGrain DataLoader

Set up a PyGrain DataLoader for sequential batch sampling.

```{code-cell}
:id: 9RuFTcsCs2Ac

sampler = pygrain.SequentialSampler(
    num_records=len(mnist_dataset),
    shard_options=pygrain.NoSharding()) # Single-device, no sharding

def pygrain_training_generator():
    """Grain DataLoader generator for training."""
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
outputId: 3f624366-b683-4d20-9d0a-777d345b0e21
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
outputId: 684e445f-d23e-4924-9e76-2c2c9359f0be
---
!pip install datasets
```

+++ {"id": "be0h_dZv0593"}

Import Library

```{code-cell}
:id: 8v1N59p76zn0

from datasets import load_dataset
```

+++ {"id": "8Gaj11tO7C86"}

### Load and Format MNIST Dataset

Load the MNIST dataset from Hugging Face and format it as `numpy` arrays for quick access or `jax` to get JAX arrays.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 301
  referenced_widgets: [32f6132a31aa4c508d3c3c5ef70348bb, d7c2ffa6b143463c91cbf8befca6ca01,
    fd964ecd3926419d92927c67f955d5d0, 60feca3fde7c4447ad8393b0542eb999, 3354a0baeca94d18bc6b2a8b8b465b58,
    a0d0d052772b46deac7657ad052991a4, fb34783b9cba462e9b690e0979c4b07a, 8d8170c1ed99490589969cd753c40748,
    f1ecb6db00a54e088f1e09164222d637, 3cf5dd8d29aa4619b39dc2542df7e42e, 2e5d42ca710441b389895f2d3b611d0a,
    5d8202da24244dc896e9a8cba6a4ed4f, a6d64c953631412b8bd8f0ba53ae4d32, 69240c5cbfbb4e91961f5b49812a26f0,
    865f38532b784a7c971f5d33b87b443e, ceb1c004191947cdaa10af9b9c03c80d, 64c6041037914779b5e8e9cf5a80ad04,
    562fa6a0e7b846a180ac4b423c5511c5, b3b922288f9c4df2a4088279ff6d1531, 75a1a8ffda554318890cf74c345ed9a9,
    3bae06cacf394a5998c2326199da94f5, ff6428a3daa5496c81d5e664aba01f97, 1ba3f86870724f55b94a35cb6b4173af,
    b3e163fd8b8a4f289d5a25611cb66d23, abd2daba215e4f7c9ddabde04d6eb382, e22ee019049144d5aba573cdf4dbe4fc,
    6ac765dac67841a69218140785f024c6, 7b057411a54e434fb74804b90daa8d44, 563f71b3c67d47c3ab1100f5dc1b98f3,
    d81a657361ab4bba8bcc0cf309d2ff64, 20316312ab88471ba90cbb954be3e964, 698fda742f834473a23fb7e5e4cf239c,
    289b52c5a38146b8b467a5f4678f6271, d07c2f37cf914894b1551a8104e6cb70, 5b55c73d551d483baaa6a1411c2597b1,
    2308f77723f54ac898588f48d1853b65, 54d2589714d04b2e928b816258cb0df4, f84b795348c04c7a950165301a643671,
    bc853a4a8d3c4dbda23d183f0a3b4f27, 1012ddc0343842d8b913a7d85df8ab8f, 771a73a8f5084a57afc5654d72e022f0,
    311a43449f074841b6df4130b0871ac9, cd4d29cb01134469b52d6936c35eb943, 013cf89ee6174d29bb3f4fdff7b36049,
    9237d877d84e4b3ab69698ecf56915bb, 337ef4d37e6b4ff6bf6e8bd4ca93383f, b4096d3837b84ccdb8f1186435c87281,
    7259d3b7e11b4736b4d2aa8e9c55e994, 1ad1f8e99a864fc4a2bc532d9a4ff110, b2b50451eabd40978ef46db5e7dd08c4,
    2dad5c5541e243128e23c3dd3e420ac2, a3de458b61e5493081d6bb9cf7e923db, 37760f8a7b164e6f9c1a23d621e9fe6b,
    745a2aedcfab491fb9cffba19958b0c5, 2f6c670640d048d2af453638cfde3a1e]
id: a22kTvgk6_fJ
outputId: 35fc38b9-a6ab-4b02-ffa4-ab27fac69df4
---
mnist_dataset = load_dataset("mnist").with_format("numpy")
```

+++ {"id": "IFjTyGxY19b0"}

### Extract images and labels

Get image shape and flatten for model input

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: NHrKatD_7HbH
outputId: deec1739-2fc0-4e71-8567-f2e0c9db198b
---
train_images = mnist_dataset["train"]["image"]
train_labels = mnist_dataset["train"]["label"]
test_images = mnist_dataset["test"]["image"]
test_labels = mnist_dataset["test"]["label"]

# Flatten images and one-hot encode labels
image_shape = train_images.shape[1:]
num_features = image_shape[0] * image_shape[1]

train_images = train_images.reshape(-1, num_features)
test_images = test_images.reshape(-1, num_features)

train_labels = one_hot(train_labels, n_targets)
test_labels = one_hot(test_labels, n_targets)

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
id: RhloYGsw6nPf
outputId: d49c1cd2-a546-46a6-84fb-d9507c38f4ca
---
train_model(num_epochs, params, hf_training_generator)
```

+++ {"id": "qXylIOwidWI3"}

## Summary

This notebook has introduced efficient strategies for data loading on a CPU with JAX, demonstrating how to integrate popular libraries like PyTorch DataLoader, TensorFlow Datasets, Grain, and Hugging Face Datasets. Each library offers distinct advantages, enabling you to streamline the data loading process for machine learning tasks. By understanding the strengths of these methods, you can select the approach that best suits your project's specific requirements.
