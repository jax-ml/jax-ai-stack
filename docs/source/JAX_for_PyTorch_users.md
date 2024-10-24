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

+++ {"id": "muUJUDNosKla"}

# JAX for PyTorch users

This is a quick overview of JAX and the JAX AI stack written for those who are famiilar with PyTorch.

First, we cover how to manipulate JAX Arrays following the [well-known PyTorch's tensors tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html). Next, we explore automatic differentiation with JAX, followed by how to build a model and optimize its parameters.
Finally, we will introduce `jax.jit` and compare it to its PyTorch counterpart `torchscript`.

## Setup

Let's get started by importing JAX and checking the installed version.
For details on how to install JAX check [installation guide](https://jax.readthedocs.io/en/latest/installation.html).

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: f_hNMIu0sL5l
outputId: 838ea182-2a22-4e9a-b3bc-3d91506e9ecf
---
import jax
import jax.numpy as jnp
print(jax.__version__)
```

+++ {"id": "LNBvB_hRDteB"}

## JAX Arrays manipulation

In this section, we will learn about JAX Arrays and how to manipulate them compared to PyTorch tensors.

### Initializing a JAX Array

The primary array object in JAX is the `jax.Array`, which is the JAX counterpart of `torch.Tensor`.
As with `torch.Tensor`, `jax.Array` objects are never constructed directly, but rather constructed via array creation APIs that populate the new array with constant numbers, random numbers, or data drawn from lists, numpy arrays, torch tensors, and more.
Let's see some examples of this.

To initialize an array from a Python data:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 9J4m79evD0fJ
outputId: 7b8196fb-4f16-4c26-864f-e3c08697fe19
---
# From data
data = [[1, 2, 3], [3, 4, 5]]
x_array = jnp.array(data)
assert isinstance(x_array, jax.Array)
print(x_array, x_array.shape, x_array.dtype)
```

Or from an existing NumPy array:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 7ofhffR0FJO7
outputId: a528065c-6298-41db-efca-a874acf8f2c5
---
import numpy as np

np_array = np.array(data)
x_np = jnp.array(np_array)
assert isinstance(x_np, jax.Array)
print(x_np, x_np.shape, x_np.dtype)
# x_np is a copy of np_array
```

You can create arrays with the same shape and `dtype` as existing JAX Arrays:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: BTjssf9rFf0A
outputId: ce963ac0-c25c-4513-dde1-70fe09d910b6
---
x_ones = jnp.ones_like(x_array)
print(x_ones, x_ones.shape, x_ones.dtype)

x_zeros = jnp.zeros_like(x_array)
print(x_zeros, x_zeros.shape, x_zeros.dtype)
```

You can even initialize arrays with constants or random values. For example:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: PQhflPwAImj1
outputId: a75045a7-3a97-49d0-a215-42426f4e0ff4
---
shape = (2, 3)
ones_tensor = jnp.ones(shape)
zeros_tensor = jnp.zeros(shape)

seed = 123
key = jax.random.key(seed)
rand_tensor = jax.random.uniform(key, shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

JAX avoids implicit global random state and instead tracks state explicitly via a random `key`.
If we create two random arrays using the same `key` we will obtain two identical random arrays.
We can also split the random `key` into multiple keys to create two different random arrays.

```{code-cell} ipython3
seed = 124
key = jax.random.key(seed)
rand_tensor1 = jax.random.uniform(key, (2, 3))
rand_tensor2 = jax.random.uniform(key, (2, 3))
assert (rand_tensor1 == rand_tensor2).all()

k1, k2 = jax.random.split(key, num=2)
rand_tensor1 = jax.random.uniform(k1, (2, 3))
rand_tensor2 = jax.random.uniform(k2, (2, 3))
assert (rand_tensor1 != rand_tensor2).all()
```

For further discussion on random numbers in NumPy and JAX check [this tutorial](https://jax.readthedocs.io/en/latest/random-numbers.html).

+++

Finally, if you have a PyTorch tensor, you can use it to initialize a JAX Array:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: iWb6OdUpJvE0
outputId: 7483461a-fb29-4e7f-d892-231397419834
---
import torch

x_torch = torch.rand(3, 4)

# Create JAX Array as a copy of x_torch tensor
x_jax = jnp.asarray(x_torch)
assert isinstance(x_jax, jax.Array)
print(x_jax, x_jax.shape, x_jax.dtype)

# Use dlpack to create JAX Array without copying
x_jax = jax.dlpack.from_dlpack(x_torch.to(device="cuda"), copy=False)
print(x_jax, x_jax.shape, x_jax.dtype)
```

+++ {"id": "oTXSGITNNnuY"}

### Attributes of a JAX Array


Similarly to PyTorch tensors, JAX Array attributes describe the array's shape, dtype and device:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 4AwolCS1OHw2
outputId: 3defd0bb-4548-4afa-fe9c-3e84b446df24
---
x_jax = jnp.ones((3, 4))
print(f"Shape of tensor: {x_jax.shape}")
print(f"Datatype of tensor: {x_jax.dtype}")
print(f"Device tensor is stored on: {x_jax.device}")
```

+++ {"id": "S4CxmHSaKz-r"}

However, there are some notable differences between PyTorch tensors and JAX Arrays:
- JAX Arrays are immutable
- The default integer and float dtypes are int32 and float32
- The default device corresponds to the available accelerator, e.g. cuda:0 if one or multiple GPUs are available.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: cDVk9NjNKqJ2
outputId: 3769b2fe-c241-447a-a9d9-c622b568e963
---
try:
  x_jax[0, 0] = 100.0
except TypeError as e:
  print(e)


x_torch = torch.tensor([1, 2, 3, 4])
x_jax = jnp.array([1, 2, 3, 4])
print(f"Default integer dtypes, PyTorch: {x_torch.dtype} and Jax: {x_jax.dtype}")

x_torch = torch.zeros(3, 4)
x_jax = jnp.zeros((3, 4))
print(f"Default float dtypes, PyTorch: {x_torch.dtype} and Jax: {x_jax.dtype}")
print(f"Default devices, PyTorch: {x_torch.device} and Jax: {x_jax.device}")
```

+++ {"id": "1o1u2N1lL7I3"}

For some discussion of JAX's alternative to in-place mutation, refer to https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html.

+++ {"id": "_x7LutxoC3Eq"}

### Devices and accelerators

Using the PyTorch API, we can check whether we have GPU accelerators available with `torch.cuda.is_available()`. In JAX, we can check available devices as follows:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: s0k84UEMQSwL
outputId: 71b3658b-45ee-4fe1-a972-986f2e0da950
---
print(f"Available devices given a backend (gpu or tpu or cpu): {jax.devices()}")
# Define CPU and CUDA devices
cpu_device = jax.devices("cpu")[0]
cuda_device = jax.devices("cuda")[0]
print(cpu_device, cuda_device)
```

+++ {"id": "u4zGIZt9RyDF"}

Let's briefly explore how to create arrays on CPU and CUDA devices.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 9ZSyG8Q6Q5Jw
outputId: 68da1b2b-6803-4382-c851-b1c1062c60dd
---
# create an array on CPU and check the device
x_cpu = jnp.ones((3, 4), device=cpu_device)
print(x_cpu.device, )
# create an array on GPU
x_gpu = jnp.ones((3, 4), device=cuda_device)
print(x_gpu.device)
```

In PyTorch we are used to device placement always being explicit. JAX can operate this way via explicit device placement as above, but unless the device is specified the array will remain *uncommitted*: i.e. it will be stored on the default device, but allow implicit movement to other devices when necessary:

```{code-cell} ipython3
x = jnp.ones((3, 4))

x.device, (x_cpu + x).device
```

However, if we make a computation with two arrays with explicitly specified devices, e.g. CPU and CUDA, similarly to PyTorch, an error will be raised.

```{code-cell} ipython3
try:
    x_cpu + x_gpu
except ValueError as e:
    print(e)
```

To move from one device to another, we can use `jax.device_put` function:

```{code-cell} ipython3
x = jnp.ones((3, 4))
x_cpu = jax.device_put(x, device=jax.devices("cpu")[0])
x_cuda = jax.device_put(x_cpu, device=jax.devices("cuda")[0])
print(f"{x.device} -> {x_cpu.device} -> {x_cuda.device}")
```

+++ {"id": "o2481PfoPdFG"}

### Operations on JAX Arrays

There is a large list of operations (arithmetics, linear algebra, matrix manipulation, etc) that can be directly performed on JAX Arrays. JAX API contains important modules:
- `jax.numpy` provides NumPy-like functions
- `jax.scipy` provides SciPy-like functions
- `jax.nn` provides common functions for neural networks: activations, softmax, one-hot encoding etc
- `jax.lax` provides low-level XLA APIs
- ...

More details on available ops can be found in the [API reference](https://jax.readthedocs.io/en/latest/jax.html).

All operations can be run on CPUs, GPUs or TPUs. By default, JAX Arrays are created on an accelerated device, while PyTorch tensors are created on CPUs.

We can now try out some array operations and check for similarities between the JAX, NumPy and PyTorch APIs.

+++

Standard NumPy-like indexing and slicing:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: LAV8UBOaWHeI
outputId: 53a67323-2375-4011-ba04-bc563f47abd1
---
tensor = jnp.ones((3, 4))
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")

# Equivalent PyTorch op: tensor[:, 1] = 0
tensor = tensor.at[:, 1].set(0)

print(tensor)
```

We would like to note particular out-of-bounds indexing behaviour in JAX. In JAX the index is clamped to the bounds of the array in the indexing operations.

```{code-cell} ipython3
print(jnp.arange(10)[11])
```

Join arrays similar to `torch.cat`. Note the kwarg name: `axis` vs `dim`.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: AQVP9jKcXWiR
outputId: 9f0eb8c2-bda0-4864-9be3-f3941fcb74e8
---
t1 = jnp.concat([tensor, tensor, tensor], axis=1)
print(t1)
```

Arithmetic operations. Operations below compute the matrix multiplication between two tensors. y1, y2 will have the same value.

```{code-cell} ipython3
:id: P8jcElVyYTp7

# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = jnp.matmul(tensor, tensor.T)

assert (y1 == y2).all()

# This computes the element-wise product. z1, z2 will have the same value
z1 = tensor * tensor
z2 = jnp.multiply(tensor, tensor)

assert (z1 == z2).all()
```

Single-element arrays. If you have a one-element array, for example by aggregating all values of a tensor into one value, you can convert it to a Python numerical value using `.item()`:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: icgYp1wKYnGi
outputId: a58d0290-145a-4cd0-d78b-10c4729c0bc7
---
agg = tensor.sum()
agg_value = agg.item()
print(agg_value, isinstance(agg_value, float), isinstance(agg, jax.Array))
```

JAX follows NumPy in exposing a number of reduction and other operations as array methods:
```python
jax_array = jnp.ones((2, 3))
jax_array.sum(), jax_array.mean(), jax_array.min(), jax_array.max(), jax_array.dot(jax_array.T), # ...

tensor = torch.ones(2, 3)
tensor.sum(), tensor.mean(), tensor.min(), tensor.max(), tensor.matmul(tensor.T), # ...
```

PyTorch exposes many more methods on its tensor object than either JAX or NumPy does on their respective array objects. Here are some examples of methods only available in PyTorch:
```python
tensor.sigmoid(), tensor.softmax(dim=1), tensor.sin(), # ...
```

+++ {"id": "zxxkIKevl_vT"}

## Automatic differentiation with JAX

In this section, we will learn about the fundamental applications of automatic differentiation (autodiff) in JAX. JAX has a pretty general autodiff system, and its API has inspired the `torch.func` module in PyTorch, previously known as “functorch” (JAX-like composable function transforms for PyTorch).

In PyTorch, there is an API to turn on the automatic operations graph recording (e.g., `required_grad` argument and `tensor.backward()`), but in JAX, automatic differentiation is a functional operation, i.e., there is no need to mark arrays with a flag to enable gradient tracking.

Let us follow [autodiff PyTorch tutorial](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) and consider the simplest one-layer neural network, with input `x`, parameters `w` and `b`, and some loss function. In JAX, this can be defined in the following way:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: hzm2u1MpYmVN
outputId: a55ae511-45b3-49ab-d7cf-816448b1b0e3
---
import jax
import jax.numpy as jnp


# Input tensor
x = jnp.ones(5)
# Target output
y_true = jnp.zeros(3)

# Initialize random parameters
seed = 123
key = jax.random.key(seed)
key, w_key, b_key = jax.random.split(key, 3)
w = jax.random.normal(w_key, (5, 3))
b = jax.random.normal(b_key, (3, ))


# model function
def predict(x, w, b):
  return jnp.matmul(x, w) + b

# Criterion or loss function
def compute_loss(w, b, x, y_true):
  y_pred = predict(x, w, b)
  return jnp.mean((y_true - y_pred) ** 2)


loss = compute_loss(w, b, x, y_true)
print(loss)
```

+++ {"id": "k3r8od6LtD_9"}

In our example network, `w` and `b` are parameters to optimize and we need to be able to compute the gradients of the loss function with respect to those variables. In order to do that, we use [`jax.grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html#jax.grad) function on `compute_loss` function:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: glqqg02VvP-W
outputId: 549934e4-823b-48e1-c7f0-04ecd2b6c0d5
---
# Differentiate `compute_loss` with respect to the 0 and 1 positional arguments:
w_grad, b_grad = jax.grad(compute_loss, argnums=(0, 1))(w, b, x, y_true)
print(f'{w_grad=}')
print(f'{b_grad=}')
```

```{code-cell} ipython3
# Compute w_grad, b_grad and loss value:
loss_value, (w_grad, b_grad) = jax.value_and_grad(compute_loss, argnums=(0, 1))(w, b, x, y_true)
print(f'{w_grad=}')
print(f'{b_grad=}')
print(f'{loss_value=}')
print(f'{compute_loss(w, b, x, y_true)=}')
```

### `jax.grad` and PyTrees


JAX introduced the [PyTree abstraction](https://jax.readthedocs.io/en/latest/working-with-pytrees.html#working-with-pytrees)(e.g. Python containers like dicts, tuples, lists, etc which provides a uniform system for handling nested containers of array values) and its functional API works easily on these containers. Let us consider an example where we gathered our example network parameters into a dictionary:

```{code-cell} ipython3
net_params = {
  "weights": w,
  "bias": b,
}

def compute_loss2(net_params, x, y_true):
  y_pred = predict(x, net_params["weights"], net_params["bias"])
  return jnp.mean((y_true - y_pred) ** 2)
```

```{code-cell} ipython3
:id: tiTLDUvjvO_s

jax.value_and_grad(compute_loss2, argnums=0)({"weights": w, "bias": b}, x, y_true)
```

The functional API in JAX easily allows us to compute higher order gradients by calling `jax.grad` multiple times on the function. We will not cover this topic in this tutorial, for more details we suggest  reading [JAX automatic differentiation tutorial](https://jax.readthedocs.io/en/latest/automatic-differentiation.html).

## Build and train a model


In this section we will learn how to build a simple model using Flax ([`flax.nnx` API](https://flax.readthedocs.io/en/latest/nnx_basics.html)) and optimize its parameters using training data provided by PyTorch dataloader.


Model creation with Flax is very similar to PyTorch using the `torch.nn` module. In this example, we will build the ResNet18 model.


### Build ResNet18 model

```{code-cell} ipython3
# To install Flax: `pip install -U flax treescope optax`
import jax
import jax.numpy as jnp
from flax import nnx


class BasicBlock(nnx.Module):
  def __init__(
    self, in_planes: int, out_planes: int, do_downsample: bool = False, *, rngs: nnx.Rngs
  ):
    strides = (2, 2) if do_downsample else (1, 1)
    self.conv1_bn1 = nnx.Sequential(
      nnx.Conv(
        in_planes, out_planes, kernel_size=(3, 3), strides=strides,
        padding="SAME", use_bias=False, rngs=rngs,
      ),
      nnx.BatchNorm(out_planes, momentum=0.9, epsilon=1e-5, rngs=rngs),
    )
    self.conv2_bn2 = nnx.Sequential(
      nnx.Conv(
        out_planes, out_planes, kernel_size=(3, 3), strides=(1, 1),
        padding="SAME", use_bias=False, rngs=rngs,
      ),
      nnx.BatchNorm(out_planes, momentum=0.9, epsilon=1e-5, rngs=rngs),
    )

    if do_downsample:
      self.conv3_bn3 = nnx.Sequential(
        nnx.Conv(
          in_planes, out_planes, kernel_size=(1, 1), strides=(2, 2),
          padding="VALID", use_bias=False, rngs=rngs,
        ),
        nnx.BatchNorm(out_planes, momentum=0.9, epsilon=1e-5, rngs=rngs),
      )
    else:
      self.conv3_bn3 = lambda x: x

  def __call__(self, x: jax.Array):
    out = self.conv1_bn1(x)
    out = nnx.relu(out)

    out = self.conv2_bn2(out)
    out = nnx.relu(out)

    shortcut = self.conv3_bn3(x)
    out += shortcut
    out = nnx.relu(out)
    return out


class ResNet18(nnx.Module):
  def __init__(self, num_classes: int, *, rngs: nnx.Rngs):
    self.num_classes = num_classes
    self.conv1_bn1 = nnx.Sequential(
      nnx.Conv(
        3, 64, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
        use_bias=False, rngs=rngs,
      ),
      nnx.BatchNorm(64, momentum=0.9, epsilon=1e-5, rngs=rngs),
    )
    self.layer1 = nnx.Sequential(
      BasicBlock(64, 64, rngs=rngs), BasicBlock(64, 64, rngs=rngs),
    )
    self.layer2 = nnx.Sequential(
      BasicBlock(64, 128, do_downsample=True, rngs=rngs), BasicBlock(128, 128, rngs=rngs),
    )
    self.layer3 = nnx.Sequential(
      BasicBlock(128, 256, do_downsample=True, rngs=rngs), BasicBlock(256, 256, rngs=rngs),
    )
    self.layer4 = nnx.Sequential(
      BasicBlock(256, 512, do_downsample=True, rngs=rngs), BasicBlock(512, 512, rngs=rngs),
    )
    self.fc = nnx.Linear(512, self.num_classes, rngs=rngs)

  def __call__(self, x: jax.Array):
    x = self.conv1_bn1(x)
    x = nnx.relu(x)
    x = nnx.max_pool(x, (2, 2), strides=(2, 2))

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = nnx.avg_pool(x, (x.shape[1], x.shape[2]))
    x = x.reshape((x.shape[0], -1))
    x = self.fc(x)
    return x
```

```{code-cell} ipython3
:id: -dRZPUx-vLBk

model = ResNet18(10, rngs=nnx.Rngs(0))

# Visualize the model architecture
nnx.display(model)
```

Let us test the model on a dummy data:

```{code-cell} ipython3
:id: -D9-dxhGvKPV

x = jnp.ones((4, 32, 32, 3))
y_pred = model(x)
y_pred.shape
```

Note that the input array is explicitly in the channels-last memory format. In PyTorch, the typical input tensor to a neural network has channels-first memory format and has shape `(4, 3, 32, 32)` by default.


### Dataflow using Torchvision and PyTorch data loaders


Let us now set up training and test data using the CIFAR10 dataset from `torchvision`.
We will create torch dataloaders with collate functions returning NumPy Arrays instead of PyTorch tensors.
Since JAX is a multithreaded framework, using it in multiple processes can cause issues. For this reason, we will avoid creating JAX Arrays in the dataloaders.


As an alternative, one can use [grain](https://github.com/google/grain/tree/main) for data loading and [PIX](https://github.com/google-deepmind/dm_pix) for image data augmentations.

```{code-cell} ipython3
# CIFAR10 training/testing datasets setup
import numpy as np

from torchvision.transforms import v2 as T
from torchvision.datasets import CIFAR10


def to_np_array(pil_image):
  return np.asarray(pil_image)


def normalize(image):
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  image = image.astype(np.float32) / 255.0
  return (image - mean) / std


train_transforms = T.Compose([
  T.Pad(4),
  T.RandomCrop(32, fill=128),
  T.RandomHorizontalFlip(),
  T.Lambda(to_np_array),
  T.Lambda(normalize),
])

test_transforms = T.Compose([
  T.Lambda(to_np_array),
  T.Lambda(normalize),
])

train_dataset = CIFAR10("./data", train=True, download=True, transform=train_transforms)
test_dataset = CIFAR10("./data", train=True, download=False, transform=test_transforms)
```

```{code-cell} ipython3
# Data loaders setup
from torch.utils.data import DataLoader


batch_size = 512


def np_arrays_collate_fn(list_of_datapoints):
  list_of_images = [dp[0] for dp in list_of_datapoints]
  list_of_targets = [dp[1] for dp in list_of_datapoints]
  return np.stack(list_of_images, axis=0), np.asarray(list_of_targets)


train_loader = DataLoader(
  train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=np_arrays_collate_fn,
)
test_loader = DataLoader(
  test_dataset, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=np_arrays_collate_fn,
)
```

```{code-cell} ipython3
# Let us check training dataloader:
trl_iter = iter(train_loader)
batch = next(trl_iter)
print(batch[0].shape, batch[0].dtype, batch[1].shape, batch[1].dtype)
```

Note: when executing the code above you may see this warning: `RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.`. This warning can be ignored as dataloaders are not using JAX in forked processes.


### Model training


Let us now define the optimizer, loss function, train and test steps using Flax API.
PyTorch users can find the code using Flax NNX API very similar to PyTorch.

```{code-cell} ipython3
import optax

learning_rate = 0.005
momentum = 0.9

optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
```

```{code-cell} ipython3
def compute_loss_and_logits(model: nnx.Module, batch):
  logits = model(batch[0])
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch[1]
  ).mean()
  return loss, logits


@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
  """Train for a single step."""
  # convert numpy arrays to jnp.array on GPU
  x, y_true = jnp.asarray(batch[0]), jnp.asarray(batch[1])

  grad_fn = nnx.value_and_grad(compute_loss_and_logits, has_aux=True)
  (loss, logits), grads = grad_fn(model, (x, y_true))

  metrics.update(loss=loss, logits=logits, labels=y_true)  # In-place updates.

  optimizer.update(grads)  # In-place updates.
  return loss


@nnx.jit
def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch):
  # convert numpy arrays to jnp.array on GPU
  x, y_true = jnp.asarray(batch[0]), jnp.asarray(batch[1])

  loss, logits = compute_loss_and_logits(model, (x, y_true))

  metrics.update(loss=loss, logits=logits, labels=y_true)  # In-place updates.
```

Readers may note the [`nnx.jit`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.jit) decorator of `train_step` and `eval_step` methods which is used to jit-compile the functions. JIT compilation in JAX is explored in the last section of this tutorial.

```{code-cell} ipython3
# Define helper object to compute train/test metrics
metrics = nnx.MultiMetric(
  accuracy=nnx.metrics.Accuracy(),
  loss=nnx.metrics.Average('loss'),
)

metrics_history = {
  'train_loss': [],
  'train_accuracy': [],
  'test_loss': [],
  'test_accuracy': [],
}
```

```{code-cell} ipython3
# Start the training

num_epochs = 3

for epoch in range(num_epochs):
  metrics.reset()  # Reset the metrics for the test set.

  model.train()  # Set model to the training mode: e.g. update batch statistics
  for step, batch in enumerate(train_loader):

    loss = train_step(model, optimizer, metrics, batch)

    print(f"\r[train] epoch: {epoch + 1}/{num_epochs}, iteration: {step}, batch loss: {loss.item():.4f}", end="")
  print("\r", end="")

  for metric, value in metrics.compute().items():  # Compute the metrics.
    metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
  metrics.reset()  # Reset the metrics for the test set.

  # Compute the metrics on the test set after each training epoch.
  model.eval()  # Set model to evaluation model: e.g. use stored batch statistics
  for test_batch in test_loader:
    eval_step(model, metrics, test_batch)

  # Log the test metrics.
  for metric, value in metrics.compute().items():
    metrics_history[f'test_{metric}'].append(value)
  metrics.reset()  # Reset the metrics for the next training epoch.

  print(
    f"[train] epoch: {epoch + 1}/{num_epochs}, "
    f"loss: {metrics_history['train_loss'][-1]:0.4f}, "
    f" accuracy: {metrics_history['train_accuracy'][-1] * 100:0.4f}"
  )
  print(
    f"[test] epoch: {epoch + 1}/{num_epochs}, "
    f"loss: {metrics_history['test_loss'][-1]:0.4f}, "
    f"accuracy: {metrics_history['test_accuracy'][-1] * 100:0.4f}"
    "\n"
  )
```

### Further reading

More details about Flax NNX API, how to save and load the model's state and about available optimizers, we suggest to check out the links below:
- [FLAX NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html)
- [Save & Load model's state](https://flax.readthedocs.io/en/latest/guides/checkpointing.html)
- [Optax](https://optax.readthedocs.io/en/latest/)


Other AI/ML tutorials to check out:
- [JAX AI Stack tutorials](https://jax-ai-stack.readthedocs.io/en/latest/tutorials.html)

## Just-In-Time (JIT) compilation in JAX


PyTorch users know very well about the eager mode execution of the operations in PyTorch, e.g. the operations are executed one by one without any high-level optimizations on sets of operations. Similarly, almost everywhere in this tutorial we used JAX in the eager mode as well.


In PyTorch 1.0 [TorchScript](https://pytorch.org/docs/stable/jit.html) was introduced to optimize and serialize PyTorch models by capturing the execution graph into TorchScript programs, which can then be run independently from Python, e.g. as a C++ program.


In JAX, there is a similar transformation: [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit). It performs JIT compilation of a Python function for efficient execution in XLA. Behind the scenes, `jax.jit` wraps the input into tracers and is tracing the function to record all JAX operations. By default, JAX JIT is compiling the function on the first call and reusing the cached compiled XLA code on subsequent calls.

```{code-cell} ipython3
def matmul_relu_add(x, y):
    z = x * y
    return jax.nn.relu(z) + x
```

```{code-cell} ipython3
key = jax.random.key(123)
key1, key2 = jax.random.split(key)
x = jax.random.uniform(key1, (2500, 3000))
y = jax.random.uniform(key2, (2500, 3000))
```

```{code-cell} ipython3
%%timeit
matmul_relu_add(x, y)
```

```{code-cell} ipython3
jit_matmul_relu = jax.jit(matmul_relu_add)
# Warm-up: compile the function
_ = jit_matmul_relu(x, y)
```

```{code-cell} ipython3
%%timeit
jit_matmul_relu(x, y)
```

### Further reading

- https://jax.readthedocs.io/en/latest/jit-compilation.html
