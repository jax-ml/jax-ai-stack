---
jupytext:
  cell_metadata_filter: -all
  formats: ipynb,md:myst
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
---

# Getting started with JAX for AI

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jax-ml/jax-ai-stack/blob/main/docs/getting_started_with_jax_for_AI.ipynb)

+++

In this tutorial, you will learn how to get started using [JAX](https://jax.readthedocs.io) and JAX-based libraries to build and train a simple neural network model. JAX is a Python package for hardware accelerator-oriented array computation and program transformation, and is the engine behind cutting-edge AI research and production models at Google and beyond.

The JAX API focuses on [array-based](https://jax.readthedocs.io/en/latest/key-concepts.html#jax-arrays-jax-array) computation, and is at the core of a growing [ecosystem](https://jax.readthedocs.io/en/latest/index.html#ecosystem) of various domain-specific tools. This tutorial introduces a part of that ecosystem with a focus on AI, namely:

- [Flax](https://flax.readthedocs.io): A machine learning library designed for building and training scalable neural networks in JAX.
- [Optax](https://optax.readthedocs.io): A high-performance function optimization library that comes with built-in optimizers and loss functions. It also allows you to create your own such functions.

+++

You should be familiar with numerical computing in Python with [NumPy](http://numpy.org) and fundamental concepts of defining, training, and evaluating machine learning models.

Once you've worked through this content, visit the [JAX documentation site](http://jax.readthedocs.io/) for a deeper dive into the high performance numerical computing library, and [Flax](https://flax.readthedocs.io) to learn about neural networks in and for JAX.

+++

## Example: A simple neural network

Let's start with a very simple example of using JAX with [Flax](https://flax.readthedocs.io) to define a model and train it on the hand-written digits dataset with the help of [Optax](https://optax.readthedocs.io) for optimization during training.

+++

### Loading the data

JAX can work with a variety of data loaders, including [Grain](https://github.com/google/grain), [TensorFlow Datasets](https://github.com/tensorflow/datasets) and [TorchData](https://github.com/pytorch/data), but for simplicity here you can use the well-known [scikit-learn `digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) dataset.

```{code-cell}
from sklearn.datasets import load_digits
digits = load_digits()

print(f"{digits.data.shape=}")
print(f"{digits.target.shape=}")
```

This dataset consists of `8x8` pixelated images of hand-written digits and their corresponding labels. You can visualize a handful of them this way with [`matplotlib`](https://matplotlib.org/stable/tutorials/index):

```{code-cell}
import matplotlib.pyplot as plt

fig, axes = plt.subplots(10, 10, figsize=(6, 6),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='gaussian')
    ax.text(0.05, 0.05, str(digits.target[i]), transform=ax.transAxes, color='green')
```

Next, split the dataset into a training and testing set, and convert these splits into [`jax.Array`s]https://jax.readthedocs.io/en/latest/key-concepts.html#jax-arrays-jax-array) before you feed them into the model.

> **Note:** A [`jax.Array`](https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.html#jax.Array) is similar to [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray). You can learn more about `jax.Array`s in the [Key concepts](https://jax.readthedocs.io/en/latest/key-concepts.html#jax-arrays-jax-array) doc on the [JAX documentation site](https://jax.readthedocs.io/).

```{code-cell}
from sklearn.model_selection import train_test_split
splits = train_test_split(digits.images, digits.target, random_state=0)
```

```{code-cell}
import jax.numpy as jnp
images_train, images_test, label_train, label_test = map(jnp.asarray, splits)
print(f"{images_train.shape=} {label_train.shape=}")
print(f"{images_test.shape=}  {label_test.shape=}")
```

### Defining the Flax model

Next, subclass [`flax.nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html#flax.nnx.Module) to create a simple [feed-forward](https://en.wikipedia.org/wiki/Feedforward_neural_network) neural network called `SimpleNN`, which is made up of [`flax.nnx.Linear`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.Linear) layers with *scaled exponential linear unit* (SELU) activation functions (using the built-in [`flax.nnx.selu`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.selu)):

```{code-cell}
from flax import nnx

class SimpleNN(nnx.Module):

  def __init__(self, n_features: int = 64, n_hidden: int = 100, n_targets: int = 10,
               *, rngs: nnx.Rngs):
    self.n_features = n_features
    self.layer1 = nnx.Linear(n_features, n_hidden, rngs=rngs)
    self.layer2 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
    self.layer3 = nnx.Linear(n_hidden, n_targets, rngs=rngs)

  def __call__(self, x):
    x = x.reshape(x.shape[0], self.n_features) # flatten images
    x = nnx.selu(self.layer1(x))
    x = nnx.selu(self.layer2(x))
    x = self.layer3(x)
    return x

model = SimpleNN(rngs=nnx.Rngs(0))

nnx.display(model)  # Interactive display if penzai is installed.
```

> **Note:** The [`flax.nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html#flax.nnx.Module) system is covered in more detail in the [Flax basics](https://flax.readthedocs.io/en/latest/nnx_basics.html#the-flax-nnx-module-system) tutorial. Notice the use of the [`flax.nnx.Rngs`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/rnglib.html#flax.nnx.Rngs) type - this is the main convenience API in Flax for managing random state. The `Rngs` object can be used to get new unique JAX [pseudorandom number generator (PRNG)](https://jax.readthedocs.io/en/latest/random-numbers.html) keys based on a root PRNG key passed to the constructor. Learn more about PRNGs in the Flax [Randomness](https://flax.readthedocs.io/en/latest/guides/randomness.html) and the [JAX PRNG](https://jax.readthedocs.io/en/latest/random-numbers.html) tutorials.

+++

### Training the model

With the `SimpleNN` model defined, utilize the [Optax](http://optax.readthedocs.io) library to define the `optimizer` and the loss function (`loss_fun()`), and then create a training step (`train_step()`:

- Optax has several [common optimizers available](https://optax.readthedocs.io/en/latest/api/optimizers.html#optimizers), such as the Stochastic Gradient Descent [`optax.sgd`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.sgd). 
- Since you have an output layer with each node corresponding to an integer label, an appropriate loss metric would be [`optax.softmax_cross_entropy_with_integer_labels`](https://optax.readthedocs.io/en/latest/api/losses.html#optax.losses.softmax_cross_entropy_with_integer_labels).
- Then, define a `train_step()` based on this optimizer.

```{code-cell}
import jax
import optax

optimizer = nnx.Optimizer(model, optax.sgd(learning_rate=0.05))

def loss_fun(
    model: nnx.Module,
    data: jax.Array,
    labels: jax.Array):
  logits = model(data)
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=labels
  ).mean()
  return loss, logits

@nnx.jit  # JIT-compile the function.
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    data: jax.Array,
    labels: jax.Array):
  loss_gradient = nnx.value_and_grad(loss_fun, has_aux=True)  # Gradient transform!
  (loss, logits), grads = loss_gradient(model, data, labels)
  optimizer.update(grads)  # In-place update.
```

Notice here the use of transformations called [`flax.nnx.jit`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.jit) and [`flax.nnx.value_and_grad`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.value_and_grad), which are built on [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) and [`jax.grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html)/[`jax.value_and_grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.value_and_grad.html):

- `jax.jit` stands for a [just-in-time (JIT) compilation](https://jax.readthedocs.io/en/latest/quickstart.html#just-in-time-compilation-with-jax-jit) transformation. It will cause the function to be passed to the [XLA](https://openxla.org/xla) compiler for fast repeated execution.
- `jax.grad` is a gradient transformation. It uses JAX's [automatic differentiation](https://jax.readthedocs.io/en/latest/automatic-differentiation.html) for fast optimization of large networks. `jax.value_and_grad` not only takes the gradient of a function, but also evaluates it.

You will return to these transformations further below.

> **Note:** For in-depth guides on transformations, check out [Flax transforms](https://flax.readthedocs.io/en/latest/guides/transforms.html) and [JAX](https://jax.readthedocs.io/en/latest/quickstart.html#just-in-time-compilation-with-jax-jit) [transforms](https://jax.readthedocs.io/en/latest/key-concepts.html#transformations).

Now, define a training loop to repeatedly perform the `train_step()` over the training data, periodically printing the `loss_fun` against the test set to monitor convergence:

```{code-cell}
for i in range(301):  # 300 training epochs
  train_step(model, optimizer, images_train, label_train)
  if i % 50 == 0:  # Print metrics.
    loss, _ = loss_fun(model, images_test, label_test)
    print(f"epoch {i}: loss={loss:.2f}")
```

After 300 training epochs, the model appears to have converged to a target loss of `0.10`. You can check what this does to the accuracy of the labels for each image:

```{code-cell}
label_pred = model(images_test).argmax(axis=1)
num_matches = jnp.count_nonzero(label_pred == label_test)
num_total = len(label_test)
accuracy = num_matches / num_total
print(f"{num_matches} labels match out of {num_total}:"
      f" accuracy = {num_matches/num_total:%}")
```

The simple feed-forward network achieved approximately 98% accuracy on the test set.

Create a visualization to check how the model predicted some digits correctly (in green) and incorrectly (in red):

```{code-cell}
fig, axes = plt.subplots(10, 10, figsize=(6, 6),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(images_test[i], cmap='binary', interpolation='gaussian')
    color = 'green' if label_pred[i] == label_test[i] else 'red'
    ax.text(0.05, 0.05, str(label_pred[i]), transform=ax.transAxes, color=color)
```

In this section, you have learned the basics of using JAX for machine learning with Flax and Optax.

> **Note:** To learn more about the Flax fundamentals, go to the [Flax basics](https://flax.readthedocs.io/en/latest/nnx_basics.html) tutorial, which also covers Optax. Flax also includes a number of useful APIs for tracking metrics during training - check them action in the [Flax MNIST tutorial](https://flax.readthedocs.io/en/latest/nnx/mnist_tutorial.html) on the Flax website.

+++

## Key features of JAX

The Flax neural network API demonstrated in the first example takes advantage of a number of key [JAX API features](https://jax.readthedocs.io/en/latest/quickstart.html), designed into the library from the ground up. In particular:

- [**JAX provides a familiar NumPy-like API for array computing**](https://jax.readthedocs.io/en/latest/quickstart.html#jax-as-numpy).
  This means that when processing data and outputs you can reach for APIs like [`jax.numpy.count_nonzero`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.count_nonzero.html), which mirror the familiar APIs of the NumPy package; in this case - [`numpy.count_nonzero`](https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html).

- [**JAX provides just-in-time (JIT) compilation**](https://jax.readthedocs.io/en/latest/quickstart.html#just-in-time-compilation-with-jax-jit).
  This means that you can implement your code easily in Python, but count on fast compiled execution on CPU, GPU, and TPU backends via the [XLA](https://openxla.org/xla) compiler by wrapping your code with a simple [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) transformation.

- [**JAX provides automatic differentiation (autodiff).**](https://jax.readthedocs.io/en/latest/quickstart.html#just-in-time-compilation-with-jax-jit)
  This means that when fitting models, `optax` and `flax` can compute closed-form gradient functions for fast optimization of models, using the [`jax.grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html) transformation.

- [**JAX provides automatic vectorization.**](https://jax.readthedocs.io/en/latest/quickstart.html#auto-vectorization-with-jax-vmap)
  Though you didn't use it directly in the previous example, but under the hood Flax takes advantage of JAX's vectorized map ([`jax.vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html)) transform to automatically convert the loss and gradient functions to efficient batch-aware functions that are just as fast as hand-written versions. This makes JAX implementations simpler and less error-prone.

In the next sections, you’ll learn about `jax.numpy`, and the `jax.jit`, `jax.grad` and `jax.vmap` transforms through additional examples.

> **Note:** You can also check out the JAX [Quickstart](https://jax.readthedocs.io/en/latest/quickstart.html) and [Key concepts](https://jax.readthedocs.io/en/latest/key-concepts.html) docs to learn more about the JAX NumPy API and various JAX transforms.

+++

### JAX NumPy interface

The foundational array computing package in Python is NumPy, and JAX provides a matching API via the [`jax.numpy`](https://jax.readthedocs.io/en/latest/jax.numpy.html#module-jax.numpy) subpackage.

Additionally, JAX arrays behave much like NumPy arrays in their attributes, and in terms of [indexing](https://numpy.org/doc/stable/user/basics.indexing.html) and [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) semantics.

When setting up the `SimpleNN` model in the first example, you used the built-in [`flax.nnx.selu`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.selu) activation function. You can also implement your own version of SELU using the [`jax.numpy`](https://jax.readthedocs.io/en/latest/jax.numpy.html#module-jax.numpy) API:

```{code-cell}
import jax.numpy as jnp

def selu(x, alpha=1.67, lam=1.05):
  return lam * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(5.0)
print(selu(x))
```

> **Caution:** Despite the broad similarities, be aware that JAX does have some well-motivated differences from NumPy that you can read about in [JAX – the Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html).

+++

### Just-in-time compilation

JAX is built on the [XLA](https://openxla.org/xla) compiler, and allows sequences of operations to be JIT-compiled using the [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) transformation.

In the example above, you used the similar [`flax.nnx.jit`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.jit) API, which has some special handling for Flax objects to speed up neural network training.

Returning to the custom `selu()` function, you can create a JIT-compiled version this way:

```{code-cell}
import jax
selu_jit = jax.jit(selu)
```

`selu_jit` is now a compiled version of the original function, which returns the same result to typical floating-point precision:

```{code-cell}
x = jnp.arange(1E6)
jnp.allclose(selu(x), selu_jit(x))  # results match
```

You can use IPython's `%timeit` magic to see the speedup (note the use of `.block_until_ready()`, which you use to account for JAX's [asynchronous dispatch](https://jax.readthedocs.io/en/latest/async_dispatch.html)):

```{code-cell}
%timeit selu(x).block_until_ready()
```

```{code-cell}
%timeit selu_jit(x).block_until_ready()
```

For this computation, running on CPU, JIT compilation gives an order of magnitude speedup.
JAX's documentation has more discussion of JIT compilation in [this in-depth tutorial](https://jax.readthedocs.io/en/latest/jit-compilation.html).

> **Note:** Learn more about `jax.jit` in the JAX [Quickstart](https://jax.readthedocs.io/en/latest/quickstart.html#just-in-time-compilation-with-jax-jit) and the [Just-in-time compilation tutorial](https://jax.readthedocs.io/en/latest/jit-compilation.html).

+++

### Automatic differentiation (`jax.grad`)

For efficient optimization of neural network models, fast gradient computations are essential. JAX enables this via its [automatic differentiation](https://jax.readthedocs.io/en/latest/automatic-differentiation.html) transformations like [`jax.grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html), which computes a closed-form gradient of a JAX function. In the example above, you used the similar [`flax.nnx.grad`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.grad) function, which has special handling for `flax.nnx` objects.

Here's how to compute the gradient of a function with [`jax.grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html):

```{code-cell}
x = jnp.float32(-1.0)
jax.grad(selu)(x)
```

You can briefly check with a finite-difference approximation that this is giving the expected value:

```{code-cell}
eps = 1E-3
(selu(x + eps) - selu(x)) / eps
```

Importantly, the automatic differentiation approach is both more accurate and efficient than computing numerical gradients. JAX's documentation has more discussion of autodiff at [Automatic differentiation](https://jax.readthedocs.io/en/latest/automatic-differentiation.html).

> **Note:** Learn more about `jax.grad` in the JAX [Quickstart](https://jax.readthedocs.io/en/latest/quickstart.html#taking-derivatives-with-jax-grad), as well as the [Automatic differentiation (autodiff)](https://jax.readthedocs.io/en/latest/automatic-differentiation.html) and [Advanced autodiff](https://jax.readthedocs.io/en/latest/advanced-autodiff.html) tutorials.

+++

### Automatic vectorization (`jax.vmap`)

In the training loop above, you defined the loss function in terms of a single input data vector of shape `n_features`, but trained the model by passing batches of data (of shape `[n_samples, n_features]`). Rather than requiring a naive and slow loop over batches in Flax and Optax internals, they instead use JAX's automatic vectorization via the `jax.vmap` transformation to construct a batched version of the kernel automatically.

Consider a simple loss function that looks like this:

```{code-cell}
def loss(x: jax.Array, x0: jax.Array):
  return jnp.sum((x - x0) ** 2)
```

You can evaluate it on a single data vector this way:

```{code-cell}
x = jnp.arange(3.)
x0 = jnp.ones(3)
loss(x, x0)
```

But if you attempt to evaluate it on a batch of vectors, it does not correctly return a batch of 4 losses:

```{code-cell}
batched_x = jnp.arange(12).reshape(4, 3)  # batch of 4 vectors
loss(batched_x, x0)  # wrong!
```

The problem is that the loss function is not batch-aware. Without automatic vectorization, there are two ways you can address this:

1. Re-write your loss function by hand to operate on batched data. However, as functions become more complicated, this becomes difficult and error-prone.
2. Naively loop over unbatched calls to your original function. This is easy to code, but can be slow because it doesn't take advantage of vectorized compute.

The `jax.vmap` transformation offers a third way: it automatically transforms your original function into a batch-aware version, so you get the speed of option 1 with the ease of option 2:

```{code-cell}
loss_batched = jax.vmap(loss, in_axes=(0, None))  # batch x over axis 0, do not batch x0
loss_batched(batched_x, x0)
```

In the neural network example you learned to build and train, both Flax and Optax make use of JAX's `jax.vmap` to allow for efficient batched computations over the unbatched loss function.

+++

> **Note:** Learn more about `jax.vmap` in the JAX [Quickstart](https://jax.readthedocs.io/en/latest/quickstart.html#auto-vectorization-with-jax-vmap) and the [Automatic vectorization](https://jax.readthedocs.io/en/latest/automatic-vectorization.html) tutorial.
