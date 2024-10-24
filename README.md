# JAX AI Stack
[![Continuous integration](https://github.com/jax-ml/jax-ai-stack/actions/workflows/test.yaml/badge.svg)](https://github.com/jax-ml/jax-ai-stack/actions/workflows/test.yaml/)
[![PyPI version](https://img.shields.io/pypi/v/jax-ai-stack)](https://pypi.org/project/jax-ai-stack/)

[JAX](http://github.com/jax-ml/jax) is a Python package for array-oriented
computation and program transformation. Built around it is a growing ecosystem
of packages for specialized numerical computing across a range of domains; an
up-to-date list of such projects can be found at
[Awesome JAX](https://github.com/n2cholas/awesome-jax).

Though JAX is often compared to neural network libraries like PyTorch, the JAX
core package itself contains very little that is specific to neural network
models. Instead, JAX encourages modularity, where domain-specific libraries
are developed separately from the core package: this helps drive innovation
as researchers and other users explore what is possible.

Within this larger, distributed ecosystem, there are a number of projects that
Google researchers and engineers have found useful for implementing and deploying
the models behind generative AI tools like [Imagen](https://imagen.research.google/),
[Gemini](https://gemini.google.com/), and more. The JAX AI stack serves as a
single point-of-entry for this suite of libraries, so you can install and begin
using many of the same open source packages that Google developers are using
in their everyday work.

To get started with the JAX AI stack, you can check out [Getting started with JAX](
https://github.com/jax-ml/jax-ai-stack/blob/main/docs/getting_started_with_jax_for_AI.ipynb).
This is still a work-in-progress, please check back for more documentation and tutorials
in the coming weeks!

## Installing the stack

The stack can be installed with the following command:
```
pip install jax-ai-stack
```
This pins particular versions of component projects which are known to work correctly
together via the integration tests in this repository. Packages include:

- [JAX](http://github.com/google/jax): the core JAX package, which includes array operations
  and program transformations like `jit`, `vmap`, `grad`, etc.
- [flax](http://github.com/google/flax): build neural networks with JAX
- [ml_dtypes](http://github.com/jax-ml/ml_dtypes): NumPy dtype extensions for machine learning.
- [optax](https://github.com/google-deepmind/optax): gradient processing and optimization in JAX.
- [orbax](https://github.com/google/orbax): checkpointing and persistence utilities for JAX.

### Optional packages

Additionally, there are optional packages you can install with `pip` extras.
The following command:
```
pip install jax-ai-stack[grain]
```
will install a compatible version of the [grain](https://github.com/google/grain) data
loader (currently linux-only).

Similarly, the following command:
```
pip install jax-ai-stack[tfds]
```
will install a compatible version of [tensorflow](https://github.com/tensorflow/tensorflow)
and [tensorflow-datasets](https://github.com/tensorflow/datasets).
