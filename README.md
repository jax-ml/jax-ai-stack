# jax-ai-stack

This is a work-in-progress meta-package for the AI/ML stack built on top of the
[jax](http://github.com/google/jax/) package. It is intended as a location for
tests, documentation, and installation instructions that cover multiple
packages in the JAX ecosystem.

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
- [ml_dtypes](http://github.com/jax-ml/ml_dtypes): NumPy dtype extensions for machine learing.
- [optax](https://github.com/google-deepmind/optax): gradient processing and optimization in JAX.
- [orbax](https://github.com/google/orbax): checkpointing and persistence utilities for JAX.

### Optional packages

Additionally, there are optional packages you can install with `pip` extras.
The following command:
```
pip install jax-ai-stack[grain]
```
will install a compatible version of the [grain](https://github.com/google/grain) data loader.

Similarly, the following command:
```
pip install jax-ai-stack[tfds]
```
will install a compatible version of [tensorflow](https://github.com/tensorflow/tensorflow)
and [tensorflow-datasets](https://github.com/tensorflow/datasets).
