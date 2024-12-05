# Installing the stack

`jax-ai-stack` is a metapackage that can be installed with the following command:
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

# Optional packages

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

## Pinned versions

The `jax-ai-stack` meta-package does periodic releases, with date-based version strings. For
example, if you'd like to pin the set of packages from November 2024, you can use this installation
command:
```
pip install jax-ai-stack==2024.11.1
```
For the full list of released versions and the pinned packages, refer to the [Change log](https://github.com/jax-ml/jax-ai-stack/blob/main/CHANGELOG.md).
