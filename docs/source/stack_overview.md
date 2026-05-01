# Stack overview

The JAX AI Stack is comprised of the following packages:

* [JAX{material-regular}`open_in_new`](https://jax.dev): high-performance array
  computing
* [Flax
  NNX{material-regular}`open_in_new`](https://flax.readthedocs.io/en/latest/):
  object-oriented neural nets
* [Optax{material-regular}`open_in_new`](https://optax.readthedocs.io/en/latest/index.html):
  optimizers
* [Orbax{material-regular}`open_in_new`](https://orbax.readthedocs.io/en/latest/):
  checkpointing and model export
* [Grain{material-regular}`open_in_new`](https://google-grain.readthedocs.io/en/latest/):
  JAX-native data loading
* [Chex{material-regular}`open_in_new`](https://chex.readthedocs.io/en/latest/):
  JAX test utilities

The `jax-ai-stack` metapackage installs compatible versions of all of these
libraries, as well as shared compatible versions of shared dependencies.

In addition, there is an optional `jax-ai-stack[tfds]` installation that
includes [TensorFlow
Datasets{material-regular}`open_in_new`](https://www.tensorflow.org/datasets),
for those who wish to use TFDS for data loading. This includes a compatible
version of TensorFlow as well.
