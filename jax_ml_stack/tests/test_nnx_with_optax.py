# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import jax
import jax.numpy as jnp
from flax import nnx
import optax


class SimpleModel(nnx.Module):

  def __init__(self, rngs):
    self.layer1 = nnx.Linear(2, 5, rngs=rngs)
    self.layer2 = nnx.Linear(5, 3, rngs=rngs)

  def __call__(self, x):
    for layer in [self.layer1, self.layer2]:
      x = layer(x)
    return x


class NNXOptaxTest(unittest.TestCase):

  def test_nnx_optax(self):
    key = jax.random.key(1701)
    x = jax.random.normal(key, (1, 2))
    y = jnp.ones((1, 3))

    model = SimpleModel(nnx.Rngs(0))
    optimizer = optax.adam(learning_rate=1e-3)
    state = nnx.Optimizer(model, optimizer)

    def loss(model, x=x, y=y):
      return jnp.mean((model(x) - y) ** 2)

    initial_loss = loss(model)
    grads = nnx.grad(loss)(state.model)
    state.update(grads)
    final_loss = loss(model)

    self.assertNotAlmostEqual(initial_loss, final_loss)


if __name__ == '__main__':
  unittest.main()
