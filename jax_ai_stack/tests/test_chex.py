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
import chex
import jax
import jax.numpy as jnp


class ChexTest(unittest.TestCase):

  def test_chex_dataclass(self):
    @chex.dataclass
    class Params:
      x: chex.ArrayDevice
      y: chex.ArrayDevice

    params = Params(
        x=jnp.arange(4),
        y=jnp.ones(10),
    )

    updated = jax.tree.map(lambda x: 2.0 * x, params)

    chex.assert_trees_all_close(updated.x, jnp.arange(0, 8, 2))
    chex.assert_trees_all_close(updated.y, jnp.full(10, fill_value=2.0))


if __name__ == '__main__':
  unittest.main()
