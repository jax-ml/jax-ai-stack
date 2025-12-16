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


import contextlib
import platform
import tempfile
import unittest

from flax import nnx
import jax
import numpy as np
import orbax.checkpoint


class SimpleModel(nnx.Module):

  def __init__(self, rngs):
    self.layer1 = nnx.Linear(2, 5, rngs=rngs)
    self.layer2 = nnx.Linear(5, 3, rngs=rngs)

  def __call__(self, x):
    for layer in [self.layer1, self.layer2]:
      x = layer(x)
    return x


class NNXOrbaxTest(unittest.TestCase):

  def setUp(self):
    self.tmp_dir = tempfile.TemporaryDirectory()

    if hasattr(self, 'enterContext'):  # Python 3.11 or newer
      self.enterContext(self.tmp_dir)
    else:
      with contextlib.ExitStack() as stack:
        stack.enter_context(self.tmp_dir)
        self.addCleanup(stack.pop_all().close)

  def test_nnx_orbax_checkpoint(self):
    model = SimpleModel(nnx.Rngs(0))

    # Create the checkpoint
    state = nnx.state(model)
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpointer.save(f'{self.tmp_dir.name}/state', item=state)
    restore_args = orbax.checkpoint.checkpoint_utils.construct_restore_args(
        state
    )

    # update the model with the loaded state
    restored_model = nnx.eval_shape(SimpleModel, nnx.Rngs(1))
    restored_state = checkpointer.restore(
        f'{self.tmp_dir.name}/state',
        item=nnx.state(restored_model),
        restore_args=restore_args,
    )
    nnx.update(restored_model, restored_state)

    self.assertEqual(type(model), type(restored_model))
    jax.tree.map(np.testing.assert_array_equal, state, restored_state)


if __name__ == '__main__':
  unittest.main()
