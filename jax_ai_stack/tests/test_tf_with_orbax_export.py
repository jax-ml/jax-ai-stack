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
import sys
import tempfile
import unittest

import numpy as np


class NNXTFDSTest(unittest.TestCase):
  tmp_dir: tempfile.TemporaryDirectory

  def setUp(self):
    self.tmp_dir = tempfile.TemporaryDirectory()

    if hasattr(self, 'enterContext'):  # Python 3.11 or newer
      self.enterContext(self.tmp_dir)
    else:
      with contextlib.ExitStack() as stack:
        stack.enter_context(self.tmp_dir)
        self.addCleanup(stack.pop_all().close)

  @unittest.skipIf(sys.platform == 'darwin', 'TODO(emilyaf): Fix MacOS CI failure.')
  def test_tf_model_with_checkpoint(self):
    # Import locally to ensure fork-safety for parallel testing (`pytest -n`).
    # This prevents heavy libraries from initializing before worker processes are
    # created, avoiding crashes on platforms like macOS.
    import jax.numpy as jnp
    import tensorflow as tf
    from orbax.export import ExportManager, JaxModule, ServingConfig

    params = {'a': np.array(5.0), 'b': np.array(1.1), 'c': np.array(0.55)}

    def model_fn(params, inputs):
      a, b, c = params['a'], params['b'], params['c']
      return a * jnp.sin(inputs) + b * inputs + c

    def preprocess(inputs):
      norm_inputs = tf.nest.map_structure(
          lambda x: x / tf.math.reduce_max(x), inputs
      )
      return norm_inputs

    def postprocess(model_fn_outputs):
      return {'outputs': model_fn_outputs}

    inputs = tf.random.normal([16], dtype=tf.float32)

    model_outputs = postprocess(model_fn(params, np.array(preprocess(inputs))))

    jax_module = JaxModule(params, model_fn)
    export_mgr = ExportManager(
        jax_module,
        [
            ServingConfig(
                'serving_default',
                input_signature=[tf.TensorSpec(shape=[16], dtype=tf.float32)],
                tf_preprocessor=preprocess,
                tf_postprocessor=postprocess,
            ),
        ],
    )
    export_mgr.save(self.tmp_dir.name)
    loaded_model = tf.saved_model.load(self.tmp_dir.name)
    loaded_model_outputs = loaded_model(inputs)

    np.testing.assert_allclose(
        model_outputs['outputs'],
        loaded_model_outputs['outputs'],
        atol=1e-5,
        rtol=1e-5,
    )


if __name__ == '__main__':
  unittest.main()
