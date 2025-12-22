# Copyright 2025 The JAX Authors.
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

from flax import nnx
import jax
import numpy as np
# import qwix


class SimpleModel(nnx.Module):

  def __init__(self, din, dhidden, dout, *, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(din, dhidden, use_bias=False, rngs=rngs)
    self.linear2 = nnx.Linear(dhidden, dout, use_bias=False, rngs=rngs)

  def __call__(self, x):
    x = self.linear1(x)
    x = nnx.relu(x)
    return self.linear2(x)


class NNXQwixIntegrationTest(unittest.TestCase):

  def test_nnx_qwix_ptq(self):
    din = 16
    dhidden = 32
    dout = 8
    batch_size = 4

    key = jax.random.PRNGKey(42)
    model_key, input_key = jax.random.split(key)

    # Instantiate the float model
    fp_model = SimpleModel(din, dhidden, dout, rngs=nnx.Rngs(model_key))
    model_input = jax.random.uniform(input_key, (batch_size, din))

    # # Define Qwix quantization rules
    # rules = [
    #     qwix.QuantizationRule(
    #         module_path='.*',  # Apply to all modules
    #         weight_qtype='int8',  # Quantize weights to int8
    #     )
    # ]

    # # Apply post-training quantization to the NNX model
    # ptq_provider = qwix.PtqProvider(rules)
    # q_model = qwix.quantize_model(fp_model, ptq_provider, model_input)

    # @nnx.jit
    # def predict(model, inputs):
    #   return model(inputs)

    # output = predict(fp_model, model_input)
    # try:
    #   q_output = predict(q_model, model_input)
    #   self.assertEqual(q_output.shape, (batch_size, dout))
    #   self.assertTrue(np.all(np.isfinite(q_output)))
    #   self.assertFalse(np.any(q_output == output))
    # except Exception as e:
    #   self.fail(f'Forward pass with quantized model failed: {e}')


if __name__ == '__main__':
  unittest.main()
