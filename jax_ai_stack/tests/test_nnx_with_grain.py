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
import functools
import unittest
from flax import nnx
import grain.python as grain
import numpy as np
import optax
import tensorflow_datasets as tfds


# Simple CNN from https://flax.readthedocs.io/en/latest/nnx/mnist_tutorial.html
class CNN(nnx.Module):

  def __init__(self, *, rngs: nnx.Rngs):
    self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
    self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
    self.avg_pool = functools.partial(
        nnx.avg_pool, window_shape=(2, 2), strides=(2, 2)
    )
    self.linear1 = nnx.Linear(448, 64, rngs=rngs)
    self.linear2 = nnx.Linear(64, 10, rngs=rngs)

  def __call__(self, x):
    x = self.avg_pool(nnx.relu(self.conv1(x)))
    x = self.avg_pool(nnx.relu(self.conv2(x)))
    x = x.reshape(x.shape[0], -1)  # flatten
    x = nnx.relu(self.linear1(x))
    x = self.linear2(x)
    return x


class NNXTFDSTest(unittest.TestCase):

  def setUp(self):
    if hasattr(self, 'enterContext'):  # Python 3.11 or newer
      self.enterContext(tfds.testing.mock_data(num_examples=5))
    else:
      with contextlib.ExitStack() as stack:
        stack.enter_context(tfds.testing.mock_data(num_examples=5))
        self.addCleanup(stack.pop_all().close)

  def test_nnx_with_grain(self):
    data_source = tfds.data_source('mnist', split='train')

    sampler = grain.IndexSampler(
        num_records=5,
        num_epochs=1,
        shard_options=grain.NoSharding(),
        shuffle=True,
        seed=0,
    )

    class DownSample(grain.MapTransform):
      shape: tuple[int, int]

      def __init__(self, shape: tuple[int, int]):
        self.shape = shape

      def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        image = element['image']
        element['image_scaled'] = image - image.mean()
        return element

    operations = [DownSample((16, 16))]

    loader = grain.DataLoader(
        data_source=data_source,
        operations=operations,
        sampler=sampler,
        worker_count=0,  # Scale to multiple workers in multiprocessing
    )

    model = CNN(rngs=nnx.Rngs(0))
    optimizer = nnx.ModelAndOptimizer(model, optax.adamw(learning_rate=0.005))

    def loss_fn(model, batch):
      logits = model(batch['image_scaled'])
      loss = optax.softmax_cross_entropy_with_integer_labels(
          logits=logits, labels=np.ravel(batch['label'])
      ).mean()
      return loss, logits

    for batch in loader:
      grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
      _, grads = grad_fn(model, batch)
      optimizer.update(grads)


if __name__ == '__main__':
  unittest.main()
