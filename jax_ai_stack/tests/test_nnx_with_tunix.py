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
import jax.numpy as jnp
import numpy as np
import optax
import grain.python as pygrain
import qwix
from tunix.sft import peft_trainer


class SimpleModel(nnx.Module):

  def __init__(self, rngs, max_len):
    self.layer1 = nnx.Linear(max_len, 5, rngs=rngs)
    self.layer2 = nnx.Linear(5, 2, rngs=rngs)

  def __call__(self, inputs, training=False):
    x = nnx.relu(self.layer1(inputs))
    return self.layer2(x)


def gen_model_input_fn(x: peft_trainer.TrainingInput):
  return {'inputs': x.input_tokens, 'training': True}


def lora_loss_fn(model, inputs, training):
  targets = jnp.zeros([inputs.shape[0]], dtype=jnp.int32)
  logits = model(inputs, training=training)
  return optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=targets
  ).mean()


def load_test_dataset(batch_size, max_len, num_epochs):
  # Create a simple data source from the tokens
  class TokenDataSource(pygrain.RandomAccessDataSource):

    def __len__(self):
      return 64

    def __getitem__(self, index):
      # Return a sequence of max_len tokens
      return jax.random.randint(
          jax.random.key(index), [max_len], minval=0, maxval=1e6
      )

  # Create the data source
  data_source = TokenDataSource()

  # Create a sampler
  sampler = pygrain.IndexSampler(
      num_records=len(data_source),
      shuffle=True,
      seed=42,
      num_epochs=1,
      shard_options=pygrain.NoSharding(),
  )

  # Create transformations
  class ToTrainingInputDict(pygrain.MapTransform):

    def map(self, batch):
      return {'input_tokens': batch, 'input_mask': np.ones_like(batch)}

  # Create the data loader
  loader = pygrain.DataLoader(
      data_source=data_source,
      sampler=sampler,
      operations=[
          pygrain.Batch(batch_size=batch_size, drop_remainder=True),
          ToTrainingInputDict(),
      ],
      worker_count=0,  # Use main thread
  )

  def to_training_input(loader):
    # The trainer expects an iterable of `peft_trainer.TrainingInput`.
    for item in loader:
      yield peft_trainer.TrainingInput(**item)

  return to_training_input(loader)


class NNXTunixTest(unittest.TestCase):

  def test_nnx_tunix_sft_with_checkpointing(self):
    max_len = 6
    model = SimpleModel(rngs=nnx.Rngs(0), max_len=max_len)
    mesh = jax.make_mesh(
        (jax.device_count(), 1),
        ('batch', 'model'),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )

    with jax.set_mesh(mesh):
      # Apply LoRA to the model
      lora_provider = qwix.LoraProvider(
          module_path='.*mha|.*linear1|.*linear2', rank=4, alpha=2.0
      )

      lora_model = qwix.apply_lora_to_model(
          model, lora_provider, inputs=jnp.zeros((4, max_len)), rngs=nnx.Rngs(0)
      )

      # Setup Tunix PeftTrainer
      train_steps = 3
      training_config = peft_trainer.TrainingConfig(
          eval_every_n_steps=None,
          max_steps=train_steps,
          data_sharding_axis=('batch',),
      )
      lora_trainer = (
          peft_trainer.PeftTrainer(
              lora_model, optax.adamw(1e-2), training_config
          )
          .with_gen_model_input_fn(gen_model_input_fn)
          .with_loss_fn(lora_loss_fn)
      )

      # Run LoRA training
      lora_train_ds = load_test_dataset(8, max_len=max_len, num_epochs=1)
      lora_trainer.train(lora_train_ds)

    self.assertEqual(lora_trainer.train_steps, train_steps)


if __name__ == '__main__':
  unittest.main()
