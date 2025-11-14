---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "YdtfHhtq7esh"}

# Train a miniGPT language model with JAX

+++ {"id": "dNvPJpcW7esj"}

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/jax-ml/jax-ai-stack/blob/main/docs/source/JAX_for_LLM_pretraining.ipynb"><img src="https://www.kaggle.com/static/images/logos/kaggle-logo-transparent-300.png" height="32" width="70"/>Run in Kaggle</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/jax-ml/jax-ai-stack/blob/main/docs/source/JAX_for_LLM_pretraining.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/jax-ml/jax-ai-stack/blob/main/docs/source/JAX_for_LLM_pretraining.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>

+++ {"id": "NIOXoY1xgiww"}

[REVAMP 0]: Expand intro.

This tutorial demonstrates how to use JAX, [Flax NNX](http://flax.readthedocs.io) and [Optax](http://optax.readthedocs.io) for language model (pre)training using data and tensor [parallelism](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization) for [Single-Program Multi-Data](https://en.wikipedia.org/wiki/Single_program,_multiple_data)). It was originally inspired by the [Keras miniGPT tutorial](https://keras.io/examples/generative/text_generation_with_miniature_gpt/).

Here, you will learn how to:

- Define the miniGPT model with Flax and JAX automatic parallelism
- Load and preprocess the dataset
- Create the loss and training step functions
- Train the model on TPUs on Kaggle or Google Colab
- Profile for hyperparameter tuning

If you are new to JAX for AI, check out the [introductory tutorial](https://jax-ai-stack.readthedocs.io/en/latest/neural_net_basics.html), which covers neural network building with [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html).

+++ {"id": "hTmz5Cbco7n_"}

## Setup

JAX installation is covered in [this guide](https://jax.readthedocs.io/en/latest/installation.html) on the JAX documentation site. We will use [Tiktoken](https://github.com/openai/tiktoken) for tokenization and [Grain](https://google-grain.readthedocs.io/en/latest/index.html) for data loading.

```{code-cell}
:id: 6zMsOIc7ouCO

!pip install -Uq tiktoken jax-ai-stack[grain] matplotlib
```

+++ {"id": "Rcji_799n4eA"}

**Note:** If you are using [Kaggle](https://www.kaggle.com/), select the free TPU v5e-8 as the hardware accelerator. If you are using [Google Colab](https://colab.research.google.com/), select the free Google Cloud TPU v5e-1 as the hardware accelerator. You may also use Google Cloud TPUs.

Check the available JAX devices, or [`jax.Device`](https://jax.readthedocs.io/en/latest/_autosummary/jax.Device.html), with [`jax.devices()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.devices.html). The output of the cell below will show a list of 8 (eight) devices.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: LS9sQEY3n0mB
outputId: 6b9ee4b0-eed0-4bae-dd99-ffed14289ad7
---
import jax
jax.devices()
```

+++ {"id": "OHzJ_bokoovZ"}

Get the [TinyStories dataset from Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories). We only use the training split.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: wUjQsgQEmI1N
outputId: a704b8b3-2a1e-48bc-8915-122329a5df52
---
!wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt?download=true -O TinyStories-train.txt
```

+++ {"id": "sKE2uUafLobI"}

Import the necessary modules, including JAX NumPy, Flax NNX, Optax, Grain, pandas, and Tiktoken:

```{code-cell}
:id: MKYFNOhdLq98

import jax
import jax.numpy as jnp

# For data and model parallelism (explained in more detail later)
from jax.sharding import PartitionSpec as P, NamedSharding

import flax.nnx as nnx
import optax

from dataclasses import dataclass
import grain.python as pygrain
import pandas as pd
import tiktoken
import time
```

+++ {"id": "rPyt7MV6prz1"}

## Define the miniGPT model with NNX and JAX automatic parallelism

### NNX: A JAX-based neural network library
[REVAMP 1]: Introduce NNX.

### Leveraging JAX's data and tensor parallelism

One of the most powerful features of JAX is [device parallelism](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization) for SPMD.

- The data parallelism technique enables, for example, the training data to run via multiple parts (this is called sharding) - batches - in parallel and simultaneously across different devices, such as GPUs and Google TPUs. This allows to use larger batch sizes to speed up training.
- Tensor parallelism allows us to split the model parameter tensors across several devices (sharding model tensors).
- You can learn more about the basics of JAX parallelism in more detail in the [Introduction to parallel programming](https://jax.readthedocs.io/en/latest/sharded-computation.html) on the JAX documentation site.

In this example, we'll utilize a 4-way data parallel and 2-way tensor parallel setup, which is aligned with Kaggle TPU v5e-8 or newer GCP TPUs chips.

Note that as of October 2025, free-tier Colab only offers TPU v5e-1, which can no longer support SPMD.

### jax.sharding.Mesh

[`jax.sharding.Mesh`](https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.Mesh) is a multidimensional NumPy array of JAX devices, where each axis of the mesh has a name, such as `'x'` or `'y'`. This will help encapsulate the information about the TPU resource organization for distributing computations across the devices.

Our `Mesh` will have two arguments:
- `devices`: This will take the value of [`jax.make_mesh((4, 2), ('batch', 'model'))`](https://jax.readthedocs.io/en/latest/jax.experimental.mesh_utils.html), enabling us to build a device mesh. It is a NumPy ndarray with JAX devices (a list of devices from the JAX backend as obtained from [`jax.devices()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.devices.html#jax.devices))..
- `axis_names`, where:
  - `batch`: 4 devices along the first axis - i.e. sharded into 4 - for data parallelism; and
  - `model`: 2 devices along the second axis - i.e. sharded into 2 -  for tensor parallism

This matches the structure in the Kaggle TPU v5e setup.

Let's instantiate `Mesh` as `mesh` and declare the TPU configuration to define how data and model parameters are distributed across the devices:

```{code-cell}
:id: xuMlCK3Q8WJD

# Create a `Mesh` object representing TPU device arrangement.
# For example, for Kaggle TPU v5e-8:
if jax.device_count() == 8:
    mesh = jax.make_mesh((4, 2), ('batch', 'model'))

    ### Alternatively, we could use the 8-way data parallelism with only one line of code change.
    ### JAX enables quick experimentation with different partitioning strategies
    ### like this. We will come back to this point at the end of this tutorial.
    mesh = jax.make_mesh((8, 1), ('batch', 'model'))

### For free-tier Colab TPU, which only has a single TPU core
if jax.device_count() == 1:
    mesh = jax.make_mesh((1, 1), ("batch", "model"))
```

+++ {"id": "_ZKdhNo98NgG"}

We will use the GPT-2 tokenizer from the [Tiktoken](https://github.com/openai/tiktoken) library:

```{code-cell}
:id: iWbkk1V7-Isg

tokenizer = tiktoken.get_encoding("gpt2")
```

+++ {"id": "0XHQ0BQ9-KIj"}

To leverage model parallelism, we need to instruct the JAX compiler how to shard the model tensors across the TPU devices. We'll use Flax NNX's [`flax.nnx.with_partitioning`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/spmd.html#flax.nnx.with_partitioning) to let each model layer know that the model weights or tensors need to be sharded according to our specification. We need to do this for every tensor/layer in the model.

`nnx.with_partitioning` will take two arguments, such as the `initializer` (e.g. [`flax.nnx.initializers.xavier_uniform`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/initializers.html#flax.nnx.initializers.xavier_uniform) and [`flax.nnx.initializers.zeros_init`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/initializers.html#flax.nnx.initializers.zeros_init)), and a sharding tuple (e.g. `(None, 'model')` in our case) wrapped in [`jax.sharding.PartitionSpec`](https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.PartitionSpec). The sharding tuple describe how to shard a tensor across, for example, the `model` axis or be replicated on other dimensions (which is denoted by `None`).

For a more detailed discussion of Flax NNX sharding, please refer to [this SPMD guide](https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html).

```{code-cell}
:id: z0p-IHurrB9i


def causal_attention_mask(seq_len):
    return jnp.tril(jnp.ones((seq_len, seq_len)))

class TransformerBlock(nnx.Module):
    """ A single Transformer block.

    Each Transformer block processes input sequences via self-attention and feed-forward networks.

    Args:
        embed_dim (int): Embedding dimensionality.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward network.
        rngs (flax.nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
        rate (float): Dropout rate. Defaults to 0.1.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, *, rngs: nnx.Rngs, rate: float = 0.1):
        self.mha = nnx.MultiHeadAttention(num_heads=num_heads,
                                          in_features=embed_dim,
                                          kernel_init=nnx.with_partitioning(
                                              nnx.initializers.xavier_uniform(), P(None, 'model')),
                                          bias_init=nnx.with_partitioning(
                                              nnx.initializers.zeros_init(), P('model')),
                                          rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=rate, rngs=rngs)
        self.layer_norm1 = nnx.LayerNorm(epsilon=1e-6,
                                         num_features=embed_dim,
                                         scale_init=nnx.with_partitioning(
                                             nnx.initializers.ones_init(), P('model')),
                                         bias_init=nnx.with_partitioning(
                                             nnx.initializers.zeros_init(), P('model')),
                                         rngs=rngs)
        self.linear1 = nnx.Linear(in_features=embed_dim,
                                  out_features=ff_dim,
                                  kernel_init=nnx.with_partitioning(
                                      nnx.initializers.xavier_uniform(), P(None, 'model')),
                                  bias_init=nnx.with_partitioning(
                                      nnx.initializers.zeros_init(), P('model')),
                                  rngs=rngs)
        self.linear2 = nnx.Linear(in_features=ff_dim,
                                  out_features=embed_dim,
                                  kernel_init=nnx.with_partitioning(
                                      nnx.initializers.xavier_uniform(), P(None, 'model')),
                                  bias_init=nnx.with_partitioning(
                                      nnx.initializers.zeros_init(), P('model')),
                                  rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=rate, rngs=rngs)
        self.layer_norm2 = nnx.LayerNorm(epsilon=1e-6,
                                         num_features=embed_dim,
                                         scale_init=nnx.with_partitioning(
                                             nnx.initializers.ones_init(), P('model')),
                                         bias_init=nnx.with_partitioning(
                                             nnx.initializers.zeros_init(), P('model')),
                                         rngs=rngs)


    def __call__(self, inputs, training: bool = False):
        input_shape = inputs.shape
        _, seq_len, _ = input_shape

        mask = causal_attention_mask(seq_len)

        attention_output = self.mha(
            inputs_q=inputs,
            mask=mask,
            decode=False
        )
        attention_output = self.dropout1(attention_output, deterministic=not training)
        out1 = self.layer_norm1(inputs + attention_output)

        ffn_output = self.linear1(out1)
        ffn_output = nnx.relu(ffn_output)
        ffn_output = self.linear2(ffn_output)
        ffn_output = self.dropout2(ffn_output, deterministic=not training)
        return self.layer_norm2(out1 + ffn_output)

class TokenAndPositionEmbedding(nnx.Module):
    """ Combines token embeddings (words in an input sentence) with
    positional embeddings (the position of each word in a sentence).

    Args:
        maxlen (int): Matimum sequence length.
        vocal_size (int): Vocabulary size.
        embed_dim (int): Embedding dimensionality.
        rngs (flax.nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
    """
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.token_emb = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(num_embeddings=maxlen, features=embed_dim, rngs=rngs)

    def __call__(self, x):
        positions = jnp.arange(0, x.shape[1])[None, :]
        position_embedding = self.pos_emb(positions)
        token_embedding = self.token_emb(x)
        return token_embedding + position_embedding

class MiniGPT(nnx.Module):
    """ A miniGPT transformer model, inherits from `flax.nnx.Module`.

    Args:
        maxlen (int): Maximum sequence length.
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimensionality.
        num_heads (int): Number of attention heads.
        feed_forward_dim (int): Dimensionality of the feed-forward network.
        num_transformer_blocks (int): Number of transformer blocks. Each block contains attention and feed-forward networks.
        rngs (nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
    """
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, num_heads: int, feed_forward_dim: int, num_transformer_blocks: int, rngs: nnx.Rngs):
        self.embedding_layer = TokenAndPositionEmbedding(
                    maxlen, vocab_size, embed_dim, rngs=rngs
                )
        self.transformer_blocks = nnx.List([TransformerBlock(
            embed_dim, num_heads, feed_forward_dim, rngs=rngs
        ) for _ in range(num_transformer_blocks)])
        self.output_layer = nnx.Linear(in_features=embed_dim,
                                       out_features=vocab_size,
                                       kernel_init=nnx.with_partitioning(
                                           nnx.initializers.xavier_uniform(), P(None, 'model')),
                                       bias_init=nnx.with_partitioning(
                                           nnx.initializers.zeros_init(), P('model')),
                                       rngs=rngs)

    def __call__(self, inputs, training: bool = False):
        x = self.embedding_layer(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        outputs = self.output_layer(x)
        return outputs

    # For Tunix use later
    def get_model_input(self):
        return dict(
            inputs=jnp.zeros((batch_size, maxlen), dtype=jnp.int32),
            training=False
        )

    @nnx.jit
    def sample_from(self, rng_key, logits):
        logits, indices = jax.lax.top_k(logits, k=top_k)
        logits = nnx.softmax(logits)
        return jax.random.choice(rng_key, indices, p=logits)

    @nnx.jit
    def generate_step(self, rng_key, padded_tokens, sample_index):
        logits = self(padded_tokens)
        next_token = self.sample_from(rng_key, logits[0][sample_index])
        return next_token

    def generate_text(self, max_tokens, start_tokens):
        generated = []
        rng_key = jax.random.PRNGKey(0) # Create the initial key
        print(tokenizer.decode(start_tokens), flush=True, end='')
        for i in range(max_tokens):
            sample_index = len(start_tokens) + len(generated) - 1

            # Split the key for each step
            rng_key, step_key = jax.random.split(rng_key)

            padded_tokens = jnp.array((start_tokens + generated + [0] * (maxlen - len(start_tokens) - len(generated))))[None, :]
            next_token = int(self.generate_step(step_key, padded_tokens, sample_index))
            if next_token == tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]:
              break
            generated.append(next_token)
            print(tokenizer.decode([next_token]), flush=True, end='')
        return tokenizer.decode(start_tokens + generated)

def create_model(rngs):
    return MiniGPT(maxlen, vocab_size, embed_dim, num_heads, feed_forward_dim, num_transformer_blocks=4, rngs=rngs)
```

+++ {"id": "igX_eoGNMTGR"}

Set some hyperparameters.

```{code-cell}
:id: GRhiDsCrMZRp

vocab_size = tokenizer.n_vocab
num_transformer_blocks = 8
maxlen = 256
embed_dim = 256
num_heads = 8
feed_forward_dim = 256
batch_size = 144 * jax.device_count() / 2  # divide by 2 in case of model parallelism
if jax.device_count() == 1:
    batch_size = 144
num_epochs = 1
top_k = 10
```

+++ {"id": "mI1ci-HyMspJ"}

## Grain: Load and preprocess the data

[REVAMP 2]: Expand the intro to Grain.

Data loading and preprocessing with [Grain](https://github.com/google/grain).

```{code-cell}
:id: rGUFsn1GMuzh

@dataclass
class TextDataset:
    data: list
    maxlen: int

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        # Use Tiktoken for tokenization
        encoding = tokenizer.encode(self.data[idx], allowed_special={'<|endoftext|>'})[:self.maxlen]  # Tokenize and truncate
        return encoding + [0] * (self.maxlen - len(encoding))  # Pad to maxlen

def load_and_preprocess_data(file_path, batch_size, maxlen):

    with open(file_path, 'r') as f:
      text = f.read()

    stories = text.split('<|endoftext|>')
    stories = [story+'<|endoftext|>' for story in stories if story.strip()]
    df = pd.DataFrame({'text': stories})
    data = df['text'].dropna().tolist()
    dataset = TextDataset(data, maxlen)

    sampler = pygrain.IndexSampler(
        len(dataset),
        shuffle=False,
        seed=42,
        shard_options=pygrain.NoSharding(),
        num_epochs=num_epochs,
    )

    dl = pygrain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[pygrain.Batch(batch_size=batch_size, drop_remainder=True)],
    )

    return dl
```

```{code-cell}
:cellView: form
:id: b9tmfMzj7eso

# @title [hidden cell; used for testing]
# This cell is run only in the JAX AI Stack's CI testing and should otherwise be ignored.
import os
AI_STACK_TEST_MODE = os.getenv('AI_STACK_TEST_MODE') == 'true'

if AI_STACK_TEST_MODE:
    num_transformer_blocks = 2
    maxlen = 16
    embed_dim = 16
    num_heads = 2
    feed_forward_dim = 8

    def load_and_preprocess_data(file_path, batch_size, maxlen):
        del file_path

        @dataclass
        class TestTextDataset:
            maxlen: int

            def __len__(self):
                return 64

            def __getitem__(self, idx: int):
                encoding = jax.random.randint(jax.random.key(idx), [self.maxlen], minval=0, maxval=1e6)
                return jnp.unstack(encoding)

        dataset = TestTextDataset(maxlen)

        sampler = pygrain.IndexSampler(
            len(dataset),
            shuffle=False,
            seed=42,
            shard_options=pygrain.NoSharding(),
            num_epochs=num_epochs,
        )

        dl = pygrain.DataLoader(
            data_source=dataset,
            sampler=sampler,
            operations=[pygrain.Batch(batch_size=batch_size, drop_remainder=True)],
        )

        return dl
```

```{code-cell}
:id: VnfW4Z7l7eso

text_dl = load_and_preprocess_data('TinyStories-train.txt', batch_size, maxlen)
```

+++ {"id": "BKVSD8KSM1um"}

## Defining the loss function and training step function

```{code-cell}
:id: 8rRuTmABNV4b

# Defines the loss function using `optax.softmax_cross_entropy_with_integer_labels`.
def loss_fn(model, batch):
    logits = model(batch[0])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch[1]).mean()
    return loss, logits

# Define the training step with the `flax.nnx.jit` transformation decorator.
@nnx.jit
def train_step(model: MiniGPT, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, lables=batch[1])
    optimizer.update(model, grads)
```

+++ {"id": "5um2vkeUNckm"}

## Optax: Train the model

[REVAMP 3]: Expand the intro to Optax.

Start training. It takes ~20 minutes on Colab TPU v5e-1.

Note that for data parallel, we are sharding the training data along the `batch` axis using `jax.device_put`.

We are also using the `jax.vmap` transformation to produce the target sequences faster.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Ysl6CsfENeJN
outputId: ba3051ad-0e11-4570-a223-28f35ca505e0
---
with mesh:
  model = create_model(rngs=nnx.Rngs(0))
  optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)
rng = jax.random.PRNGKey(0)

start_prompt = "Once upon a time"
start_tokens = tokenizer.encode(start_prompt)[:maxlen]
print("Initial generated text:")
generated_text = model.generate_text(maxlen, start_tokens)

metrics_history = {
    "train_loss": [],
}

prep_target_batch = jax.vmap(
    lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0])))
)

step = 0
for epoch in range(num_epochs):
    start_time = time.time()
    for batch in text_dl:
        if len(batch) % len(jax.devices()) != 0:
            continue  # skip the remaining elements
        input_batch = jnp.array(jnp.array(batch).T)
        target_batch = prep_target_batch(input_batch)
        train_step(
            model,
            optimizer,
            metrics,
            jax.device_put(
                (input_batch, target_batch), NamedSharding(mesh, P("batch", None))
            ),
        )

        if (step + 1) % 200 == 0:
            for metric, value in metrics.compute().items():
                metrics_history[f"train_{metric}"].append(value)
            metrics.reset()

            elapsed_time = time.time() - start_time
            print(
                f"\n\nStep {step + 1}, Loss: {metrics_history['train_loss'][-1]}, Elapsed Time: {elapsed_time:.2f} seconds"
            )
            start_time = time.time()

            print("Generated text:")
            generated_text = model.generate_text(maxlen, start_tokens)

        step += 1
        # break

# Final text generation
print("Final generated text:")
generated_text = model.generate_text(maxlen, start_tokens)
```

+++ {"id": "thaLs6TD0lt5"}

Visualize the training loss.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 472
id: B6Eg1Cz2y_iP
outputId: 60e54019-dcdc-425b-e95a-c03e301d688f
---
import matplotlib.pyplot as plt
plt.plot(metrics_history['train_loss'])
plt.title('Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()
```

+++ {"id": "WB-ExEt1Zl1C"}

As you can see, the model goes from generating completely random words at the beginning to generating sensible tiny stories at the end of the training. So essentially we have pretrained a small LLM to write tiny stories for us.

+++ {"id": "d1_l1Pr-7eso"}

## Debugging the model

[REVAMP 4]: Add a debugging section.

+++ {"id": "soPqiR1JNmjf"}

## Orbax: Save the checkpoint

[REVAMP 5] Introduce Orbax.

Save the model checkpoint.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: EkoFGCgSZ1yz
outputId: 593af986-77a7-4665-eee4-5deb2d891b7f
---
import orbax.checkpoint as orbax

state = nnx.state(model)

checkpointer = orbax.PyTreeCheckpointer()
checkpointer.save('/content/save', args=orbax.args.PyTreeSave(state), force=True)

# Make sure the files are there
!ls /content/save/
```

+++ {"id": "azzwiMeP7eso"}

## Tunix: Fine-tuning

[Tunix](https://github.com/google/tunix) is a JAX-native LLM post-training library open sourced by Google. It supports a range of post-training techniques including supervised finetuning, preference tuning, reinforcement learning and model distillation. So in this section, we are going to use Tunix to finetune the miniGPT model we just pretrained using LoRA ([Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)) so that the finetuned model generates output of a different style.

+++ {"id": "-kJShd9n_iTl"}

First we install Tunix and its dependencies, and import necessary libraries.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: uipLvy9E7eso
outputId: 93cf001f-2964-467c-9848-b106be6cacf5
---
!pip install google-tunix[prod] --no-deps
!pip install qwix tensorboardX tensorflow

import qwix
from tunix.sft import peft_trainer
import tensorflow as tf
import tensorflow_datasets as tfds
```

+++ {"id": "cE0Rx6Q3_q4n"}

We set some hyperparameters.

```{code-cell}
:id: YzEDIR_N7_D4

# LoRA Hyperparameters
lora_rank = 16
lora_alpha = 2.0
lora_max_steps = 400
lora_num_epochs = 10
lora_batch_size = 80
```

+++ {"id": "mfnj-S02_yM5"}

Previously we used Grain to load the Tiny Stories dataset. JAX is actually flexible to work with other dataloaders as well. So for LoRA fintuning we use TensorFlow Datasets to load the [Tiny Shakespeare](https://www.tensorflow.org/datasets/catalog/tiny_shakespeare) dataset.

```{code-cell}
:id: Mtzb0NXb8TVY

def load_shakespeare_dataset(batch_size, max_len, num_epochs):
    # Load dataset from TFDS
    train_ds_raw = tfds.load("tiny_shakespeare", split="train", as_supervised=False)

    # Define a wrapper for the tokenizer to use in tf.data.map
    def tokenize_fn(text_tensor):
        text = text_tensor.numpy().decode("utf-8")
        tokens = tokenizer.encode(text)
        return tf.constant(tokens, dtype=tf.int32)

    def py_tokenize_fn(text_tensor):
        # tf.py_function is needed to run python code in a tf.data pipeline
        return tf.py_function(tokenize_fn, [text_tensor], tf.int32)

    # Map to the final training input format for both datasets
    def to_training_input_dict(batch):
        return {
            "input_tokens": batch,
            "input_mask": tf.ones_like(batch)
        }

    # Process Training Data in a single pipeline
    train_dataset = (
        train_ds_raw.map(lambda x: py_tokenize_fn(x['text']))
        .flat_map(tf.data.Dataset.from_tensor_slices)
        .batch(max_len, drop_remainder=True)
        .shuffle(buffer_size=10000)
        .repeat(num_epochs)
        .batch(batch_size, drop_remainder=True)
        .map(to_training_input_dict)
    )

    def to_training_input(ds):
        # The trainer expects an iterable of `peft_trainer.TrainingInput`.
        for item in tfds.as_numpy(ds):
            yield peft_trainer.TrainingInput(**item)

    return to_training_input(train_dataset)

lora_train_ds = load_shakespeare_dataset(lora_batch_size, maxlen, lora_num_epochs)
```

+++ {"id": "qvxSxbbcBim7"}

We define a few helper functions to create the LoRA model, loss and etc.

```{code-cell}
:id: oX9F-ZsN8ima

def get_lora_model(base_model, mesh):
  lora_provider = qwix.LoraProvider(
      module_path=".*mha|.*linear1|.*linear2",
      rank=lora_rank,
      alpha=lora_alpha,
  )

  model_input = base_model.get_model_input()
  lora_model = qwix.apply_lora_to_model(
      base_model, lora_provider, **model_input
  )

  with mesh:
    state = nnx.state(lora_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_model, sharded_state)

  return lora_model

def gen_model_input_fn(x: peft_trainer.TrainingInput):
  return {
      'inputs': x.input_tokens,
      'training': True
  }

def lora_loss_fn(model, inputs, training):
    inputs = inputs
    targets = jnp.concatenate([inputs[:, 1:], jnp.zeros((inputs.shape[0], 1), dtype=jnp.int32)], axis=1)
    logits = model(inputs, training=training)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets).mean()
    return loss
```

+++ {"id": "h4Miifz_BpXf"}

Now we can start the finetuning.

```{code-cell}
:id: pIQ5Obif8pfO

print("Starting LoRA Finetuning...")
with mesh:
    # Apply LoRA to the model
    lora_model = get_lora_model(model, mesh)

    # Setup Tunix PeftTrainer
    training_config = peft_trainer.TrainingConfig(
        eval_every_n_steps=None,
        max_steps=lora_max_steps,
        data_sharding_axis=('batch',),
    )
    lora_optimizer = optax.adamw(1e-2)
    lora_trainer = peft_trainer.PeftTrainer(
        lora_model, lora_optimizer, training_config
    ).with_gen_model_input_fn(gen_model_input_fn).with_loss_fn(lora_loss_fn)

    # Run LoRA training
    lora_trainer.train(lora_train_ds)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 799
id: -uTeXvXaLCZb
outputId: aa52154f-9583-46d3-896a-05e28de603de
---
# Generate text with LoRA-finetuned model
print("\nGenerating text after LoRA finetuning:\n\n")
lora_model.generate_text(maxlen, start_tokens)
```

+++ {"id": "UWVdZGK9COj0"}

After the finetuning, you can easily see that now the model produces text of a different style, kind of like Shakespeare's work, which means our finetuning works.

+++ {"id": "3813cbf2"}

## Xprof: profiling for hyperparameter tuning

[REVAMP 7]: Intro xprof

**Note:** this section assume multiple TPU cores. Free-tier Colab TPU v5e-1 cannot run here.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: b5d933c6
outputId: f39e18fc-a25f-4202-ab39-9aadfb232522
---
!pip install -Uq tensorboard-plugin-profile tensorflow tensorboard
```

+++ {"id": "2ac5fc4d"}

Load the tensorboard colab extension.

```{code-cell}
:id: 74f0c212

%load_ext tensorboard
```

+++ {"id": "17c6131f"}

As we're going to be running this model a number of times, we need some scaffolding to more easily compare our work. For a baseline, we'll need to perform some warmup to guarantee that our code is JIT'd and that our TPUs are warm. For improved comparability, we'll only start tracing after we've finished warmup.

```{code-cell}
:id: ddfd576e

trace_dir = "/tmp/jax-trace/"

def loop_step(batch, step):
    input_batch = jnp.array(jnp.array(batch).T)
    target_batch = prep_target_batch(input_batch)
    with jax.set_mesh(mesh):
        train_step(model, optimizer, metrics, jax.device_put((input_batch, target_batch), P('batch', None)))

def generate_trace():
    tracing_steps = 30
    warmup_steps = 5
    for current_step in range(warmup_steps + tracing_steps):
        if current_step == warmup_steps:
            jax.profiler.start_trace(trace_dir)
        with jax.profiler.StepTraceAnnotation("train", step_num=current_step):
            batch = next(text_dl)
            loop_step(batch, current_step)

    jax.profiler.stop_trace()
```

+++ {"id": "de70f5b7"}

Now we'll perform some traces to compare results of different batch sizes. This will take several minutes as we need to reprocess our input data to prepare new batches each time.

```{code-cell}
:id: bc9452a6

trace_dir = "/tmp/jax-trace-batch-comparison/"

batch_size = 64
text_dl = iter(load_and_preprocess_data('TinyStories-train.txt', batch_size, maxlen))
generate_trace()

batch_size = 256
text_dl = iter(load_and_preprocess_data('TinyStories-train.txt', batch_size, maxlen))
generate_trace()
```

+++ {"id": "ea379965"}

Run Tensorboard with the Profiler Plugin to compare our runs. Runs are listed in order from newest to oldest, so the top run in the list will be have `batch_size = 256`.

The key metrics to focus on here for this hyperparameter are FLOPS Utilization and Average Step Time.

In general, we want to maximize FLOPS Utilization while minimizing the step time per training example. In this case, we can see that increasing the batch size from 64 -> 256 achieves both of those. FLOPS increases from 16% to 27%. Average Step Time increase from 100ms to 260ms, however we increased our batch size by 300%. This means we move from 1.5ms per training example to 1.02ms per training example.

```{code-cell}
:id: b86c565a

%tensorboard --logdir=$trace_dir
```

+++ {"id": "657967a5"}

Next, we can explore alternative parallelism methods. In cell #4, we used 4-way data parallel and 2-way tensor parallel. 8-way data parallel is another popular way. Let's compare results between them. To switch to 8-way data parallel, we'll replace the `Mesh` definition with:

`mesh = jax.make_mesh((8, 1), ('batch', 'model'))`

JAX will automatically figure out how to shard the model and data to use the new partition strategy and nothing else need to be done. Re-connect the TPU runtime and run it again to see how it runs.

How simple and powerful is this! And that's the beauty of JAX automatic parallelism.

```{code-cell}
:id: 80daa8dc

trace_dir = "/tmp/jax-trace-parallelism-comparison/"

mesh_dims = (4, 2) if jax.device_count() == 8 else (1, 1)
mesh = jax.make_mesh(mesh_dims, ('batch', 'model'))
generate_trace()

mesh = jax.make_mesh((jax.device_count(), 1), ('batch', 'model'))
generate_trace()
```

+++ {"id": "ad96e72b"}

Once again we'll run tensorboard.

Looking at the results, we see that the step times are nearly the same, however the FLOPS Utilization is at 13% for 8-way data parallelism compared to 27% or 4-way data parallelism.

By looking at the Trace Viewer tool and looking under each TPU's ops, we can see that the TPUs spend a large amount of time idle while waiting for the host, as well as spending a good amount of time in `reduce_sum` operations.

```{code-cell}
:id: 780e9c72

%tensorboard --logdir=$trace_dir
```

+++ {"id": "deca486e"}

By changing hyperparameters and comparing profiles, we're able to gain significant insights into our bottlenecks and limitations. These are just two examples of hyperparameters to tune, but plenty more of them will have significant effects on training speed and resource utilization.

+++ {"id": "Q3Z_0hQu7est"}

## Inference with vLLM

After training the miniGPT model, we can also serve it on Google TPUs for high-performance inference.

[vLLM TPU](https://github.com/vllm-project/tpu-inference/) supports running LLMs on TPUs. It takes some additional work to make it work, which is beyond the scope of this tutorial. But feel free to checkout vLLM TPU [docuementation](https://docs.vllm.ai/projects/tpu/en/latest/developer_guides/jax_model_development/) if you want to learn more about it.
