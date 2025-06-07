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

+++ {"id": "NIOXoY1xgiww"}

# Train a miniGPT language model with JAX

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jax-ml/jax-ai-stack/blob/main/docs/source/JAX_for_LLM_pretraining.ipynb)

This tutorial demonstrates how to use JAX, [Flax NNX](http://flax.readthedocs.io) and [Optax](http://optax.readthedocs.io) for language model (pre)training using data and tensor [parallelism](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization) for [Single-Program Multi-Data](https://en.wikipedia.org/wiki/Single_program,_multiple_data)). It was originally inspired by the [Keras miniGPT tutorial](https://keras.io/examples/generative/text_generation_with_miniature_gpt/).

Here, you will learn how to:

- Define the miniGPT model with Flax and JAX automatic parallelism
- Load and preprocess the dataset
- Create the loss and training step functions
- Train the model on Google Colab’s Cloud TPU v2
- Profile for hyperparameter tuning

If you are new to JAX for AI, check out the [introductory tutorial](https://jax-ai-stack.readthedocs.io/en/latest/neural_net_basics.html), which covers neural network building with [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html).

+++ {"id": "hTmz5Cbco7n_"}

## Setup

JAX installation is covered in [this guide](https://jax.readthedocs.io/en/latest/installation.html) on the JAX documentation site. We will use [Tiktoken](https://github.com/openai/tiktoken) for tokenization and [Grain](https://google-grain.readthedocs.io/en/latest/index.html) for data loading.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 6zMsOIc7ouCO
outputId: 037d56a9-b18f-4504-f80a-3a4fa2945068
---
!pip install -Uq tiktoken grain matplotlib
```

+++ {"id": "Rcji_799n4eA"}

**Note:** If you are using [Google Colab](https://colab.research.google.com/), select the free Google Cloud TPU v2 as the hardware accelerator.

Check the available JAX devices, or [`jax.Device`](https://jax.readthedocs.io/en/latest/_autosummary/jax.Device.html), with [`jax.devices()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.devices.html). The output of the cell below will show a list of 8 (eight) devices.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: LS9sQEY3n0mB
outputId: 9ffcf3a6-20ef-4f80-b006-f5d3c5644a15
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
outputId: e6eff24e-5578-4277-a0f9-24e27bd91ee0
---
!wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt?download=true -O TinyStories-train.txt
```

+++ {"id": "sKE2uUafLobI"}

Import the necessary modules, including JAX NumPy, Flax NNX, Optax, Grain, pandas, and Tiktoken:

```{code-cell}
:id: MKYFNOhdLq98

import jax
import jax.numpy as jnp

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding # For data and model parallelism (explained in more detail later)
from jax.experimental import mesh_utils

import flax.nnx as nnx
import optax

from dataclasses import dataclass
import grain.python as pygrain
import pandas as pd
import tiktoken
import time
```

+++ {"id": "rPyt7MV6prz1"}

## Define the miniGPT model with Flax and JAX automatic parallelism

### Leveraging JAX's data and tensor parallelism

One of the most powerful features of JAX is [device parallelism](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization) for SPMD.

- The data parallelism technique enables, for example, the training data to run via multiple parts (this is called sharding) - batches - in parallel and simultaneously across different devices, such as GPUs and Google TPUs. This allows to use larger batch sizes to speed up training.
- Tensor parallelism allows us to split the model parameter tensors across several devices (sharding model tensors).
- You can learn more about the basics of JAX parallelism in more detail in the [Introduction to parallel programming](https://jax.readthedocs.io/en/latest/sharded-computation.html) on the JAX documentation site.

In this example, we'll utilize a 4-way data parallel and 2-way tensor parallel setup. The free Google Cloud TPU v2 on Google Colab offers 4 chips, each with 2 TPU cores. The TPU v2 architeture aligns with the proposed setup.

### jax.sharding.Mesh

Earlier, we imported [`jax.sharding.Mesh`](https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.Mesh) - is a multidimensional NumPy array of JAX devices, where each axis of the mesh has a name, such as `'x'` or `'y'`. This will help encapsulate the information about the TPU resource organization for distributing computations across the devices.

Our `Mesh` will have two arguments:
- `devices`: This will take the value of [`jax.experimental.mesh_utils((4, 2))`](https://jax.readthedocs.io/en/latest/jax.experimental.mesh_utils.html), enabling us to build a device mesh. It is a NumPy ndarray with JAX devices (a list of devices from the JAX backend as obtained from [`jax.devices()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.devices.html#jax.devices))..
- `axis_names`, where:
  - `batch`: 4 devices along the first axis - i.e. sharded into 4 - for data parallelism; and
  - `model`: 2 devices along the second axis - i.e. sharded into 2 -  for tensor paralleism, mapping to the TPU v2 cores.

This matches the `(4, 2)` structure in the Colab's TPU v2 setup.

Let's instantiate `Mesh` as `mesh` and declare the TPU configuration to define how data and model parameters are distributed across the devices:

```{code-cell}
:id: xuMlCK3Q8WJD

# Create a `Mesh` object representing TPU device arrangement.
mesh = Mesh(mesh_utils.create_device_mesh((4, 2)), ('batch', 'model'))

### Alternatively, we could use the 8-way data parallelism with only one line of code change.
### JAX enables quick experimentation with different partitioning strategies
### like this. We will come back to this point at the end of this tutorial.
# mesh = Mesh(mesh_utils.create_device_mesh((8, 1)), ('batch', 'model'))
```

+++ {"id": "_ZKdhNo98NgG"}

We will use the GPT-2 tokenizer from the [Tiktoken](https://github.com/openai/tiktoken) library:

```{code-cell}
:id: iWbkk1V7-Isg

tokenizer = tiktoken.get_encoding("gpt2")
```

+++ {"id": "0XHQ0BQ9-KIj"}

To leverage model parallelism, we need to instruct the JAX compiler how to shard the model tensors across the TPU devices. Earlier, we also imported [`jax.sharding.PartitionSpec`](https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.PartitionSpec) and [`jax.sharding.NamedSharding`](https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.NamedSharding):
- [`PartitionSpec`](https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.PartitionSpec) (using alias `P`) defines how tensors are sharded across the devices in our `Mesh`. Its elements describe how an input dimension is partitioned across mesh dimensions. For example, in `PartitionSpec('x', 'y')` the first dimension of data is sharded across `x` axis of the mesh, and the second one - across the `y` axis.
  - We'll use `PartitionSpec` to describe how to shard a tensor across, for example, the `model` axis or be replicated on other dimensions (which is denoted by `None`).
- [`NamedSharding`](https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.NamedSharding) is a (`Mesh`, `PartitionSpec`) pair that describes how to shard a model tensor across our `mesh`.
- We combine `Mesh` (the TPU resources) with `PartitionSpec` and create a `NamedSharding`, which instructs how to shard each model tensor across the TPU devices.

Additionally, we'll use Flax NNX's [`flax.nnx.with_partitioning`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/spmd.html#flax.nnx.with_partitioning) to let each model layer know that the model weights or tensors need to be sharded according to our specification. We need to do this for every tensor/layer in the model.
- `nnx.with_partitioning` will take two arguments, such as the `initializer` (such as [`flax.nnx.initializers.xavier_uniform`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/initializers.html#flax.nnx.initializers.xavier_uniform) and [`flax.nnx.initializers.zeros_init`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/initializers.html#flax.nnx.initializers.zeros_init)) and `sharding` (e.g. `NamedSharding(Mesh, PartitionSpec)` or `NamedSharding(mesh, P('model')` in our case).

```{code-cell}
:id: z0p-IHurrB9i

# Define a triangular mask for causal attention with `jax.numpy.tril` and `jax.numpy.ones`.
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
        # Multi-Head Attention (MHA) with `flax.nnx.MultiHeadAttention`.
        # Specifies tensor sharding (depending on the mesh configuration)
        # where we shard the weights across devices for parallel computation.
        self.mha = nnx.MultiHeadAttention(num_heads=num_heads,
                                          in_features=embed_dim,
                                          kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
                                          bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
                                          rngs=rngs)
        # The first dropout with `flax.nnx.Dropout`.
        self.dropout1 = nnx.Dropout(rate=rate)
        # First layer normalization with `flax.nnx.LayerNorm`.
        self.layer_norm1 = nnx.LayerNorm(epsilon=1e-6,
                                         num_features=embed_dim,
                                         scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P('model'))),
                                         bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
                                         rngs=rngs)
        # The first linear transformation for the feed-forward network with `flax.nnx.Linear`.
        self.linear1 = nnx.Linear(in_features=embed_dim,
                                  out_features=ff_dim,
                                  kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
                                  bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
                                  rngs=rngs)
        # The second linear transformation for the feed-forward network with `flax.nnx.Linear`.
        self.linear2 = nnx.Linear(in_features=ff_dim,
                                  out_features=embed_dim,
                                  kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
                                  bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
                                  rngs=rngs)
        # The second dropout with `flax.nnx.Dropout`.
        self.dropout2 = nnx.Dropout(rate=rate)
        # Second layer normalization with `flax.nnx.LayerNorm`.
        self.layer_norm2 = nnx.LayerNorm(epsilon=1e-6,
                                         num_features=embed_dim,
                                         scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model'))),
                                         bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P(None, 'model'))),
                                         rngs=rngs)


    # Apply the Transformer block to the input sequence.
    def __call__(self, inputs, training: bool = False):
        input_shape = inputs.shape
        _, seq_len, _ = input_shape

        # Instantiate the causal attention mask.
        mask = causal_attention_mask(seq_len)

        # Apply Multi-Head Attention with the causal attention mask.
        attention_output = self.mha(
            inputs_q=inputs,
            mask=mask,
            decode=False
        )
        # Apply the first dropout.
        attention_output = self.dropout1(attention_output, deterministic=not training)
        # Apply the first layer normalization.
        out1 = self.layer_norm1(inputs + attention_output)

        # The feed-forward network.
        # Apply the first linear transformation.
        ffn_output = self.linear1(out1)
        # Apply the ReLU activation with `flax.nnx.relu`.
        ffn_output = nnx.relu(ffn_output)
        # Apply the second linear transformation.
        ffn_output = self.linear2(ffn_output)
        # Apply the second dropout.
        ffn_output = self.dropout2(ffn_output, deterministic=not training)
        # Apply the second layer normalization and return the output of the Transformer block.
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
        # Initialize token embeddings (using `flax.nnx.Embed`).
        # Each unique word has an embedding vector.
        self.token_emb = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
        # Initialize positional embeddings (using `flax.nnx.Embed`).
        self.pos_emb = nnx.Embed(num_embeddings=maxlen, features=embed_dim, rngs=rngs)

    # Takes a token sequence (integers) and returns the combined token and positional embeddings.
    def __call__(self, x):
        # Generate a sequence of positions for the input tokens.
        positions = jnp.arange(0, x.shape[1])[None, :]
        # Look up the positional embeddings for each position in the input sequence.
        position_embedding = self.pos_emb(positions)
        # Look up the token embeddings for each token in the input sequence.
        token_embedding = self.token_emb(x)
        # Combine token and positional embeddings.
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
    # Initialize miniGPT model components.
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, num_heads: int, feed_forward_dim: int, num_transformer_blocks: int, rngs: nnx.Rngs):
        # Initiliaze the `TokenAndPositionEmbedding` that combines token and positional embeddings.
        self.embedding_layer = TokenAndPositionEmbedding(
                    maxlen, vocab_size, embed_dim, rngs=rngs
                )
        # Create a list of `TransformerBlock` instances.
        # Each block processes input sequences using attention and feed-forward networks.
        self.transformer_blocks = [TransformerBlock(
            embed_dim, num_heads, feed_forward_dim, rngs=rngs
        ) for _ in range(num_transformer_blocks)]
        # Initialize the output `flax.nnx.Linear` layer producing logits over the vocabulary for next-token prediction.
        self.output_layer = nnx.Linear(in_features=embed_dim,
                                       out_features=vocab_size,
                                       kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
                                       bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P(None, 'model'))),
                                       rngs=rngs)

    def __call__(self, inputs, training: bool = False):
        # Pass the input tokens through the `embedding_layer` to get token embeddings.
        # Apply each transformer block sequentially to the embedded input, use the `training` flag for the behavior of `flax.nnx.Dropout`.
        x = self.embedding_layer(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        # Pass the output of the transformer blocks through the output layer,
        # and obtain logits for each token in the vocabulary (for next token prediction).
        outputs = self.output_layer(x)
        return outputs

    # Text generation.
    def generate_text(self, max_tokens: int, start_tokens: [int], top_k=10):
        # Sample the next token from a probability distribution based on
        # `logits` and `tok_k` (top-k) sampling strategy.
        def sample_from(logits):
            logits, indices = jax.lax.top_k(logits, k=top_k)
            # Convert logits to probabilities (using `flax.nnx.softmax`).
            logits = nnx.softmax(logits)
            return jax.random.choice(jax.random.PRNGKey(0), indices, p=logits)

        # Generate text one token at a time until the maximum token limit is reached (`maxlen`).
        def generate_step(start_tokens):
            pad_len = maxlen - len(start_tokens)
            # Index of the last token in the current sequence.
            sample_index = len(start_tokens) - 1
            # If the input is longer than `maxlen`, then truncate it.
            if pad_len < 0:
                x = jnp.array(start_tokens[:maxlen])
                sample_index = maxlen - 1
            # If the input is shorter than `maxlen`, then pad it (`pad_len`).
            elif pad_len > 0:
                x = jnp.array(start_tokens + [0] * pad_len)
            else:
                x = jnp.array(start_tokens)

            # Add a batch dimension.
            x = x[None, :]
            logits = self(x)
            next_token = sample_from(logits[0][sample_index])
            return next_token

        # Store generated tokens.
        generated = []
        # Generate tokens until the end-of-text token is encountered or the maximum token limit is reached.
        for _ in range(max_tokens):
            next_token = generate_step(start_tokens + generated)
            # Truncate whatever is after '<|endoftext|>' (stop word)
            if next_token == tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]:
              # Stop text generation if the end-of-text token is encountered.
              break
            generated.append(int(next_token))
        # Decode the generated token IDs into text.
        return tokenizer.decode(start_tokens + generated)

# Creates the miniGPT model with 4 transformer blocks.
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
batch_size = 256 # You can set a bigger batch size if you use Kaggle's Cloud TPU.
num_epochs = 1
```

+++ {"id": "mI1ci-HyMspJ"}

## Loading and preprocessing the data

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
    optimizer.update(grads)
```

+++ {"id": "5um2vkeUNckm"}

## Training the model

Start training. It takes ~50 minutes on Colab.

Note that for data parallel, we are sharding the training data along the `batch` axis using `jax.device_put` with `NamedSharding`.

We are also using the `jax.vmap` transformation to produce the target sequences faster.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Ysl6CsfENeJN
outputId: 5dd06dca-f030-4927-a9b6-35d412da535c
---
model = create_model(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3))
metrics = nnx.MultiMetric(
  loss=nnx.metrics.Average('loss'),
)
rng = jax.random.PRNGKey(0)

start_prompt = "Once upon a time"
start_tokens = tokenizer.encode(start_prompt)[:maxlen]
generated_text = model.generate_text(
    maxlen, start_tokens
)
print(f"Initial generated text:\n{generated_text}\n")


metrics_history = {
  'train_loss': [],
}

prep_target_batch = jax.vmap(lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0]))))

step = 0
for epoch in range(num_epochs):
    start_time = time.time()
    for batch in text_dl:
        if len(batch) % len(jax.devices()) != 0:
          continue  # skip the remaining elements
        input_batch = jnp.array(jnp.array(batch).T)
        target_batch = prep_target_batch(input_batch)
        train_step(model, optimizer, metrics, jax.device_put((input_batch, target_batch), NamedSharding(mesh, P('batch', None))))

        if (step + 1) % 200 == 0:
          for metric, value in metrics.compute().items():
              metrics_history[f'train_{metric}'].append(value)
          metrics.reset()

          elapsed_time = time.time() - start_time
          print(f"Step {step + 1}, Loss: {metrics_history['train_loss'][-1]}, Elapsed Time: {elapsed_time:.2f} seconds")
          start_time = time.time()

          generated_text = model.generate_text(
              maxlen, start_tokens
          )
          print(f"Generated text:\n{generated_text}\n")
        step += 1

# Final text generation
generated_text = model.generate_text(
    maxlen, start_tokens
)
print(f"Final generated text:\n{generated_text}")
```

+++ {"id": "thaLs6TD0lt5"}

Visualize the training loss.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 472
id: B6Eg1Cz2y_iP
outputId: 7cafe711-1ae4-4eb9-fd37-e1bde54cbfc5
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

+++ {"id": "soPqiR1JNmjf"}

## Saving the checkpoint

Save the model checkpoint.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: EkoFGCgSZ1yz
outputId: 3467b8ba-ce05-42f0-fb89-75922cc91e31
---
import orbax.checkpoint as orbax

state = nnx.state(model)

checkpointer = orbax.PyTreeCheckpointer()
checkpointer.save('/content/save', state)

# Make sure the files are there
!ls /content/save/
```

## Profiling for hyperparameter tuning

```{code-cell}
!pip install -Uq tensorboard-plugin-profile tensorflow tensorboard
```

Load the tensorboard colab extension.

```{code-cell}
%load_ext tensorboard
```

As we're going to be running this model a number of times, we need some scaffolding to more easily compare our work. For a baseline, we'll need to perform some warmup to guarantee that our code is JIT'd and that our TPUs are warm. For improved comparability, we'll only start tracing after we've finished warmup.

```{code-cell}
trace_dir = "/tmp/jax-trace/"

def loop_step(batch, step):
    input_batch = jnp.array(jnp.array(batch).T)
    target_batch = prep_target_batch(input_batch)
    train_step(model, optimizer, metrics, jax.device_put((input_batch, target_batch), NamedSharding(mesh, P('batch', None))))

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

Now we'll perform some traces to compare results of different batch sizes. This will take several minutes as we need to reprocess our input data to prepare new batches each time.

```{code-cell}
trace_dir = "/tmp/jax-trace-batch-comparison/"

batch_size = 64
text_dl = iter(load_and_preprocess_data('TinyStories-train.txt', batch_size, maxlen))
generate_trace()

batch_size = 256
text_dl = iter(load_and_preprocess_data('TinyStories-train.txt', batch_size, maxlen))
generate_trace()
```

Run Tensorboard with the Profiler Plugin to compare our runs. Runs are listed in order from newest to oldest, so the top run in the list will be have `batch_size = 256`.

The key metrics to focus on here for this hyperparameter are FLOPS Utilization and Average Step Time.

In general, we want to maximize FLOPS Utilization while minimizing the step time per training example. In this case, we can see that increasing the batch size from 64 -> 256 achieves both of those. FLOPS increases from 16% to 27%. Average Step Time increase from 100ms to 260ms, however we increased our batch size by 300%. This means we move from 1.5ms per training example to 1.02ms per training example.

```{code-cell}
%tensorboard --logdir=$trace_dir
```

Next, we can explore alternative parallelism methods. In cell #4, we used 4-way data parallel and 2-way tensor parallel. 8-way data parallel is another popular way. Let's compare results between them. To switch to 8-way data parallel, we'll replace the `Mesh` definition with:

`mesh = Mesh(mesh_utils.create_device_mesh((8, 1)), ('batch', 'model'))`

JAX will automatically figure out how to shard the model and data to use the new partition strategy and nothing else need to be done. Re-connect the TPU runtime and run it again to see how it runs.

How simple and powerful is this! And that's the beauty of JAX automatic parallelism.

```{code-cell}
trace_dir = "/tmp/jax-trace-parallelism-comparison/"

mesh = Mesh(mesh_utils.create_device_mesh((4, 2)), ('batch', 'model'))
generate_trace()

mesh = Mesh(mesh_utils.create_device_mesh((8, 1)), ('batch', 'model'))
generate_trace()
```

Once again we'll run tensorboard.

Looking at the results, we see that the step times are nearly the same, however the FLOPS Utilization is at 13% for 8-way data parallelism compared to 27% or 4-way data parallelism.

By looking at the Trace Viewer tool and looking under each TPU's ops, we can see that the TPUs spend a large amount of time idle while waiting for the host, as well as spending a good amount of time in `reduce_sum` operations.

```{code-cell}
%tensorboard --logdir=$trace_dir
```

By changing hyperparameters and comparing profiles, we're able to gain significant insights into our bottlenecks and limitations. These are just two examples of hyperparameters to tune, but plenty more of them will have significant effects on training speed and resource utilization.
