---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "QEhawzCcCcFR"}

#Using LoRA in Jax

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jax-ml/jax-ai-stack/blob/main/docs/source/JAX_using_LoRA.ipynb)


This tutorial demonstrates how to implement LoRA for efficient fine-tuning of language models in JAX.
It builds upon the [JAX for LLM pretraining](https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html) tutorial by showing how to replace standard linear
layers with LoRA-enabled linear layers to significantly reduce the number of trainable parameters.

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that:
- Keeps pre-trained model weights frozen
- Adds small trainable low-rank decomposition matrices to certain layers
- Drastically reduces the number of trainable parameters (often by 90%+)

In the first chapter we will buildi a LoRA-enabled model from scratch, while the next chapter: "2. Fine-tuning a pre-trained LLM with LoRA" will demonstrate the more common and practical workflow of applying LoRA to existing pre-trained models.

Both chapters show how to implement these techniques using JAX and Flax's NNX library.

+++ {"id": "NIOXoY1xgiww"}

# 1.Creating a LoRa enabled LLM in Jax from scratch

In this chapter, we'll take an unconventional approach by implementing a language model with LoRA from scratch. This is different from standard practice, where LoRA is typically applied to already pre-trained models as a fine-tuning technique.

Why are we doing it this way? While not the optimal approach to train a model that achives good preformace (as we'll see in our results), building from scratch makes the integration of LoRA components within the model architecture more clear.

If you're interested in the more practical approach of applying LoRA to an existing pre-trained model, you can skip to the next chapter where we demonstrate that workflow.

+++ {"id": "hTmz5Cbco7n_"}

## Setup
Install required packages

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 6zMsOIc7ouCO
outputId: 40d84dff-b5c6-45ed-df08-fb22d3eeb01a
---
!pip install -q jax-ai-stack
!pip install -Uq tiktoken grain matplotlib
```

+++ {"id": "Rcji_799n4eA"}

Confirm we have TPUs set up.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: LS9sQEY3n0mB
outputId: b516c248-777f-4a59-a550-26e12bc2e2fc
---
import jax
jax.devices()
```

+++ {"id": "OHzJ_bokoovZ"}

Get the [TinyStories dataset from Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories). We only use the training split.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: wUjQsgQEmI1N
outputId: 90fc683c-696f-4f25-a75c-6a2a5b032cef
---
!wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt?download=true -O TinyStories-train.txt
```

+++ {"id": "sKE2uUafLobI"}

Import necessary libraries

```{code-cell} ipython3
:id: MKYFNOhdLq98

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from flax.nnx.nn.lora import LoRALinear  # Import LoRALinear
import optax
from dataclasses import dataclass
import grain.python as pygrain
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import pandas as pd
import tiktoken
import time
```

+++ {"id": "rPyt7MV6prz1"}

## Building the Model with LoRA

We'll use the same tokenizer and parallelism strategy as in the [pre-training tutorial](https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html).
The mesh defines how our computation will be distributed across TPU cores.

```{code-cell} ipython3
:id: xuMlCK3Q8WJD

tokenizer = tiktoken.get_encoding("gpt2")
mesh = Mesh(mesh_utils.create_device_mesh((4, 2)), ('batch', 'model'))
```

+++ {"id": "0XHQ0BQ9-KIj"}

The key difference from the original pre-training model is that we replace standard
`nnx.Linear` layers with `LoRALinear` layers from [Flax](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/lora.html).


This way, only the small rank decomposition matrices need to be trained.

```{code-cell} ipython3
:id: z0p-IHurrB9i

def causal_attention_mask(seq_len):
    return jnp.tril(jnp.ones((seq_len, seq_len)))

class TransformerBlock(nnx.Module):
    # update the __init__ function arguments to include lora_rank
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, *, rngs: nnx.Rngs, rate: float = 0.1, lora_rank=8):
        self.mha = nnx.MultiHeadAttention(num_heads=num_heads,
                                          in_features=embed_dim,
                                          kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
                                          bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
                                          rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=rate)
        self.layer_norm1 = nnx.LayerNorm(epsilon=1e-6,
                                         num_features=embed_dim,
                                         scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P('model'))),
                                         bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
                                         rngs=rngs)
        # here we replace the regular linear layer with the LoRALinea layer
        self.linear1 = LoRALinear(
            in_features=embed_dim,
            out_features=ff_dim,
            lora_rank=lora_rank,  # set the rank for the low-rank matrices
            kernel_init=nnx.with_partitioning(nnx.initializers.normal(0.02), P('model', None)),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, None),
            rngs=rngs
        )
        # here we replace the regular linear layer with the LoRALinea layer
        self.linear2 = LoRALinear(
            in_features=ff_dim,
            out_features=embed_dim,
            lora_rank=lora_rank,
            kernel_init=nnx.with_partitioning(nnx.initializers.normal(0.02), P('model', None)),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, None),
            rngs=rngs
        )
        self.dropout2 = nnx.Dropout(rate=rate)
        self.layer_norm2 = nnx.LayerNorm(epsilon=1e-6,
                                         num_features=embed_dim,
                                         scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model'))),
                                         bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P(None, 'model'))),
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
        # feed-forward network with LoRA layer
        ffn_output = self.linear1(out1)
        ffn_output = nnx.relu(ffn_output)
        ffn_output = self.linear2(ffn_output)
        ffn_output = self.dropout2(ffn_output, deterministic=not training)

        return self.layer_norm2(out1 + ffn_output)


class TokenAndPositionEmbedding(nnx.Module):

    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.token_emb = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(num_embeddings=maxlen, features=embed_dim, rngs=rngs)

    def __call__(self, x):
        positions = jnp.arange(0, x.shape[1])[None, :]
        position_embedding = self.pos_emb(positions)
        token_embedding = self.token_emb(x)
        return token_embedding + position_embedding


class MiniGPT(nnx.Module):
    # update the __init__ function arguments to include lora_rank
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, num_heads: int, feed_forward_dim: int, num_transformer_blocks: int, rngs: nnx.Rngs, lora_rank=8):
        self.embedding_layer = TokenAndPositionEmbedding(
                    maxlen, vocab_size, embed_dim, rngs=rngs
                )
        # create transformer blocks with LoRA
        self.transformer_blocks = [TransformerBlock(
            embed_dim, num_heads, feed_forward_dim, rngs=rngs, lora_rank=lora_rank
        ) for _ in range(num_transformer_blocks)]

        # modify the output layer to use LoRALinear instead of regular linear layer
        self.output_layer = LoRALinear(
            in_features=embed_dim,
            out_features=vocab_size,
            lora_rank=lora_rank,
            kernel_init=nnx.with_partitioning(nnx.initializers.normal(0.02), P('model', None)),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, None),
            rngs=rngs
        )


    def __call__(self, inputs, training: bool = False):
        x = self.embedding_layer(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        outputs = self.output_layer(x)
        return outputs

    def generate_text(self, max_tokens: int, start_tokens: [int], top_k=10):
        def sample_from(logits):
            logits, indices = jax.lax.top_k(logits, k=top_k)
            logits = nnx.softmax(logits)
            return jax.random.choice(jax.random.PRNGKey(0), indices, p=logits)

        def generate_step(start_tokens):
            pad_len = maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = jnp.array(start_tokens[:maxlen])
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = jnp.array(start_tokens + [0] * pad_len)
            else:
                x = jnp.array(start_tokens)

            x = x[None, :]
            logits = self(x)
            next_token = sample_from(logits[0][sample_index])
            return next_token

        generated = []
        for _ in range(max_tokens):
            next_token = generate_step(start_tokens + generated)
            if next_token == tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]:
              break
            generated.append(int(next_token))
        return tokenizer.decode(start_tokens + generated)

# modify the function arguments to include lora_rank
def create_model(rngs, lora_rank=8):
    return MiniGPT(maxlen, vocab_size, embed_dim, num_heads, feed_forward_dim, num_transformer_blocks=4, rngs=rngs,
                   lora_rank=lora_rank)
```

+++ {"id": "igX_eoGNMTGR"}

## Set Hyperparameters

We'll use the same hyperparameters as in the [pre-training tutorial](https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html) for consistency.

```{code-cell} ipython3
:id: GRhiDsCrMZRp

vocab_size = tokenizer.n_vocab
num_transformer_blocks = 8
maxlen = 256
embed_dim = 256
num_heads = 8
feed_forward_dim = 256
batch_size = 256  # You can adjust batch size based on your TP
num_epochs = 1
lora_rank = 128 # A higher rank will capture more complex patterns in the LLM, and will also increase the number of trainable parameters
```

+++ {"id": "mI1ci-HyMspJ"}

## Prepare data

Data loading and preprocessing remains the same as in the [pre-training tutorial](https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html).
We create a TextDataset class to handle tokenization and padding.

```{code-cell} ipython3
:id: rGUFsn1GMuzh

@dataclass
class TextDataset:
    data: list
    maxlen: int

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
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

## Train the model with LoRA

+++ {"id": "WbSt_MuyaG48"}

LoRA's efficiency lies in how we train only the small adapter matrices while keeping the rest of the model frozen. Let's look at how we implement this in JAX:

```{code-cell} ipython3
:id: h9hXS0NngSAw

# Create the model with LoRA
lora_model = create_model(rngs=nnx.Rngs(0), lora_rank=lora_rank)
# Filter for LoRA parameters only (look for lora_a and lora_b in the parameter path)
lora_params = nnx.All(nnx.Param, nnx.PathContains('lora_a') or nnx.PathContains('lora_b'))
# Create optimizer to only update LoRA parameters
optimizer = nnx.Optimizer(lora_model, optax.adam(1e-3), wrt=lora_params)
```

+++ {"id": "e5hooDhBadPb"}

 Using `nnx.All` create a mask that identifies only our LoRA parameters, looking for lora_a or lora_b in the parameter paths. Then we:

- Configure the optimizer to only update these selected parameters using the `wrt` argument
-Create a special `diff_state` that directs gradient computation to only flow to these parameters

Now we can use this `diff_state` when computing gradients in our training step:

```{code-cell} ipython3
:id: reUqnpEtiy0e

def loss_fn(model, batch):
    logits = model(batch[0])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch[1]).mean()
    return loss, logits

@nnx.jit
def train_step(model: MiniGPT, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    # Create differentiable state that only includes LoRA parameters
    diff_state = nnx.DiffState(0, lora_params)
    grad_fn = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, lables=batch[1])
    optimizer.update(grads)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: Ysl6CsfENeJN
outputId: 3236e7a1-4d6b-4378-b580-65a509236b23
---
metrics = nnx.MultiMetric(
  loss=nnx.metrics.Average('loss'),
)
rng = jax.random.PRNGKey(0)

start_prompt = "Once upon a time"
start_tokens = tokenizer.encode(start_prompt)[:maxlen]
generated_text = lora_model.generate_text(
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
          continue
        input_batch = jnp.array(jnp.array(batch).T)
        target_batch = prep_target_batch(input_batch)
        train_step(lora_model, optimizer, metrics, jax.device_put((input_batch, target_batch), NamedSharding(mesh, P('batch', None))))

        if (step + 1) % 200 == 0:
          for metric, value in metrics.compute().items():
              metrics_history[f'train_{metric}'].append(value)
          metrics.reset()

          elapsed_time = time.time() - start_time
          print(f"Step {step + 1}, Loss: {metrics_history['train_loss'][-1]}, Elapsed Time: {elapsed_time:.2f} seconds")
          start_time = time.time()

          generated_text = lora_model.generate_text(
              maxlen, start_tokens
          )
          print(f"Generated text:\n{generated_text}\n")
        step += 1

generated_text = lora_model.generate_text(
    maxlen, start_tokens
)
print(f"Final generated text:\n{generated_text}")
```

+++ {"id": "thaLs6TD0lt5"}

Visualize the training loss.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 472
id: B6Eg1Cz2y_iP
outputId: 227b6ad5-21de-45d8-ab0b-7dc9a931834f
---
import matplotlib.pyplot as plt
plt.plot(metrics_history['train_loss'])
plt.title('Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: Sy74Hsey8bBR
outputId: 78d9ce88-54aa-4501-fca9-42b4df44b466
---
# Analysis of LoRA Parameter Efficiency using proper module iteration
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Initialize counters
total_params = 0
lora_a_params = 0
lora_b_params = 0
other_params = 0
layer_counts = {}

# Iterate through all modules and count parameters
for path, module in lora_model.iter_modules():
    module_name = '.'.join(str(p) for p in path) if path else "root"

    module_params = 0
    module_lora_params = 0

    for name, attr in vars(module).items():
        if name.startswith('_') or not hasattr(attr, 'value'):
            continue

        # Get the parameter array
        param_array = attr.value
        if not isinstance(param_array, jnp.ndarray):
            continue

        param_count = param_array.size
        module_params += param_count
        total_params += param_count

        # Check if this is a LoRA parameter
        if name == 'lora_a':
            lora_a_params += param_count
            module_lora_params += param_count
        elif name == 'lora_b':
            lora_b_params += param_count
            module_lora_params += param_count
        else:
            other_params += param_count

# Calculate total LoRA parameters and ratios
lora_params = lora_a_params + lora_b_params
trainable_ratio = lora_params / total_params
frozen_ratio = other_params / total_params

print(f"Total model parameters: {total_params:,}")
print(f"Frozen parameters: {other_params:,} ({frozen_ratio:.2%})")
print(f"Trainable LoRA parameters: {lora_params:,} ({trainable_ratio:.2%})")
print(f"  - LoRA A matrices: {lora_a_params:,}")
print(f"  - LoRA B matrices: {lora_b_params:,}")
```

+++ {"id": "bOl4CY61Llcp"}

This example has demonstrated the implementation of LoRA in JAX, showing how to replace standard linear layers with LoRA-enabled versions and train only these adapter parameters while keeping the base model frozen.

As we've seen from our experiment results, this approach of applying LoRA to a model trained from scratch produced limited generation quality. The text outputs were repetitive and lacked coherence.

That is because LoRA is designed to make incremental adaptations to already capable models, not to carry the full burden of learning language structure from scratch. The small parameter space of the LoRA matrices (even with rank=128) simply cannot capture the full complexity of language when starting from random initialization.


###Next Steps
In a subsequent chapter, we'll explore how to integrate LoRA into existing pre-trained language models rather than training from scratch. If you want to stop here and save your progress you can run the follwing cells:

+++ {"id": "soPqiR1JNmjf"}

### Saving

```{code-cell} ipython3
:id: EkoFGCgSZ1yz

import orbax.checkpoint as orbax

state = nnx.state(lora_model)

checkpointer = orbax.PyTreeCheckpointer()
checkpointer.save('/content/save', state)

# Make sure the files are there
!ls /content/save/
```

+++ {"id": "jCApVd7671c1"}

### Disconnect the Colab runtime

```{code-cell} ipython3
:id: NsqYdbrDVKSq

from google.colab import runtime
runtime.unassign()
```

+++ {"id": "BRE3MBAfRS7i"}

# 2. Fine-tuning a pre-trained LLM with LoRA
