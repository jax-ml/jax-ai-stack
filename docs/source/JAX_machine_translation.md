---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Machine translation with a transformer using JAX

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jax-ml/jax-ai-stack/blob/main/docs/source/JAX_machine_translation.ipynb)

+++

This tutorial will demonstrate how to use JAX, [Flax NNX](http://flax.readthedocs.io) and [Optax](http://optax.readthedocs.io) to perform machine translation. It was originally inspired by the [Keras English-to-Spanish translation tutorial](https://keras.io/examples/nlp/neural_machine_translation_with_transformer/) (which was adapted from [Deep Learning with Python, Second Edition by François Chollet](https://www.manning.com/books/deep-learning-with-python-second-edition)).

Here, you will learn how to:

- Load and preprocess the dataset
- Define the transformer model - the encoder, decoder and positional embedding classes - with Flax and JAX
- Create the loss and training step functions
- Train the model

If you are new to JAX for AI, check out the [introductory tutorial](https://jax-ai-stack.readthedocs.io/en/latest/getting_started_with_jax_for_AI.html), which covers neural network building with [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html).


## Setup

JAX installation is covered in [this guide](https://jax.readthedocs.io/en/latest/installation.html) on the JAX documentation site. We will use [Tiktoken](https://github.com/openai/tiktoken) for tokenization and [Grain](https://google-grain.readthedocs.io/en/latest/index.html) for data loading (`!pip install -Uq tiktoken grain)

Import the necessary modules, including JAX NumPy, Flax NNX, Optax, Tiktoken, and tqdm:

```{code-cell} ipython3
import pathlib
import random
import string
import re
import numpy as np

import jax.numpy as jnp
import optax

from flax import nnx

import tiktoken
import grain.python as grain
import tqdm
```

## Loading and preprocessing the data

For simplicity, we will download the Spanish-to-English dataset to a temporary location, extract it, and read it into a Python object.

```{code-cell} ipython3
import requests
import zipfile
import tempfile

url = "http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"

with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = pathlib.Path(temp_dir)
    zip_file_path = temp_path / "spa-eng.zip"

    response = requests.get(url)
    zip_file_path.write_bytes(response.content)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(temp_path)

    text_file = temp_path / "spa-eng" / "spa.txt"

    with open(text_file) as f:
        lines = f.read().split("\n")[:-1]
    text_pairs = []
    for line in lines:
        eng, spa = line.split("\t")
        spa = "[start] " + spa + " [end]"
        text_pairs.append((eng, spa))
```

We will stay close to the original Keras tutorial, but use the "off-the-shelf" `cl100k_base` tokenizer from the [Tiktoken](https://github.com/openai/tiktoken) library, as it has a wide range of language understanding (and it's fast).


We need to extract the data, format it, and tokenize the phrases with padding.

```{code-cell} ipython3
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")
```

Instantiate the `cl100k_base` tokenizer:

```{code-cell} ipython3
tokenizer = tiktoken.get_encoding("cl100k_base")
```

Remove any punctuation to keep things simple and in line with the original tutorial. The square brackets `[` `]` are kept to preserve `[start]` and `[end]` formatting.

```{code-cell} ipython3
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

vocab_size = tokenizer.n_vocab
sequence_length = 20
```

Define the input standardization function:

```{code-cell} ipython3
def custom_standardization(input_string):
    lowercase = input_string.lower()
    return re.sub(f"[{re.escape(strip_chars)}]", "", lowercase)
```

Define the tokenizer function that also adding padding:

```{code-cell} ipython3
def tokenize_and_pad(text, tokenizer, max_length):
    tokens = tokenizer.encode(text)[:max_length]
    padded = tokens + [0] * (max_length - len(tokens)) if len(tokens) < max_length else tokens ##assumes list-like - (https://github.com/openai/tiktoken/blob/main/tiktoken/core.py#L81 current tiktoken out)
    return padded
```

Define the dataset formatting function that applies both `custom_standardization` and `tokenize_and_pad`:

```{code-cell} ipython3
def format_dataset(eng, spa, tokenizer, sequence_length):
    eng = custom_standardization(eng)
    spa = custom_standardization(spa)
    eng = tokenize_and_pad(eng, tokenizer, sequence_length)
    spa = tokenize_and_pad(spa, tokenizer, sequence_length)
    return {
            "encoder_inputs": eng,
            "decoder_inputs": spa[:-1],
            "target_output": spa[1:],
            }
```

Format the dataset:

```{code-cell} ipython3
train_data = [format_dataset(eng, spa, tokenizer, sequence_length) for eng, spa in train_pairs]
val_data = [format_dataset(eng, spa, tokenizer, sequence_length) for eng, spa in val_pairs]
test_data = [format_dataset(eng, spa, tokenizer, sequence_length) for eng, spa in test_pairs]
```

At this point we have extracted the data, applied formatting, and tokenized the phrases with padding. The data is kept in training, validate and test sets, with dictionary entries that look like this:

```{code-cell} ipython3
## data selection example
print(train_data[135])
```

The output should look something like:

{'encoder_inputs': [9514, 265, 3339, 264, 2466, 16930, 1618, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'decoder_inputs': [29563, 60, 1826, 7206, 71086, 37116, 653, 16109, 1493, 54189, 510, 408, 60, 0, 0, 0, 0, 0, 0], 'target_output': [60, 1826, 7206, 71086, 37116, 653, 16109, 1493, 54189, 510, 408, 60, 0, 0, 0, 0, 0, 0, 0]}

+++

## Defining the transformer model with Flax and JAX: Encoder, decoder, positional embedding

Next, we will construct the transformer model using JAX and Flax NNX. In many ways our approach tries to stay close to the original Keras machine translation tutorial, with `ops` changing to `jnp` (JAX NumPy) and `keras` or `layers` becoming Flax NNX's `nnx` layers. Certain `Module`-specific arguments are also different, such as Flax NNX (`flax.nxx.rngs`), while `decode=False` in the `MultiHeadAttention` call.

Let's build with the transformer encoder class - `TransformerEncoder()`, `TransformerDecoder` and a token embedding class - `PositionalEmbedding()` - by subclassing `flax.nnx.Module`. The `PositionalEmbedding()` class transforms tokens and positions into embeddings that will be fed into the transformer. It will combine token embeddings (words in an input sentence) with positional embeddings (the position of each word in a sentence). (It handles embedding both word tokens and their positions within the sequence.)

```{code-cell} ipython3
class TransformerEncoder(nnx.Module):
    """ A single Transformer encoder that processes the embedded sequences.

    Args:
        embed_dim (int): Embedding dimensionality.
        dense_dim (int): Dimensionality of the linear layers.
        rngs (flax.nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
    """
    def __init__(self, embed_dim: int, dense_dim: int, num_heads: int, rngs: nnx.Rngs, **kwargs):
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        # Multi-Head Attention (MHA) with `flax.nnx.MultiHeadAttention`.
        self.attention = nnx.MultiHeadAttention(num_heads=num_heads,
                                          in_features=embed_dim,
                                          decode=False,
                                          rngs=rngs)
        # Linear transformation with ReLU activation for the feed-forward network with `flax.nnx.Linear`
        # and `flax.nnx.relu` activation.
        self.dense_proj = nnx.Sequential(
                nnx.Linear(embed_dim, dense_dim, rngs=rngs),
                nnx.relu,
                nnx.Linear(dense_dim, embed_dim, rngs=rngs),
        )

        # First layer normalization with `flax.nnx.LayerNorm`.
        self.layernorm_1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        # Second layer normalization with `flax.nnx.LayerNorm`.
        self.layernorm_2 = nnx.LayerNorm(embed_dim, rngs=rngs)

    def __call__(self, inputs, mask=None):
        # The padding mask for attention.
        if mask is not None:
            padding_mask = jnp.expand_dims(mask, axis=1).astype(jnp.int32)
        else:
            padding_mask = None

        # Apply Multi-Head Attention (with/without a mask).
        attention_output = self.attention(
            inputs_q = inputs, inputs_k = inputs, inputs_v = inputs, mask=padding_mask, decode = False
        )
        # Apply the first layer normalization.
        proj_input = self.layernorm_1(inputs + attention_output)
        # The feed-forward network.
        # Apply the first linear transformation.
        proj_output = self.dense_proj(proj_input)
        # Apply the second linear transformation.
        return self.layernorm_2(proj_input + proj_output)


class PositionalEmbedding(nnx.Module):
    """ Combines token embeddings (words in an input sentence) with positional embeddings
    (the position of each word in a sentence).

    Args:
        sequence_length (int): Matimum sequence length.
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimensionality.
        rngs (flax.nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
    """

    # Initializes the token embedding layer (using `flax.nnx.Embed`).
    # Handles token and positional embeddings.
    def __init__(self, sequence_length: int, vocab_size: int, embed_dim: int, rngs: nnx.Rngs, **kwargs):
        self.token_embeddings = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
        self.position_embeddings = nnx.Embed(num_embeddings=sequence_length, features=embed_dim, rngs=rngs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    # Generates embeddings for the input tokens and their positions.
    # Takes a token sequence (integers) and returns the combined token and positional embeddings.
    def __call__(self, inputs):
        length = inputs.shape[1]
        positions = jnp.arange(0, length)[None, :]
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    # Computes the attention mask.
    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        else:
            return jnp.not_equal(inputs, 0)

class TransformerDecoder(nnx.Module):
    """ A single Transformer encoder that processes the embedded sequences.

    Args:
        embed_dim (int): Embedding dimensionality.
        latent_dim (int):
        num_heads (int):
        rngs (flax.nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
    """
    def __init__(self, embed_dim: int, latent_dim: int, num_heads: int, rngs: nnx.Rngs, **kwargs):
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = nnx.MultiHeadAttention(num_heads=num_heads,
                                  in_features=embed_dim,
                                  decode=False,
                                  rngs=rngs)
        self.attention_2 = nnx.MultiHeadAttention(num_heads=num_heads,
                                  in_features=embed_dim,
                                  decode=False,
                                  rngs=rngs)

        self.dense_proj = nnx.Sequential(
                nnx.Linear(embed_dim, latent_dim, rngs=rngs),
                nnx.relu,
                nnx.Linear(latent_dim, embed_dim, rngs=rngs),
        )
        self.layernorm_1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.layernorm_2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.layernorm_3 = nnx.LayerNorm(embed_dim, rngs=rngs)

    def __call__(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs.shape[1])
        if mask is not None:
            padding_mask = jnp.expand_dims(mask, axis=1).astype(jnp.int32)
            padding_mask = jnp.minimum(padding_mask, causal_mask)
        else:
            padding_mask = None
        attention_output_1 = self.attention_1(
            inputs_q=inputs, inputs_v=inputs, inputs_k=inputs,  mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2( ## https://github.com/google/flax/blob/main/flax/nnx/nn/attention.py#L403-L405
            inputs_q=out_1,
            inputs_v=encoder_outputs,
            inputs_k=encoder_outputs,
            mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, sequence_length):
        i = jnp.arange(sequence_length)[:, None]
        j = jnp.arange(sequence_length)
        mask = (i >= j).astype(jnp.int32)
        mask = jnp.reshape(mask, (1, 1, sequence_length, sequence_length))
        return mask
```

Here we finally use our earlier encoder, decoder, and positional embedding classes to construct the transformer class that we'll train and later use for inference:

```{code-cell} ipython3
class TransformerModel(nnx.Module):
    def __init__(self, sequence_length: int, vocab_size: int, embed_dim: int, latent_dim: int, num_heads: int, dropout_rate: float, rngs: nnx.Rngs):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.encoder = TransformerEncoder(embed_dim, latent_dim, num_heads, rngs=rngs)
        self.positional_embedding = PositionalEmbedding(sequence_length, vocab_size, embed_dim, rngs=rngs)
        self.decoder = TransformerDecoder(embed_dim, latent_dim, num_heads, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.dense = nnx.Linear(embed_dim, vocab_size, rngs=rngs)

    def __call__(self, encoder_inputs: jnp.array, decoder_inputs: jnp.array, mask: jnp.array = None, deterministic: bool = False):
        x = self.positional_embedding(encoder_inputs)
        encoder_outputs = self.encoder(x, mask=mask)

        x = self.positional_embedding(decoder_inputs)
        decoder_outputs = self.decoder(x, encoder_outputs, mask=mask)
        # per nnx.Dropout - disable (deterministic=True) for eval, keep (False) for training
        decoder_outputs = self.dropout(decoder_outputs, deterministic=deterministic)

        logits = self.dense(decoder_outputs)
        return logits
```

## Building the Grain data loader

It can be more computationally efficient to use pygrain for the data load stage, but this way it's abundandtly clear what's happening: data pairs go in and sets of jnp arrays come out, in step with our original dictionaries. 'Encoder_inputs', 'decoder_inputs' and 'target_output'.

```{code-cell} ipython3
batch_size = 512 #set here for the loader and model train later on

class CustomPreprocessing(grain.MapTransform):
    def __init__(self):
        pass

    def map(self, data):
        return {
            "encoder_inputs": np.array(data["encoder_inputs"]),
            "decoder_inputs": np.array(data["decoder_inputs"]),
            "target_output": np.array(data["target_output"]),
        }

train_sampler = grain.IndexSampler(
    len(train_data),
    shuffle=True,
    seed=12,                        # Seed for reproducibility
    shard_options=grain.NoSharding(), # No sharding since it's a single-device setup
    num_epochs=1,                    # Iterate over the dataset for one epoch
)

val_sampler = grain.IndexSampler(
    len(val_data),
    shuffle=False,
    seed=12,
    shard_options=grain.NoSharding(),
    num_epochs=1,
)

train_loader = grain.DataLoader(
    data_source=train_data,
    sampler=train_sampler,                 # Sampler to determine how to access the data
    worker_count=4,                        # Number of child processes launched to parallelize the transformations
    worker_buffer_size=2,                  # Count of output batches to produce in advance per worker
    operations=[
        CustomPreprocessing(),
        grain.Batch(batch_size=batch_size, drop_remainder=True),
    ]
)

val_loader = grain.DataLoader(
    data_source=val_data,
    sampler=val_sampler,
    worker_count=4,
    worker_buffer_size=2,
    operations=[
        CustomPreprocessing(),
        grain.Batch(batch_size=batch_size),
    ]
)
```

Optax doesn't have the identical loss function that the source tutorial uses, but this softmax cross entropy works well here - you can one_hot_encode if you don't use the `_with_integer_labels` version of the loss.

```{code-cell} ipython3
def compute_loss(logits, labels):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels)
    return jnp.mean(loss)
```

While in the original tutorial most of the model and training details happen inside keras, we make them explicit here in our step functions, which are later used in `train_one_epoch` and `eval_model`.

```{code-cell} ipython3
@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model, train_encoder_input, train_decoder_input, train_target_input):
        logits = model(train_encoder_input, train_decoder_input)
        loss = compute_loss(logits, train_target_input)
        return loss

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, jnp.array(batch["encoder_inputs"]), jnp.array(batch["decoder_inputs"]), jnp.array(batch["target_output"]))
    optimizer.update(grads)
    return loss

@nnx.jit
def eval_step(model, batch, eval_metrics):
    logits = model(jnp.array(batch["encoder_inputs"]), jnp.array(batch["decoder_inputs"]))
    loss = compute_loss(logits, jnp.array(batch["target_output"]))
    labels = jnp.array(batch["target_output"])

    eval_metrics.update(
        loss=loss,
        logits=logits,
        labels=labels,
    )
```

Here, `nnx.MultiMetric` helps us keep track of general training statistics, while we make our own dictionaries to hold historical values

```{code-cell} ipython3
eval_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average('loss'),
    accuracy=nnx.metrics.Accuracy(),
)

train_metrics_history = {
    "train_loss": [],
}

eval_metrics_history = {
    "test_loss": [],
    "test_accuracy": [],
}
```

```{code-cell} ipython3
## Hyperparameters
rng = nnx.Rngs(0)
embed_dim = 256
latent_dim = 2048
num_heads = 8
dropout_rate = 0.5
vocab_size = tokenizer.n_vocab
sequence_length = 20
learning_rate = 1.5e-3
num_epochs = 10
```

```{code-cell} ipython3
bar_format = "{desc}[{n_fmt}/{total_fmt}]{postfix} [{elapsed}<{remaining}]"
train_total_steps = len(train_data) // batch_size

def train_one_epoch(epoch):
    model.train()  # Set model to the training mode: e.g. update batch statistics
    with tqdm.tqdm(
        desc=f"[train] epoch: {epoch}/{num_epochs}, ",
        total=train_total_steps,
        bar_format=bar_format,
        leave=True,
    ) as pbar:
        for batch in train_loader:
            loss = train_step(model, optimizer, batch)
            train_metrics_history["train_loss"].append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)


def evaluate_model(epoch):
    # Compute the metrics on the train and val sets after each training epoch.
    model.eval()  # Set model to evaluation model: e.g. use stored batch statistics

    eval_metrics.reset()  # Reset the eval metrics
    for val_batch in val_loader:
        eval_step(model, val_batch, eval_metrics)

    for metric, value in eval_metrics.compute().items():
        eval_metrics_history[f'test_{metric}'].append(value)

    print(f"[test] epoch: {epoch + 1}/{num_epochs}")
    print(f"- total loss: {eval_metrics_history['test_loss'][-1]:0.4f}")
    print(f"- Accuracy: {eval_metrics_history['test_accuracy'][-1]:0.4f}")
```

```{code-cell} ipython3
model = TransformerModel(sequence_length, vocab_size, embed_dim, latent_dim, num_heads, dropout_rate, rngs=rng)
optimizer = nnx.ModelAndOptimizer(model, optax.adamw(learning_rate))
```

## Start the Training!
With our data loaders set and the model, optimizer, and epoch train/eval functions set up - time to finally press go - on a 3090, this is roughly 19GB VRAM and each epoch is roughly 18 seconds with batch_size set to 512.

```{code-cell} ipython3
for epoch in range(num_epochs):
    train_one_epoch(epoch)
    evaluate_model(epoch)
```

We can then plot the loss over training time. That log-plot comes in handy here, or it's hard to appreciate the progress after 1000 steps or so.

```{code-cell} ipython3
import matplotlib.pyplot as plt

plt.plot(train_metrics_history["train_loss"], label="Loss value during the training")
plt.yscale('log')
plt.legend()
```

And eval set Loss and Accuracy - Accuracy does continue to rise, though it's hard-earned progress after about the 5th epoch. Based on the training statistics, it's fair to say the process starts overfitting after roughly that 5th epoch.

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(10, 10))
axs[0].set_title("Loss value on eval set")
axs[0].plot(eval_metrics_history["test_loss"])
axs[1].set_title("Accuracy on eval set")
axs[1].plot(eval_metrics_history["test_accuracy"])
```

## Use Model for Inference
After all that, the product of what we were working for: a trained model we can save and load for inference. For people using LLMs recently, this pattern may look rather familiar: an input sentence tokenized into an array and computed 'next' token-by-token. While many recent LLMs are decoder-only, this was an encoder/decoder architecture with the very specific english-to-spanish pattern baked in.

We've changed a couple things from the source 'use' function, here - because of the tokenizer used, things like `[start]` and `[end]` are no longer single tokens - instead `[start]` is `[29563, 60] = "[start" + "]"` and `[end]` is `[58308, 60] = "[end" + "]"` - thus we start with only a single token `[start` and can't only test on `last_token = "[end"]`. Otherwise, the main change here is that the input is assumed a single sentence, rather than batch inference.

```{code-cell} ipython3
def decode_sequence(input_sentence):

    input_sentence = custom_standardization(input_sentence)
    tokenized_input_sentence = tokenize_and_pad(input_sentence, tokenizer, sequence_length)

    decoded_sentence = "[start"
    for i in range(sequence_length):
        tokenized_target_sentence = tokenize_and_pad(decoded_sentence, tokenizer, sequence_length)[:-1]
        predictions = model(jnp.array([tokenized_input_sentence]), jnp.array([tokenized_target_sentence]))

        sampled_token_index = np.argmax(predictions[0,i, :]).item(0)
        sampled_token = tokenizer.decode([sampled_token_index])
        decoded_sentence += "" + sampled_token

        if decoded_sentence[-5:] == "[end]":
            break
    return decoded_sentence
```

```{code-cell} ipython3
test_eng_texts = [pair[0] for pair in test_pairs]
```

```{code-cell} ipython3
test_result_pairs = []
for _ in range(10):
    input_sentence = random.choice(test_eng_texts)
    translated = decode_sequence(input_sentence)

    test_result_pairs.append(f"[Input]: {input_sentence} [Translation]: {translated}")
```

## Test Results
For the model and the data, not too shabby - It's definitely spanish-ish. Though when 'making' friends, please don't confuse 'hacer' (to make) with 'comer' (to eat).

```{code-cell} ipython3
for i in test_result_pairs:
    print(i)
```

Example output from the above cell:

      [Input]: We're going to have a baby. [Translation]: [start] nosotros vamos a tener un bebé [end]
      [Input]: You drive too fast. [Translation]: [start] conducís demasiado rápido [end]
      [Input]: Let me know if there's anything I can do. [Translation]: [start] déjame saber si hay cualquier cosa que yo pueda hacer [end]
      [Input]: Let's go to the kitchen. [Translation]: [start] vayamos a la cocina [end]
      [Input]: Tom gasped. [Translation]: [start] tom se quedó sin aliento [end]
      [Input]: I was just hanging out with some of my friends. [Translation]: [start] estaba escquieto con algunos de mi amigos [end]
      [Input]: Tom is in the bathroom. [Translation]: [start] tom está en el cuarto de baño [end]
      [Input]: I feel safe here. [Translation]: [start] me siento segura [end]
      [Input]: I'm going to need you later. [Translation]: [start] me voy a necesitar después [end]
      [Input]: A party is a good place to make friends with other people. [Translation]: [start] una fiesta es un buen lugar de comer amigos con otras personas [end]
