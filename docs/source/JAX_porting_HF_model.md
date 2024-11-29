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

+++ {"id": "wXMWNV2Rw-wE"}

# Converting the LLama 3 8B Instruct model from Hugging Face to JAX

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jax-ml/jax-ai-stack/blob/main/docs/JAX_porting_HF_model.ipynb)

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/jax-ml/jax-ai-stack/blob/main/docs/JAX_porting_HF_model.ipynb)

This tutorial demonstrates to convert Meta's [Llama 3 8B Instruct model](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) from Hugging Face to a JAX model and run it on a CPU.

You need some familiarity with [Flax](https://flax.readthedocs.io/en/latest/index.html), a library for building Neural Networks in JAX, to follow along. If you are getting started, check out the tutorials on [Getting started with JAX for AI](https://jax-ai-stack.readthedocs.io/en/latest/getting_started_with_jax_for_AI.html#example-a-simple-neural-network-with-flax) and [Flax's MNIST tutorial](https://flax.readthedocs.io/en/latest/mnist_tutorial.html).

Since the model is fairly large, you are going to need a **high RAM** machine, for example, a Colab's high RAM instance (which requires [Pro subscription](https://colab.research.google.com/signup)), a Kaggle VM, or your own computer with lots of RAM.

+++ {"id": "Iuq-_y1qyXLF"}

## Setup

Let's install the `jax-ai-stack`, we'll use the `jax` and `flax` libraries from the stack in this tutorial. We will also need `huggingface_hub` for downloading model weights and `transformers` for tokenization.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: q5ueiKcIw8Sl
outputId: cfafaf60-954e-46ee-9d76-cf54b1cc3d2d
---
!pip install -q jax-ai-stack
!pip install -Uq transformers huggingface_hub
!huggingface-cli login
```

+++ {"id": "dLCAc5Wbyl4N"}

Take care of the imports.

```{code-cell}
:id: VnGrbnX1yjsQ

import jax
import jax.numpy as jnp
from flax import nnx
from safetensors import safe_open
from pathlib import Path
import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
```

+++ {"id": "7oUeDSNozsF0"}

## Define the configuration

+++ {"id": "ZlmwhxkcyD9V"}

The Hugging Face Transformers library has [some tips for using the Llama3 model](https://huggingface.co/docs/transformers/main/en/model_doc/llama3). For further reference, the modeling code in the transformers library lives in [the `models/llama/modeling_llama.py` file](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py).

Before we create the model in JAX, we need to define some parameters. You can refer to the [Llama2 documentation for the configuration options](https://huggingface.co/docs/transformers/main/en/model_doc/llama2#transformers.LlamaConfig).

```{code-cell}
:id: 9WErwCTtzv7x

class LlamaConfig:
    def __init__(self):
        self.dim = 4096
        self.n_layers = 32
        self.n_heads = 32
        self.n_kv_heads = 8
        self.head_dim = self.dim // self.n_heads
        self.intermediate_size = 14336
        self.vocab_size = 128256
        self.multiple_of = 1024
        self.norm_eps = 1e-05
        self.rope_theta = 500000.0

config = LlamaConfig()
```

+++ {"id": "mRNFx-eBywHO"}

## Load the model weights

We'll use the transformers library to download the model weights.

Meta requires [acceptance of the license](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/130) before you can access the files. You will also need a Hugging Face access token, please refer to [Hugging Face documentation](https://huggingface.co/docs/hub/en/security-tokens) to set it up.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 67
  referenced_widgets: [f92401a14f734ee4a12f1914677176a5, 06ae04d84d1c42368325378f0ea78a3c,
    60c1e76587ce44988a14ff0d64ff23a3, 5075c5c1a9d04e2e958d3f1bf609fcd7, 30b461fa084b49c39b512f0d1dee1a58,
    572d284c16a448c3876d7070b436556e, aaae7410f2bf4ee989c2954c0cb30979, ddc40e128f9b433e8ce3cb66df8ecabb,
    f55de353cf8541ca8deb98eba743392e, d3a8ccf2b66e43a28c9b47e7fb92ec8f, 88388fce5ed048c8b0e9af55ad27bb19]
id: oVAJOEEOzEA9
outputId: e9514113-4eb0-476e-84fa-9c7fe80c717e
---
model_id = "meta-llama/Meta-Llama-3-8B"
if os.path.exists('/kaggle'):
    weights_base_dir = '/kaggle/tmp'
elif os.path.exists('/content'):
    # Colab
    weights_base_dir = '/content'
else:
    # Local machine
    weights_base_dir = '.'

path_to_model_weights = os.path.join(weights_base_dir, model_id)

snapshot_download(repo_id=model_id, local_dir=path_to_model_weights)
```

+++ {"id": "4Xo6QyVR0UkF"}

Then extract the model weights from the safetensors file and store them in the `weights` dict. These weights will be loaded into our JAX model soon.

```{code-cell}
:id: pERpzsPS0fLj

def load_safetensors():
    weights = {}
    safetensors_files = Path(path_to_model_weights).glob('*.safetensors')

    for file in safetensors_files:
        with safe_open(file, framework="jax", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    return weights

weights = load_safetensors()
```

+++ {"id": "6ToKOCWIPJl1"}

Note that the weights are stored as `bfloat16`.

+++ {"id": "vQ7Bmvj70jjA"}

## Define the Flax model

+++ {"id": "HdHQwfSO1xfY"}

Now we can define the model in Flax.

[This Transformer vs Llama diagram](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/_images/transformer_vs_llama.svg) from Nvidia visualizes the model architecture pretty nicely. We will define each layer using [Flax's NNX.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html#flax.nnx.Module).

We will start by defining the RMS normalization layer. Note how we load the parameters from the `weights` dict.

```{code-cell}
:id: 8Zi7p42-0qya

class LlamaRMSNorm(nnx.Module):

    def __init__(self, name=None, layer_idx=None, rngs=None):
        if name is None and layer_idx is None:
            # Final normalization layer
            self.norm_weights = nnx.Param(weights["model.norm.weight"], rngs=rngs)
        else:
            self.norm_weights = nnx.Param(weights[f"model.layers.{layer_idx}.{name}.weight"], rngs=rngs)

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        squared_mean = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jnp.reciprocal(jnp.sqrt(squared_mean + config.norm_eps))
        return self.norm_weights * hidden_states.astype(input_dtype)
```

+++ {"id": "8ZecTSm_2SVT"}

Llama 3 uses [Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864) to encode both token and positional embeddings. For a gentle introduction to RoPE, please refer to the [CMU lecture slides](https://www.cs.cmu.edu/~mgormley/courses/10423-s24//slides/lecture5-vit-ink.pdf) and this awesome [EleutherAI blog](https://blog.eleuther.ai/rotary-embeddings/).

```{code-cell}
:id: ZgrersK60ycn

class LlamaRotaryEmbedding(nnx.Module):

    def __init__(self, dim, base=10000, rngs=None):
        self.dim = dim
        self.base = base

    def __call__(self, position_ids):
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        inv_freq_expanded = jnp.expand_dims(inv_freq, axis=(0, 1))
        position_ids_expanded = jnp.expand_dims(position_ids, axis=(0, 2)).astype(jnp.float32)
        freqs = jnp.einsum('bij,bjk->bijk', position_ids_expanded, inv_freq_expanded)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(emb).squeeze(2).astype(jnp.bfloat16)
        sin = jnp.sin(emb).squeeze(2).astype(jnp.bfloat16)
        return cos, sin
```

+++ {"id": "8omcCyDO4OdS"}

Now we create the attention layers. Note how we load the weights into the q, k and v projection layers.

```{code-cell}
:id: vkPG8ILr0zwg

class LlamaAttention(nnx.Module):

    def __init__(self, layer_idx, rngs=None):
        self.q_proj = nnx.Linear(config.dim, config.n_heads * config.head_dim, use_bias=False, rngs=rngs)
        self.q_proj.kernel.value = weights[f"model.layers.{layer_idx}.self_attn.q_proj.weight"].T
        self.k_proj = nnx.Linear(config.dim, config.n_kv_heads * config.head_dim, use_bias=False, rngs=rngs)
        self.k_proj.kernel.value = weights[f"model.layers.{layer_idx}.self_attn.k_proj.weight"].T
        self.v_proj = nnx.Linear(config.dim, config.n_kv_heads * config.head_dim, use_bias=False, rngs=rngs)
        self.v_proj.kernel.value = weights[f"model.layers.{layer_idx}.self_attn.v_proj.weight"].T
        self.o_proj = nnx.Linear(config.n_heads * config.head_dim, config.dim, use_bias=False, rngs=rngs)
        self.o_proj.kernel.value = weights[f"model.layers.{layer_idx}.self_attn.o_proj.weight"].T
        self.rotary_emb = LlamaRotaryEmbedding(config.head_dim, base=config.rope_theta, rngs=rngs)

    # Alternative implementation: https://github.com/google/flax/blob/5d896bc1a2c68e2099d147cd2bc18ebb6a46a0bd/examples/gemma/positional_embeddings.py#L45
    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=1):
        cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
        sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return jnp.concatenate([-x2, x1], axis=-1)

    def repeat_kv(self, hidden_states, n_repeat):
        batch, n_kv_heads, seq_len, head_dim = hidden_states.shape
        if n_repeat == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].repeat(n_repeat, axis=2)
        return hidden_states.reshape(batch, n_kv_heads * n_repeat, seq_len, head_dim)

    def __call__(self, x, position_ids):
        batch_size, seq_len, _ = x.shape
        query = self.q_proj(x).reshape(batch_size, seq_len, config.n_heads, config.head_dim).transpose((0, 2, 1, 3))
        key = self.k_proj(x).reshape(batch_size, seq_len, config.n_kv_heads, config.head_dim).transpose((0, 2, 1, 3))
        value = self.v_proj(x).reshape(batch_size, seq_len, config.n_kv_heads, config.head_dim).transpose((0, 2, 1, 3))
        # Assuming batch_size=1
        cos, sin = self.rotary_emb(position_ids[0])
        query, key = self.apply_rotary_pos_emb(query, key, cos, sin)

        key = self.repeat_kv(key, config.n_heads // config.n_kv_heads)
        value = self.repeat_kv(value, config.n_heads // config.n_kv_heads)

        attn_weights = jnp.matmul(query, jnp.transpose(key, (0, 1, 3, 2))) / jnp.sqrt(config.head_dim)
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)
        attn_output = jnp.matmul(attn_weights, value).transpose((0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        return output
```

+++ {"id": "J0hDkS4N3yor"}

MLP layer follows the attention layer. Similarly we load the weights into the gate, up and down projection layers.

```{code-cell}
:id: y5qP9b82047Y

class LlamaMLP(nnx.Module):

    def __init__(self, layer_idx, rngs=None):
        self.gate_proj = nnx.Linear(config.dim, config.intermediate_size, use_bias=False, rngs=rngs)
        self.gate_proj.kernel.value = weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"].T
        self.up_proj = nnx.Linear(config.dim, config.intermediate_size, use_bias=False, rngs=rngs)
        self.up_proj.kernel.value = weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"].T
        self.down_proj = nnx.Linear(config.intermediate_size, config.dim, use_bias=False, rngs=rngs)
        self.down_proj.kernel.value = weights[f"model.layers.{layer_idx}.mlp.down_proj.weight"].T

    def __call__(self, x):
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))
```

+++ {"id": "keogeCxz30Tw"}

We assemble the decoder block.

```{code-cell}
:id: dW--6pPv1EQI

class LlamaTransformerBlock(nnx.Module):

    def __init__(self, layer_idx, rngs=None):
        self.input_layernorm = LlamaRMSNorm(name="input_layernorm", layer_idx=layer_idx, rngs=rngs)
        self.attention = LlamaAttention(layer_idx=layer_idx, rngs=rngs)
        self.post_attention_layernorm = LlamaRMSNorm(name="post_attention_layernorm", layer_idx=layer_idx, rngs=rngs)
        self.mlp = LlamaMLP(layer_idx=layer_idx, rngs=rngs)

    def __call__(self, x, position_ids):
        residual = x
        x = self.input_layernorm(x)
        x = self.attention(x, position_ids)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x
```

+++ {"id": "5mvjJWU54Biz"}

Finally we have the enire model.

```{code-cell}
:id: W85ioRay1HYz

class LlamaForCausalLM(nnx.Module):

    def __init__(self, rngs=None):
        self.token_embed = nnx.Embed(num_embeddings=config.vocab_size, features=config.dim, dtype=jnp.bfloat16, rngs=rngs)
        self.token_embed.embedding.value = weights["model.embed_tokens.weight"]

        self.layers = [LlamaTransformerBlock(layer_idx=idx, rngs=rngs) for idx in range(config.n_layers)]
        self.lm_head = nnx.Linear(config.dim, config.vocab_size, use_bias=False, rngs=rngs)
        self.lm_head.kernel.value = weights["lm_head.weight"].T
        self.norm = LlamaRMSNorm(name=None, layer_idx=None, rngs=rngs)

    def __call__(self, input_ids, position_ids):
        assert input_ids.shape[0] == 1, "Only batch size 1 is supported"
        x = self.token_embed(input_ids)
        for layer in self.layers:
            x = layer(x, position_ids)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
```

+++ {"id": "gBfFRlNx1b8Z"}

## Run the Flax model

+++ {"id": "ZWcxgcsI4ESh"}

Let's take it for a spin! We are still going to use the tokenizer from Hugging Face (since our primary focus is re-building the model instead of the tokenizer).

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: HsQAmr-h1ZYa
outputId: 8026a7dc-4485-4107-c428-ecf624c75fea
---
model = LlamaForCausalLM(rngs=nnx.Rngs(0))

# We no longer need `weights`
del weights

tokenizer = AutoTokenizer.from_pretrained(model_id)
input_text = "The capital of Japan is"

input_ids = tokenizer(input_text, return_tensors="jax")["input_ids"]
position_ids = jnp.asarray([jnp.arange(input_ids.shape[1])])

for _ in range(5):
    logits = model(input_ids, position_ids)
    next_token = jnp.argmax(logits[:, -1, :], axis=-1)
    input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=1)
    position_ids = jnp.asarray([jnp.arange(input_ids.shape[1])])
    print(f"Generated token: {next_token[0]}")

print(tokenizer.decode(input_ids[0]))
```

+++ {"id": "QLAM95pf5RMu"}

There you have it. We have successfully converted the Hugging Face model weights from the safetensors file, loaded them up in our JAX model, and run the model.

For simplicity, we have left out many optimizations (JIT, batch inference, KV cache, leveraging accelerators, SPMD and etc.) to speed things up. Feel free to implement them as an exercise.

```{code-cell}
:id: sj8FQAMAiyC3

del model
del tokenizer
```

+++ {"id": "IJfrKpEa7DLb"}

## Convert weights from other frameworks

You can also convert weights from other frameworks. Afer all, those are just numbers. But note that the tensor names and layouts could be different (this is expected since different frameworks may implement differently), so you may need to adjust your code accordingly.

**Before proceeding, you may need to restart the runtime to release the memory.**

+++ {"id": "GrzUleJt78Lm"}

### Keras Hub

+++ {"id": "kRdNtJj_TlaF"}

Keras Hub supports Llama 3 models. We can use Keras Hub to load the model and extract the weights. Note that the model files are hosted on [Kaggle](https://www.kaggle.com/models/keras/llama3), so you need to accept the license first and provide a Kaggle API key to access the files below.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: AoMY1AbX69-Z
outputId: 0ea99f63-e58b-466f-8615-d387b9e51b4a
---
!pip install -Uq keras-hub

import keras_hub

llama_lm = keras_hub.models.Llama3CausalLM.from_preset("llama3_8b_en", dtype="bfloat16")

weights_dict = {}
for layer in llama_lm.backbone.layers:
    weights = layer.get_weights()
    if weights:
        weights_dict[layer.name] = weights
```

+++ {"id": "AD1cA9GcUjR4"}

For example, here are the embeddings. You then need to extract every weight tensor and load them up in JAX like we did for the Hugging Face model.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: jDt2YqTtSyiO
outputId: d7d88eb9-e1c4-429b-ccfa-16e5d40f9371
---
weights_dict["token_embedding"]
```

```{code-cell}
:id: ixOo05DQiqpS

del llama_lm
del weights_dict
```

+++ {"id": "KE-hPLhG8Hrs"}

### PyTorch

When you download the Hugging Face model, the original PyTorch model weights released by Meta are automatically downloaded as well. They are located in the `original` subfolder (another way to access them is to visit [Meta's Llama website](https://www.llama.com/llama-downloads/) and go from there). We can load them up like this:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: r5qUM0O-8J4C
outputId: 3f444b2a-998c-4ab7-ae2d-f38d5d54638e
---
import torch
import os

if os.path.exists('/kaggle'):
    weights_base_dir = '/kaggle/tmp'
elif os.path.exists('/content'):
    # Colab
    weights_base_dir = '/content'
else:
    # Local machine
    weights_base_dir = '.'
path_to_model_weights = os.path.join(weights_base_dir, "meta-llama/Meta-Llama-3-8B/original")
model_weights = torch.load(os.path.join(path_to_model_weights, "consolidated.00.pth"))
```

+++ {"id": "3qBM83yqUaEM"}

For example, here are the embeddings. You then need to extract every weight tensor and load them up in JAX like we did for the Hugging Face model.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: CC5DiPvTOWU1
outputId: d2f82cf8-ecca-4f2f-cd7d-34fcefde6e41
---
model_weights["tok_embeddings.weight"]
```

```{code-cell}
:id: BnV5PtjG1_By


```
