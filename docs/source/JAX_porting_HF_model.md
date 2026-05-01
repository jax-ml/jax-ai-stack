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

+++ {"id": "wXMWNV2Rw-wE"}

# Converting the LLama 3.2 1B model from Hugging Face to JAX

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jax-ml/jax-ai-stack/blob/main/docs/JAX_porting_HF_model.ipynb)

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/jax-ml/jax-ai-stack/blob/main/docs/JAX_porting_HF_model.ipynb)

This tutorial demonstrates to convert Meta's [Llama 3.2 1B model](https://huggingface.co/meta-llama/Llama-3.2-1B) from Hugging Face to a JAX model and run it on T4 GPU.

You need some familiarity with [Flax](https://flax.readthedocs.io/en/latest/index.html), a library for building Neural Networks in JAX, to follow along. If you are getting started, check out the tutorials on [Getting started with JAX for AI](https://jax-ai-stack.readthedocs.io/en/latest/getting_started_with_jax_for_AI.html#example-a-simple-neural-network-with-flax) and [Flax's MNIST tutorial](https://flax.readthedocs.io/en/latest/mnist_tutorial.html).

+++ {"id": "Iuq-_y1qyXLF"}

## Setup

Let's install the `jax-ai-stack`, we'll use the `jax` and `flax` libraries from the stack in this tutorial. We will also need `huggingface_hub` for downloading model weights and `transformers` for tokenization.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: q5ueiKcIw8Sl
outputId: 79f0281a-3b98-462f-8cc4-9a4450e7124a
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
from dataclasses import dataclass
```

+++ {"id": "7oUeDSNozsF0"}

## Define the configuration

+++ {"id": "ZlmwhxkcyD9V"}

The Hugging Face Transformers library has [some tips for using the Llama3 model](https://huggingface.co/docs/transformers/main/en/model_doc/llama3). For further reference, the modeling code in the transformers library lives in the [`models/llama/modeling_llama.py` file](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py).

Before we create the model in JAX, we need to define some parameters. You can refer to the [Llama2 documentation for the configuration options](https://huggingface.co/docs/transformers/main/en/model_doc/llama2#transformers.LlamaConfig).

```{code-cell}
:id: 9WErwCTtzv7x

@dataclass
class LlamaConfig:
    def __init__(self):
        self.dim = 2048
        self.n_layers = 16
        self.n_heads = 32
        self.n_kv_heads = 8
        self.head_dim = self.dim // self.n_heads
        self.intermediate_size = 14336
        self.vocab_size = 128256
        self.multiple_of = 256
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
  height: 483
  referenced_widgets: [fad155559da44fdb88eca53c7336b623, 7bff39f20c8d47d896e67d0c12d3876a,
    065de156ffc04f6c90f6d0acab5cf79d, e368b2711e4f4eaa9208713d4505d4ca, fa203d94c35b4d3f8d3eb116a27f09f7,
    f941be815cd2460084bd06ed8512cdf3, 5946811b94d3457dbe4dc2346cae03db, bd6e501bfc234d1d8a0aeeacb857d6b3,
    c5ffb6869bf242f6b5d62daa94e88700, b3680e93364d4dd7a95ce078c947bf5d, d37fbf370c144435bc81fc67d1e9ae20,
    bb75a58ea37f470aadfd36471f6f2d54, c79d201995224a359485b65fb2d4ac3d, 237d55df9c6c49b8bbb89bbf76eb1eb0,
    65bd947bb2e64a6783efd9b18a6f22d6, 56d3affb3f564c9d8f5269311ad8c9a0, 55359e5ff8f7468a88cc5241b233a5cf,
    d2d62d19e9ef459b8694fa8fc04f7670, 72cd64d2d0c3464b80f19e669d021504, 3f48608bb20e44e59a87dc7249e8d147,
    025fbb3ad18a4dfeb6db1f394fae4736, 911317598fa4427db665316e8db46b1f, b8000a8b6aa54446af809ad5cc7f019d,
    024be3b7d5214124b7d41d76b7a97182, d99c871aadde47af8449f3431ba7a886, 2b231602bb554268b099b2baa14511cf,
    b1755ba29f8741088c20fb681b681c00, b43ce79062894691a1bf9886c2b8e6b8, f5437b12aae84e4d9658e6f6c3846a0f,
    5eb8545b4db54496add70dcf4357e0f3, 8c7b08d959d2470f9e7a46d81eb0ebea, 06191421335c4de89310c24207b631a6,
    2d0083b45f8247069bd72d9066e987a0, 700d269b34e64f85a7e111c61f0a3b7e, 74788132cad749719f76ab3972a899ed,
    79549aac63d7460296593a3e3ce7f09b, 3a9bed3445d24beeb69ab9402dc34d13, 85bed253cdd9456e83bb8b33e54aab9e,
    99d44ee67bad499b9bc2a00a28c34bc3, 0c35c1a8ac804b1faab2f18088b43c23, 4ac2e640b7964f47b6d017e4159eaf3a,
    c15cfdc0727d471b9a5a218f12c26d02, d5205f2eb06e45a894c3ca20f4261fab, 520359af946a4032a9635194db487774,
    012bbe7b346b47eaa9227581c669975c, 1ac1326418b44cef96160767e3831096, 10a92b00f7014a9e9aedb6574581705d,
    726878209f214f25b7492f2f7a9b2ee6, ef73037c897e487d97e2ce89724b60bd, 32b0fa4dfb5347ebb793431275c8d72d,
    3228aca1d0194c86bdb5f86c6058c72c, b2dbeb90af1044808e8848eb030b8766, 3507db8a8fd34390aa9e15d9105f1d0a,
    b937de3f7db84e74b80e28d6fea6bfc1, 5c67cb28b3e749ea8e9aeffbbc165468, c4a6eb5b48fc46b898e1ac513d3772c7,
    a3a69a28441c468d84871dd17744e7c2, b2a9193016534cada856067258345f56, 56ec8c7bc5294637b5a1a11d3c3767e7,
    7dc2253fac944438817793c45a5bb9ab, 4b021d221f65417b8628494ae789b701, 1fbbccba07b14f8e99e761b250d35964,
    d7776768213d415daf2241f0ac1f2187, 8301b9cb00614b65a81017b21a6985c2, f62c9d394bb34fe2b9d127366e921f94,
    f0bf0884100a4ebcb1377fa459de2de3, 0cbe17f150144c5aaaa0eba198b611c1, 03142c82be754e08a24d4aa19575d4e5,
    644dcf0d2ff34ee195064d1f89ab057e, 3c742306b7984844b53084f86e1d010c, 01861f72c37c498899edf210a74b7e85,
    b799afa71a054a50b72c6de0b54386b5, 0a8f989898864470bd2916dfe3b60fd5, 35683985e8d14dfaa4403a1acd7cdae1,
    a3dd2174043f4192bf2a2b3a6341b65d, 6ff5f4651a0245f6b17faccca91cb9c1, dc4b72a79c3245bb9c0b02afbae086dd,
    722265be6b854f978734b3cbf87b18fb, 1bb372afbb734e2483a58250257a5fd9, b7020b71b9c24032b00fc1821bf82228,
    0a6244614b194f9d9c424d308b65a8a0, 071cada4af0440bf8116f2bc854ab8c0, 276fdd8b53714675b36575cc71a3df02,
    8a40b083ab3b4d689294afef37f3d147, 217a6d792eb54aa78cc638759b00e1c1, ad80a83d5abb4b5f923281d3a7d63c24,
    7979764d457849bb94bbdc5505d55d3b, aa42716485a14d3683e16307eb6c6e0c, 382149dd5bcb4624895569c8ac170372,
    3c75a409325d43bea8f6d43279afd9d9, 3908b6b8d7134dd69473194179da8fea, 1779c500960a43a5a7ee9bb58dad5207,
    5ed2bd8b3a644fffaa302e77bada2e2e, 382a319b54f14e4ca09b4170598b75b7, da8122dc54ef4a1084ede9392a74db43,
    41f944e91b1740dd928a80e6d0590e3f, 21a9438874234d018b77b3380329cd07, c94b67f0f2124b14bf49866c20b62b8c,
    5e83455db1d048d5ae686cdf729710ef, 73ae1efa75824e1a989fac364a2b4e09, f78f2721fbae44b997386f066db00b29,
    563979534a11436ebee23fe2bb6c8dca, 3349480e1c644cdf80b58163884ab2f7, c858a04e271b412b8f8446e7eb68889b,
    801ced0f3a6c49adab0023d21ab0d4d7, 9665d42f9d5447b19335f7439fe9b2d5, b14d4b1b435d409db1af150bbdf020c1,
    2abd9963b97449b2b0d8ef3e05ef1e3d, 4dcf5d9d66804b49aaee7fa30767f132, ab0a80d6acdf4aafb7e5a57457cbc9d0,
    39fd896e2a264cd78379f08c9add13e9, 23d21e1a6acb4a718be7c18c5be2c93f, 715fdf4b1efc46f5907534f68d7fdd92,
    e915104b595a4b74b00d2468be0a587c, 4d6e4c16e3b14d538594874dea58da6d, 1ef70d5159ae49f5a80075beeae2fb17,
    c9a1b1a8afa346df8147a04452954835, d46a3bffadf64e228ea0aaeff39fe6d1, 7c47c3e59e9f4fae9a8d61588b9ebe3d,
    0158f2d795a447cbaedc022a212f8105, 04728a4d860f46b3a26403572dd9106b, d9048c4f46094a2e885af9fb06470633,
    8b9d3f9eef254d739a65e6fb087354ab, d9912461020141f1af9cad9bfe744af8, 764e9f8ad0004be8884893947ce8fdb0,
    2bc1f621cf504672a1bd36dea5e1a93e, 5d4ab4e1171d4c8f823f8e194500d81c, 7982f63e70f341f3917f05d124ec06cf,
    e5ce5b265b014766876319b383c788b9, 8e234d1b27e04131af7a2bc31921d8d7, 68685cb68293451b89d8d9f314d7e644,
    f05997ae6c414c13ad8d9b120f92174c, 887709ff4cb741f8b347810e21da6382, ed6e5f48072f48709c47a6e8e10497ad,
    5997c80b0e56441db8004eeb92fc6012, 0456360e7c604781bc8fe20cd4865d28, 93ef5a0fee6f418d90c4dd660d2a0069,
    3a3f098bf41d45c39e85eadae66f807a, d9846792be4946a1b6b0498b30fa6e34, a071743d438941d7ab0fb454097aeb45,
    a3564e4cab5f45c6b4060c07d6834afd, d13d2c418680468eb424ae66d97c3144, 0301038b0e7446169e2556436e37a8ad,
    719fd442f6f14e58b73b7f8e0d39ae9d, e7aaad97570b4d99ac62b0a47e93ac34, 1d230d9fd3d34f00be74e857f2beda39,
    39fe533e3ae94b7ca0cdffd0200b6a92, 3d2fe7971efd4ee18ec33361ac5cc031, bfa096d185d0485ab0b8b6817e94f86f,
    4a9c1bbdab4244339bf7d3a0b1786569, f4e6583d1f3148d3ba2613b696cd71c5, a5d84bf775b549ceb4bf09f5f42a5b4c,
    7aaa91cb372d4513b8de20281999e0a9, df7a459d5b2d456a814cacd85c07b8d2]
id: oVAJOEEOzEA9
outputId: 652b44c5-c9e5-4c88-b5ea-cb703be948ed
---
model_id = "meta-llama/Llama-3.2-1B"
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

        attn_weights = jnp.matmul(query, jnp.transpose(key, (0, 1, 3, 2)))
        attn_weights = (attn_weights.astype(jnp.float32) / jnp.sqrt(config.head_dim)).astype(jnp.bfloat16)
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
        self.lm_head.kernel.value = weights["model.embed_tokens.weight"].T
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
  height: 391
  referenced_widgets: [db648ae2225e4b218939567f4a281f0f, 380c07798cb74ae6be9b2cf685147612,
    702f71d9c479447b808416ae78cc6655, 12dcaca581244bdd8da549e2f46e673a, eee35be4166146fa8dce1e510981d548,
    a3ea606f028a468a90d344e5c7f37350, 7777b8d4c67143dfbfc0cfaaef7e8590, 04bffebb0634453f8131067a166da151,
    862950930b404af2b04b99d767a3f720, d2a87f06b3ab451eb7a05091853a285f, 6c5ff424c33a4d0fb4e652903e7dfae6,
    2af63ae9858e4a829c873e6e07ee16fa, 102504d1910b49f5aaefad9367eeb0a6, 0524a8c9f6d4430dab46b41c963909e1,
    176ce9bb62224fd9a329bf0c1d5309b3, e9601f2f9fb849029cb6a6aeac381e70, bab68630e8204fe5a00a1c53942499c3,
    c57de29a1b414584a242e4aed467f395, 280fe6e6deec40a6a3bb0351b13b3cc5, e94bbdd0b8a046a8aa933fd78d5f370b,
    287728ac8e8144cd96e2c758fecbaf69, 7c97faa07a61453ea7fdb2cad69e07b2, d8a684387f654b81bec5036fffd89494,
    7a1e5e604d3645b19d218d03f4cfab1a, d419362b5245474594099068513ef888, 394ca11fa4294e34837de20b1dd9cd9b,
    adc9d4338d8143e79f122d2a5145354b, 95fda494a07a4966bc3db9444fd39093, 2d407f3736824068a1f069037b079bc9,
    caf5ea5ffc3d48dc98967ee2560fa6df, 341da67846ac4a3ba78072b554b4688d, 47266581f7fb4189bb274c3e11af2a6c,
    216dbc57cb1d4d668468619ae28c1f1c]
id: HsQAmr-h1ZYa
outputId: 1c4df76f-e505-4664-8976-27631a4731e4
---
model = LlamaForCausalLM(rngs=nnx.Rngs(0))

# We no longer need `weights`
del weights

tokenizer = AutoTokenizer.from_pretrained(model_id)
input_text = "The capital of Japan is"

input_ids = tokenizer(input_text, return_tensors="jax")["input_ids"]
position_ids = jnp.asarray([jnp.arange(input_ids.shape[1])])

for _ in range(15):
    logits = model(input_ids, position_ids)
    next_token = jnp.argmax(logits[:, -1, :], axis=-1)
    input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=1)
    position_ids = jnp.asarray([jnp.arange(input_ids.shape[1])])
    print(f"Generated token: {next_token[0]}")

print(tokenizer.decode(input_ids[0]))
```

+++ {"id": "QLAM95pf5RMu"}

There you have it. We have successfully converted the Hugging Face model weights from the safetensors file, loaded them up in our JAX model, and run the model.

For simplicity, we have left out many optimizations (JIT, batch inference, KV cache, accelerators, SPMD and etc.) to speed things up. Feel free to implement them as an exercise.

```{code-cell}
:id: 40d8e941

del model
del tokenizer
```

+++ {"id": "IJfrKpEa7DLb"}

## Convert weights from other frameworks

You can also convert weights from other frameworks. Afer all, weights are just numbers. But note that the tensor names and layouts could be different (this is expected since different frameworks may implement differently), so you may need to adjust your code accordingly.

**Before proceeding, you may need to restart the runtime to release the GPU memory.**

+++ {"id": "GrzUleJt78Lm"}

### Keras Hub

+++ {"id": "kRdNtJj_TlaF"}

We will use Keras Hub to load the Llama 3.2 1B model  and extract the weights.

```{code-cell}
:id: AoMY1AbX69-Z

!pip install -Uq keras-hub

import keras_hub

llama_lm = keras_hub.models.Llama3CausalLM.from_preset("hf://meta-llama/Llama-3.2-1B", dtype="bfloat16")

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
outputId: cbccfa16-10ad-42a3-b7fc-5f355b748530
---
weights_dict["token_embedding"]
```

```{code-cell}
:id: 039c5e4a

del llama_lm
del weights_dict
```

+++ {"id": "KE-hPLhG8Hrs"}

### PyTorch

When you download the Hugging Face model, the original PyTorch model weights released by Meta are automatically downloaded as well. They are located in the `original` subfolder (another way to access them is to visit [Meta's Llama website](https://www.llama.com/llama-downloads/) and go from there). We can load the weights like this:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: r5qUM0O-8J4C
outputId: 64d6745a-c885-4aff-925e-cb3312ad717c
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
path_to_model_weights = os.path.join(weights_base_dir, "meta-llama/Llama-3.2-1B/original")
model_weights = torch.load(os.path.join(path_to_model_weights, "consolidated.00.pth"))
```

+++ {"id": "3qBM83yqUaEM"}

For example, here are the embeddings. You then need to extract every weight tensor and load them up in JAX like we did for the Hugging Face model.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: CC5DiPvTOWU1
outputId: 3cfd96e5-786e-49c7-f082-e2fd2873a21f
---
model_weights["tok_embeddings.weight"]
```

```{code-cell}
:id: 0faba2f4


```
