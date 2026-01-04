# Copyright (C) 2025 Kyle Killian
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import time
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, device_put, device_get
from flax import linen as nn
from functools import partial
from safetensors.torch import load_file

# --- VELOCITY CONFIG ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

# --- MODEL CONFIG ---
MODEL_SCALE = "70B"

if MODEL_SCALE == "8B":
    NUM_LAYERS = 32
    DIM = 4096
    HEADS = 32
    KV_HEADS = 8
    INTERMEDIATE = 14336
    VOCAB_SIZE = 128256
    WEIGHTS_DIR = "llama3_weights"
elif MODEL_SCALE == "70B":
    NUM_LAYERS = 80
    DIM = 8192
    HEADS = 64
    KV_HEADS = 8
    INTERMEDIATE = 28672
    VOCAB_SIZE = 128256
    WEIGHTS_DIR = "llama3_70b_weights"

# --- GLOBAL RAM CACHE ---
MODEL_RAM_CACHE = [None] * NUM_LAYERS

# --- ARCHITECTURE ---
class LlamaMLP(nn.Module):
    dim: int
    hidden_dim: int
    @nn.compact
    def __call__(self, x):
        g = nn.Dense(self.hidden_dim, use_bias=False, name="gate_proj")(x)
        u = nn.Dense(self.hidden_dim, use_bias=False, name="up_proj")(x)
        return nn.Dense(self.dim, use_bias=False, name="down_proj")(nn.silu(g) * u)

class LlamaAttention(nn.Module):
    dim: int
    num_heads: int
    num_kv_heads: int
    @nn.compact
    def __call__(self, x, cache=None):
        batch, seq, _ = x.shape
        head_dim = self.dim // self.num_heads
        kv_dim = self.num_kv_heads * head_dim
        
        q = nn.Dense(self.dim, use_bias=False, name="q_proj")(x).reshape(batch, seq, self.num_heads, head_dim)
        k = nn.Dense(kv_dim, use_bias=False, name="k_proj")(x).reshape(batch, seq, self.num_kv_heads, head_dim)
        v = nn.Dense(kv_dim, use_bias=False, name="v_proj")(x).reshape(batch, seq, self.num_kv_heads, head_dim)
        
        num_rep = self.num_heads // self.num_kv_heads
        if num_rep > 1:
            k = jnp.repeat(k, num_rep, axis=2)
            v = jnp.repeat(v, num_rep, axis=2)
            
        attn = nn.softmax(jnp.einsum('bqhd,bkhd->bhqk', q, k) * (head_dim**-0.5), axis=-1)
        out = nn.Dense(self.dim, use_bias=False, name="o_proj")(jnp.einsum('bhqk,bkhd->bqhd', attn, v).reshape(batch, seq, self.dim))
        return out, (k, v)

class LlamaDecoderLayer(nn.Module):
    dim: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    @nn.compact
    def __call__(self, x, cache=None):
        attn_input = nn.RMSNorm(name="input_layernorm")(x)
        attn_out, kv = LlamaAttention(self.dim, self.num_heads, self.num_kv_heads, name="self_attn")(attn_input)
        x = x + attn_out
        
        mlp_input = nn.RMSNorm(name="post_attention_layernorm")(x)
        mlp_out = LlamaMLP(self.dim, self.intermediate_size, name="mlp")(mlp_input)
        x = x + mlp_out
        return x, kv

# --- HELPER: Load Embeddings ---
def load_embeddings():
    """Loads the specific tensor 'model.embed_tokens.weight' from the shards."""
    print(" ⏳ Finding embedding weights...")
    index_path = os.path.join(WEIGHTS_DIR, "model.safetensors.index.json")
    with open(index_path, 'r') as f:
        weight_map = json.load(f)["weight_map"]
    
    # Locate which shard holds the embeddings
    shard_name = weight_map.get("model.embed_tokens.weight")
    if not shard_name:
        raise ValueError("Could not find 'model.embed_tokens.weight' in index.json")
    
    # Load that specific file
    w = load_file(os.path.join(WEIGHTS_DIR, shard_name))
    embed_weights = w["model.embed_tokens.weight"].float().numpy()
    print(f" ✅ Loaded Embeddings. Shape: {embed_weights.shape}")
    return jnp.array(embed_weights)

# --- WEIGHT LOADER ---
def load_layer_weights(layer_idx):
    if MODEL_RAM_CACHE[layer_idx] is not None:
        return MODEL_RAM_CACHE[layer_idx]

    index_path = os.path.join(WEIGHTS_DIR, "model.safetensors.index.json")
    with open(index_path, 'r') as f:
        weight_map = json.load(f)["weight_map"]
    
    prefix = f"model.layers.{layer_idx}."
    target_shards = {shard for key, shard in weight_map.items() if key.startswith(prefix)}
    
    if not target_shards:
        raise ValueError(f"No shards found for layer {layer_idx}")

    weights = {}
    for shard in target_shards:
        w = load_file(os.path.join(WEIGHTS_DIR, shard))
        for k in w:
            if k.startswith(prefix):
                weights[k[len(prefix):]] = w[k]

    layer_params = {
        "input_layernorm": {"scale": weights["input_layernorm.weight"].float().numpy()},
        "post_attention_layernorm": {"scale": weights["post_attention_layernorm.weight"].float().numpy()},
        "self_attn": {
            "q_proj": {"kernel": weights["self_attn.q_proj.weight"].float().numpy().T},
            "k_proj": {"kernel": weights["self_attn.k_proj.weight"].float().numpy().T},
            "v_proj": {"kernel": weights["self_attn.v_proj.weight"].float().numpy().T},
            "o_proj": {"kernel": weights["self_attn.o_proj.weight"].float().numpy().T},
        },
        "mlp": {
            "gate_proj": {"kernel": weights["mlp.gate_proj.weight"].float().numpy().T},
            "up_proj": {"kernel": weights["mlp.up_proj.weight"].float().numpy().T},
            "down_proj": {"kernel": weights["mlp.down_proj.weight"].float().numpy().T},
        }
    }
    
    MODEL_RAM_CACHE[layer_idx] = layer_params
    return layer_params

# --- INFERENCE ENGINE ---
model_def = LlamaDecoderLayer(dim=DIM, intermediate_size=INTERMEDIATE, num_heads=HEADS, num_kv_heads=KV_HEADS)

@partial(jit)
def forward_layer(variables, x, cache=None):
    return model_def.apply(variables, x, cache=cache, mutable=False)

def generate(prompt_tokens, max_new_tokens=100):
    print(f"🚀 Streaming Inference ({MODEL_SCALE}) | Prompt: {len(prompt_tokens)} tokens")
    
    # 1. Load Embeddings (The missing link)
    embed_weights = load_embeddings()
    
    # 2. Convert Token IDs -> Vectors (Real Math)
    # Shape: (1, seq_len) -> (1, seq_len, 8192)
    prompt_ids = jnp.array(prompt_tokens, dtype=jnp.int32)[None, :]
    x = jnp.take(embed_weights, prompt_ids, axis=0) 
    
    print("   Loading layers from disk (first pass is slow — be patient)...")
    
    kv_caches = [None] * NUM_LAYERS

    try:
        # First pass (Disk -> RAM -> GPU)
        for layer_idx in range(NUM_LAYERS):
            print(f"   Loading layer {layer_idx + 1}/{NUM_LAYERS}...", end="\r")
            layer_params = load_layer_weights(layer_idx)
            variables = {'params': layer_params}
            x, kv = forward_layer(variables, x, kv_caches[layer_idx])
            kv_caches[layer_idx] = kv
            x = device_get(x)
        
        print("\n   All layers loaded and cached! Starting generation...")
    except Exception as e:
        print(f"\n❌ Error during layer loading: {e}")
        import traceback
        traceback.print_exc()
        return []

    # Generation loop
    generated = []
    for step in range(max_new_tokens):
        # NOTE: This LM Head is currently a dummy placeholder for speed testing.
        # To get real text, we would also need to load `lm_head.weight` 
        # and matrix multiply here. For now, this tests the 80 layers of heavy lifting.
        logits = x[:, -1, :] @ jnp.ones((DIM, VOCAB_SIZE)) 
        next_token = jnp.argmax(logits, axis=-1)
        generated.append(int(next_token[0]))
        
        # Look up embedding for the NEW token
        new_token_id = next_token[:, None]
        new_x = jnp.take(embed_weights, new_token_id, axis=0)
        
        # Append to stream
        x = jnp.concatenate([x, new_x], axis=1)
        
        for layer_idx in range(NUM_LAYERS):
            variables = {'params': MODEL_RAM_CACHE[layer_idx]}
            x, kv = forward_layer(variables, x, kv_caches[layer_idx])
            kv_caches[layer_idx] = kv
            x = device_get(x)
            
        print(f"   Generated token {step+1}/{max_new_tokens}", end="\r")
    
    return generated

if __name__ == "__main__":
    prompt_tokens = [1, 15043, 3186, 25] # "Hello world" IDs roughly
    output = generate(prompt_tokens, max_new_tokens=20)
    print("\nGenerated IDs:", output)