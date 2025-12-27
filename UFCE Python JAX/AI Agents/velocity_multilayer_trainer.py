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
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit
from flax import linen as nn
from safetensors.torch import load_file
from functools import partial

# --- VELOCITY CONFIG ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85" # Give VRAM breathing room
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

# --- LLAMA-3-8B CONFIG ---
NUM_LAYERS = 32  # The Real Scale
DIM = 4096
HEADS = 32
KV_HEADS = 8     # GQA
INTERMEDIATE = 14336
BATCH_SIZE = 1   # Start small for full-chain verification
SEQ_LEN = 128    # Short sequence for speed test

WEIGHTS_DIR = "./llama3_weights"

# --- 1. ARCHITECTURE (GQA ENABLED) ---
class LlamaMLP(nn.Module):
    dim: int
    hidden_dim: int
    @nn.compact
    def __call__(self, x):
        g = nn.Dense(self.hidden_dim, use_bias=False, name="gate_proj")(x)
        u = nn.Dense(self.hidden_dim, use_bias=False, name="up_proj")(x)
        d = nn.Dense(self.dim, use_bias=False, name="down_proj")
        return d(nn.silu(g) * u)

class LlamaAttention(nn.Module):
    dim: int
    num_heads: int
    num_kv_heads: int
    @nn.compact
    def __call__(self, x):
        batch, seq, _ = x.shape
        head_dim = self.dim // self.num_heads
        kv_dim = self.num_kv_heads * head_dim
        
        q = nn.Dense(self.dim, use_bias=False, name="q_proj")(x).reshape(batch, seq, self.num_heads, head_dim)
        k = nn.Dense(kv_dim, use_bias=False, name="k_proj")(x).reshape(batch, seq, self.num_kv_heads, head_dim)
        v = nn.Dense(kv_dim, use_bias=False, name="v_proj")(x).reshape(batch, seq, self.num_kv_heads, head_dim)
        
        # GQA Repeat
        num_rep = self.num_heads // self.num_kv_heads
        if num_rep > 1:
            k = jnp.repeat(k, num_rep, axis=2)
            v = jnp.repeat(v, num_rep, axis=2)
            
        attn = nn.softmax(jnp.einsum('bqhd,bkhd->bhqk', q, k) * (head_dim**-0.5), axis=-1)
        return nn.Dense(self.dim, use_bias=False, name="o_proj")(jnp.einsum('bhqk,bkhd->bqhd', attn, v).reshape(batch, seq, self.dim))

class LlamaDecoderLayer(nn.Module):
    dim: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    @nn.compact
    def __call__(self, x):
        x = x + nn.remat(LlamaAttention)(self.dim, self.num_heads, self.num_kv_heads, name="self_attn")(nn.RMSNorm(name="input_layernorm")(x))
        x = x + nn.remat(LlamaMLP)(self.dim, self.intermediate_size, name="mlp")(nn.RMSNorm(name="post_attention_layernorm")(x))
        return x

# --- 2. FOUNDER MODE: SHARD-AWARE LOADER ---
def load_layer_weights(layer_idx):
    index_path = os.path.join(WEIGHTS_DIR, "model.safetensors.index.json")
    if not os.path.exists(index_path): return None # Handle single file later if needed

    with open(index_path, 'r') as f:
        weight_map = json.load(f)["weight_map"]
    
    prefix = f"model.layers.{layer_idx}."
    target_shards = set()
    for key, shard in weight_map.items():
        if key.startswith(prefix): target_shards.add(shard)
    
    # print(f"   -> Fetching Layer {layer_idx} from {len(target_shards)} shards...")
    weights = {}
    for shard in target_shards:
        w = load_file(os.path.join(WEIGHTS_DIR, shard))
        for k in w.keys():
            if k.startswith(prefix): weights[k] = w[k]
            
    # Map & Cast to FP32
    def get_w(name): return weights[f"{prefix}{name}.weight"].float().numpy().T
    return {
        "input_layernorm": {"scale": weights[f"{prefix}input_layernorm.weight"].float().numpy()},
        "post_attention_layernorm": {"scale": weights[f"{prefix}post_attention_layernorm.weight"].float().numpy()},
        "self_attn": {
            "q_proj": {"kernel": get_w("self_attn.q_proj")},
            "k_proj": {"kernel": get_w("self_attn.k_proj")},
            "v_proj": {"kernel": get_w("self_attn.v_proj")},
            "o_proj": {"kernel": get_w("self_attn.o_proj")},
        },
        "mlp": {
            "gate_proj": {"kernel": get_w("mlp.gate_proj")},
            "up_proj":   {"kernel": get_w("mlp.up_proj")},
            "down_proj": {"kernel": get_w("mlp.down_proj")},
        }
    }

# --- 3. EXECUTION ENGINE ---
@partial(jit, static_argnums=(2,))
def forward_layer(params, x, model_def):
    return model_def.apply({'params': params}, x)

def run_pipeline():
    print(f"🚀 VELOCITY MULTI-LAYER STREAMER")
    print(f"🎯 Goal: Stream {NUM_LAYERS} Layers of Llama-3 8B (Real Weights)")
    print("-" * 60)
    
    # 1. Initialize "Baton" (Input Embeddings)
    # In a real run, this comes from the Embedding Layer. We use random for the chain test.
    print("🎬 Initializing Input Signal...")
    current_activations = jax.random.normal(random.PRNGKey(0), (BATCH_SIZE, SEQ_LEN, DIM))
    
    total_start = time.time()
    
    # 2. THE VELOCITY LOOP
    for i in range(NUM_LAYERS):
        iter_start = time.time()
        
        # A. Ingest (CPU)
        layer_params = load_layer_weights(i)
        if layer_params is None:
            print(f"❌ Failed to load Layer {i}")
            break
            
        # B. Teleport (GPU)
        # Device Put happens implicitly or explicitly. 
        # We explicitly put the weights to measure transfer time.
        # Note: In JAX, passing them to JIT handles this, but let's be explicit.
        
        model = LlamaDecoderLayer(DIM, INTERMEDIATE, HEADS, KV_HEADS)
        
        # C. Compute
        # We pass the PREVIOUS activations (on GPU) into the NEW layer
        current_activations = forward_layer(layer_params, current_activations, model)
        current_activations.block_until_ready()
        
        # D. Offload (Optional for Inference, Mandatory for Training)
        # For this test, we just keep the 'current_activations' in VRAM as the baton.
        # (It's small: 1 * 128 * 4096 * 4 bytes = ~2MB).
        
        t = time.time() - iter_start
        print(f"Layer {i:<2} | ✅ Executed | Time: {t:.4f}s")
        
        # E. Garbage Collection (Manual Founder Mode)
        # We delete the heavy weights from CPU RAM to prevent buildup
        del layer_params

    total_time = time.time() - total_start
    print("-" * 60)
    print(f"🏁 Full Model Complete.")
    print(f"⏱️ Total Time: {total_time:.2f}s")
    print(f"⚡ Avg Layer Latency: {total_time/NUM_LAYERS:.4f}s")

if __name__ == "__main__":
    run_pipeline()