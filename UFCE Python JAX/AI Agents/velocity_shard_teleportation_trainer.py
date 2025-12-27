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
import optax
from collections import deque
from functools import partial
from safetensors.torch import load_file

# --- VELOCITY CONFIG ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

# --- LLAMA-3-8B CONFIGURATION ---
DIM = 4096
HEADS = 32
KV_HEADS = 8  # CRITICAL FIX: Llama 3 uses Grouped Query Attention (GQA)
INTERMEDIATE = 14336
BATCH_SIZE, SEQ_LEN = 4, 512
LEARNING_RATE = 2e-5

WEIGHTS_DIR = "./llama3_weights"
DB_PATH = "knowledge_base_full.dat" if os.path.exists("knowledge_base_full.dat") else "knowledge_base/knowledge_base_full.dat"
TX = optax.adamw(LEARNING_RATE)

# --- 1. THE SHARD-AWARE LOADER ---
def load_actual_weights(layer_idx):
    index_path = os.path.join(WEIGHTS_DIR, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        # Fallback for single file
        single = os.path.join(WEIGHTS_DIR, "model.safetensors")
        if os.path.exists(single):
            print(f"📡 Teleporting Layer {layer_idx} from single file...")
            return map_weights(load_file(single), layer_idx)
        return None

    with open(index_path, 'r') as f:
        weight_map = json.load(f)["weight_map"]
    
    prefix = f"model.layers.{layer_idx}."
    shard_to_keys = {}
    for key, shard_name in weight_map.items():
        if key.startswith(prefix):
            if shard_name not in shard_to_keys: shard_to_keys[shard_name] = []
            shard_to_keys[shard_name].append(key)
    
    print(f"📡 Teleporting Layer {layer_idx} from {len(shard_to_keys)} shards...")
    layer_weights_flat = {}
    for shard_name, keys in shard_to_keys.items():
        shard_path = os.path.join(WEIGHTS_DIR, shard_name)
        shard_data = load_file(shard_path)
        for key in keys:
            layer_weights_flat[key] = shard_data[key]
            
    return map_weights(layer_weights_flat, layer_idx)

def map_weights(source_weights, layer_idx):
    prefix = f"model.layers.{layer_idx}."
    # FIX: Cast BFloat16 to Float32 before NumPy conversion
    def get_w(name):
        return source_weights[f"{prefix}{name}.weight"].float().numpy().T

    return {
        "input_layernorm": {"scale": source_weights[f"{prefix}input_layernorm.weight"].float().numpy()},
        "post_attention_layernorm": {"scale": source_weights[f"{prefix}post_attention_layernorm.weight"].float().numpy()},
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

# --- 2. ARCHITECTURE (UPDATED FOR GQA) ---
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
    num_kv_heads: int  # New Arg for GQA

    @nn.compact
    def __call__(self, x):
        batch, seq, _ = x.shape
        head_dim = self.dim // self.num_heads
        
        # Dimensions for GQA
        kv_dim = self.num_kv_heads * head_dim
        
        # Projections
        q = nn.Dense(self.dim, use_bias=False, name="q_proj")(x)
        k = nn.Dense(kv_dim, use_bias=False, name="k_proj")(x) # Smaller Dim
        v = nn.Dense(kv_dim, use_bias=False, name="v_proj")(x) # Smaller Dim
        o_proj = nn.Dense(self.dim, use_bias=False, name="o_proj")
        
        # Reshape for Heads
        q = q.reshape(batch, seq, self.num_heads, head_dim)
        k = k.reshape(batch, seq, self.num_kv_heads, head_dim)
        v = v.reshape(batch, seq, self.num_kv_heads, head_dim)
        
        # GQA Replication: Repeat K/V heads to match Q heads
        # 32 / 8 = 4 repeats per head
        num_rep = self.num_heads // self.num_kv_heads
        if num_rep > 1:
            k = jnp.repeat(k, num_rep, axis=2)
            v = jnp.repeat(v, num_rep, axis=2)
        
        # Standard Attention Math
        attn_weights = jnp.einsum('bqhd,bkhd->bhqk', q, k) * (head_dim ** -0.5)
        attn_weights = nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v).reshape(batch, seq, self.dim)
        
        return o_proj(attn_output)

class LlamaDecoderLayer(nn.Module):
    dim: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int # Pass it down
    
    @nn.compact
    def __call__(self, x):
        # Pass num_kv_heads to Attention
        AttentionBlock = nn.remat(LlamaAttention)
        MLPBlock = nn.remat(LlamaMLP)
        
        # Self Attention
        h = AttentionBlock(self.dim, self.num_heads, self.num_kv_heads, name="self_attn")(
            nn.RMSNorm(name="input_layernorm")(x)
        )
        x = x + h
        
        # MLP
        h = MLPBlock(self.dim, self.intermediate_size, name="mlp")(
            nn.RMSNorm(name="post_attention_layernorm")(x)
        )
        x = x + h
        return x

# --- 3. TRAINING ENGINE ---
@partial(jit, static_argnums=(3,))
def train_step(params, opt_state, batch, model):
    def loss_fn(p):
        out = model.apply({'params': p}, batch)
        return jnp.mean((out - batch) ** 2)
    val, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = TX.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), new_opt_state, val

# --- 4. MAIN LOOP ---
def run_simulation():
    print("🚀 VELOCITY: Real Llama-3 (GQA) Weight Ingestion")
    
    try:
        raw_vectors = np.memmap(DB_PATH, dtype='float32', mode='r')
        elements_per_batch = BATCH_SIZE * SEQ_LEN * DIM
        num_batches = raw_vectors.shape[0] // elements_per_batch
        print(f"✅ Data Reservoir: {num_batches} batches available.")
    except:
        print(f"❌ DB not found at {DB_PATH}.")
        return

    # Initialize with GQA Config
    model = LlamaDecoderLayer(dim=DIM, intermediate_size=INTERMEDIATE, num_heads=HEADS, num_kv_heads=KV_HEADS)
    
    real_params = load_actual_weights(0)
    if real_params:
        print("✅ Weights Loaded & Cast to FP32.")
        params = real_params
    else:
        print("⚠️ Random Init (Sim Mode)")
        params = model.init(random.PRNGKey(0), jnp.ones((BATCH_SIZE, SEQ_LEN, DIM)))['params']
    
    opt_state = TX.init(params)
    ring = deque(maxlen=4)
    
    print("\n⚡ STARTING PIPELINE ⚡")
    for i in range(num_batches):
        iter_start = time.time()
        start_idx = i * elements_per_batch
        batch_cpu = raw_vectors[start_idx : start_idx + elements_per_batch].reshape((BATCH_SIZE, SEQ_LEN, DIM))
        
        ring.append(batch_cpu)
        batch_gpu = ring.popleft()
        
        batch_jax = jax.device_put(batch_gpu)
        params, opt_state, loss = train_step(params, opt_state, batch_jax, model)
        loss.block_until_ready()
        
        tps = (BATCH_SIZE * SEQ_LEN) / (time.time() - iter_start)
        if i % 1 == 0:
            print(f"Step {i:<5} | Loss: {loss:.4f} | Tokens/s: {tps:,.0f}")

if __name__ == "__main__":
    run_simulation()