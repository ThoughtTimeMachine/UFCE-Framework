import os

# --- VELOCITY CONFIG ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit
from flax import linen as nn
import optax
from collections import deque

# --- LLAMA-3-8B CONFIGURATION ---
# --- LLAMA-3-8B CONFIGURATION ---
DIM = 4096           
HEADS = 32           
HEAD_DIM = 128       
INTERMEDIATE = 14336 
# CRITICAL FIX: Dropped from 16 -> 4 to fit in 12GB VRAM @ FP32
BATCH_SIZE = 4      
SEQ_LEN = 512        
LEARNING_RATE = 2e-5

# DB Path Logic
if os.path.exists("knowledge_base/knowledge_base_full.dat"):
    DB_PATH = "knowledge_base/knowledge_base_full.dat"
elif os.path.exists("../knowledge_base/knowledge_base_full.dat"):
    DB_PATH = "../knowledge_base/knowledge_base_full.dat"
else:
    DB_PATH = "knowledge_base_full.dat"

# Global Optimizer
TX = optax.adamw(LEARNING_RATE)

# --- 1. MODEL DEFINITION (WITH REMATERIALIZATION) ---
class LlamaMLP(nn.Module):
    dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        gate_proj = nn.Dense(self.hidden_dim, use_bias=False)(x)
        up_proj = nn.Dense(self.hidden_dim, use_bias=False)(x)
        down_proj = nn.Dense(self.dim, use_bias=False)
        x = nn.silu(gate_proj) * up_proj
        return down_proj(x)

class LlamaAttention(nn.Module):
    dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        q_proj = nn.Dense(self.dim, use_bias=False)(x)
        k_proj = nn.Dense(self.dim, use_bias=False)(x)
        v_proj = nn.Dense(self.dim, use_bias=False)(x)
        o_proj = nn.Dense(self.dim, use_bias=False)
        
        batch, seq, _ = x.shape
        head_dim = self.dim // self.num_heads
        
        q = q_proj.reshape(batch, seq, self.num_heads, head_dim)
        k = k_proj.reshape(batch, seq, self.num_heads, head_dim)
        v = v_proj.reshape(batch, seq, self.num_heads, head_dim)
        
        attn_weights = jnp.einsum('bqhd,bkhd->bhqk', q, k) * (head_dim ** -0.5)
        attn_weights = nn.softmax(attn_weights, axis=-1)
        
        attn_output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        attn_output = attn_output.reshape(batch, seq, self.dim)
        
        return o_proj(attn_output)

class LlamaDecoderLayer(nn.Module):
    dim: int
    intermediate_size: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        # FIX: Wrap massive blocks in nn.remat to save VRAM
        # This trades slightly more compute for significantly less memory
        AttentionBlock = nn.remat(LlamaAttention)
        MLPBlock = nn.remat(LlamaMLP)

        norm_1 = nn.RMSNorm()(x)
        attn_out = AttentionBlock(self.dim, self.num_heads)(norm_1)
        x = x + attn_out 
        
        norm_2 = nn.RMSNorm()(x)
        mlp_out = MLPBlock(self.dim, self.intermediate_size)(norm_2)
        x = x + mlp_out 
        return x

# --- 2. VELOCITY RING BUFFER ---
class RingBuffer:
    def __init__(self, capacity=4):
        self.buffer = deque(maxlen=capacity)
    def push(self, chunk):
        self.buffer.append(chunk)
    def get(self):
        if len(self.buffer) > 0: return self.buffer.popleft()
        return None

# --- 3. TRAINING STATE ---
def create_train_state(rng, model, sample_input):
    params = model.init(rng, sample_input)['params']
    return TX.init(params), params

def train_step(params, opt_state, batch, model):
    def loss_fn(p):
        output = model.apply({'params': p}, batch)
        return jnp.mean((output - batch) ** 2)

    val, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = TX.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, val

# Manual JIT
train_step = jit(train_step, static_argnums=(3,))

# --- 4. MAIN LOOP ---
def run_simulation():
    print("🦙 Initializing Llama-3 (8B) Decoder Layer Simulation (w/ Gradient Checkpointing)...")
    print(f"   Hidden Dim: {DIM} | MLP Dim: {INTERMEDIATE} | Heads: {HEADS}")
    
    try:
        raw_vectors = np.memmap(DB_PATH, dtype='float32', mode='r')
        total_elements = raw_vectors.shape[0]
        elements_per_batch = BATCH_SIZE * SEQ_LEN * DIM
        num_batches = total_elements // elements_per_batch
        print(f"✅ Data Source: {num_batches} batches available.")
    except:
        print(f"❌ DB not found at {DB_PATH}.")
        return

    model = LlamaDecoderLayer(dim=DIM, intermediate_size=INTERMEDIATE, num_heads=HEADS)
    rng = random.PRNGKey(0)
    
    dummy_input = jnp.ones((BATCH_SIZE, SEQ_LEN, DIM))
    print("🧠 Allocating Weights...")
    opt_state, params = create_train_state(rng, model, dummy_input)
    
    ring = RingBuffer()
    print("\n⚡ STARTING REAL-LAYER TRAINING ⚡")
    print(f"{'Step':<10} | {'Loss':<10} | {'Throughput (Tokens/s)':<25}")
    print("-" * 75)
    
    for i in range(num_batches):
        iter_start = time.time()
        
        # 1. CPU Load
        start_idx = i * elements_per_batch
        batch_cpu = raw_vectors[start_idx : start_idx + elements_per_batch]
        if batch_cpu.shape[0] != elements_per_batch: continue
        batch_cpu = batch_cpu.reshape((BATCH_SIZE, SEQ_LEN, DIM))
        
        # 2. Pipeline
        ring.push(batch_cpu)
        batch_gpu = ring.get()
        if batch_gpu is None: continue
        
        batch_jax = jax.device_put(batch_gpu)
        
        # 3. Compute
        params, opt_state, loss = train_step(params, opt_state, batch_jax, model)
        loss.block_until_ready()
        
        # Metrics
        t = time.time() - iter_start
        tokens = BATCH_SIZE * SEQ_LEN
        tps = tokens / t
        
        if i % 1 == 0:
            print(f"{i:<10} | {loss:.4f}     | {tps:,.0f}")

    print("✅ Done.")

if __name__ == "__main__":
    run_simulation()