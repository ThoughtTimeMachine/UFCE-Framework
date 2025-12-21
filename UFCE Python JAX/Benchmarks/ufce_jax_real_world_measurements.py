
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
# SAFETY: Prevent VRAM crashes
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# REALISM: Use Float32 (AI Mode).
import jax
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
from jax import jit, devices
from jax.lax import scan, top_k
import time

# --- OPTIMIZED CONFIGURATION ---
# We reduced Queries to 1,000 to make the benchmark finish in ~10 seconds.
# Total Ops = 100 Billion (Still massive, but manageable for sorting).
N_QUERIES = 1_000         
N_KEYS    = 100_000_000   # 100 Million Keys (The massive context window)
BLOCK_SIZE = 4096         # Smaller block size is often faster for sorting logic
TOP_K     = 10            

def print_header(mode):
    print("\n" + "="*60)
    print(f" UFCE REAL-WORLD MEASUREMENT: {mode}")
    print(f" Context: {N_KEYS/1e6:.0f}M Tokens | Queries: {N_QUERIES}")
    print(f" Precision: Float32 | Block Size: {BLOCK_SIZE}")
    print("="*60)

# =========================================================================
# KERNEL 1: TOP-K ANOMALY DETECTION (Security/Finance)
# =========================================================================
@jit
def ufce_topk_kernel(queries, keys_blocked):
    n_q = queries.shape[0]
    
    # Initialize with very low values
    init_val = jnp.full((n_q, TOP_K), -1e9, dtype=jnp.float32)
    init_idx = jnp.zeros((n_q, TOP_K), dtype=jnp.int32)

    def body(carry, key_block_tuple):
        curr_vals, curr_idxs = carry
        key_block, block_start_idx = key_block_tuple 
        
        # 1. Compute Scores (Outer Product)
        # (N_Q, 1) * (1, Block) -> (N_Q, Block)
        scores = jnp.expand_dims(queries, 1) * jnp.expand_dims(key_block, 0)
        
        # 2. Find Top-K in this block (Local Sort)
        block_vals, block_rel_idxs = top_k(scores, k=TOP_K)
        block_abs_idxs = block_rel_idxs + block_start_idx
        
        # 3. Merge with current bests (Global Sort)
        merged_vals = jnp.concatenate([curr_vals, block_vals], axis=1)
        merged_idxs = jnp.concatenate([curr_idxs, block_abs_idxs], axis=1)
        
        # Find Top-K from the merged set of 20 candidates
        best_vals, best_sorting_indices = top_k(merged_vals, k=TOP_K)
        best_idxs = jnp.take_along_axis(merged_idxs, best_sorting_indices, axis=1)
        
        return (best_vals, best_idxs), None

    (final_vals, final_idxs), _ = scan(body, (init_val, init_idx), keys_blocked)
    return final_vals, final_idxs

# =========================================================================
# KERNEL 2: STREAMING SOFTMAX (LLM Attention)
# =========================================================================
@jit
def ufce_softmax_kernel(queries, keys_blocked):
    n_q = queries.shape[0]
    init_max = jnp.full((n_q,), -1e9, dtype=jnp.float32)
    init_sum = jnp.zeros((n_q,), dtype=jnp.float32)
    
    def body(carry, key_block):
        curr_max, curr_sum = carry
        
        scores = jnp.expand_dims(queries, 1) * jnp.expand_dims(key_block, 0)
        block_max = jnp.max(scores, axis=1)
        
        new_max = jnp.maximum(curr_max, block_max)
        correction = jnp.exp(curr_max - new_max)
        
        block_exp_sum = jnp.sum(jnp.exp(scores - jnp.expand_dims(new_max, 1)), axis=1)
        new_sum = (curr_sum * correction) + block_exp_sum
        
        return (new_max, new_sum), None

    (final_max, final_sum_exp), _ = scan(body, (init_max, init_sum), keys_blocked)
    return final_max + jnp.log(final_sum_exp)


def run_benchmark():
    # --- DATA GENERATION ---
    print("Generating Data (Float32)...", end="\r")
    key = jax.random.PRNGKey(42)
    
    d_q = jax.random.normal(key, (N_QUERIES,), dtype=jnp.float32)
    
    # Keys reshaped into blocks
    num_blocks = N_KEYS // BLOCK_SIZE
    d_k_blocks = jax.random.normal(key, (num_blocks, BLOCK_SIZE), dtype=jnp.float32)
    d_idxs = jnp.arange(num_blocks) * BLOCK_SIZE
    scan_args = (d_k_blocks, d_idxs)
    
    d_k_blocks.block_until_ready()
    print("Generating Data... DONE            ")

    # --- BENCHMARK 1: TOP-K ---
    print_header("TOP-K ANOMALY DETECTION")
    print("Compiling...", end="\r")
    _ = ufce_topk_kernel(d_q[:10], (d_k_blocks[:2], d_idxs[:2]))
    print("Compiling... DONE")
    
    print("Running Top-K Benchmark...")
    start = time.time()
    vals, idxs = ufce_topk_kernel(d_q, scan_args)
    vals.block_until_ready()
    end = time.time()
    
    ops = N_QUERIES * N_KEYS
    print(f"Time: {end-start:.4f}s")
    print(f"Throughput: {ops/(end-start)/1e9:.2f} Billion Ops/s")
    print(f"Check: Top score {vals[0,0]:.2f}")

    # --- BENCHMARK 2: ONLINE SOFTMAX ---
    print_header("STREAMING SOFTMAX (ATTENTION)")
    print("Compiling...", end="\r")
    _ = ufce_softmax_kernel(d_q[:10], d_k_blocks[:2])
    print("Compiling... DONE")
    
    print("Running Softmax Benchmark...")
    start = time.time()
    lse = ufce_softmax_kernel(d_q, d_k_blocks)
    lse.block_until_ready()
    end = time.time()
    
    print(f"Time: {end-start:.4f}s")
    print(f"Throughput: {ops/(end-start)/1e9:.2f} Billion Ops/s")
    print(f"Check: LogSumExp sample {lse[0]:.2f}")

if __name__ == "__main__":
    run_benchmark()