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

# Infinite Attention Core 
import os
# SAFETY: Prevent JAX from pre-allocating all VRAM, crucial for hybrid workflows
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
# SPEED: Use Float32 (Standard AI Precision). Set to True only for scientific verification.
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
from jax import jit, random
from jax.lax import scan, top_k
import time

# === CONFIGURATION ===
# Real LLM Dimensions for Benchmark
BATCH_SIZE = 1
NUM_HEADS  = 1        # Simplified for demo (real transformers use 8-32)
HEAD_DIM   = 128      # Standard head dimension (e.g., Llama-3 uses 128)
N_QUERIES  = 128      # Tokens generating attention (The "Prompt")
N_KEYS     = 1_000_000 # The "Infinite" Context (1 Million Tokens)
TOP_K      = 50       # We only care about the top 50 influential tokens
BLOCK_SIZE = 4096     # Streaming Block Size

class UFCEInfiniteAttention:
    def __init__(self, top_k=50):
        self.top_k = top_k

    # =========================================================================
    # KERNEL 1: GLOBAL STREAMING STATS (Zero-Memory Aggregates)
    # =========================================================================
    @jit
    def _global_streaming_stats(self, queries, keys_blocked):
        """
        Computes Mean and Variance of the interaction flux across the entire stream.
        Optimized with Dot Product to avoid N*B*D memory explosion.
        """
        init_sum = 0.0
        init_sq_sum = 0.0
        
        def body(carry, key_block):
            curr_sum, curr_sq = carry
            
            # OPTIMIZATION: Use dot product (batch matmul)
            # (N_Q, D) @ (Block, D).T -> (N_Q, Block)
            # This avoids creating the massive (N_Q, Block, D) broadcast tensor
            scores = jnp.dot(queries, key_block.T)
            
            block_sum = jnp.sum(scores)
            block_sq = jnp.sum(scores ** 2)
            
            return (curr_sum + block_sum, curr_sq + block_sq), None
        
        (total_sum, total_sq), _ = scan(body, (init_sum, init_sq_sum), keys_blocked)
        return total_sum, total_sq

    # =========================================================================
    # KERNEL 2: GLOBAL TOP-K DISCOVERY (Zero-Memory Scan)
    # =========================================================================
    @jit
    def _global_topk_flux(self, queries, key_blocks_tuple):
        """
        PASS 1: The "Radar Scan"
        Finds the indices of the most relevant tokens in the infinite stream.
        """
        keys_blocked, block_offsets = key_blocks_tuple
        n_q, dim = queries.shape
        
        # Init: (N_Queries, Top_K) - Start with negative infinity
        init_vals = jnp.full((n_q, self.top_k), -jnp.inf, dtype=jnp.float32)
        init_idxs = jnp.zeros((n_q, self.top_k), dtype=jnp.int32)

        def body(carry, inputs):
            curr_vals, curr_idxs = carry
            k_block, offset = inputs
            
            # 1. Compute Flux/Attention Scores
            scores = jnp.dot(queries, k_block.T)
            
            # 2. Local Top-K (The Filter)
            block_vals, block_rel_idxs = top_k(scores, k=self.top_k)
            block_abs_idxs = block_rel_idxs + offset
            
            # 3. Merge with Global Top-K (The Accumulator)
            merged_vals = jnp.concatenate([curr_vals, block_vals], axis=1)
            merged_idxs = jnp.concatenate([curr_idxs, block_abs_idxs], axis=1)
            
            # 4. Sort and Prune to keep only the absolute best
            final_vals, sort_indices = top_k(merged_vals, k=self.top_k)
            final_idxs = jnp.take_along_axis(merged_idxs, sort_indices, axis=1)
            
            return (final_vals, final_idxs), None

        # Scan over the massive key stream
        (final_scores, final_indices), _ = scan(body, (init_vals, init_idxs), (keys_blocked, block_offsets))
        return final_indices

    # =========================================================================
    # KERNEL 3: LOCAL EXACT SOFTMAX (Drill-Down Precision)
    # =========================================================================
    @jit
    def _local_softmax(self, queries, selected_keys):
        """
        PASS 2: The "Precision Strike"
        Compute exact Softmax weights on just the Top-K survivors.
        """
        # Einstein Summation for batch dot product on the specific indices
        # query: (Q, D), keys: (Q, K, D) -> scores: (Q, K)
        scores = jnp.einsum('qd,qkd->qk', queries, selected_keys)
        
        # Standard Transformer Scaling
        scale = 1.0 / jnp.sqrt(queries.shape[-1])
        scores = scores * scale
        
        # Numerically Stable Softmax
        max_scores = jnp.max(scores, axis=-1, keepdims=True)
        exp_scores = jnp.exp(scores - max_scores)
        weights = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)
        
        return weights

    # =========================================================================
    # THE ORCHESTRATOR
    # =========================================================================
    def forward(self, queries, all_keys, return_stats=True, return_topk=True, return_softmax=True):
        # 1. Reshape keys into blocks for the stream
        num_blocks = all_keys.shape[0] // BLOCK_SIZE
        # Ensure we trim any excess that doesn't fit a block (or pad in production)
        keys_blocked = all_keys[:num_blocks*BLOCK_SIZE].reshape(num_blocks, BLOCK_SIZE, HEAD_DIM)
        block_offsets = jnp.arange(num_blocks) * BLOCK_SIZE
        
        results = {}
        
        # MODE A: Global Statistics (Mean/Variance)
        if return_stats:
            total_sum, total_sq = self._global_streaming_stats(queries, keys_blocked)
            n_total = queries.shape[0] * (num_blocks * BLOCK_SIZE)
            mean = total_sum / n_total
            variance = (total_sq / n_total) - mean ** 2
            results['global_mean'] = mean
            results['global_variance'] = variance

        # MODE B: Top-K Discovery & Drill Down
        if return_topk or return_softmax:
            # Pass 1: Scan
            top_indices = self._global_topk_flux(queries, (keys_blocked, block_offsets))
            
            if return_topk:
                results['top_k_indices'] = top_indices
            
            if return_softmax:
                # Pass 2: Gather & Compute
                # In a disk-based system, this 'all_keys[top_indices]' is the seek/read step
                selected_keys = all_keys[top_indices]
                weights = self._local_softmax(queries, selected_keys)
                results['local_softmax_weights'] = weights

        return results

# === BENCHMARK ===
if __name__ == "__main__":
    print(f"🚀 UFCE INFINITE ATTENTION CORE")
    print(f"Context: {N_KEYS} Tokens | Queries: {N_QUERIES}")
    print(f"Strategy: Zero-Memory Scan -> Top-{TOP_K} Drill Down")
    
    key = random.PRNGKey(42)
    
    # Generate Mock Embeddings (Float32)
    print("Generating Vectors...", end="\r")
    Q = random.normal(key, (N_QUERIES, HEAD_DIM))
    K = random.normal(key, (N_KEYS, HEAD_DIM)) # The Infinite Context
    print("Generating Vectors... DONE      ")
    
    layer = UFCEInfiniteAttention(top_k=TOP_K)
    
    print("Warming JIT...", end="\r")
    # Warmup on a small subset to compile the graph
    _ = layer.forward(Q, K[:BLOCK_SIZE*2], return_stats=True, return_topk=True, return_softmax=True)
    print("Warming JIT... DONE       ")
    
    print("Running Full Context Analysis...")
    start = time.time()
    results = layer.forward(Q, K)
    # Block until JAX is actually done
    if 'local_softmax_weights' in results:
        results['local_softmax_weights'].block_until_ready()
    elif 'global_mean' in results:
        results['global_mean'].block_until_ready()
    end = time.time()
    
    throughput = (N_QUERIES * N_KEYS) / (end - start)
    
    print(f"\n=== RESULTS ===")
    print(f"Time:       {end - start:.4f} s")
    print(f"Throughput: {throughput / 1e9:.2f} Billion Pairs/s")
    
    # Validate Stats
    print(f"\n[Global Analytics]")
    print(f"Mean Flux:  {results['global_mean']:.4f}")
    print(f"Variance:   {results['global_variance']:.4f}")
    
    # Validate Drill-Down
    print(f"\n[Drill-Down Precision]")
    weights = results['local_softmax_weights']
    print(f"Weight Shape: {weights.shape} (Queries, Top_K)")
    print(f"Sum Check:    {jnp.sum(weights[0]):.4f} (Should be ~1.0)")
    print(f"Top-1 Index:  {results['top_k_indices'][0,0]}")