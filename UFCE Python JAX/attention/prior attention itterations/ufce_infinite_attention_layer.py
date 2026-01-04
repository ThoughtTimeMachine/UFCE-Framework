
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
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
from jax import jit, random
from jax.lax import scan, top_k
import time

# === CONFIGURATION ===
# Real LLM Dimensions
BATCH_SIZE = 1
NUM_HEADS  = 1        # Simplified for demo (normally 8-32)
HEAD_DIM   = 128      # Standard head dimension (e.g., Llama-3 uses 128)
N_QUERIES  = 128      # Tokens generating attention (The "Prompt")
N_KEYS     = 1_000_000 # The "Infinite" Context (1 Million Tokens)
TOP_K      = 50       # We only care about the top 50 influential tokens
BLOCK_SIZE = 4096     # Streaming Block Size

class UFCEInfiniteAttention:
    def __init__(self, top_k=50):
        self.top_k = top_k

    @jit
    def _stream_topk_indices(self, query_batch, key_blocks_tuple):
        """
        PASS 1: The "Radar Scan" (Zero-Memory)
        Finds the indices of the most relevant tokens in the infinite stream.
        """
        keys_blocked, block_offsets = key_blocks_tuple
        n_q, dim = query_batch.shape
        
        # Init: (N_Queries, Top_K) - Start with negative infinity
        init_vals = jnp.full((n_q, self.top_k), -jnp.inf, dtype=jnp.float32)
        init_idxs = jnp.zeros((n_q, self.top_k), dtype=jnp.int32)

        def body(carry, inputs):
            curr_vals, curr_idxs = carry
            k_block, offset = inputs
            
            # 1. Compute Raw Attention Scores (Dot Product)
            # (N_Q, Dim) @ (Block, Dim).T -> (N_Q, Block)
            # We use half-precision for the scan to be insanely fast, if supported
            scores = jnp.dot(query_batch, k_block.T)
            
            # 2. Local Top-K (The Filter)
            block_vals, block_rel_idxs = top_k(scores, k=self.top_k)
            block_abs_idxs = block_rel_idxs + offset
            
            # 3. Merge with Global Top-K (The Accumulator)
            merged_vals = jnp.concatenate([curr_vals, block_vals], axis=1)
            merged_idxs = jnp.concatenate([curr_idxs, block_abs_idxs], axis=1)
            
            # 4. Sort and Prune
            final_vals, sort_indices = top_k(merged_vals, k=self.top_k)
            final_idxs = jnp.take_along_axis(merged_idxs, sort_indices, axis=1)
            
            return (final_vals, final_idxs), None

        # Scan over the massive key stream
        (final_scores, final_indices), _ = scan(body, (init_vals, init_idxs), (keys_blocked, block_offsets))
        return final_indices

    @jit
    def _compute_exact_softmax(self, query_batch, selected_keys):
        """
        PASS 2: The "Precision Strike" (Standard Attention)
        Compute exact Softmax weights on just the survivors.
        """
        # (N_Q, Dim) @ (N_Q, Top_K, Dim).T ?? 
        # Actually easier: Einstein Summation for batch dot product
        # query: (Q, D), keys: (Q, K, D) -> scores: (Q, K)
        scores = jnp.einsum('qd,qkd->qk', query_batch, selected_keys)
        
        # Scale (standard Transformer scaling)
        scale = 1.0 / jnp.sqrt(query_batch.shape[-1])
        scores = scores * scale
        
        # Softmax (Numerically Stable)
        max_scores = jnp.max(scores, axis=-1, keepdims=True)
        exp_scores = jnp.exp(scores - max_scores)
        weights = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)
        
        return weights

    def forward(self, queries, all_keys):
        # 1. Reshape keys into blocks for the stream
        num_blocks = all_keys.shape[0] // BLOCK_SIZE
        keys_blocked = all_keys[:num_blocks*BLOCK_SIZE].reshape(num_blocks, BLOCK_SIZE, HEAD_DIM)
        offsets = jnp.arange(num_blocks) * BLOCK_SIZE
        
        # 2. PASS 1: Find the Needles
        # Note: We pass the WHOLE stream to the scan
        top_indices = self._stream_topk_indices(queries, (keys_blocked, offsets))
        
        # 3. GATHER: Fetch the actual vectors for the top indices
        # In a real system (SSD), this would be a disk read.
        # Here (RAM), it's a fancy indexing operation.
        # selected_keys shape: (N_Queries, Top_K, Dim)
        selected_keys = all_keys[top_indices]
        
        # 4. PASS 2: Compute Weights
        weights = self._compute_exact_softmax(queries, selected_keys)
        
        return weights, top_indices

# === BENCHMARK ===
if __name__ == "__main__":
    print(f"🚀 UFCE INFINITE ATTENTION LAYER")
    print(f"Context: {N_KEYS} Tokens | Queries: {N_QUERIES}")
    print(f"Strategy: Zero-Memory Scan -> Top-{TOP_K} Drill Down")
    
    key = random.PRNGKey(0)
    
    # Generate Mock Embeddings (Float32)
    print("Generating Vectors...", end="\r")
    Q = random.normal(key, (N_QUERIES, HEAD_DIM))
    K = random.normal(key, (N_KEYS, HEAD_DIM)) # The Infinite Context
    print("Generating Vectors... DONE      ")
    
    layer = UFCEInfiniteAttention(top_k=TOP_K)
    
    print("Warming JIT...", end="\r")
    layer.forward(Q, K[:BLOCK_SIZE*2]) # Warmup on small subset
    print("Warming JIT... DONE       ")
    
    print("Running Full Context Attention...")
    start = time.time()
    weights, indices = layer.forward(Q, K)
    weights.block_until_ready()
    end = time.time()
    
    throughput = (N_QUERIES * N_KEYS) / (end - start)
    
    print(f"\n=== RESULTS ===")
    print(f"Time:       {end - start:.4f} s")
    print(f"Throughput: {throughput / 1e9:.2f} Billion Pairs/s")
    print(f"Sparsity:   {100 * (1 - (TOP_K / N_KEYS)):.4f}% (Tokens ignored)")
    print(f"Sample Indices (First Query): {indices[0]}")