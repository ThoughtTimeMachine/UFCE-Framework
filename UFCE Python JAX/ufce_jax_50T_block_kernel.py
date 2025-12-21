import os
# 1. SAFETY: Prevent VRAM allocation crashes
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# 2. SPEED: Force Float32 (AI Mode) - Crucial for RTX 4070 Ti
import jax
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
from jax import jit, devices
from jax.lax import scan
import time

# --- CONFIGURATION: MATCHING THE C++ BENCHMARK ---
N_QUERIES = 50_000
N_KEYS    = 1_000_000_000  # 1 Billion
TOTAL_OPS = N_QUERIES * N_KEYS

# Block size tuning: 
# We process Keys in chunks. 
# A chunk of 4096 keys means we compute a (50,000 x 4096) matrix in VRAM.
# 50k * 4096 * 4 bytes = ~800 MB per step. This fits easily in the 12GB 4070 Ti.
BLOCK_SIZE = 4096 

def print_header():
    try:
        device_name = devices("gpu")[0].device_kind
    except:
        device_name = "CPU (Warning: Will be slow)"
    
    print("-" * 50)
    print(" UFCE GOD MODE BENCHMARK (JAX PORT)")
    print(f" Device:      {device_name}")
    print(f" Scenario:    {N_KEYS / 1e9:.0f} BILLION Token Context Window")
    print(f" Precision:   Float32 (AI Mode)")
    print(f" Total Ops:   {TOTAL_OPS / 1e12:.0f} TRILLION")
    print(f" Block Size:  {BLOCK_SIZE}")
    print("-" * 50)

# --- THE JAX KERNEL ---
# We use 'scan' to loop over the blocks of Keys.
# This prevents us from trying to materialize a (50k x 1B) matrix which is 200TB.
@jit
def god_mode_kernel(queries, keys_blocked):
    
    # Initialize 'global_max' with a very small number (like -1e9 in C++)
    init_max = jnp.full((N_QUERIES,), -1e9, dtype=jnp.float32)

    def body(current_max, key_block):
        # MATRIX MATH TRICK:
        # Instead of looping k=0..N like C++, we do a massive Matrix Multiply.
        # Queries: (50000, 1)
        # Key_Block: (1, 4096)
        # Result: (50000, 4096) -> This saturates the Tensor Cores
        
        # 1. Compute scores for this block (Outer Product)
        # We use implicit broadcasting: (N, 1) * (M,) -> (N, M)
        scores = jnp.expand_dims(queries, 1) * key_block
        
        # 2. Find the max for this block (Reduce along the block axis)
        block_max = jnp.max(scores, axis=1)
        
        # 3. Update the running global max
        new_max = jnp.maximum(current_max, block_max)
        
        return new_max, None

    # Run the loop
    final_max, _ = scan(body, init_max, keys_blocked)
    return final_max

def run_benchmark():
    print_header()

    # 1. DATA GENERATION
    print("Allocating & Generating 4GB Data on GPU...", end="\r")
    key = jax.random.PRNGKey(0)
    
    # Generate Queries (50k)
    d_q = jax.random.normal(key, (N_QUERIES,), dtype=jnp.float32)
    
    # Generate Keys (1 Billion)
    # We generate them directly into the reshaped block format to save time
    # Shape: (Num_Blocks, Block_Size)
    num_blocks = N_KEYS // BLOCK_SIZE
    d_k_blocked = jax.random.normal(key, (num_blocks, BLOCK_SIZE), dtype=jnp.float32)
    
    # Force synchronization to ensure allocation is done
    d_k_blocked.block_until_ready()
    print("Allocating & Generating 4GB Data on GPU... DONE")

    # 2. WARMUP (Compiles the JAX/XLA Graph)
    print("Compiling Kernel...", end="\r")
    # Run on just 1 block to compile
    _ = god_mode_kernel(d_q, d_k_blocked[:1])
    print("Compiling Kernel... DONE           ")

    # 3. EXECUTION
    print(f"Launching GOD MODE Kernel ({TOTAL_OPS/1e12:.0f} Trillion Ops)...")
    start = time.time()
    
    final_scores = god_mode_kernel(d_q, d_k_blocked)
    
    final_scores.block_until_ready()
    end = time.time()

    # 4. VALIDATION
    duration = end - start
    throughput_ops = TOTAL_OPS / duration
    throughput_trillions = throughput_ops / 1e12

    print("\n--- GOD MODE RESULTS ---")
    print(f"Time:        {duration:.4f} s")
    print(f"Throughput:  {throughput_trillions:.2f} TRILLION ops/sec")
    
    if throughput_trillions > 1.0:
        print("\n✅ SUCCESS: Broke the 1 Trillion Barrier.")

if __name__ == "__main__":
    run_benchmark()