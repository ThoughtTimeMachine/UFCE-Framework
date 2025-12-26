import os

# --- VELOCITY ARCHITECTURE CONFIG ---
# 1. Disable JAX Preallocation (Crucial for Pinned Memory behavior)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# 2. Reserve 90% of VRAM for Compute (Weights), leaving 10% for the Streaming Buffer
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90" 
# 3. Force "Strict" picking to avoid re-compilation overhead during streaming
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
import psutil
from collections import deque

# --- CONFIGURATION ---
DB_PATH = "knowledge_base_full.dat"
# Dimensionality of your "Simulated" Model (e.g., Llama-3 Hidden Size is 4096)
MODEL_DIM = 4096 
BATCH_SIZE = 1024  # How many vectors to train on at once
LEARNING_RATE = 0.01

print("🚀 Initializing VELOCITY Trainer (JAX)...")

# --- 1. THE VELOCITY RING BUFFER ---
class RingBuffer:
    def __init__(self, capacity=4):
        """
        Implements a Quad-Buffered Pipeline (4 Zones).
        In a real low-level C++ implementation, these would be physical RAM addresses.
        In Python, we simulate this with a Deque to manage the 'Ownership' of data chunks.
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, data_chunk):
        """Zone 1 -> Zone 2: Ingest Data"""
        self.buffer.append(data_chunk)
        
    def get_batch(self):
        """Zone 3: Pin/Feed to GPU"""
        if len(self.buffer) > 0:
            return self.buffer.popleft()
        return None

# --- 2. JAX COMPUTE KERNELS (The GPU Brain) ---

def init_params(key, dim):
    """Initialize a simple Dummy Neural Network (Linear Layer)"""
    w_key, b_key = jax.random.split(key)
    # Simulating a massive matrix multiplication (The "Heavy Lifting" of LLMs)
    return {
        'w': jax.random.normal(w_key, (dim, dim)) * 0.02,
        'b': jax.random.normal(b_key, (dim,))
    }

def loss_fn(params, batch_data):
    """
    Self-Supervised Loss: Try to reconstruct the input (Autoencoder logic).
    This forces the GPU to actually crunch numbers (MatMul).
    """
    # Forward Pass: Wx + b
    prediction = jnp.dot(batch_data, params['w']) + params['b']
    # Loss: Mean Squared Error (Input vs Output)
    loss = jnp.mean((prediction - batch_data) ** 2)
    return loss

# JIT Compile the Update Step (The "Backward Pass")
# This creates the optimized XLA kernel that runs on the GPU
@jit
def update_step(params, batch_data):
    # Calculate Gradients (Automatic Differentiation)
    grads = grad(loss_fn)(params, batch_data)
    
    # Optimizer Step (SGD) - Zone 4: Writeback
    # In a real run, we'd apply these to the weights.
    # We return the loss to track progress.
    loss_val = loss_fn(params, batch_data)
    
    # Update weights (Simple SGD for demonstration)
    new_w = params['w'] - LEARNING_RATE * grads['w']
    new_b = params['b'] - LEARNING_RATE * grads['b']
    
    return {'w': new_w, 'b': new_b}, loss_val

# --- 3. MAIN TRAINING LOOP ---
def run_training():
    print(f"📂 Mapping Database: {DB_PATH}")
    try:
        # Load the Data Map (Zero-Copy)
        raw_vectors = np.memmap(DB_PATH, dtype='float32', mode='r')
        # We reshape it to match our "Model Dimension" to simulate training tokens
        # Note: We truncate slightly to fit exact dimensions
        num_samples = raw_vectors.shape[0] // MODEL_DIM
        dataset = raw_vectors[:num_samples*MODEL_DIM].reshape((num_samples, MODEL_DIM))
        print(f"✅ Linked to {num_samples:,} training samples (Virtual Memory).")
    except FileNotFoundError:
        print("❌ Data file not found. Run merge_shards.py first.")
        return

    # Initialize Model
    print("🧠 Initializing Model Parameters on GPU...")
    key = jax.random.PRNGKey(42)
    params = init_params(key, MODEL_DIM)
    
    # Initialize Ring Buffer
    ring = RingBuffer(capacity=4)
    
    print("\n⚡ STARTING VELOCITY TRAINING LOOP ⚡")
    print("="*60)
    print(f"{'Batch':<10} | {'Loss':<15} | {'Throughput (Tokens/s)':<25} | {'RAM Usage':<15}")
    print("-" * 75)

    start_time = time.time()
    processed_tokens = 0
    
    # The Loop
    try:
        # We iterate through the dataset in batches
        for i in range(0, len(dataset), BATCH_SIZE):
            loop_start = time.time()
            
            # --- STAGE 1: INGEST (CPU) ---
            # Grab a chunk from the disk/memmap
            batch_cpu = dataset[i : i + BATCH_SIZE]
            
            # If batch is too small (end of file), skip
            if batch_cpu.shape[0] != BATCH_SIZE:
                continue
                
            # --- STAGE 2: PIPELINE PUSH ---
            ring.push(batch_cpu)
            
            # --- STAGE 3: GPU COMPUTE (JAX) ---
            # Pull from Ring Buffer
            batch_gpu = ring.get_batch()
            if batch_gpu is None:
                continue
                
            # Move to Device (Pinned Transfer)
            batch_jax = jax.device_put(batch_gpu)
            
            # Perform Training Step (Forward + Backward + Update)
            params, loss = update_step(params, batch_jax)
            
            # Make sure JAX finishes async execution for timing accuracy
            loss.block_until_ready()
            
            # --- METRICS ---
            tokens_in_batch = BATCH_SIZE * MODEL_DIM
            processed_tokens += tokens_in_batch
            loop_time = time.time() - loop_start
            tps = tokens_in_batch / loop_time  # Tokens per second
            
            if i % (BATCH_SIZE * 10) == 0:  # Print every 10 batches
                ram_gb = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
                print(f"{i//BATCH_SIZE:<10} | {loss:.6f}        | {tps:,.0f}                   | {ram_gb:.2f} GB")

    except KeyboardInterrupt:
        print("\n🛑 Training Paused by User.")

    total_time = time.time() - start_time
    print("="*60)
    print(f"✅ Training Complete.")
    print(f"⏱️  Total Time: {total_time:.2f}s")
    print(f"📊 Total Throughput: {processed_tokens / total_time:,.0f} tokens/second")

if __name__ == "__main__":
    run_training()