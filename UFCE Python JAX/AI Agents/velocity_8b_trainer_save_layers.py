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
from jax import random, jit, vjp
from flax import linen as nn
import optax
from collections import deque
from functools import partial
from safetensors.torch import load_file
# --- ADDED: Import for saving weights ---
from safetensors.numpy import save_file

# --- VELOCITY CONFIG ---
# Critical: Disable preallocation to allow VRAM to clear between layers
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

# --- MODEL CONFIG (FOUNDER MODE SWITCH) ---
# To train 70B, change: NUM_LAYERS=80, DIM=8192, INTERMEDIATE=28672, HEADS=64, KV_HEADS=8
MODEL_SCALE = "8B"

if MODEL_SCALE == "8B":
    NUM_LAYERS = 32
    DIM = 4096
    HEADS = 32
    KV_HEADS = 8
    INTERMEDIATE = 14336
    BATCH_SIZE = 1  # Keep small for initial full-loop verification
    SEQ_LEN = 128
elif MODEL_SCALE == "70B":
    NUM_LAYERS = 80
    DIM = 8192
    HEADS = 64
    KV_HEADS = 8
    INTERMEDIATE = 28672
    BATCH_SIZE = 1
    SEQ_LEN = 128

LEARNING_RATE = 2e-5
WEIGHTS_DIR = "./llama3_weights"

# Global Optimizer Definition
TX = optax.adamw(LEARNING_RATE)

# --- 1. ARCHITECTURE (GQA) ---
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

# --- 2. FOUNDER MODE: SHARD LOADER ---
def load_layer_weights(layer_idx):
    index_path = os.path.join(WEIGHTS_DIR, "model.safetensors.index.json")
    if not os.path.exists(index_path): return None 

    with open(index_path, 'r') as f:
        weight_map = json.load(f)["weight_map"]
    
    prefix = f"model.layers.{layer_idx}."
    target_shards = set()
    for key, shard in weight_map.items():
        if key.startswith(prefix): target_shards.add(shard)
    
    weights = {}
    for shard in target_shards:
        w = load_file(os.path.join(WEIGHTS_DIR, shard))
        for k in w.keys():
            if k.startswith(prefix): weights[k] = w[k]
            
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

# --- ADDED: SAVE FUNCTION (PERSISTENCE) ---
def save_layer_weights(layer_idx, params):
    """Saves the trained JAX parameters back to SafeTensors format."""
    print(f"   💾 Saving Layer {layer_idx} to disk...")
    flat_weights = {}
    prefix = f"model.layers.{layer_idx}."
    
    # Helper: JAX Array -> Numpy
    def to_np(x): return np.array(x)

    # Flatten Attention
    flat_weights[f"{prefix}self_attn.q_proj.weight"] = to_np(params['self_attn']['q_proj']['kernel'].T)
    flat_weights[f"{prefix}self_attn.k_proj.weight"] = to_np(params['self_attn']['k_proj']['kernel'].T)
    flat_weights[f"{prefix}self_attn.v_proj.weight"] = to_np(params['self_attn']['v_proj']['kernel'].T)
    flat_weights[f"{prefix}self_attn.o_proj.weight"] = to_np(params['self_attn']['o_proj']['kernel'].T)
    
    # Flatten MLP
    flat_weights[f"{prefix}mlp.gate_proj.weight"] = to_np(params['mlp']['gate_proj']['kernel'].T)
    flat_weights[f"{prefix}mlp.up_proj.weight"]   = to_np(params['mlp']['up_proj']['kernel'].T)
    flat_weights[f"{prefix}mlp.down_proj.weight"] = to_np(params['mlp']['down_proj']['kernel'].T)
    
    # Flatten Norms
    flat_weights[f"{prefix}input_layernorm.weight"] = to_np(params['input_layernorm']['scale'])
    flat_weights[f"{prefix}post_attention_layernorm.weight"] = to_np(params['post_attention_layernorm']['scale'])
    
    # Save to a new file (avoiding immediate overwrite of source for safety)
    # In a real run, you might overwrite or use a checkpoint folder.
    save_file(flat_weights, os.path.join(WEIGHTS_DIR, f"layer_{layer_idx}_tuned.safetensors"))

# --- 3. THE VELOCITY ENGINES ---

# A. Forward Engine (Returns Output)
@partial(jit, static_argnums=(2,))
def forward_step(params, inputs, model_def):
    return model_def.apply({'params': params}, inputs)

# B. Backward Engine (The Magic: Returns Gradients for Weights AND Inputs)
@partial(jit, static_argnums=(4,))
def backward_step(params, inputs, grad_from_above, opt_state, model_def):
    def fwd(p, x):
        return model_def.apply({'params': p}, x)
    
    # VJP calculates vector-jacobian product (chain rule)
    # We get gradients w.r.t parameters AND inputs
    (output, vjp_fn) = vjp(fwd, params, inputs)
    grad_params, grad_inputs = vjp_fn(grad_from_above)
    
    # Optimizer Update
    updates, new_opt_state = TX.update(grad_params, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, grad_inputs

# --- 4. THE V-CYCLE MAIN LOOP ---
def run_training_loop():
    print(f"🚀 VELOCITY 70B-CAPABLE TRAINER")
    print(f"🎯 Target: {MODEL_SCALE} Model ({NUM_LAYERS} Layers)")
    print(f"🧠 System RAM Strategy: Storing {NUM_LAYERS} Optimizer States + Activations")
    print("-" * 60)
    
    # 1. INITIALIZATION
    print("🎬 Initializing Optimizer States in RAM (Zero-Cost)...")
    # We create ONE empty optimizer state and replicate it to save time/memory for the demo
    # In production, each layer has unique state.
    model = LlamaDecoderLayer(DIM, INTERMEDIATE, HEADS, KV_HEADS)
    dummy_params = model.init(random.PRNGKey(0), jnp.ones((BATCH_SIZE, SEQ_LEN, DIM)))['params']
    base_opt_state = TX.init(dummy_params)
    
    # The "Storage Rack" in System RAM
    optimizer_states = [base_opt_state for _ in range(NUM_LAYERS)]
    activations_store = [None] * (NUM_LAYERS + 1) # +1 for initial input
    
    # Random Input (The "Prompt")
    activations_store[0] = jax.random.normal(random.PRNGKey(0), (BATCH_SIZE, SEQ_LEN, DIM))
    
    total_start = time.time()
    
    # --- PHASE 1: FORWARD PASS (DOWNSTREAM) ---
    print("\n⬇️  FORWARD PASS (Streaming Layers 0 -> 31)")
    for i in range(NUM_LAYERS):
        t0 = time.time()
        
        # Load
        layer_params = load_layer_weights(i)
        if layer_params is None: break
        
        # Compute & Save Output
        # Input comes from the store
        current_input = jax.device_put(activations_store[i]) 
        
        output = forward_step(layer_params, current_input, model)
        output.block_until_ready()
        
        # Offload Output to RAM (Next Layer's Input)
        activations_store[i+1] = jax.device_get(output)
        
        # Cleanup
        del layer_params, current_input, output
        
        print(f"   Layer {i} FWD | Time: {time.time()-t0:.2f}s")

   # --- REAL LOSS CALCULATION ---
    print(f"\n⚡ CALCULATING TRUE CROSS-ENTROPY LOSS...")
    
    # 1. Get Final Embeddings (Output of Layer 31)
    final_embeddings = jax.device_put(activations_store[NUM_LAYERS])
    
    # 2. Apply "LM Head" (The Final Prediction Layer)
    # Note: In a real Llama model, there is a final Norm + Dense layer here.
    # For this trainer, we will project embeddings back to vocabulary size.
    # We assume 'w_head' exists or we simulate it for the gradient test.
    vocab_size = 128256 # Llama 3 Vocab
    
    # Create a dummy LM Head for the demo (Real training loads this from safetensors)
    lm_head_kernel = jax.random.normal(random.PRNGKey(0), (DIM, vocab_size)) * (DIM**-0.5)
    logits = final_embeddings @ lm_head_kernel
    
    # 3. Create Targets (Shifted Input)
    # In Causal LM, if input is "A B C", target is "B C D"
    # Here we just generate random targets for the mechanics test
    targets = jax.random.randint(random.PRNGKey(1), (BATCH_SIZE, SEQ_LEN), 0, vocab_size)
    
    # 4. Calculate Cross Entropy Loss & Gradient
    def loss_function(logits, targets):
        one_hot = jax.nn.one_hot(targets, vocab_size)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        return loss

    # We want the gradient w.r.t 'final_embeddings' to pass back into Layer 31
    loss_val, grad_fn = vjp(lambda x: loss_function(x @ lm_head_kernel, targets), final_embeddings)
    
    # This is the "Hot Potato" we pass to Layer 31
    final_grad = grad_fn(1.0)[0] 
    
    print(f"   ✅ Loss: {loss_val:.4f} | Initial Grad Norm: {jnp.linalg.norm(final_grad):.2f}")
    
    current_grad = final_grad
    
    # --- PHASE 2: BACKWARD PASS (UPSTREAM) ---
    print("\n⬆️  BACKWARD PASS (Streaming Layers 31 -> 0)")
    for i in reversed(range(NUM_LAYERS)):
        t0 = time.time()
        
        # Load Weights & Opt State
        layer_params = load_layer_weights(i)
        opt_state = optimizer_states[i]
        
        # Load Input (Saved from Forward Pass)
        layer_input = jax.device_put(activations_store[i])
        
        # Compute Gradients & Update
        new_params, new_opt_state, input_grad = backward_step(
            layer_params, layer_input, current_grad, opt_state, model
        )
        input_grad.block_until_ready()
        
        # Pass the "Hot Potato" (Gradient) down
        current_grad = input_grad
        
        # Save updated optimizer state back to RAM
        optimizer_states[i] = jax.device_get(new_opt_state)
        
        # --- ADDED: PERSISTENCE CALL ---
        # Saves the trained weights to disk so they aren't lost
        save_layer_weights(i, new_params)
        # -------------------------------
        
        # Cleanup
        del layer_params, layer_input, new_params, new_opt_state
        
        print(f"   Layer {i} BWD | Time: {time.time()-t0:.2f}s | Grad Norm: {jnp.linalg.norm(current_grad):.2f}")

    print("-" * 60)
    print(f"✅ Training Step Complete.")
    print(f"⏱️  Total Time: {time.time() - total_start:.2f}s")
    print(f"💡 You have successfully backpropagated through a {MODEL_SCALE} model.")

if __name__ == "__main__":
    run_training_loop()