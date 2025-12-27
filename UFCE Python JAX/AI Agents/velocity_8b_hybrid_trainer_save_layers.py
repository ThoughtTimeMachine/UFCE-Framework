import os
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vjp
from flax import linen as nn
from functools import partial
import optax
from safetensors.torch import load_file
from safetensors.numpy import save_file

# --- VELOCITY CONFIG ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

# --- TRAINING CONFIG ---
MODEL_SCALE = "8B"
TOTAL_STEPS = 10       # Run for more steps now that it's fast
SAVE_EVERY_STEPS = 5   # Only pay the "Disk Tax" every 5 steps
CHECKPOINT_ROOT = "./checkpoints"

if MODEL_SCALE == "8B":
    NUM_LAYERS = 32
    DIM = 4096
    HEADS = 32
    KV_HEADS = 8
    INTERMEDIATE = 14336
    BATCH_SIZE = 1
    SEQ_LEN = 128
    WEIGHTS_DIR = "./llama3_weights"

LEARNING_RATE = 2e-5
TX = optax.adamw(LEARNING_RATE)

# --- GLOBAL RAM CACHE (The Speed Hack) ---
# Holds the live weights in System RAM so we don't read/write disk every step.
MODEL_RAM_CACHE = [None] * NUM_LAYERS

# --- 1. REAL DATA LOADER ---
def get_real_batch(step):
    key = random.PRNGKey(step)
    fake_tokens = random.randint(key, (BATCH_SIZE, SEQ_LEN), 0, 128256)
    return fake_tokens

# --- 2. ARCHITECTURE ---
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

# --- 3. HYBRID LOADER (RAM FIRST, DISK FALLBACK) ---
def load_layer_weights(layer_idx):
    # 1. FAST PATH: Check RAM Cache
    if MODEL_RAM_CACHE[layer_idx] is not None:
        return MODEL_RAM_CACHE[layer_idx]

    # 2. SLOW PATH: Load from Disk (First run only)
    # print(f"   💿 Loading Layer {layer_idx} from Disk...")
    
    index_path = os.path.join(WEIGHTS_DIR, "model.safetensors.index.json")
    if not os.path.exists(index_path): return None 

    with open(index_path, 'r') as f: weight_map = json.load(f)["weight_map"]
    
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
    
    # Create the JAX Dict
    layer_params = {
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
    
    # Update Cache so we never read disk again for this layer
    MODEL_RAM_CACHE[layer_idx] = layer_params
    return layer_params

def save_checkpoint(layer_idx, params, step):
    """Saves to checkpoints/step_X/layer_Y.safetensors"""
    save_dir = os.path.join(CHECKPOINT_ROOT, f"step_{step}")
    os.makedirs(save_dir, exist_ok=True)
    
    flat_weights = {}
    prefix = f"model.layers.{layer_idx}."
    def to_disk(x): return np.array(x).astype(np.float16) # FP16 Save

    flat_weights[f"{prefix}self_attn.q_proj.weight"] = to_disk(params['self_attn']['q_proj']['kernel'].T)
    flat_weights[f"{prefix}self_attn.k_proj.weight"] = to_disk(params['self_attn']['k_proj']['kernel'].T)
    flat_weights[f"{prefix}self_attn.v_proj.weight"] = to_disk(params['self_attn']['v_proj']['kernel'].T)
    flat_weights[f"{prefix}self_attn.o_proj.weight"] = to_disk(params['self_attn']['o_proj']['kernel'].T)
    
    flat_weights[f"{prefix}mlp.gate_proj.weight"] = to_disk(params['mlp']['gate_proj']['kernel'].T)
    flat_weights[f"{prefix}mlp.up_proj.weight"]   = to_disk(params['mlp']['up_proj']['kernel'].T)
    flat_weights[f"{prefix}mlp.down_proj.weight"] = to_disk(params['mlp']['down_proj']['kernel'].T)
    
    flat_weights[f"{prefix}input_layernorm.weight"] = to_disk(params['input_layernorm']['scale'])
    flat_weights[f"{prefix}post_attention_layernorm.weight"] = to_disk(params['post_attention_layernorm']['scale'])
    
    save_file(flat_weights, os.path.join(save_dir, f"layer_{layer_idx}.safetensors"))

# --- 4. ENGINES ---
@partial(jit, static_argnums=(2,))
def forward_step(params, inputs, model_def):
    return model_def.apply({'params': params}, inputs)

@partial(jit, static_argnums=(4,))
def backward_step(params, inputs, grad_from_above, opt_state, model_def):
    def fwd(p, x): return model_def.apply({'params': p}, x)
    (output, vjp_fn) = vjp(fwd, params, inputs)
    grad_params, grad_inputs = vjp_fn(grad_from_above)
    updates, new_opt_state = TX.update(grad_params, opt_state, params)
    return optax.apply_updates(params, updates), new_opt_state, grad_inputs

# --- 5. MAIN TRAINING LOOP (VERBOSE EDITION) ---
def run_training():
    print(f"🚀 VELOCITY HYBRID TRAINER")
    print(f"🎯 8B Model | RAM Cache: ON | Save Every: {SAVE_EVERY_STEPS} Steps")
    print("-" * 60)
    
    model = LlamaDecoderLayer(DIM, INTERMEDIATE, HEADS, KV_HEADS)
    dummy_params = model.init(random.PRNGKey(0), jnp.ones((BATCH_SIZE, SEQ_LEN, DIM)))['params']
    base_opt_state = TX.init(dummy_params)
    
    optimizer_states = [base_opt_state for _ in range(NUM_LAYERS)]
    activations_store = [None] * (NUM_LAYERS + 1)
    
    # START STEP LOOP
    for step in range(TOTAL_STEPS):
        print(f"\n🚩 STARTING STEP {step+1}/{TOTAL_STEPS}")
        step_start = time.time()
        
        # 1. GET DATA
        activations_store[0] = jax.random.normal(random.PRNGKey(step), (BATCH_SIZE, SEQ_LEN, DIM))

        # 2. FORWARD
        print(f"⬇️  Step {step+1}: Forward Pass")
        for i in range(NUM_LAYERS):
            t0 = time.time()
            # print(f"    Load L{i}...", end="\r") # Debug print
            layer_params = load_layer_weights(i) 
            
            current_input = jax.device_put(activations_store[i])
            output = forward_step(layer_params, current_input, model)
            output.block_until_ready() # Force sync to measure true time
            activations_store[i+1] = jax.device_get(output)
            
            del current_input, output
            # Only print every 4 layers to reduce clutter, or all layers to verify life
            print(f"   Layer {i} FWD | {time.time()-t0:.2f}s")

        # 3. LOSS
        print(f"⚡ Step {step+1}: Calculating Loss")
        final_embeddings = jax.device_put(activations_store[NUM_LAYERS])
        loss_grad = jax.random.normal(random.PRNGKey(step), final_embeddings.shape)
        current_grad = loss_grad

        # 4. BACKWARD + UPDATE
        print(f"⬆️  Step {step+1}: Backward & Update")
        for i in reversed(range(NUM_LAYERS)):
            t0 = time.time()
            
            layer_params = load_layer_weights(i)
            opt_state = optimizer_states[i]
            layer_input = jax.device_put(activations_store[i])
            
            # FIRST RUN JIT WARNING
            if step == 0 and i == NUM_LAYERS - 1:
                print("   (Compiling Backward Kernel... this takes ~30s)...")
            
            new_params, new_opt_state, input_grad = backward_step(
                layer_params, layer_input, current_grad, opt_state, model
            )
            input_grad.block_until_ready()
            current_grad = input_grad
            optimizer_states[i] = jax.device_get(new_opt_state)
            
            # Update RAM Cache (Instant)
            MODEL_RAM_CACHE[i] = jax.device_get(new_params)
            
            # Checkpoint (Slow)
            saved_msg = ""
            if (step + 1) % SAVE_EVERY_STEPS == 0:
                save_checkpoint(i, MODEL_RAM_CACHE[i], step+1)
                saved_msg = "| Saved 💾"
            
            del layer_params, layer_input, new_params, new_opt_state
            
            print(f"   Layer {i} BWD | {time.time()-t0:.2f}s {saved_msg}")

        step_time = time.time()-step_start
        print(f"✅ Step {step+1} Complete: {step_time:.2f}s")
if __name__ == "__main__":
    run_training()