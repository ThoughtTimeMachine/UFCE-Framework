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
import glob
import numpy as np
import json
from llama_cpp import Llama 
from tqdm import tqdm

# --- LOAD CONFIGURATION ---
CONFIG_FILE = "velocity_config.json"

def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ Error: {CONFIG_FILE} not found. Please create it first.")
        exit()
        
    with open(CONFIG_FILE, 'r') as f:
        data = json.load(f)
    dataset_key = data["active_dataset"]
    print(f"🔧 Loaded Config: {dataset_key} ({data['datasets'][dataset_key]['description']})")
    return data["datasets"][dataset_key]

cfg = load_config()

# --- DYNAMIC CONFIG ---
SHARDS_DIR = cfg["shards_input_dir"]
OUTPUT_DIR = cfg["vectors_output_dir"]
# Note: GGUF processes sequentially, so 'batch_size' is less relevant here 
# but we keep the config variable for reference.
MAX_TOKENS = cfg["max_tokens"] 
EMBEDDING_DIM = cfg["embedding_dim"]

# --- SETUP MODEL (GGUF) ---
# Ensure this path points to your actual downloaded GGUF file
MODEL_PATH = "./binaries/nomic-embed-text-v1.5.Q5_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model not found at {MODEL_PATH}")
    print("Please download 'nomic-embed-text-v1.5.Q5_K_M.gguf' to the binaries folder.")
    exit()

print("🚀 Loading GGUF Model & Tokenizer...")
# n_ctx=8192 is critical for Nomic v1.5
model = Llama(
    model_path=MODEL_PATH, 
    embedding=True, 
    n_ctx=8192, 
    verbose=False,
    n_gpu_layers=-1 # Use all GPU layers
)

def stream_chunks(filename, max_tokens=256):
    """
    Reads a single shard and yields chunks via GGUF tokenization.
    """
    buffer_ids = []
    
    # Nomic needs 'search_document: ' prefix, which takes up tokens.
    # We reserve space for it.
    prefix_tokens = model.tokenize(b"search_document: ", add_bos=False)
    prefix_len = len(prefix_tokens)
    effective_limit = max_tokens - prefix_len

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Llama.cpp tokenize expects bytes
            line_ids = model.tokenize(line.encode("utf-8"), add_bos=False)
            buffer_ids.extend(line_ids)
            
            # Yield full chunks
            while len(buffer_ids) >= effective_limit:
                chunk_ids = buffer_ids[:effective_limit]
                # Decode back to string
                chunk_text = model.detokenize(chunk_ids).decode("utf-8", errors="ignore")
                yield chunk_text
                buffer_ids = buffer_ids[effective_limit:]
        
        # Yield remainder
        if buffer_ids:
            chunk_text = model.detokenize(buffer_ids).decode("utf-8", errors="ignore")
            yield chunk_text

def process_single_shard(shard_path):
    """
    Processes one text file -> .vec.npy (Vectors) + .meta (Text)
    """
    base_name = os.path.splitext(os.path.basename(shard_path))[0]
    
    output_vec = os.path.join(OUTPUT_DIR, f"{base_name}.vec.npy")
    output_meta = os.path.join(OUTPUT_DIR, f"{base_name}.meta")

    # --- RESUME LOGIC ---
    if os.path.exists(output_vec) and os.path.exists(output_meta):
        print(f"⏩ Skipping {base_name} (Already processed)")
        return False

    print(f"\n⚡ Processing: {base_name}...")

    # --- PASS 1: Tokenize & Chunk ---
    chunks_text = []
    chunk_stream = stream_chunks(shard_path, MAX_TOKENS)
    
    # We collect all chunks first to ensure clean writing
    for chunk in chunk_stream:
        # Collapse newlines for cleaner metadata
        clean = chunk.replace("\n", " ")
        if clean.strip(): # Skip empty chunks
            chunks_text.append(clean)

    num_chunks = len(chunks_text)
    if num_chunks == 0:
        print(f"⚠️  Warning: {base_name} resulted in 0 chunks.")
        return False

    # --- PASS 2: Embedding Loop ---
    print(f"   Embedding {num_chunks} chunks...")
    embeddings_list = []

    # GGUF models are not optimized for large batching like PyTorch.
    # We loop through them. It is still very fast on GPU.
    for text in tqdm(chunks_text, desc="Embedding", unit="chunk"):
        # IMPORTANT: Add the Nomic prefix!
        # This tells the model "This is a document to be stored"
        prefixed_text = "search_document: " + text
        
        # Generate embedding
        # Returns a list of floats
        vector = model.create_embedding(prefixed_text)['data'][0]['embedding']
        embeddings_list.append(vector)

    # Convert to Numpy
    embeddings = np.array(embeddings_list, dtype='float32')

    # Normalize (L2) - Critical for Cosine Similarity
    # (Nomic GGUF usually returns normalized, but we double-check)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # --- SAVE ---
    np.save(output_vec, embeddings)
    
    with open(output_meta, "w", encoding="utf-8") as f:
        for chunk in chunks_text:
            f.write(chunk + "\n")
    
    return True

def run_ingestion_pipeline():
    """Main entry point."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    shard_files = sorted(glob.glob(os.path.join(SHARDS_DIR, "*.txt")))
    
    if not shard_files:
        print(f"❌ No shards found in '{SHARDS_DIR}'. Check your config.")
        return

    print(f"Found {len(shard_files)} shards.")
    print("🚀 Running single-threaded (GGUF Safe Mode)...")

    # NOTE: We removed Multiprocessing.
    # Passing GGUF C++ pointers across processes is unstable.
    # Sequential processing on GPU is usually plenty fast.
    for shard in tqdm(shard_files, desc="Processing Shards", unit="shard"):
        process_single_shard(shard)

    print("-" * 50)
    print("✅ Ingestion complete!")

if __name__ == "__main__":
    run_ingestion_pipeline()
