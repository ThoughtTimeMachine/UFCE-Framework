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
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm

# --- LOAD CONFIGURATION ---
CONFIG_FILE = "velocity_config.json"

def load_config():
    # Fallback if config doesn't exist yet
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
BATCH_SIZE = cfg["batch_size"]
MAX_TOKENS = cfg["max_tokens"]
EMBEDDING_DIM = cfg["embedding_dim"]

# --- SETUP MODEL ---
print("Loading Model & Tokenizer...")
model = SentenceTransformer('all-MiniLM-L6-v2')
model.max_seq_length = MAX_TOKENS
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def stream_chunks(filename, max_tokens=256):
    """Reads a single shard and yields chunks via proper tokenization."""
    buffer_ids = []
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Encode without special tokens first
            line_ids = tokenizer.encode(line, add_special_tokens=False)
            buffer_ids.extend(line_ids)
            
            # Yield full chunks of exactly MAX_TOKENS size
            while len(buffer_ids) >= max_tokens:
                chunk = tokenizer.decode(buffer_ids[:max_tokens])
                yield chunk
                buffer_ids = buffer_ids[max_tokens:]
        
        # Yield the remainder
        if buffer_ids:
            yield tokenizer.decode(buffer_ids)

def process_single_shard(shard_path):
    """
    Processes one text file -> .vec.npy (Vectors) + .meta (Text)
    Returns: True if processed, False if skipped
    """
    # FIX: Strip extension so we don't get .txt.vec.npy
    base_name = os.path.splitext(os.path.basename(shard_path))[0]
    
    # Define Output Filenames (Must match merge_shards expectations)
    output_vec = os.path.join(OUTPUT_DIR, f"{base_name}.vec.npy")
    output_meta = os.path.join(OUTPUT_DIR, f"{base_name}.meta")

    # --- RESUME LOGIC ---
    if os.path.exists(output_vec) and os.path.exists(output_meta):
        print(f"⏩ Skipping {base_name} (Already processed)")
        return False

    print(f"\n⚡ Processing: {base_name}...")

    # --- PASS 1: Count & Create Metadata ---
    # We write metadata first to count how many chunks we have
    temp_meta = output_meta + ".tmp"
    chunks_text = []
    
    chunk_stream = stream_chunks(shard_path, MAX_TOKENS)
    
    with open(temp_meta, "w", encoding="utf-8") as f:
        for chunk in tqdm(chunk_stream, desc="Pass 1 (Tokenizing)", unit=" chunks"):
            clean_chunk = chunk.replace("\n", " ")
            f.write(clean_chunk + "\n")
            chunks_text.append(clean_chunk)

    num_chunks = len(chunks_text)
    
    if num_chunks == 0:
        print(f"⚠️  Warning: {base_name} resulted in 0 chunks.")
        if os.path.exists(temp_meta): os.remove(temp_meta)
        return False

    # --- PASS 2: Embedding ---
    # We use model.encode directly on the list of strings for efficiency
    print(f"   Embedding {num_chunks} chunks...")
    
    embeddings = model.encode(
        chunks_text, 
        batch_size=BATCH_SIZE, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        normalize_embeddings=True # Good for cosine similarity
    )

    # --- SAVE ---
    # Save as standard .npy so merge_shards can read it easily
    np.save(output_vec, embeddings.astype('float32'))
    
    # Rename temp metadata to final
    if os.path.exists(output_meta): os.remove(output_meta)
    os.rename(temp_meta, output_meta)
    
    return True

from multiprocessing import Pool, cpu_count

def run_ingestion_pipeline(parallel=True):
    """Main entry point — supports both parallel and single-threaded modes."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    shard_files = sorted(glob.glob(os.path.join(SHARDS_DIR, "*.txt")))
    
    if not shard_files:
        print(f"❌ No shards found in '{SHARDS_DIR}'. Check your config or folder.")
        return

    print(f"Found {len(shard_files)} shards.")

    if parallel and len(shard_files) > 1:
        print(f"🚀 Running in parallel on {cpu_count()} cores...")
        with Pool(cpu_count()) as pool:
            # Use imap for ordered progress bar
            list(tqdm(pool.imap(process_single_shard, shard_files), 
                      total=len(shard_files), desc="Overall Progress", unit="shard"))
    else:
        print("Running single-threaded...")
        for shard in tqdm(shard_files, desc="Processing Shards", unit="shard"):
            process_single_shard(shard)

    print("-" * 50)
    print("✅ Ingestion complete!")

if __name__ == "__main__":
    run_ingestion_pipeline(parallel=True)  # Default to parallel