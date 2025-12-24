
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
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm

# --- CONFIG ---
SHARDS_DIR = "wiki_shards"         # Where the text files are
OUTPUT_DIR = "knowledge_base"      # Where to save the vectors
BATCH_SIZE = 64                    
MAX_TOKENS = 256                   
EMBEDDING_DIM = 384                

# Load Model & Tokenizer
print("Loading Model & Tokenizer...")
model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def stream_chunks(filename, max_tokens=256):
    """Reads a single shard and yields chunks."""
    buffer_ids = []
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_ids = tokenizer.encode(line, add_special_tokens=False)
            buffer_ids.extend(line_ids)
            
            while len(buffer_ids) >= max_tokens:
                chunk = tokenizer.decode(buffer_ids[:max_tokens])
                yield chunk
                buffer_ids = buffer_ids[max_tokens:]
        
        if buffer_ids:
            yield tokenizer.decode(buffer_ids)

def process_single_shard(shard_path):
    """
    Processes one text file -> one .dat file
    Returns: True if processed, False if skipped (already exists)
    """
    base_name = os.path.splitext(os.path.basename(shard_path))[0]
    output_bin = os.path.join(OUTPUT_DIR, f"{base_name}.dat")
    output_meta = os.path.join(OUTPUT_DIR, f"{base_name}_meta.txt")

    # --- RESUME LOGIC: Check if already done ---
    if os.path.exists(output_bin) and os.path.exists(output_meta):
        if os.path.getsize(output_bin) > 0:
            print(f"⏩ Skipping {base_name} (Already processed)")
            return False

    print(f"\n⚡ Processing: {base_name}...")

    # --- PASS 1: Count & Metadata (Now with Progress Bar!) ---
    temp_meta = output_meta + ".tmp"
    num_chunks = 0
    
    # We wrap the generator in tqdm. We don't know the 'total' yet, so it shows iterations/sec.
    with open(temp_meta, "w", encoding="utf-8") as meta_f:
        chunk_stream = stream_chunks(shard_path, MAX_TOKENS)
        for chunk in tqdm(chunk_stream, desc="Pass 1 (Counting)", unit=" chunks"):
            clean_chunk = chunk.replace("\n", " ")
            meta_f.write(clean_chunk + "\n")
            num_chunks += 1
            
    if num_chunks == 0:
        print(f"⚠️  Warning: {base_name} was empty.")
        if os.path.exists(temp_meta):
            os.remove(temp_meta)
        return False

    # --- PASS 2: Vectorization ---
    fp = np.memmap(output_bin, dtype='float32', mode='w+', shape=(num_chunks, EMBEDDING_DIM))
    
    chunk_generator = stream_chunks(shard_path, MAX_TOKENS)
    batch_buffer = []
    write_idx = 0
    
    # Pass 2 already had a bar, but we ensure it looks good
    for chunk in tqdm(chunk_generator, total=num_chunks, desc="Pass 2 (Embedding)", leave=False):
        batch_buffer.append(chunk)
        
        if len(batch_buffer) >= BATCH_SIZE:
            embeddings = model.encode(batch_buffer, convert_to_numpy=True)
            fp[write_idx : write_idx + len(embeddings)] = embeddings
            write_idx += len(embeddings)
            batch_buffer = []

    # Final Batch
    if batch_buffer:
        embeddings = model.encode(batch_buffer, convert_to_numpy=True)
        fp[write_idx : write_idx + len(embeddings)] = embeddings
    
    fp.flush()
    
    # --- FINALIZE ---
    os.rename(temp_meta, output_meta)
    return True

def run_ingestion_pipeline():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Find all shards
    shard_files = sorted(glob.glob(os.path.join(SHARDS_DIR, "*.txt")))
    
    if not shard_files:
        print(f"❌ No shards found in {SHARDS_DIR}. Did you run the dump processor?")
        return

    print(f"Found {len(shard_files)} shards. Starting Pipeline...")
    print("This script can be stopped (Ctrl+C) and restarted safely.\n")

    total_processed = 0
    for shard in shard_files:
        try:
            was_processed = process_single_shard(shard)
            if was_processed:
                total_processed += 1
        except Exception as e:
            print(f"\n❌ Error processing {shard}: {e}")
            # We continue to the next shard instead of crashing the whole pipeline
            continue

    print("-" * 50)
    print(f"✅ Job Complete. Processed {total_processed} new shards.")

if __name__ == "__main__":
    run_ingestion_pipeline()