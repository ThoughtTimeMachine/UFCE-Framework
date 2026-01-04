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

# --- HARDWARE CONFIG (The Speed Cheat) ---
# This must happen BEFORE importing JAX to lock the RAM (Pinned Memory).
# It prevents JAX from gobbling all VRAM instantly, allowing for streaming.

import os
import glob
import numpy as np
import json
from tqdm import tqdm

# --- LOAD CONFIGURATION ---
CONFIG_FILE = "velocity_config.json"

def load_config():
    with open(CONFIG_FILE, 'r') as f:
        data = json.load(f)
    dataset_key = data["active_dataset"]
    print(f"🔧 Loaded Config: {dataset_key} ({data['datasets'][dataset_key]['description']})")
    return data["datasets"][dataset_key]

cfg = load_config()

# --- DYNAMIC CONFIG ---
KB_DIR = cfg["vectors_output_dir"]
OUTPUT_DAT = os.path.join(KB_DIR, cfg["final_dat_name"])
OUTPUT_META = os.path.join(KB_DIR, cfg["final_meta_name"])
EMBEDDING_DIM = cfg["embedding_dim"]

def merge_database():
    print(f"🚀 UFCE Database Merger")
    print(f"📂 Input Source: {KB_DIR}")
    print(f"💾 Output Target: {OUTPUT_DAT}")

    # 1. Find all processed vector shards
    vec_files = sorted(glob.glob(os.path.join(KB_DIR, "*.vec.npy")))
    meta_files = sorted(glob.glob(os.path.join(KB_DIR, "*.meta")))

    if len(vec_files) == 0:
        print("❌ No vector shards found! Run ingestion_pipeline.py first.")
        return

    # 2. Calculate Total Size
    total_vectors = 0
    print("Scanning shards...")
    for f in vec_files:
        # Quick header read to get shape without loading data
        shape = np.load(f, mmap_mode='r').shape
        total_vectors += shape[0]

    print(f"∑ Total Vectors: {total_vectors:,}")
    
    # 3. Create/Overwrite the massive .dat file (Memmap)
    # Mode 'w+' creates a new file or overwrites existing
    print("Allocating disk space...")
    fp = np.memmap(OUTPUT_DAT, dtype='float32', mode='w+', shape=(total_vectors, EMBEDDING_DIM))
    
    # 4. Stream chunks into the big file
    current_idx = 0
    print("Streaming vectors...")
    for f in tqdm(vec_files):
        data = np.load(f)
        n = data.shape[0]
        fp[current_idx : current_idx + n] = data
        current_idx += n
        
    fp.flush() # Ensure write to disk
    print("✅ Vector database merged.")

    # 5. Merge Metadata (Text)
    print("Merging metadata...")
    with open(OUTPUT_META, 'w', encoding='utf-8') as outfile:
        for f in tqdm(meta_files):
            with open(f, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                
    print(f"✅ Full Knowledge Base Created at: {OUTPUT_DAT}")

if __name__ == "__main__":
    merge_database()