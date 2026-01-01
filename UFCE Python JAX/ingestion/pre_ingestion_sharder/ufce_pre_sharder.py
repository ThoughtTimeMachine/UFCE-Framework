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
import math
import json
import multiprocessing  # For core detection
from tqdm import tqdm

# --- LOAD CONFIGURATION ---
CONFIG_FILE = "velocity_config.json"

def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ Error: {CONFIG_FILE} not found.")
        exit()
    
    with open(CONFIG_FILE, 'r') as f:
        data = json.load(f)
    
    active_key = data["active_dataset"]
    print(f"🔧 Loaded Config: {active_key} ({data['datasets'][active_key]['description']})")
    return data["datasets"][active_key]

cfg = load_config()

# --- DYNAMIC CONFIG ---
SHARDS_DIR = cfg["shards_input_dir"]
FULL_INPUT_FILE = os.path.join(SHARDS_DIR, "full.txt")  # From preprocessor
TARGET_SHARD_SIZE_MB = 500  # Fallback size-based target

def smart_shard_document():
    if not os.path.exists(FULL_INPUT_FILE):
        print(f"❌ Error: Full input file not found: {FULL_INPUT_FILE}")
        print("   Run your dataset preprocessor first.")
        return

    file_size_mb = os.path.getsize(FULL_INPUT_FILE) / (1024**2)
    print(f"📂 Input: {FULL_INPUT_FILE} ({file_size_mb:.1f} MB)")

    # === SMART CORE-BASED SHARDING (Primary) ===
    try:
        available_cores = multiprocessing.cpu_count()
        # Use 80% of cores to leave room for system/other processes
        target_shards = max(1, int(available_cores * 0.8))
        print(f"🖥️  Detected {available_cores} CPU cores → Targeting {target_shards} shards (auto)")
    except Exception:
        print("⚠️  Could not detect CPU cores — falling back to size-based sharding")
        target_shards = max(1, math.ceil(file_size_mb / TARGET_SHARD_SIZE_MB))
        print(f"🎯 Fallback: Targeting {target_shards} shards (~{TARGET_SHARD_SIZE_MB}MB each)")

    # === SHARDING LOGIC ===
    with open(FULL_INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    if "\n\n" in content:
        paragraphs = content.split("\n\n")
        boundary = "\n\n"
    else:
        paragraphs = content.split("\n")
        boundary = "\n"

    total_chars = len(content)
    chars_per_shard = total_chars // target_shards if target_shards > 1 else total_chars

    current_shard = []
    current_size = 0
    shard_idx = 0

    print("✂️ Sharding...")
    for para in tqdm(paragraphs, desc="Building shards"):
        para_size = len(para.encode('utf-8'))
        
        if current_size + para_size > chars_per_shard * 1.1 and current_shard:
            shard_path = os.path.join(SHARDS_DIR, f"shard_{shard_idx:04d}.txt")
            with open(shard_path, "w", encoding="utf-8") as out_f:
                out_f.write(boundary.join(current_shard))
            
            current_shard = [para]
            current_size = para_size
            shard_idx += 1
        else:
            current_shard.append(para)
            current_size += para_size + len(boundary.encode('utf-8'))

    # Final shard
    if current_shard:
        shard_path = os.path.join(SHARDS_DIR, f"shard_{shard_idx:04d}.txt")
        with open(shard_path, "w", encoding="utf-8") as out_f:
            out_f.write(boundary.join(current_shard))

    print(f"✅ Sharding complete: {shard_idx + 1} shards in {SHARDS_DIR}")
    print("   Now run ufce_ingestion_pipeline_shard.py — parallel processing will engage automatically!")

if __name__ == "__main__":
    smart_shard_document()