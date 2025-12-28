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
import gzip
import requests
import json
from tqdm import tqdm

# --- LOAD CONFIGURATION ---
CONFIG_FILE = "velocity_config.json"

def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ Error: {CONFIG_FILE} not found. Please create it first.")
        exit()

    with open(CONFIG_FILE, 'r') as f:
        data = json.load(f)
    
    # We specifically need the genome dataset config for this script
    # regardless of what 'active_dataset' is set to, though usually they match.
    dataset_key = data["active_dataset"]
    if dataset_key not in data["datasets"]:
        print(f"❌ Error: '{dataset_key}' not in config.")
        exit()
         
    return data["datasets"]["genome"]

cfg = load_config()
prep_cfg = cfg.get("preprocessing", {})

if not prep_cfg:
    print("❌ Error: 'preprocessing' block missing in genome config.")
    exit()

# --- DYNAMIC CONFIG ---
DOWNLOAD_URL = prep_cfg["download_url"]
GENOME_DATA_DIR = prep_cfg["raw_data_dir"]
SHARDS_DIR = cfg["shards_input_dir"] # This is where the output goes so ingestion can find it

INPUT_GZ = os.path.join(GENOME_DATA_DIR, prep_cfg["raw_filename"])
OUTPUT_TXT = os.path.join(SHARDS_DIR, prep_cfg["preprocessed_filename"])

# --- SETUP ---
os.makedirs(GENOME_DATA_DIR, exist_ok=True)
os.makedirs(SHARDS_DIR, exist_ok=True)

def preprocess_genome():
    print(f"🚀 UFCE Genome Preprocessor")
    print(f"📂 Raw Data Dir: {GENOME_DATA_DIR}")
    print(f"📂 Output Shard Dir: {SHARDS_DIR}")

    # --- STEP 1: Download if missing ---
    if not os.path.exists(INPUT_GZ):
        print(f"\n⬇️ Downloading genome from NCBI...")
        print(f"   URL: {DOWNLOAD_URL}")
        
        try:
            response = requests.get(DOWNLOAD_URL, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(INPUT_GZ, "wb") as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc="Download"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            print(f"✅ Downloaded to {INPUT_GZ}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return
    else:
        print(f"✅ Genome already downloaded: {INPUT_GZ}")

    # --- STEP 2: Preprocess FASTA → Clean Text ---
    print("\n🧬 Preprocessing FASTA → Clean Text Shard...")
    
    if os.path.exists(OUTPUT_TXT):
        print(f"⏩ Output file {OUTPUT_TXT} already exists. Skipping.")
        return

    try:
        with gzip.open(INPUT_GZ, "rt") as f_in, open(OUTPUT_TXT, "w", encoding="utf-8") as f_out:
            line_count = 0
            for line in tqdm(f_in, desc="Cleaning Sequence"):
                line = line.strip()
                if line.startswith(">"):
                    continue  # Skip header lines
                if line:
                    f_out.write(line + "\n")
                    line_count += 1
                    
        print(f"✅ Clean E. coli genome text saved to: {OUTPUT_TXT}")
        print(f"   Lines processed: {line_count}")
        print(f"\n👉 Next Step: Run 'ufce_ingestion_pipeline_shard.py' to embed this data.")
        
    except Exception as e:
         print(f"❌ Error during preprocessing: {e}")

if __name__ == "__main__":
    preprocess_genome()