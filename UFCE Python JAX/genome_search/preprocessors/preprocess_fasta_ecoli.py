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
from tqdm import tqdm

# --- CONFIG ---
GENOME_DATA_DIR = "genome_data"
ECOLI_SHARDS_DIR = "ecoli_shards"  # New dedicated folder for E. coli

FASTA_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz"
INPUT_GZ = os.path.join(GENOME_DATA_DIR, "ecoli_mg1655.fna.gz")
OUTPUT_TXT = os.path.join(ECOLI_SHARDS_DIR, "ecoli_genome.txt")

# Create directories
os.makedirs(GENOME_DATA_DIR, exist_ok=True)
os.makedirs(ECOLI_SHARDS_DIR, exist_ok=True)

# --- STEP 1: Download if missing ---
if not os.path.exists(INPUT_GZ):
    print("⬇️ Downloading E. coli MG1655 genome (~4.6MB compressed)...")
    response = requests.get(FASTA_URL, stream=True)
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
else:
    print(f"✅ Genome already downloaded: {INPUT_GZ}")

# --- STEP 2: Preprocess FASTA → Clean Text ---
print("🧬 Preprocessing FASTA → clean text sequence...")
with gzip.open(INPUT_GZ, "rt") as f_in, open(OUTPUT_TXT, "w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if line.startswith(">"):
            continue  # Skip header lines (e.g., >NC_000913.3 ...)
        if line:  # Only write non-empty sequence lines
            f_out.write(line + "\n")

print(f"✅ Clean E. coli genome text saved to {OUTPUT_TXT}")
print(f"   Ready for ingestion: Run ufce_ingestion_pipeline_shard.py with SHARDS_DIR = 'ecoli_shards'")