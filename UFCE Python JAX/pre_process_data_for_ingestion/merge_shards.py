import os
import glob
import shutil

# --- CONFIG ---
KB_DIR = "knowledge_base"
OUTPUT_DAT = "knowledge_base_full.dat"
OUTPUT_META = "metadata_full.txt"

def merge_database():
    print("🚧 Starting Knowledge Base Merger...")
    
    # 1. Find all parts
    # We look for files ending in .dat (but NOT the full one we are about to create)
    dat_files = sorted([f for f in glob.glob(os.path.join(KB_DIR, "*.dat")) 
                        if "full.dat" not in f and "wiki_test" not in f])
    
    meta_files = sorted([f for f in glob.glob(os.path.join(KB_DIR, "*_meta.txt")) 
                         if "full.txt" not in f and "wiki_test" not in f])

    if len(dat_files) != len(meta_files):
        print("❌ Error: Mismatch between .dat and .txt files. Re-run ingestion.")
        return

    print(f"Found {len(dat_files)} shards to merge.")

    # 2. Merge Vectors (Binary Concatenation)
    print("Merging Vectors...")
    with open(OUTPUT_DAT, 'wb') as outfile:
        for fname in dat_files:
            print(f"  -> Adding {os.path.basename(fname)}")
            with open(fname, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)

    # 3. Merge Metadata (Text Concatenation)
    print("Merging Text Index...")
    with open(OUTPUT_META, 'w', encoding="utf-8") as outfile:
        for fname in meta_files:
            with open(fname, 'r', encoding="utf-8") as infile:
                shutil.copyfileobj(infile, outfile)

    print("-" * 50)
    print(f"✅ FINAL DATABASE READY: {OUTPUT_DAT}")
    print(f"✅ FINAL METADATA READY: {OUTPUT_META}")
    print("🚀 You can now point 'ufce_agent.py' to these files!")

if __name__ == "__main__":
    merge_database()