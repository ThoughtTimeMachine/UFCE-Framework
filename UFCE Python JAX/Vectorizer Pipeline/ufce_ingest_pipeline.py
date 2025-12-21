import os
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = "large_dataset.txt"   # Your massive text file (Logs, Books, etc.)
OUTPUT_BIN = "knowledge_base.dat"  # The optimized vector file
OUTPUT_META = "metadata.txt"       # Keeps track of which text matches which vector
CHUNK_SIZE = 500                   # Characters per chunk (approx. 100-150 tokens)
BATCH_SIZE = 32                    # How many chunks to embed at once (GPU batch)
EMBEDDING_DIM = 384                # Depends on model (MiniLM=384, BGE=1024)

# Load a fast, local embedding model
# 'all-MiniLM-L6-v2' is incredibly fast and good for general English
print("Loading Embedding Model...")
model = SentenceTransformer('all-MiniLM-L6-v2') 

def count_lines(filename):
    """Counts lines quickly for the progress bar."""
    with open(filename, 'rb') as f:
        return sum(1 for _ in f)

def ingest_data():
    # 1. PRE-ALLOCATION
    # We need to know how big the file will be roughly.
    # For a real massive file, we might append, but pre-allocating is faster.
    # Let's count chunks first (simplification for this script).
    
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    # Simple chunking by character count (Advanced: use a tokenizer)
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    num_chunks = len(chunks)
    
    print(f"Total Chunks to Embed: {num_chunks}")
    
    # 2. CREATE MEMORY MAP ON DISK
    # This creates the file on the hard drive immediately.
    # It acts like an array, but data lives on disk.
    fp = np.memmap(OUTPUT_BIN, dtype='float32', mode='w+', shape=(num_chunks, EMBEDDING_DIM))
    
    # 3. STREAMING EMBEDDING LOOP
    print("Starting Ingestion Pipeline...")
    
    # We process in batches to keep the GPU/CPU busy
    for i in tqdm(range(0, num_chunks, BATCH_SIZE)):
        # a. Get Batch of Text
        batch_text = chunks[i : i + BATCH_SIZE]
        
        # b. Embed (Text -> Vector)
        # This runs on your GPU/CPU automatically via Torch
        embeddings = model.encode(batch_text, convert_to_numpy=True)
        
        # c. Write to Disk (Directly into the memmap slot)
        # No RAM spike here!
        current_batch_size = len(batch_text)
        fp[i : i + current_batch_size] = embeddings

    # 4. FLUSH & SAVE METADATA
    # Ensure all data is written to disk
    fp.flush()
    
    # Save the text chunks so we can retrieve the answer later!
    # (In production, use a lightweight database like SQLite, but this works for now)
    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.replace("\n", " ") + "\n")
            
    print(f"\n✅ Ingestion Complete!")
    print(f"Vector Database: {OUTPUT_BIN} ({os.path.getsize(OUTPUT_BIN)/1e6:.2f} MB)")
    print(f"Metadata: {OUTPUT_META}")

if __name__ == "__main__":
    # Create dummy data if file doesn't exist
    if not os.path.exists(INPUT_FILE):
        print("Generating dummy data...")
        with open(INPUT_FILE, "w") as f:
            for _ in range(10000):
                f.write("System Error: Kernel panic at address 0x0045F. CPU Overload.\n")
                f.write("User Login: Admin access granted to user 'root'.\n")
    
    ingest_data()