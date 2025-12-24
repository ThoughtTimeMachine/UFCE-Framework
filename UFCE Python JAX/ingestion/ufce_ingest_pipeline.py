
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
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm

# --- CONFIG ---
INPUT_FILE = "large_dataset.txt"   # Must be the CLEAN text file (from Step 1)
OUTPUT_BIN = "knowledge_base.dat"
OUTPUT_META = "metadata.txt"
BATCH_SIZE = 64        # GPU batch size (Higher = Faster, more VRAM)
MAX_TOKENS = 256       # Professional semantic chunk size
EMBEDDING_DIM = 384    # MiniLM dimension

# Load Model & Tokenizer
print("Loading Model & Tokenizer...")
model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def stream_chunks(filename, max_tokens=256):
    """
    Grok's Logic: Stream file + Tokenizer-aware chunking.
    Benefit: Ensures chunks are semantically valid (don't cut words in half).
    """
    buffer_ids = []
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Tokenize line and add to buffer
            line_ids = tokenizer.encode(line, add_special_tokens=False)
            buffer_ids.extend(line_ids)
            
            # While buffer is large enough, yield chunks
            while len(buffer_ids) >= max_tokens:
                chunk = tokenizer.decode(buffer_ids[:max_tokens])
                yield chunk
                buffer_ids = buffer_ids[max_tokens:]
        
        # Final cleanup
        if buffer_ids:
            yield tokenizer.decode(buffer_ids)

def run_ingestion():
    print(f"Starting True Streaming Ingestion of {INPUT_FILE}...")
    
    # --- PASS 1: Indexing (CPU Bound) ---
    # We read the file once just to write metadata and count chunks.
    # We DO NOT save the text to a list variable.
    print("Pass 1: Writing Metadata & Counting Chunks...")
    
    num_chunks = 0
    with open(OUTPUT_META, "w", encoding="utf-8") as meta_f:
        # We stream the file line by line
        for chunk in tqdm(stream_chunks(INPUT_FILE, MAX_TOKENS)):
            clean_chunk = chunk.replace("\n", " ")
            meta_f.write(clean_chunk + "\n")
            num_chunks += 1
            
    print(f"Total Chunks Found: {num_chunks}")
    if num_chunks == 0:
        print("Error: No chunks found. Check your input file!")
        return

    # --- PASS 2: Vectorization (GPU Bound) ---
    # We re-open the stream and feed the GPU in small batches.
    print(f"Pass 2: Encoding {num_chunks} vectors to SSD...")
    
    # Create the memmap (Allocates 0 RAM, just disk space)
    fp = np.memmap(OUTPUT_BIN, dtype='float32', mode='w+', shape=(num_chunks, EMBEDDING_DIM))

    # Re-initialize the stream generator for the second pass
    chunk_generator = stream_chunks(INPUT_FILE, MAX_TOKENS)
    
    batch_buffer = []
    write_idx = 0
    
    # Process with progress bar
    for chunk in tqdm(chunk_generator, total=num_chunks):
        batch_buffer.append(chunk)
        
        # When buffer hits batch size, send to GPU
        if len(batch_buffer) >= BATCH_SIZE:
            # 1. Embed (GPU)
            embeddings = model.encode(batch_buffer, convert_to_numpy=True)
            
            # 2. Write (SSD)
            current_batch_len = len(embeddings)
            fp[write_idx : write_idx + current_batch_len] = embeddings
            
            # 3. Cleanup (RAM)
            batch_buffer = [] 
            write_idx += current_batch_len
            
            # Optional: Flush periodically to save progress
            if write_idx % (BATCH_SIZE * 10) == 0:
                fp.flush()

    # Process any remaining items in the buffer
    if batch_buffer:
        embeddings = model.encode(batch_buffer, convert_to_numpy=True)
        fp[write_idx : write_idx + len(embeddings)] = embeddings
        fp.flush()

    print(f"✅ Success! Knowledge Base created: {OUTPUT_BIN}")
    print(f"✅ Metadata saved: {OUTPUT_META}")

if __name__ == "__main__":
    run_ingestion()