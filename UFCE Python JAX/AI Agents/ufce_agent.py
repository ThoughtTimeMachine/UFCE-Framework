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

import numpy as np
import jax.numpy as jnp
from jax import jit, device_put
from sentence_transformers import SentenceTransformer
import requests
import json
import time

# --- CONFIG ---
# UPDATED PATHS FOR TEST RUN:
DB_PATH = "knowledge_base/wiki_test_subset.dat"
META_PATH = "knowledge_base/wiki_test_subset_meta.txt"

EMBEDDING_DIM = 384  # Must match your ingest model
TOP_K = 5            # How many chunks to give the LLM
OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
MODEL_NAME = "llama3" # Or "mistral", "gemma", etc.

# --- LOAD RESOURCES ---
print("Loading Embedding Model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Mapping Massive Database...")
# This is instant (Zero-Copy)
try:
    # Use 'r' mode (Read Only) to protect data
    vectors = np.memmap(DB_PATH, dtype='float32', mode='r')
    # Reshape it to (N, Dim) - we infer N from file size
    num_vectors = vectors.shape[0] // EMBEDDING_DIM
    vectors = vectors.reshape((num_vectors, EMBEDDING_DIM))
    print(f"✅ Linked to {num_vectors} vectors (Virtual Memory).")
except FileNotFoundError:
    print(f"❌ Error: Database file not found at {DB_PATH}")
    print("   Did you run ufce_ingestion_pipeline_sharded.py?")
    exit()

# Load Metadata (The actual text)
print("Loading Text Index...")
try:
    with open(META_PATH, "r", encoding="utf-8") as f:
        text_chunks = f.readlines()
except FileNotFoundError:
    print(f"❌ Error: Metadata file not found at {META_PATH}")
    exit()

# --- THE JAX KERNEL (The Engine) ---
@jit
def fast_scanner(query_vec, db_chunk):
    # Normalize query for Cosine Similarity
    q_norm = query_vec / jnp.linalg.norm(query_vec)
    
    # Dot Product (Batch Matrix Multiply)
    scores = jnp.dot(db_chunk, q_norm)
    
    return scores

def query_ollama(prompt):
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=data)
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: Ollama returned status {response.status_code}"
    except Exception as e:
        return f"Error connecting to Ollama: {e}"

# Define a Safe Chunk Size for 12GB VRAM
STREAM_BATCH_SIZE = 500_000 

def print_banner():
    print(r"""
   __  ________________
  / / / / ____/ ____/ ____/
 / / / / /_  / /   / __/
/ /_/ / __/ / /___/ /___
\____/_/    \____/_____/
    ___   _____________   ______
   /   | / ____/ ____/ | / /_  __/
  / /| |/ / __/ __/ /  |/ / / /
 / ___ / /_/ / /___/ /|  / / /
/_/  |_\____/_____/_/ |_/ /_/

    :: UFCE Framework ::  (v1.0.0 - JAX Accelerated)
    [Mode: Infinite Context] [Device: GPU/NVIDIA]
    """)
    print("-" * 60)

def run_agent():
    print_banner()
    print(f"🤖 Agent Model: {MODEL_NAME}")
    print(f"🌊 Streaming Batch Size: {STREAM_BATCH_SIZE}")
    print(f"📚 Knowledge Base: {num_vectors} vectors")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("User: ")
        if query.lower() == 'exit': break
        
        t0 = time.time()
        
        # 1. Embed Query
        q_vec = embedder.encode(query)
        q_jax = device_put(q_vec)
        
        # 2. UFCE Streaming Scan (The Loop)
        all_scores = []
        
        # We iterate over the massive memmap in chunks
        for i in range(0, len(vectors), STREAM_BATCH_SIZE):
            chunk = vectors[i : i + STREAM_BATCH_SIZE]
            scores_chunk = fast_scanner(q_jax, chunk)
            all_scores.append(np.array(scores_chunk))
            
        final_scores = np.concatenate(all_scores)
        
        # 3. Get Top-K (Exact Brute Force)
        top_k_indices = np.argpartition(final_scores, -TOP_K)[-TOP_K:]
        
        # Retrieve Text
        retrieved_context = []
        for idx in top_k_indices:
            if idx < len(text_chunks):
                retrieved_context.append(text_chunks[idx].strip())
            
        t1 = time.time()
        scan_time = t1 - t0
        
        # 3. Construct Prompt
        context_block = "\n---\n".join(retrieved_context)
        prompt = f"""
        Use the following retrieved data to answer the user question.
        DATA:
        {context_block}
        
        QUESTION: {query}
        """
        
        # 4. LLM Answer
        print(f"\n[System] Scanned {num_vectors} vectors in {scan_time:.4f}s.")
        print("[System] Thinking...")
        answer = query_ollama(prompt)
        
        print(f"\nAI: {answer}\n")
        print("-" * 50)

if __name__ == "__main__":
    run_agent()