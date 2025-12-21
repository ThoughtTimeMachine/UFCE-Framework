
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
DB_PATH = "knowledge_base.dat"
META_PATH = "metadata.txt"
EMBEDDING_DIM = 384  # Must match your ingest model
TOP_K = 5            # How many chunks to give the LLM
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3" # Or "mistral", "gemma", etc.

# --- LOAD RESOURCES ---
print("Loading Embedding Model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Mapping Massive Database...")
# This is instant (Zero-Copy)
# We assume the file exists from the previous ingest step
try:
    # Use 'r' mode (Read Only) to protect data
    vectors = np.memmap(DB_PATH, dtype='float32', mode='r')
    # Reshape it to (N, Dim) - we infer N from file size
    num_vectors = vectors.shape[0] // EMBEDDING_DIM
    vectors = vectors.reshape((num_vectors, EMBEDDING_DIM))
    print(f"✅ Linked to {num_vectors} vectors (Virtual Memory).")
except FileNotFoundError:
    print("❌ Error: database not found. Run ingest_pipeline.py first.")
    exit()

# Load Metadata (The actual text)
# For 100GB files, use SQLite instead of a list. For <1GB, a list is fine.
print("Loading Text Index...")
with open(META_PATH, "r", encoding="utf-8") as f:
    text_chunks = f.readlines()

# --- THE JAX KERNEL (The Engine) ---
@jit
def fast_scanner(query_vec, db_chunk):
    # Normalize query for Cosine Similarity
    q_norm = query_vec / jnp.linalg.norm(query_vec)
    
    # Dot Product (Batch Matrix Multiply)
    # (Chunk_Size, Dim) @ (Dim, 1) -> (Chunk_Size, )
    scores = jnp.dot(db_chunk, q_norm)
    
    # We return the top scores and indices for this chunk
    # (In a real massive system, we'd use the block-streaming logic here)
    # For simplicity in this demo, we scan in one go if VRAM allows, 
    # or you wrap this in the Python loop we discussed.
    return scores

def query_ollama(prompt):
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=data)
    return response.json()['response']

def run_agent():
    print(f"\n🤖 UFCE Infinite-Context Agent ({MODEL_NAME})")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("User: ")
        if query.lower() == 'exit': break
        
        t0 = time.time()
        
        # 1. Embed Query
        q_vec = embedder.encode(query)
        q_jax = device_put(q_vec)
        
        # 2. UFCE Scan (The Magic)
        # Note: If vectors > VRAM, you loop here using the chunk logic we discussed.
        # Here we assume "Big RAM" setup for simplicity of the demo script.
        scores = fast_scanner(q_jax, vectors) # JAX reads directly from mmap!
        
        # Get Top-K (We do the sort on CPU for the final aggregate if list is huge)
        # Convert to numpy to use argpartition (fast top-k)
        scores_np = np.array(scores) 
        top_indices = np.argpartition(scores_np, -TOP_K)[-TOP_K:]
        
        retrieved_context = []
        for idx in top_indices:
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