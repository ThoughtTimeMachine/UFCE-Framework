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

# --- HARDWARE CONFIG (The Speed Cheat) ---
# This must happen BEFORE importing JAX to lock the RAM (Pinned Memory).
# It prevents JAX from gobbling all VRAM instantly, allowing for streaming.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90" # Use 90% of GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

import numpy as np
import jax.numpy as jnp
from jax import jit, device_put
from sentence_transformers import SentenceTransformer
import requests
import json
import time
import psutil  # For verifying the "Zero-Memory" Proof

class UFCEAgent:
    def __init__(self, db_path="knowledge_base_full.dat", meta_path="metadata_full.txt",
                 embedding_dim=384, top_k=5, model_name="llama3",
                 ollama_url="http://host.docker.internal:11434/api/generate",
                 stream_batch_size=500_000):
        
        # --- CONFIG ---
        self.db_path = db_path
        self.meta_path = meta_path
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.stream_batch_size = stream_batch_size
        
        # --- LOAD RESOURCES ---
        print("Loading Embedding Model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        print(f"Mapping Database: {self.db_path}")
        try:
            # Check file size for the user
            file_size_gb = os.path.getsize(self.db_path) / (1024**3)
            print(f"📂 Database Size on Disk: {file_size_gb:.2f} GB")

            # The "Zero-Copy" Magic: Memmap
            vectors = np.memmap(self.db_path, dtype='float32', mode='r')
            self.num_vectors = vectors.shape[0] // self.embedding_dim
            self.vectors = vectors.reshape((self.num_vectors, self.embedding_dim))
            print(f"✅ Linked to {self.num_vectors:,} vectors (Virtual Memory).")
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Error: Database file not found at {self.db_path}. Did you run merge_shards.py?")

        print("Loading Text Index...")
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.text_chunks = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Error: Metadata file not found at {self.meta_path}")

    # --- THE JAX KERNEL (The Engine) ---
    @staticmethod
    @jit
    def _fast_scanner(query_vec, db_chunk):
        # Normalize query for Cosine Similarity
        q_norm = query_vec / jnp.linalg.norm(query_vec)
        # Dot Product (Batch Matrix Multiply)
        scores = jnp.dot(db_chunk, q_norm)
        return scores

    def query_ollama(self, prompt):
        data = { "model": self.model_name, "prompt": prompt, "stream": False }
        try:
            response = requests.post(self.ollama_url, json=data)
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: Ollama returned status {response.status_code}"
        except Exception as e:
            return f"Error connecting to Ollama: {e}"

    def get_ram_usage(self):
        """Returns the RAM usage of just THIS process in GB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)

    def print_banner(self):
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
          
      NEURAL WEB (JAX)         DATA WORMHOLE
       o        o              \ . . . . . /
        \      /                \         /
    o---( U F )---o              \       /
       /   |   \                  |     |
      o    C    o                 |     |
       \   |   /                 /       \
    o---( E A )---o             /         \
       /       \               / . . . . . \
      o         o

    :: UFCE Framework ::  (v2.0.0 - JAX Accelerated)
    [Mode: Infinite Context] [Device: GPU/NVIDIA]
    [Architecture: Zero-Memory Streaming + Pinned RAM]
          
    Left:  The JAX agent weaving semantic connections in vector space.
    Right: The 'Wormhole' pipeline streaming 11GB+ of data instantly.
    """)
        print("-" * 60)

    def search(self, query):
        """Performs the vector search and returns context strings."""
        t0 = time.time()
        
        # 1. Embed Query
        q_vec = self.embedder.encode(query)
        q_jax = device_put(q_vec)
        
        # 2. UFCE Streaming Scan (The Loop)
        all_scores = []
        # We iterate over the massive memmap in chunks
        for i in range(0, len(self.vectors), self.stream_batch_size):
            chunk = self.vectors[i : i + self.stream_batch_size]
            scores_chunk = self._fast_scanner(q_jax, chunk)
            all_scores.append(np.array(scores_chunk))
            
        final_scores = np.concatenate(all_scores)
        
        # 3. Get Top-K (Exact Brute Force)
        top_k_indices = np.argpartition(final_scores, -self.top_k)[-self.top_k:]
        
        # Retrieve Text
        retrieved_context = []
        for idx in top_k_indices:
            if idx < len(self.text_chunks):
                retrieved_context.append(self.text_chunks[idx].strip())
        
        scan_time = time.time() - t0
        return retrieved_context, scan_time

    def run_interactive(self):
        self.print_banner()
        print(f"🤖 Agent Model: {self.model_name}")
        print(f"🌊 Streaming Batch Size: {self.stream_batch_size}")
        print(f"📚 Knowledge Base: {self.num_vectors:,} vectors")
        print(f"💾 Physical Database Size: {os.path.getsize(self.db_path) / (1024**3):.2f} GB")
        print("-" * 60)
        print("Type 'exit' to quit.\n")
        
        while True:
            query = input("User: ")
            if query.lower() in ['exit', 'quit']: break
            if not query.strip(): continue
            
            # Perform Search
            retrieved_context, scan_time = self.search(query)
            
            # RAM CHECK
            current_ram = self.get_ram_usage()
            
            # 4. Construct Prompt
            context_block = "\n---\n".join(retrieved_context)
            prompt = f"""
            Use the following retrieved data to answer the user question.
            DATA:
            {context_block}
            
            QUESTION: {query}
            """
            
            print(f"\n[System] Scanned {self.num_vectors:,} vectors in {scan_time:.4f}s.")
            print(f"[System] RAM Usage: {current_ram:.2f} GB (Proof of Infinite Context)")
            
            print("[System] Thinking...")
            answer = self.query_ollama(prompt)
            
            print(f"\nAI: {answer}\n")
            print("-" * 50)

if __name__ == "__main__":
    agent = UFCEAgent()
    agent.run_interactive()