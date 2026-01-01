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
import jax.numpy as jnp
from jax import jit, device_put
from sentence_transformers import SentenceTransformer
import requests
import time
import psutil

class UFCEAgent:
    def __init__(self, db_path="knowledge_base_full.dat", meta_path="metadata_full.txt",
                 embedding_dim=384, top_k=5, model_name="llama3",
                 ollama_url="http://host.docker.internal:11434/api/generate",
                 stream_batch_size=500_000, flux_threshold=0.15):
        
        self.db_path = db_path
        self.meta_path = meta_path
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.stream_batch_size = stream_batch_size
        self.flux_threshold = flux_threshold  # Mean flux threshold for GPU routing
        
        print("Loading Embedding Model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        print(f"Mapping Database: {self.db_path}")
        file_size_gb = os.path.getsize(self.db_path) / (1024**3)
        print(f"📂 Database Size on Disk: {file_size_gb:.2f} GB")

        vectors = np.memmap(self.db_path, dtype='float32', mode='r')
        self.num_vectors = vectors.shape[0] // self.embedding_dim
        self.vectors = vectors.reshape((self.num_vectors, self.embedding_dim))
        print(f"✅ Linked to {self.num_vectors:,} vectors (Virtual Memory).")

        print("Loading Text Index...")
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.text_chunks = f.readlines()

    # CPU fallback scanner (NumPy)
    @staticmethod
    def _cpu_scanner(query_vec, vectors):
        q_norm = query_vec / np.linalg.norm(query_vec)
        return np.dot(vectors, q_norm)

    # GPU scanner (JAX)
    @staticmethod
    @jit
    def _gpu_scanner(query_vec, db_chunk):
        q_norm = query_vec / jnp.linalg.norm(query_vec)
        return jnp.dot(db_chunk, q_norm)


    def query_ollama(self, prompt):
        data = { "model": self.model_name, "prompt": prompt, "stream": False }
        try:
            response = requests.post(self.ollama_url, json=data, timeout=120)
            return response.json().get('response', "No response") if response.status_code == 200 else f"Error: {response.status_code}"
        except Exception as e:
            return f"Error connecting to Ollama: {e}"

    def get_ram_usage(self):
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)

    def print_banner(self):
        GREEN = "\033[92m"
        RESET = "\033[0m"

        print(GREEN + r"""
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
    """+ RESET)
        print("-" * 60)

    def search(self, query):
        t0 = time.time()
        
        q_vec = self.embedder.encode(query)
        q_jax = device_put(q_vec.astype(np.float32))

        # === RESONANCE-BASED DYNAMIC SCHEDULING ===
        # Quick CPU estimate of mean flux to decide routing
        sample_size = min(100_000, self.num_vectors)
        sample_vectors = self.vectors[:sample_size]
        sample_scores = self._cpu_scanner(q_vec, sample_vectors)
        mean_flux = np.mean(np.abs(sample_scores))
        
        print(f"[System] Estimated mean flux: {mean_flux:.4f}")

        if mean_flux < self.flux_threshold:
            print(f"[System] Low resonance detected — routing to CPU for power saving")
            # Full CPU scan
            scores = self._cpu_scanner(q_vec, self.vectors)
        else:
            print(f"[System] High resonance detected — using GPU for max speed")
            # Full GPU scan
            all_scores = []
            for i in range(0, len(self.vectors), self.stream_batch_size):
                chunk = self.vectors[i:i + self.stream_batch_size]
                scores_chunk = self._gpu_scanner(q_jax, chunk)
                all_scores.append(np.array(scores_chunk))
            scores = np.concatenate(all_scores)

        # Top-K retrieval
        top_k_indices = np.argpartition(scores, -self.top_k)[-self.top_k:]
        retrieved_context = [self.text_chunks[idx].strip() for idx in top_k_indices if idx < len(self.text_chunks)]
        
        scan_time = time.time() - t0
        return retrieved_context, scan_time

    def run_interactive(self):
        print("🚀 UFCE Agent with Dynamic Scheduling (Power-Aware)")
        print(f"🤖 Model: {self.model_name}")
        print(f"📚 Vectors: {self.num_vectors:,}")
        print(f"💾 DB Size: {os.path.getsize(self.db_path) / (1024**3):.2f} GB")
        print(f"⚡ Flux Threshold: {self.flux_threshold} (low = CPU, high = GPU)")
        print("-" * 60)
        print("Type 'quit' to exit.\n")
        
        while True:
            query = input("User: ").strip()
            if query.lower() in ['quit', 'exit']: break
            if not query: continue
            
            context, scan_time = self.search(query)
            
            current_ram = self.get_ram_usage()
            print(f"\n[System] Scanned in {scan_time:.4f}s | RAM: {current_ram:.2f} GB")
            
            if context:
                prompt = f"Use this data to answer:\n\n" + "\n---\n".join(context) + f"\n\nQuestion: {query}"
                print("[System] Thinking...")
                answer = self.query_ollama(prompt)
                print(f"\nAI: {answer}\n")
            else:
                print("\nAI: No relevant context found.")
            print("-" * 50)

if __name__ == "__main__":
    agent = UFCEAgent(flux_threshold=0.15)  # Adjust threshold as needed
    agent.run_interactive()