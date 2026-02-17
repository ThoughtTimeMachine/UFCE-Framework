import os
import sys

# --- 1. USER CONFIGURATION ---
# Set this to 512. If it crashes, lower to 256.
DEFAULT_BATCH_SIZE = 512 

# --- 2. FORCE TRUE OFFLINE MODE ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 

if sys.stdout and sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# --- 3. GPU BINARY INJECTION ---
current_dir = os.getcwd()
possible_paths = [
    os.path.abspath(os.path.join(current_dir, "binaries")),
    os.path.abspath(os.path.join(os.path.dirname(current_dir), "binaries")),
    os.path.abspath(r"C:\Users\kylek\UFCE-Framework\UFCE Python JAX\binaries"),
    os.path.abspath(os.path.join(current_dir, "..", "binaries"))
]
gpu_lib_path = next((p for p in possible_paths if os.path.exists(p)), None)

if gpu_lib_path:
    print(f"--- [BOOT] Found GPU Binaries at: {gpu_lib_path} ---", flush=True)
    os.environ["PATH"] = gpu_lib_path + os.pathsep + os.environ["PATH"]
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(gpu_lib_path)
        except Exception as e:
            print(f"--- [BOOT] Failed to add DLL directory: {e} ---", flush=True)

import shutil
import json
import time
import socket
import uvicorn
import threading
import queue 
import glob
import hashlib
import pypdf
import requests
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Generator, Optional

# --- IMPORTS ---
print("--- [BOOT] Importing AI Engines... ---", flush=True)
import torch
from sentence_transformers import SentenceTransformer
import jax.numpy as jnp
from jax import jit, device_put
import jax

# --- HARDWARE CONFIG ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90" 
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

# --- CONFIG & STATE LOCATIONS ---
CONFIG_FILENAME = "velocity_config.json"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

if sys.platform.startswith('win'):
    appdata = os.getenv('LOCALAPPDATA')
    STATE_DIR = os.path.join(appdata, 'VelocityRecall')
else:
    STATE_DIR = os.path.expanduser('~/.velocity_recall')

os.makedirs(STATE_DIR, exist_ok=True)
STATE_FILE = os.path.join(STATE_DIR, "ingestion_state.json")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HELPER FUNCTIONS ---
def get_dataset_path():
    raw_path = "personal/knowledge_base/personal_vectors"
    try:
        paths_to_check = [CONFIG_FILENAME, os.path.join("..", CONFIG_FILENAME), os.path.join("..", "..", CONFIG_FILENAME)]
        config_path = next((p for p in paths_to_check if os.path.exists(p)), None)
        if config_path:
            with open(config_path, 'r') as f:
                data = json.load(f)
                active = data["active_dataset"]
                raw_path = data["datasets"][active]["vectors_output_dir"]
    except: pass
        
    if os.path.isabs(raw_path): return raw_path
    current_dir = os.getcwd()
    if "binaries" in current_dir or "src-tauri" in current_dir:
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        return os.path.join(project_root, raw_path)
    return os.path.abspath(os.path.join(current_dir, raw_path))

def load_velocity_config():
    candidates = [CONFIG_FILENAME, os.path.join("..", CONFIG_FILENAME), os.path.join("..", "..", CONFIG_FILENAME)]
    config_path = next((p for p in candidates if os.path.exists(p)), None)
    
    try:
        if not config_path: return None
        with open(config_path, 'r') as f: data = json.load(f)
        active_key = data.get("active_dataset")
        if not active_key: return None
        dataset_cfg = data["datasets"][active_key]
        return {
            "name": active_key,
            "dat_name": dataset_cfg.get("final_dat_name", "knowledge_base_full.dat"),
            "meta_name": dataset_cfg.get("final_meta_name", "metadata_full.txt"),
            "embedding_dim": dataset_cfg.get("embedding_dim", 768)
        }
    except: return None

# --- STREAMING CHUNKER ---
def chunk_text_generator(text, chunk_size=1000, overlap=100) -> Generator[str, None, None]:
    if not text: return
    words = text.split() 
    current_chunk = []
    current_length = 0
    overlap_count = int(overlap / 5) 
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1 
        if current_length >= chunk_size:
            yield " ".join(current_chunk)
            current_chunk = current_chunk[-overlap_count:] if overlap_count > 0 else []
            current_length = sum(len(w) + 1 for w in current_chunk)
    if current_chunk:
        yield " ".join(current_chunk)

# --- STATE MANAGER ---
class IngestionState:
    def __init__(self):
        self.source_queue = [] 
        self.processed_files = set() 
        self.is_ingesting = False
        self.current_status = "Idle"
        self.progress = 0.0
        self.active_bundle_name = "Default Personal"
        
        if not os.path.exists(STATE_FILE):
            print(f"[STATE] No state found. Creating new at {STATE_FILE}", flush=True)
            self.save_state()
        else:
            self.load_state()

    def load_state(self):
        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
                self.source_queue = data.get('queue', [])
                self.processed_files = set(data.get('processed', []))
                self.active_bundle_name = data.get('active_bundle', "Default Personal")
        except:
            print("[WARN] State file corrupt. Resetting.", flush=True)
            self.save_state()

    def save_state(self):
        try:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            with open(STATE_FILE, 'w') as f:
                json.dump({'queue': self.source_queue, 'processed': list(self.processed_files), 'active_bundle': self.active_bundle_name}, f)
        except Exception as e:
             print(f"[ERROR] Failed to save state: {e}", flush=True)

    def add_folder(self, path):
        if path not in self.source_queue:
            self.source_queue.append(path)
            self.save_state()
            return True
        return False

    def clear_queue(self):
        self.source_queue = []
        self.save_state()

    def clear_knowledge(self):
        self.processed_files = set()
        self.save_state()

state = IngestionState()

# --- INGESTION ENGINE (DYNAMIC BATCHING + STREAMING MERGE) ---
class IngestionEngine:
    def __init__(self):
        self.config = load_velocity_config()
        self.chunk_queue = queue.Queue(maxsize=5) 
        self.save_queue = queue.Queue(maxsize=500)   

    def get_allowed_extensions(self):
        return ['.txt', '.md', '.pdf', '.docx', '.html'] 

    def scan_files(self):
        allowed_exts = self.get_allowed_extensions()
        files_to_process = []
        for folder in state.source_queue:
            if os.path.isdir(folder):
                for root, _, files in os.walk(folder):
                    for file in files:
                        if any(file.lower().endswith(ext.lower()) for ext in allowed_exts):
                            full_path = os.path.join(root, file)
                            if full_path not in state.processed_files:
                                files_to_process.append(full_path)
        return files_to_process

    def loader_worker(self, files_subset, worker_id):
        if worker_id == 0:
             print(f"[SWARM-{worker_id}] Leader thread active.", flush=True)

        for file_path in files_subset:
            try:
                t0 = time.time()
                if file_path in state.processed_files: continue

                if worker_id == 0: 
                     state.current_status = f"Scanning {os.path.basename(file_path)}..."
                
                raw_text = ""
                if file_path.lower().endswith('.pdf'):
                    try:
                        reader = pypdf.PdfReader(file_path)
                        for page in reader.pages:
                            t = page.extract_text()
                            if t: raw_text += t + "\n"
                    except: pass
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        raw_text = f.read()

                raw_text = raw_text.strip()
                if not raw_text: continue

                chunks = list(chunk_text_generator(raw_text, chunk_size=1000, overlap=150))
                
                self.chunk_queue.put((file_path, chunks))
                
                dt = time.time() - t0
                print(f"[SWARM-{worker_id}] 🟢 Prepped {os.path.basename(file_path)} ({len(chunks)} chunks) in {dt:.2f}s", flush=True)
                
            except Exception as e:
                print(f"[LOADER-{worker_id} ERROR] {file_path}: {e}", flush=True)

    def writer_worker(self, output_dir, vecs_dir, shards_dir):
        while True:
            item = self.save_queue.get()
            if item is None: break 
            
            file_path, vectors, chunks = item
            
            try:
                if vectors.shape[1] > 768: vectors = vectors[:, :768]
                base_name = hashlib.md5(file_path.encode()).hexdigest()
                np.save(os.path.join(vecs_dir, f"{base_name}.npy"), vectors)
                
                with open(os.path.join(shards_dir, f"{base_name}.txt"), 'w', encoding='utf-8') as f:
                    for chunk in chunks:
                        clean_chunk = chunk.replace("\n", " ")
                        f.write(clean_chunk + "\n")
                
                state.processed_files.add(file_path)
                state.save_state()
            except Exception as e:
                print(f"[WRITER ERROR] {e}", flush=True)

    def run_ingestion_task(self, embedder_model, user_batch_size=None):
        state.is_ingesting = True
        
        files = self.scan_files()
        if not files:
            # If no files, we might just need to merge existing shards
            print("[PIPELINE] No new files. Checking for merge...", flush=True)
            output_dir = get_dataset_path()
            shards_dir = os.path.join(output_dir, "shards")
            vecs_dir = os.path.join(output_dir, "vectors")
            self.merge_shards(vecs_dir, shards_dir, output_dir)
            
            state.is_ingesting = False
            state.current_status = "Finished."
            state.progress = 100.0
            return

        # Determine Batch Size
        if user_batch_size:
            final_batch_size = user_batch_size
        elif torch.cuda.is_available():
            final_batch_size = DEFAULT_BATCH_SIZE
        else:
            final_batch_size = 32

        print(f"[PIPELINE] Starting FP16 Pipeline | Batch Size: {final_batch_size} | Files: {len(files)}", flush=True)

        output_dir = get_dataset_path()
        shards_dir = os.path.join(output_dir, "shards")
        vecs_dir = os.path.join(output_dir, "vectors")
        os.makedirs(shards_dir, exist_ok=True)
        os.makedirs(vecs_dir, exist_ok=True)

        # 1. SPAWN LOADERS
        NUM_LOADERS = 16 
        chunk_size = len(files) // NUM_LOADERS + 1
        file_batches = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
        
        loader_threads = []
        for i, batch in enumerate(file_batches):
            if batch:
                t = threading.Thread(target=self.loader_worker, args=(batch, i))
                t.start()
                loader_threads.append(t)
        
        print(f"[SWARM] Active Loaders: {len(loader_threads)}", flush=True)

        # 2. START WRITER
        writer_t = threading.Thread(target=self.writer_worker, args=(output_dir, vecs_dir, shards_dir))
        writer_t.start()

        # 3. GPU LOOP
        print(f"[GPU] Waiting for chunks...", flush=True)
        
        while True:
            try:
                job = self.chunk_queue.get(timeout=2) 
            except queue.Empty:
                if any(t.is_alive() for t in loader_threads):
                    continue 
                else:
                    print("\n[PIPELINE] All loaders finished. Shutting down writer...", flush=True)
                    self.save_queue.put(None) 
                    break
            
            file_path, chunks = job
            state.current_status = f"Embedding {os.path.basename(file_path)}"
            
            all_vectors = []
            
            for i in range(0, len(chunks), final_batch_size):
                batch = chunks[i : i + final_batch_size]
                
                # Mixed Precision
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        vec_batch = embedder_model.encode(batch)
                else:
                    vec_batch = embedder_model.encode(batch)
                
                all_vectors.append(vec_batch)
                
                if i % 100 == 0:
                     print(f"   [GPU] ⏩ Embedded {i}/{len(chunks)} chunks for {os.path.basename(file_path)}", flush=True)

            if all_vectors:
                final_vecs = np.vstack(all_vectors)
                self.save_queue.put((file_path, final_vecs, chunks))
            
            state.progress += (100 / len(files))

        for t in loader_threads: t.join()
        writer_t.join()
        
        state.current_status = "Merging Database..."
        self.merge_shards(vecs_dir, shards_dir, output_dir)
        agent.load_database() 
        state.is_ingesting = False
        state.progress = 100.0
        print("[PIPELINE] Finished.", flush=True)

    # --- CRITICAL FIX: STREAMING MERGE (PREVENTS RAM CRASH) ---
    def merge_shards(self, vecs_dir, shards_dir, output_dir):
        print(f"[MERGE] Starting ZERO-RAM merge sequence...", flush=True)
        files = sorted([f for f in os.listdir(vecs_dir) if f.endswith(".npy")])
        if not files: return
        
        # 1. Calculate Total Size first (Fast Scan)
        total_rows = 0
        DIM = 768
        
        print(f"[MERGE] Scanning {len(files)} shards for size...", flush=True)
        for i, fname in enumerate(files):
            try:
                # Use mmap_mode='r' to read shape without loading data
                npy_path = os.path.join(vecs_dir, fname)
                v = np.load(npy_path, mmap_mode='r')
                total_rows += v.shape[0]
                if i % 10 == 0: print(f"   scanned {i}/{len(files)}", flush=True)
            except: pass
            
        print(f"[MERGE] Total Database Size: {total_rows} vectors", flush=True)
        
        # 2. Allocate Massive File on Disk (Zero RAM usage)
        dat_name = "knowledge_base_full.dat"
        meta_name = "metadata_full.txt"
        final_dat_path = os.path.join(output_dir, dat_name)
        
        # Create the empty file container
        fp = np.memmap(final_dat_path, dtype='float32', mode='w+', shape=(total_rows, DIM))
        
        # 3. Stream Data (Load One -> Write One -> Delete One)
        current_idx = 0
        with open(os.path.join(output_dir, meta_name), 'w', encoding='utf-8') as f_meta:
            for i, fname in enumerate(files):
                try:
                    state.current_status = f"Merging {i}/{len(files)}"
                    state.progress = (i / len(files)) * 100
                    
                    # Paths
                    base_name = fname.replace(".npy", "")
                    npy_path = os.path.join(vecs_dir, fname)
                    txt_path = os.path.join(shards_dir, base_name + ".txt")
                    
                    # Load Vector (RAM Spike = Size of 1 file only)
                    v = np.load(npy_path) # Load fully into RAM for speed
                    rows = v.shape[0]
                    
                    # Load Metadata
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r', encoding='utf-8') as ft: 
                            chunks = ft.readlines()
                    else: chunks = []
                    
                    # Safety trim
                    min_len = min(rows, len(chunks))
                    
                    # WRITE TO DISK (Flush to SSD)
                    if min_len > 0:
                        fp[current_idx : current_idx + min_len] = v[:min_len]
                        for line in chunks[:min_len]:
                            f_meta.write(line.strip() + "\n")
                        
                        current_idx += min_len
                    
                    # Memory Cleanup
                    del v 
                    if i % 5 == 0:
                        fp.flush() # Force write to disk
                        print(f"[MERGE] 💾 Wrote shard {i}/{len(files)} to disk. ({(current_idx/total_rows)*100:.1f}%)", flush=True)
                        
                except Exception as e:
                    print(f"[MERGE ERROR] Failed on {fname}: {e}", flush=True)

        fp.flush()
        del fp # Close file handle
        print("[MERGE] ✅ Database merge complete.", flush=True)

# --- DATA MODELS ---
class PathRequest(BaseModel): path: str
class BundleRequest(BaseModel): path: str; name: str
class QueryRequest(BaseModel): text: str
class IngestionConfig(BaseModel): 
    batch_size: Optional[int] = None

# --- UFCE AGENT ---
class UFCEAgent:
    def __init__(self, top_k=5, stream_batch_size=500_000):
        self.stream_batch_size = stream_batch_size
        self.top_k = top_k
        self.config = load_velocity_config()
        self.db_path = "Not Loaded"
        
        print("[INIT] Loading Embedding Model...", flush=True)
        if getattr(sys, 'frozen', False):
            application_path = os.path.dirname(sys.executable)
        else:
            application_path = os.path.dirname(os.path.abspath(__file__))
            
        local_model_path = os.path.join(application_path, "embedder")

        if os.path.exists(local_model_path) and os.path.exists(os.path.join(local_model_path, "config.json")):
            print(f"[BOOT] Loading OFFLINE model from: {local_model_path}", flush=True)
            self.embedder = SentenceTransformer(local_model_path, trust_remote_code=True, local_files_only=True)
        else:
            print(f"[BOOT] Model not found at {local_model_path}. Downloading...", flush=True)
            self.embedder = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
            try:
                self.embedder.save(local_model_path)
                print(f"[BOOT] Model saved to {local_model_path} for future offline use.", flush=True)
            except Exception as e:
                print(f"[WARN] Failed to save offline copy: {e}", flush=True)

        self.load_database()

    def load_database(self):
        self.config = load_velocity_config()
        resolved_dir = get_dataset_path()
        dat_name = self.config['dat_name'] if self.config else "knowledge_base_full.dat"
        meta_name = self.config['meta_name'] if self.config else "metadata_full.txt"
        self.db_path = os.path.join(resolved_dir, dat_name)
        self.meta_path = os.path.join(resolved_dir, meta_name)
        
        if not os.path.exists(self.db_path):
            self.is_loaded = False
            return

        try:
            vectors = np.memmap(self.db_path, dtype='float32', mode='r')
            self.dim = self.config['embedding_dim'] if self.config else 768
            if vectors.shape[0] % self.dim != 0: self.dim = 384 
            self.num_vectors = vectors.shape[0] // self.dim
            self.vectors = vectors.reshape((self.num_vectors, self.dim))
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f: self.text_chunks = f.readlines()
            self.is_loaded = True
            print(f"[OK] Linked to {self.num_vectors:,} vectors (Dim: {self.dim}).", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to load DB: {e}", flush=True)
            self.is_loaded = False

    @staticmethod
    @jit
    def _fast_scanner(query_vec, db_chunk):
        q_norm = query_vec / jnp.linalg.norm(query_vec)
        scores = jnp.dot(db_chunk, q_norm)
        return scores

    def generate_response(self, query, context_chunks):
        if not context_chunks: return "No relevant information found in the Knowledge Base."
        context_block = "\n---\n".join(context_chunks)
        prompt = f"""You are an expert research assistant using the UFCE Drill-Down process.\nUse the following Retrieved Context to answer the User's Question.\nIf the answer is not in the context, strictly state that you cannot find it.\n\nUSER QUESTION: {query}\n\nRETRIEVED CONTEXT:\n{context_block}\n\nANSWER:"""
        print("[LLM] Sending prompt to Ollama...", flush=True)
        try:
            resp = requests.post(OLLAMA_API_URL, json={"model": "llama3", "prompt": prompt, "stream": False}, timeout=60)
            if resp.status_code == 200: return resp.json().get("response", "Error: Empty response from LLM")
            else: return f"Error: Ollama returned status {resp.status_code}"
        except Exception as e: return f"Error connecting to Ollama: {str(e)}. Is 'ollama run llama3' running?"

    def search(self, query):
        if not self.is_loaded: return ["Error: Database not loaded."], 0.0
        print(f"[SEARCH] Query: '{query}'", flush=True)
        t0 = time.time()
        prefix = "search_query: "
        q_vec_full = self.embedder.encode(prefix + query)
        q_vec = q_vec_full[:self.dim]
        q_jax = device_put(q_vec)
        all_scores = []
        for i in range(0, len(self.vectors), self.stream_batch_size):
            chunk = self.vectors[i : i + self.stream_batch_size]
            scores_chunk = self._fast_scanner(q_jax, chunk)
            all_scores.append(np.array(scores_chunk))
        final_scores = np.concatenate(all_scores)
        top_k_indices = np.argpartition(final_scores, -self.top_k)[-self.top_k:]
        retrieved_context = []
        for idx in top_k_indices:
            if idx < len(self.text_chunks): retrieved_context.append(self.text_chunks[idx].strip())
        return retrieved_context, time.time() - t0

agent = UFCEAgent()
ingestor = IngestionEngine()

@app.post("/add_source")
async def add_source(req: PathRequest):
    if state.add_folder(req.path): return {"status": "added", "queue": state.source_queue}
    return {"status": "duplicate", "queue": state.source_queue}

@app.get("/get_queue")
def get_queue(): return {"queue": state.source_queue, "bundle": state.active_bundle_name, "is_ingesting": state.is_ingesting, "progress": state.progress, "status_text": state.current_status}

@app.post("/clear_queue")
def clear_queue():
    state.clear_queue()
    return {"status": "cleared"}

@app.post("/clear_knowledge")
def clear_knowledge():
    target_dir = get_dataset_path()
    try:
        if os.path.exists(target_dir): shutil.rmtree(target_dir); os.makedirs(target_dir)
        state.clear_knowledge(); agent.is_loaded = False
        return {"status": "wiped"}
    except Exception as e: return {"error": str(e)}

@app.post("/start_ingestion")
def start_ingestion(background_tasks: BackgroundTasks, config: Optional[IngestionConfig] = None):
    if state.is_ingesting: return {"status": "busy"}
    chosen_batch = None
    if config and config.batch_size:
        chosen_batch = config.batch_size
    background_tasks.add_task(ingestor.run_ingestion_task, agent.embedder, chosen_batch)
    return {"status": "started", "batch_size": chosen_batch or DEFAULT_BATCH_SIZE}

@app.post("/save_bundle")
def save_bundle(req: BundleRequest):
    try: shutil.copytree(get_dataset_path(), req.path, dirs_exist_ok=True); return {"status": "saved", "location": req.path}
    except Exception as e: return {"error": str(e)}

@app.post("/load_bundle")
def load_bundle(req: PathRequest):
    state.active_bundle_name = os.path.basename(req.path); state.save_state()
    return {"status": "loaded", "bundle": state.active_bundle_name}

@app.get("/")
def read_root(): return {"status": "Velocity Engine Online", "loaded": agent.is_loaded, "dataset": state.active_bundle_name, "db_path": agent.db_path, "dim": getattr(agent, 'dim', "Unknown")}

@app.post("/search")
async def search(request: QueryRequest):
    if not agent.is_loaded: agent.load_database()
    context_chunks, time_taken = agent.search(request.text)
    llm_response = agent.generate_response(request.text, context_chunks)
    output = f"🤖 **AI Analysis (Llama 3):**\n{llm_response}\n\n⏱️ *Drill-Down completed in {time_taken:.4f}s*\n--------------------------------------------------\n📚 **Source Evidence:**\n\n" + "\n\n---\n\n".join(context_chunks)
    return {"results": output}

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: s.bind(('', 0)); return s.getsockname()[1]

if __name__ == "__main__":
    port = find_free_port()
    print(f"VELOCITY_PORT:{port}", flush=True)
    sys.stdout.flush()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")