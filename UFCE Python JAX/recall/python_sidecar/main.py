import os
import sys
import shutil
import json
import time
import socket
import uvicorn
import threading
import glob
import hashlib
import pypdf
import requests  # <--- NEW: For talking to Ollama
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# --- WINDOWS CONSOLE FIX ---
if sys.stdout and sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# --- HARDWARE CONFIG ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90" 
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

# --- IMPORTS ---
from sentence_transformers import SentenceTransformer
import jax.numpy as jnp
from jax import jit, device_put

# --- CONFIG & STATE LOCATIONS ---
CONFIG_FILENAME = "velocity_config.json"
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # <--- NEW: Local LLM URL

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

# --- GLOBAL HELPER: PATH RESOLVER ---
def get_dataset_path():
    """ Globally resolves where the vectors should be saved/loaded. """
    # 1. Default fallback
    raw_path = "personal/knowledge_base/personal_vectors"
    
    # 2. Read from Config
    try:
        paths_to_check = [
            CONFIG_FILENAME, 
            os.path.join("..", CONFIG_FILENAME), 
            os.path.join("..", "..", CONFIG_FILENAME)
        ]
        config_path = next((p for p in paths_to_check if os.path.exists(p)), None)
        
        if config_path:
            with open(config_path, 'r') as f:
                data = json.load(f)
                active = data["active_dataset"]
                raw_path = data["datasets"][active]["vectors_output_dir"]
    except:
        pass
        
    # 3. Absolute Path Check
    if os.path.isabs(raw_path):
        return raw_path

    # 4. Dev vs Prod Check
    current_dir = os.getcwd()
    if "binaries" in current_dir or "src-tauri" in current_dir:
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        return os.path.join(project_root, raw_path)
        
    return os.path.abspath(os.path.join(current_dir, raw_path))

# --- ROBUST CONFIG LOADER ---
def load_velocity_config():
    candidates = [
        CONFIG_FILENAME,
        os.path.join("..", CONFIG_FILENAME),
        os.path.join("..", "..", CONFIG_FILENAME),
        os.path.join(os.path.dirname(sys.executable), CONFIG_FILENAME),
        r"C:\Users\kylek\UFCE-Framework\UFCE Python JAX\recall\velocity_config.json" 
    ]
    
    config_path = next((p for p in candidates if os.path.exists(p)), None)
    
    if config_path:
        print(f"[INFO] Config loaded from: {os.path.abspath(config_path)}")
    else:
        print("[WARN] Config NOT found. Using defaults (768).")
        return None

    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
            
        active_key = data.get("active_dataset")
        if not active_key: return None
            
        dataset_cfg = data["datasets"][active_key]
        return {
            "name": active_key,
            "dat_name": dataset_cfg.get("final_dat_name", "knowledge_base_full.dat"),
            "meta_name": dataset_cfg.get("final_meta_name", "metadata_full.txt"),
            "embedding_dim": dataset_cfg.get("embedding_dim", 768)
        }
    except Exception as e:
        print(f"[WARN] Config load error: {e}")
        return None

# --- CHUNKING HELPER ---
def chunk_text(text, chunk_size=1000, overlap=100):
    if not text: return []
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1 
        
        if current_length >= chunk_size:
            chunk_str = " ".join(current_chunk)
            chunks.append(chunk_str)
            overlap_count = int(overlap / 5) 
            current_chunk = current_chunk[-overlap_count:] if overlap_count > 0 else []
            current_length = sum(len(w) + 1 for w in current_chunk)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# --- STATE MANAGER ---
class IngestionState:
    def __init__(self):
        self.source_queue = [] 
        self.processed_files = set() 
        self.is_ingesting = False
        self.current_status = "Idle"
        self.progress = 0.0
        self.active_bundle_name = "Default Personal"
        self.load_state()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    data = json.load(f)
                    self.source_queue = data.get('queue', [])
                    self.processed_files = set(data.get('processed', []))
                    self.active_bundle_name = data.get('active_bundle', "Default Personal")
            except: pass

    def save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump({
                'queue': self.source_queue,
                'processed': list(self.processed_files),
                'active_bundle': self.active_bundle_name
            }, f)

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

# --- INGESTION ENGINE ---
class IngestionEngine:
    def __init__(self):
        self.config = load_velocity_config()

    def get_allowed_extensions(self):
        return ['.txt', '.md', '.pdf', '.docx', '.html'] 

    def scan_files(self):
        allowed_exts = self.get_allowed_extensions()
        print(f"[SEARCH] Scanning for extensions: {allowed_exts}")
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

    def run_ingestion_task(self, embedder_model):
        state.is_ingesting = True
        state.current_status = "Scanning folders..."
        
        self.config = load_velocity_config()
        target_dim = self.config['embedding_dim'] if self.config else 768
        
        files = self.scan_files()
        total = len(files)
        
        if total == 0:
            state.current_status = "Nothing new to process."
            state.is_ingesting = False
            return

        output_dir = get_dataset_path()
        print(f"[INFO] Writing Knowledge Base to: {output_dir}")
        print(f"[INFO] Target Dimension: {target_dim}")
        
        shards_dir = os.path.join(output_dir, "shards")
        vecs_dir = os.path.join(output_dir, "vectors")
        os.makedirs(shards_dir, exist_ok=True)
        os.makedirs(vecs_dir, exist_ok=True)

        state.current_status = f"Ingesting {total} files..."
        
        for idx, file_path in enumerate(files):
            try:
                raw_text = ""
                # --- PDF HANDLING ---
                if file_path.lower().endswith('.pdf'):
                    try:
                        reader = pypdf.PdfReader(file_path)
                        for page in reader.pages:
                            t = page.extract_text()
                            if t: raw_text += t + "\n"
                    except Exception as pdf_err:
                        print(f"[WARN] Failed to parse PDF {file_path}: {pdf_err}")
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        raw_text = f.read()

                # --- CHUNKING LOGIC ---
                raw_text = raw_text.strip()
                if not raw_text: continue

                text_chunks = chunk_text(raw_text, chunk_size=1000, overlap=150)
                if not text_chunks: continue
                
                vectors = embedder_model.encode(text_chunks)
                if vectors.shape[1] > target_dim:
                    vectors = vectors[:, :target_dim]
                
                base_name = hashlib.md5(file_path.encode()).hexdigest()
                np.save(os.path.join(vecs_dir, f"{base_name}.npy"), vectors)
                
                with open(os.path.join(shards_dir, f"{base_name}.txt"), 'w', encoding='utf-8') as f:
                    for chunk in text_chunks:
                        clean_chunk = chunk.replace("\n", " ")
                        f.write(clean_chunk + "\n")

                state.processed_files.add(file_path)
                state.progress = ((idx + 1) / total) * 100
                state.save_state()

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        state.current_status = "Merging Database..."
        self.merge_shards(vecs_dir, shards_dir, output_dir)
        
        agent.load_database() 
        state.current_status = "Ingestion Complete"
        state.is_ingesting = False
        state.progress = 100.0

    def merge_shards(self, vecs_dir, shards_dir, output_dir):
        all_vecs = []
        all_meta = []
        files = sorted(os.listdir(vecs_dir))
        
        for fname in files:
            if not fname.endswith(".npy"): continue
            base_name = fname.replace(".npy", "")
            npy_path = os.path.join(vecs_dir, fname)
            txt_path = os.path.join(shards_dir, base_name + ".txt")
            
            try:
                v = np.load(npy_path)
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        chunks = f.readlines()
                else:
                    chunks = []
                
                min_len = min(v.shape[0], len(chunks))
                if min_len > 0:
                    all_vecs.append(v[:min_len])
                    all_meta.extend(chunks[:min_len])
            except Exception as e:
                print(f"[ERROR] Merge failed for {fname}: {e}")

        if not all_vecs:
            print("[WARN] No vectors found to merge.")
            return

        final_vecs = np.vstack(all_vecs)
        dat_name = "knowledge_base_full.dat"
        meta_name = "metadata_full.txt"
        
        fp = np.memmap(os.path.join(output_dir, dat_name), dtype='float32', mode='w+', shape=final_vecs.shape)
        fp[:] = final_vecs[:]
        fp.flush()
        
        with open(os.path.join(output_dir, meta_name), 'w', encoding='utf-8') as f:
            for line in all_meta:
                f.write(line.strip() + "\n")
        print(f"[SUCCESS] Merged {final_vecs.shape[0]} vectors.")

# --- DATA MODELS ---
class PathRequest(BaseModel):
    path: str
class BundleRequest(BaseModel):
    path: str
    name: str
class QueryRequest(BaseModel):
    text: str

# --- UFCE AGENT ---
class UFCEAgent:
    def __init__(self, top_k=5, stream_batch_size=500_000):
        self.stream_batch_size = stream_batch_size
        self.top_k = top_k
        self.config = load_velocity_config()
        self.db_path = "Not Loaded"
        
        print("[INIT] Loading Embedding Model...")
        self.embedder = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True) 
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
            
            if vectors.shape[0] % self.dim != 0:
                 self.dim = 384 
            
            self.num_vectors = vectors.shape[0] // self.dim
            self.vectors = vectors.reshape((self.num_vectors, self.dim))
            
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.text_chunks = f.readlines()
            
            self.is_loaded = True
            print(f"[OK] Linked to {self.num_vectors:,} vectors (Dim: {self.dim}).")
        except Exception as e:
            print(f"[ERROR] Failed to load DB: {e}")
            self.is_loaded = False

    @staticmethod
    @jit
    def _fast_scanner(query_vec, db_chunk):
        q_norm = query_vec / jnp.linalg.norm(query_vec)
        scores = jnp.dot(db_chunk, q_norm)
        return scores

    # --- THE GENERATION LAYER ---
    def generate_response(self, query, context_chunks):
        if not context_chunks:
            return "No relevant information found in the Knowledge Base."
        
        # Build the RAG Prompt
        context_block = "\n---\n".join(context_chunks)
        prompt = f"""You are an expert research assistant using the UFCE Drill-Down process.
Use the following Retrieved Context to answer the User's Question.
If the answer is not in the context, strictly state that you cannot find it.

USER QUESTION: {query}

RETRIEVED CONTEXT:
{context_block}

ANSWER:"""

        print("[LLM] Sending prompt to Ollama...")
        try:
            # Call Local Ollama API
            resp = requests.post(
                OLLAMA_API_URL, 
                json={
                    "model": "llama3", 
                    "prompt": prompt, 
                    "stream": False 
                },
                timeout=60 # Wait up to 60s for answer
            )
            if resp.status_code == 200:
                return resp.json().get("response", "Error: Empty response from LLM")
            else:
                return f"Error: Ollama returned status {resp.status_code}"
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}. Is 'ollama run llama3' running?"

    def search(self, query):
        if not self.is_loaded:
            return ["Error: Database not loaded."], 0.0

        print(f"[SEARCH] Query: '{query}'")
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
            if idx < len(self.text_chunks):
                retrieved_context.append(self.text_chunks[idx].strip())
        
        return retrieved_context, time.time() - t0

agent = UFCEAgent()
ingestor = IngestionEngine()

# --- ROUTES ---
@app.post("/add_source")
async def add_source(req: PathRequest):
    if state.add_folder(req.path):
        return {"status": "added", "queue": state.source_queue}
    return {"status": "duplicate", "queue": state.source_queue}

@app.get("/get_queue")
def get_queue():
    return {
        "queue": state.source_queue, 
        "bundle": state.active_bundle_name,
        "is_ingesting": state.is_ingesting,
        "progress": state.progress,
        "status_text": state.current_status
    }

@app.post("/clear_queue")
def clear_queue():
    state.clear_queue()
    return {"status": "cleared"}

@app.post("/clear_knowledge")
def clear_knowledge():
    target_dir = get_dataset_path()
    try:
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
            os.makedirs(target_dir)
        state.clear_knowledge()
        agent.is_loaded = False
        return {"status": "wiped"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/start_ingestion")
def start_ingestion(background_tasks: BackgroundTasks):
    if state.is_ingesting:
        return {"status": "busy"}
    background_tasks.add_task(ingestor.run_ingestion_task, agent.embedder)
    return {"status": "started"}

@app.post("/save_bundle")
def save_bundle(req: BundleRequest):
    src = get_dataset_path()
    dst = req.path
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return {"status": "saved", "location": dst}
    except Exception as e:
        return {"error": str(e)}

@app.post("/load_bundle")
def load_bundle(req: PathRequest):
    state.active_bundle_name = os.path.basename(req.path)
    state.save_state()
    return {"status": "loaded", "bundle": state.active_bundle_name}

@app.get("/")
def read_root():
    return {
        "status": "Velocity Engine Online", 
        "loaded": agent.is_loaded,
        "dataset": state.active_bundle_name,
        "db_path": agent.db_path,
        "dim": getattr(agent, 'dim', "Unknown")
    }

@app.post("/search")
async def search(request: QueryRequest):
    # 1. Load DB if needed
    if not agent.is_loaded:
        agent.load_database()
        if not agent.is_loaded:
             return {"results": f"[WARN] Database not loaded."}

    # 2. PASS 1: RETRIEVAL (The "Global Scan")
    context_chunks, time_taken = agent.search(request.text)
    
    # 3. PASS 2: GENERATION (The "Drill Down")
    # We call Ollama to synthesize the chunks into a specific answer
    llm_response = agent.generate_response(request.text, context_chunks)
    
    # 4. Format Output
    output = f"🤖 **AI Analysis (Llama 3):**\n{llm_response}\n\n"
    output += f"⏱️ *Drill-Down completed in {time_taken:.4f}s*\n"
    output += "--------------------------------------------------\n"
    output += "📚 **Source Evidence:**\n\n"
    output += "\n\n---\n\n".join(context_chunks)
    
    return {"results": output}

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

if __name__ == "__main__":
    port = find_free_port()
    print(f"VELOCITY_PORT:{port}")
    sys.stdout.flush()
    uvicorn.run(app, host="127.0.0.1", port=port)