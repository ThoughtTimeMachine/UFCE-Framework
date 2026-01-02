import sys
import socket
import uvicorn
from fastapi import FastAPI

# --- EVENTUALLY IMPORT YOUR JAX ENGINE HERE ---
# from velocity_engine import run_inference 

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Velocity Engine Active", "model": "Nomic v1.5 (GGUF)"}

@app.post("/query")
def query_endpoint(data: dict):
    # This is where we will hook up your actual UFCE logic later
    return {"response": f"Velocity received query: {data.get('query')}"}

def get_free_port():
    """Finds a free port to avoid conflicts."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

if __name__ == "__main__":
    # 1. Force stdout flush so Rust sees logs instantly
    sys.stdout.reconfigure(encoding='utf-8')
    
    # 2. Handshake: Print the port for Rust to grab
    port = get_free_port()
    print(f"PORT: {port}", flush=True)
    
    # 3. Start Server (log_config=None keeps the output clean)
    uvicorn.run(app, host="127.0.0.1", port=port, log_config=None)