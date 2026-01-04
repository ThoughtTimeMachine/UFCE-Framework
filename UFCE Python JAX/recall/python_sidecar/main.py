import sys
import uvicorn
import socket
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # <--- NEW IMPORT
from pydantic import BaseModel

app = FastAPI()

# <--- NEW BLOCK: ALLOW TAURI TO TALK TO PYTHON --->
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (simplest for local dev)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# <------------------------------------------------>

class QueryRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "Velocity Engine Online"}

@app.post("/search")
async def search(request: QueryRequest):
    return {"results": f"UFCE Search for: {request.text}"}

@app.post("/chat")
async def chat(request: QueryRequest):
    return {"response": f"AI response to: {request.text}"}

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

if __name__ == "__main__":
    port = find_free_port()
    
    # The Handshake
    print(f"VELOCITY_PORT:{port}")
    sys.stdout.flush()

    # Start Server
    uvicorn.run(app, host="127.0.0.1", port=port)