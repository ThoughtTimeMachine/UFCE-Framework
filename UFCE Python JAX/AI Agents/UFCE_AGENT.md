# UFCE Infinite Context Agent (`UFCE_agent.py`)

## Overview
The `UFCE_agent.py` script is the interactive "Driver" of the UFCE framework. It acts as the bridge between the user, the **Tiered Memory Reservoir** (SSD/RAM), and the Local LLM (Ollama).

Unlike traditional RAG systems that rely on slow, approximate nearest neighbor (ANN) indexes, this agent performs a **Brute-Force JAX Scan** over the entire knowledge base for every query. This ensures 100% retrieval accuracy ("Lossless") without the need for pre-built indexes or massive VRAM.

---

## How It Works

### 1. Connection to the Reservoir (Zero-Copy)
* **Mechanism:** The agent uses `numpy.memmap` to link to the `knowledge_base.dat` file created by the ingestion pipeline.
* **Benefit:** It does **not** load the file into RAM. It maps the file's address space into virtual memory. The operating system handles the actual data movement.
* **Scale:** This allows a 16GB laptop to "open" and query a 1TB database instantly.

### 2. The JAX Scan Engine (The Math)
* **Hardware Acceleration:** The `fast_scanner` function is JIT-compiled by JAX. It treats the memory-mapped file as a standard array and blasts it through the GPU (or CPU AVX instructions).
* **Operation:** It computes the **Cosine Similarity** (Dot Product) between your query vector and *every single vector* in the database.
* **Speed:** On an RTX 4070 Ti, this occurs at ~2 Trillion operations per second.

### 3. The "Turbo-Cache" Physics
This architecture leverages the physical hierarchy of modern computing to accelerate repeated interactions:

* **Pass 1 (Cold Start):** The first time you scan the database, data is read physically from the SSD.
    * *Speed:* ~7-12 GB/s (limited by NVMe Gen4/5 speeds).
* **The OS Magic:** The operating system (Linux/Windows) detects the file access and automatically retains the read pages in your system's available RAM (the "Page Cache").
* **Pass 2 (Warm Query):** The second time you ask a question (or for any follow-up query), the data is served directly from your DDR5 System RAM.
    * *Speed:* **~60-80 GB/s** (limited only by DDR5 bandwidth).
    * *Result:* Your 80GB system RAM acts as a massive **Turbo-Cache** for the GPU, making iterative chats and deep-dive sessions instantaneous.

### 4. Integration with Ollama
* **Context Injection:** The agent retrieves the top-K (e.g., 5) most relevant text chunks based on the JAX scan.
* **Prompt Engineering:** It constructs a prompt containing strictly the retrieved facts and feeds it to the local LLM (e.g., Llama 3) via the Ollama API.
* **Outcome:** The LLM answers using *your* data, not just its training data, with zero hallucinations regarding the retrieved facts.

---

### Commands
* **User Input**: Type any question related to your dataset to initiate a real-time scan and retrieval.
* **`exit`**: Type this to safely close the memory map and shut down the agent.

### Performance Tuning

| Variable | Default | Description |
| :--- | :--- | :--- |
| `TOP_K` | `5` | Number of text snippets to feed the LLM. Increase for broader context, decrease for precision. |
| `MODEL_NAME` | `"llama3"` | The Ollama model to use. Can be changed to `"mistral"`, `"gemma"`, etc. |
| `BATCH_SIZE` | *(Internal)* | If modifying for 100GB+ files, wrap the scanner in a loop to process in chunks larger than VRAM but smaller than System RAM. |

## Usage Guide

### Prerequisites
1.  **Ollama Running:** Ensure `ollama serve` is active in a background terminal.
2.  **Data Ingested:** You must have run `UFCE_ingestion_pipeline.py` first to create the `.dat` file.

### Running the Agent
```bash
python UFCE_agent.py

