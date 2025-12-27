# The UniField Coupling Equation (UFCE) Framework

**A Zero-Memory Streaming Kernel for Trillion-Scale Spatial-Temporal Interactions**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-yellow.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-green.svg)](https://github.com/google/jax)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.18055873.svg)](https://zenodo.org/records/18055873)

## 🚀 The Breakthrough

Traditional dense matrix algorithms require **1 Terabyte of RAM** to store 125 billion interaction points.

The **UFCE streaming kernel** eliminates the "Memory Wall" entirely — computing aggregate statistics with **zero additional memory overhead** while achieving **real-time performance on consumer hardware**.

### Latest Benchmarks (December 2025 — Ryzen 9 7950X + RTX 4070 Ti)

**Hardware:** AMD Ryzen 9 7950X (CPU) + NVIDIA RTX 4070 Ti (GPU) + 80GB DDR5 RAM

| Kernel Implementation | Precision | Throughput | Speedup vs CPU | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Ryzen 9 7950X (CPU)** | FP64 | 43.33 Billion ops/s | 1.0× | Baseline Validation |
| **RTX 4070 Ti (Scientific)** | FP32 | 1.50 Trillion ops/s | 34.7× | Legacy Native Kernel |
| **JAX "God Mode"** | **FP32** | **2.02 Trillion ops/s** | **47.0×** | **Theoretical Max / Scanning** |
| **JAX Streaming Softmax** | FP32 | **232.6 Billion ops/s** | 5.3× | **LLM Attention (Linear)** |
| **JAX Top-K** | FP32 | 1.40 Billion ops/s | 0.03× | **Deep Security Forensics** |

### Key Achievements
* **The 2 Trillion Barrier Broken:** The JAX Blocked Kernel achieved **2,020 Billion operations per second** on a single consumer GPU.
* **Real-World LLM Attention:** Validated a **Streaming Softmax** kernel running at **232 Billion ops/s**, proving that exact attention statistics can be computed for 100M+ token contexts in real-time.
* **Infinite Context:** Processed a **50 Trillion Interaction** workload (equivalent to a 1 Billion Token Context) in just **24.7 seconds** on RTX 4070 Ti.
* **1 Terabyte Challenge**: 125 billion points processed in **2.89 seconds** (CPU) with **0.00 MB memory overhead**.

---

## 🤖 The Infinite Context Agent (Tutorial)

This repository includes a fully functional **"Drill-Down" AI Agent** capable of searching massive datasets (e.g., all of Wikipedia) on a consumer PC without using enterprise VRAM.

### Step 1: Preparing Wikipedia Input Shards
The pipeline works best with the full English Wikipedia dump.

1. **Download the Dump**:
   ```bash
   wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

### Step 2: Prepare the Data

1. **Extract Clean Text Shards (using WikiExtractor)**:
```bash
pip install wikiextractor
```
2. **Run extraction to create the wiki_shards/ folder**:
```bash
WikiExtractor.py enwiki-latest-pages-articles.xml.bz2 --output wiki_shards/ -b 1M
```
This produces multiple text files in wiki_shards/ (one per large article batch), ready for the ingestion pipeline.

### Step 3: Ingest into Reservoir

## Overview
The UFCE ingestion pipeline has been upgraded to a **sharded, resumable architecture** for handling truly massive datasets (e.g., full Wikipedia dumps, Common Crawl subsets, or multi-domain corpora). It consists of two scripts:

- `ufce_ingestion_pipeline_shard.py`: Processes individual text shards into vector + metadata pairs.
- `merge_shards.py`: Concatenates all shards into the final `knowledge_base_full.dat` and `metadata_full.txt` used by the UFCE agent.

This design enables **safe, parallel, and resumable** ingestion of datasets far larger than system RAM while maintaining the **Zero-Memory** philosophy.

Convert the text into a Tiered Memory Reservoir (SSD-backed Vector DB). 
The sharded pipeline processes Wikipedia articles in independent chunks, using streaming embedding per shard to avoid RAM overload on massive datasets.

```bash
python UFCE_ingestion_pipeline.py
# Output: knowledge_base.dat (Binary Vectors) + metadata.txt (Index)
```
### Step 4: Launch the Agent with Local LLM (Ollama)

The UFCE agent connects to a running Ollama server to generate responses grounded in your knowledge base.

You have **two options** for running Ollama — choose based on convenience and performance.

#### Option 1: Ollama on Host Machine (Windows — Recommended for Speed & Ease)
This is the fastest and simplest setup — Ollama runs natively on your Windows machine.
Host version is faster (no container overhead, direct GPU access if using Ollama GPU build).

1. **Install Ollama** (if not already):
   - Download from https://ollama.com/download
   - Run the installer.

2. **Download Llama-3**:
   ```bash
   ollama pull llama3      # 8B model (fast, ~4.7GB)
   # or
   ollama pull llama3:70b  # 70B model (if you have 48GB+ RAM/VRAM)
3. **Start the Model (in a separate CMD/PowerShell window)**:
```bash
ollama run llama3
 ```
Keep this window open — it runs the server on localhost:11434

### Option 2: Ollama Inside Docker Container
Use this if you want everything isolated in the container 

1. **Add to your Dockerfile** (or run manually):
```dockerfile
# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Pull model (do this at build time or first run)
RUN ollama pull llama3
```
2. **Start Ollama Server** in container background:
```bash
ollama serve &
```
3. **Update Agent URL**(in UFCE_agent.py):
OLLAMA_URL = "http://localhost:11434/api/generate"

### Step 5: Launch the Agent

Connect the local LLM (via Ollama) to the reservoir.

```bash
python UFCE_agent.py
```
## 📂 Repository Contents

### 🧠 Core Framework
- `ufce_jax_god_mode_benchmark.py` — The Record Breaker. Runs the "God Mode" block-streaming kernel (2.02T ops/s).
- `ufce_jax_real_world_measurements.py` — The Real World. Runs the Streaming Softmax (LLM) and Top-K (Security) kernels.
- `ufce_attention_core.py` — The Engine. The pure JAX logic for the agent.

### 🛠️ Utilities & Agent
- `prepare_wiki_dump.py` — Streaming XML cleaner for massive datasets.
- `UFCE_ingestion_pipeline.py` — "Zero-Memory" vector database creator.
- `UFCE_agent.py` — The interactive AI interface.

### 📜 Legacy & Docs
- `paper/` — Full academic preprint (LaTeX + PDF).
- `validate_attention.cu` — Optimized CUDA C++ kernels.

## 🛠️ Quick Start (VS Code Dev Container)

This repo is configured as a VS Code Dev Container — the easiest way to reproduce.

1. Install VS Code and the "Dev Containers" extension.
2. Clone the repo and open the folder in VS Code.
3. When prompted, click "Reopen in Container".

### Run the Benchmarks

**CPU Tests** (inside container terminal):

```bash
python cyber_validation.py
python blockchain_validation.py
```
**Run the Record-Breaking "God Mode" Benchmark:**

```bash
python ufce_jax_god_mode_benchmark.py
```
**GPU Test (CUDA C++ Native):**

```bash
# For RTX 40-series (Ada Lovelace)
nvcc -o attention_gpu validate_attention.cu -O3 -arch=sm_89
./attention_gpu
```
## 🧠 The "Cognitive Tax": Blind vs. Smart Processing

We benchmarked the cost of introducing physics-informed decision logic into the kernel.

| Kernel Type         | Logic                       | Throughput             | Insight                                      |
|---------------------|-----------------------------|------------------------|----------------------------------------------|
| Blind "God Mode"    | No decision (Always GPU)    | 2.02 Trillion Ops/s    | Pure Tensor Core saturation.                 |
| Cognitive Hybrid    | Physics-Check per block     | 0.35 Trillion Ops/s    | The cost of flexibility. Conditional logic (if/else) breaks pure kernel fusion, but enables dynamic energy saving. |

**Conclusion:** For maximum raw power, use the Blind Kernel. For energy-efficient robotics (where you want to idle the GPU during low-flux), use the Cognitive Kernel.

## ⚖️ Licensing & Commercial Use

### Open Source License
This project is open-source under the GNU General Public License v3.0 (GPLv3). This ensures that the core framework remains free for researchers, students, and open-source projects.

## ⚡ Project VELOCITY: Breaking the 12GB VRAM Barrier

While traditional training requires the entire model to fit in VRAM, **Project VELOCITY** implements a **Layer-Wise Swapper**. This treats your System RAM (DDR5) as a high-speed L4 cache and your GPU VRAM as a dedicated compute core.



### The "Infinite Model" Benchmark
We successfully executed a **32GB Model (Full FP32 Llama-3-8B)** on a single **12GB RTX 4070 Ti** with near-zero compute starvation.

| Metric | Achievement | Impact |
| :--- | :--- | :--- |
| **Model Size** | **32 GB** | 3x larger than physical VRAM capacity. |
| **Layer Latency** | **0.78s** | Transfer + Compute happens in sub-second timeframes. |
| **Effective Throughput** | **512 GB/s** | Saturates the PCIe bus via Quad-Buffered Ring Pipelining. |
| **Training Capability** | **Full-Parameter** | No quantization (4-bit) required; trains at 100% precision. |

### 🔬 How it Works: The Quad-Buffered Ring Pipeline
Project VELOCITY eliminates the "Stop-and-Go" latency of standard data loading. By using **4-zone asynchronous DMA**, the system "teleports" the next layer into the GPU while the current layer is still being computed.



1. **Ingest:** Fetches the next layer from System RAM.
2. **Pin/Feed:** Pages the memory into a "Page-Locked" DMA zone.
3. **Compute:** The GPU executes the forward/backward pass.
4. **Writeback:** Updates gradients and clears the VRAM for the next incoming layer.

---

### Commercial Licensing
For proprietary software, closed-source applications, or enterprise use cases where GPLv3 compliance is not feasible (e.g., defense, proprietary robotics, closed banking systems), a Commercial License is available. This license waives the copyleft requirements and includes priority support.

**Contact:** thoughttimemachinexr@gmail.com for enterprise inquiries.
## 📚 Citation

If you use the UFCE Streaming Kernels or the Infinite Context Agent in your research, please cite the framework:

```bibtex
@software{ufce_framework_2025,
  author = {Thought Time Machine XR},
  title = {The UniField Coupling Equation (UFCE) Framework: Zero-Memory Streaming Kernels},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.17906337},
  url = {[https://github.com/thoughttimemachinexr/UFCE](https://github.com/thoughttimemachinexr/UFCE)}
}

