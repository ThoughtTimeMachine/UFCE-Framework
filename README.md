# The UniField Coupling Equation (UFCE) Framework

**A Zero-Memory Streaming Kernel for Trillion-Scale Spatial-Temporal Interactions**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-yellow.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-green.svg)](https://github.com/google/jax)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.17906337.svg)](https://zenodo.org/records/18012490)

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

### Step 1: Prepare the Data
Convert a raw XML dump (e.g., Wikipedia) into clean text.
```bash
# Input: enwiki-latest-pages-articles.xml (25GB+)
python prepare_wiki_dump.py
# Output: large_dataset.txt (Cleaned Text)

### Step 2: Ingest into Reservoir

Convert the text into a Tiered Memory Reservoir (SSD-backed Vector DB). This uses a 2-pass streaming method to avoid RAM crashes.

```bash
python UFCE_ingestion_pipeline.py
# Output: knowledge_base.dat (Binary Vectors) + metadata.txt (Index)

### Step 3: Launch the Agent

Connect the local LLM (via Ollama) to the reservoir.

```bash
python UFCE_agent.py

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

**Run the Record-Breaking "God Mode" Benchmark:**

```bash
python ufce_jax_god_mode_benchmark.py

**GPU Test (CUDA C++ Native):**

```bash
# For RTX 40-series (Ada Lovelace)
nvcc -o attention_gpu validate_attention.cu -O3 -arch=sm_89
./attention_gpu

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

### Commercial Licensing
For proprietary software, closed-source applications, or enterprise use cases where GPLv3 compliance is not feasible (e.g., defense, proprietary robotics, closed banking systems), a Commercial License is available. This license waives the copyleft requirements and includes priority support.

**Contact:** thoughttimemachinexr@gmail.com for enterprise inquiries.

