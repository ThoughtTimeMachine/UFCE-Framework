# The UniField Coupling Equation (UFCE) Framework

**A Zero-Memory Streaming Kernel for 100+ Billion Point Spatial-Temporal Interactions.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://zenodo.org/doi/10.5281/zenodo.XXXXXX)

## 🚀 The Breakthrough
Traditional matrix algorithms require **1 Terabyte of RAM** to process 125 Billion interaction points ($N \times M$).
The **UFCE Framework** solves this "Memory Wall" by using a streaming kernel that processes interactions in CPU/GPU registers with **Zero Memory Overhead**.

### Benchmarks (Consumer Hardware)
| Device | Precision | Throughput | Speedup |
| :--- | :--- | :--- | :--- |
| **CPU (Numba)** | FP64 | 43 Billion pts/sec | 1.0x |
| **RTX 4070 Ti** | FP64 (Scientific) | 74 Billion pts/sec | 1.7x |
| **RTX 4070 Ti** | **FP32 (AI/Real-Time)** | **465 Billion pts/sec** | **10.8x** |

---

## 📂 Repository Contents

* **`paper/`**: The full academic preprint detailing the mathematics.
* **`core/`**: Source code for CPU (Python) and GPU (CUDA C++) kernels.
* **`examples/`**: Real-world validation scripts:
    * `needle_In_a_haystack_cyber_security_validation.py`: **Cybersecurity** (Finds a cyber-attack in 100B logs in ~1.0s).
    * `validate_blockchain.py`: **FinTech** (Finds a crypto whale in 100B transactions in ~1.3s).

## 🛠️ How to Run (One-Click)
This repo includes a Dev Container.
1. Open this folder in **VS Code**.
2. Click **"Reopen in Container"** when prompted.
3. The environment (Python, CUDA, Numba) will auto-install.

**Run the Blockchain Test:**
```bash
python examples/validate_blockchain.py

## 🧠 AI / Neural Network Benchmark: "Infinite Context"
Traditional Transformers suffer from **Quadratic Complexity ($O(N^2)$)**. Doubling the context window requires $4\times$ the compute and memory.

The **UFCE Linear Attention Kernel** ($O(N)$) eliminates this bottleneck.

| Model | Context Window | Interaction Matrix | Memory Required | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Transformer** | 2 Million Tokens | $4 \times 10^{12}$ | ~16,000 GB | **OOM Crash** |
| **UFCE (CPU)** | **2 Million Tokens** | **Streamed** | **~0 GB** | **✅ 2.98s** |

**Reproduce this test:**
```bash
python examples/validate_attention.py