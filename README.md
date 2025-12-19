# The UniField Coupling Equation (UFCE) Framework

**A Zero-Memory Streaming Kernel for Trillion-Scale Spatial-Temporal Interactions**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://zenodo.org/doi/10.5281/zenodo.XXXXXX) <!-- 

## 🚀 The Breakthrough

Traditional dense matrix algorithms require **1 Terabyte of RAM** to store 125 billion interaction points.

The **UFCE streaming kernel** eliminates the "Memory Wall" entirely — computing aggregate statistics with **zero additional memory overhead** while achieving **real-time performance on consumer hardware**.

### Latest Benchmarks (December 2025 — Ryzen 9 7950X + RTX 4070 Ti)

| Device                  | Precision      | Throughput                          | Speedup vs CPU |
|-------------------------|----------------|-------------------------------------|----------------|
| Ryzen 9 7950X (CPU)     | FP64           | 43.33 Billion points/sec            | 1.0×           |
| RTX 4070 Ti (Scientific)| FP64           | 74 Billion points/sec               | 1.7×           |
| **RTX 4070 Ti (Super Turbo)** | **FP32**   | **465.9 Billion points/sec**        | **10.8×**      |
| **RTX 4070 Ti (God Mode)**    | **FP32**   | **1.502 Trillion ops/sec**          | **34.7×**      |

- **1 Terabyte Challenge**: 125 billion points processed in **2.89 seconds** (CPU) with **0.00 MB memory overhead**.
- **God Mode Run**: 50 trillion operations (1B token context equivalent) in **33.29 seconds** on RTX 4070 Ti — **1.5 Trillion ops/sec** sustained.

## 📂 Repository Contents

- `paper/` — Full academic preprint (LaTeX + PDF).
- `needle_In_a_haystack_cyber_security_validation.py` — Cybersecurity: Finds attack in 100B logs in ~1.0s.
- `validate_blockchain.py` — FinTech: Detects whale in 100B transactions in ~1.3s.
- `validate_attention_god_mode.py` — Neural Networks: Linear attention on 10M token context in ~11s.
- `validate_attention.cu` — CUDA kernel for GPU attention flux (God Mode capable).

## 🛠️ Quick Start (One-Click with VS Code Dev Container)

This repo is configured as a **VS Code Dev Container** — the easiest way to reproduce.

1. Install [VS Code](https://code.visualstudio.com/) and the "Dev Containers" extension.
2. Clone the repo and open the folder in VS Code.
3. When prompted, click **"Reopen in Container"**.
   - The environment (Python 3.12, Numba, CUDA) auto-installs.

### Run the Benchmarks

**CPU Tests** (inside container terminal):
```bash
python cyber_validation.py
python blockchain_validation.py
python attention_validation.py

**GPU Test (CUDA — requires NVIDIA drivers + CUDA toolkit installed on host)**:

The kernel is optimized for modern NVIDIA GPUs. Use the correct compute capability for your card:

```bash
# RTX 40-series (Ada Lovelace) — e.g., RTX 4070 Ti, 4080, 4090
nvcc -o attention_gpu validate_attention.cu -O3 -arch=sm_89

# RTX 30-series (Ampere) — e.g., RTX 3080, 3090
nvcc -o attention_gpu validate_attention.cu -O3 -arch=sm_86

# RTX 20-series (Turing) — e.g., RTX 2080
nvcc -o attention_gpu validate_attention.cu -O3 -arch=sm_75

# GTX 10-series (Pascal) — e.g., GTX 1080
nvcc -o attention_gpu validate_attention.cu -O3 -arch=sm_61

# For maximum compatibility (slower on newer cards)
nvcc -o attention_gpu validate_attention.cu -O3 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_89,code=sm_89

# Or GPU version:
nvcc -o attention_gpu validate_attention.cu -O3 -arch=sm_89
./attention_gpu

### Removing GPU Kernel Timeout Cap (Windows only)
Long-running CUDA kernels can be killed by Windows' default 2-second TDR limit. To disable it (use with caution — can make GPU hangs require reboot):

1. Open Registry Editor (`regedit` as Administrator).
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`
3. Create or edit these DWORD (32-bit) values:
   - `TdrDelay` = `60` (decimal) → increases timeout to 60 seconds
   - `TdrLevel` = `0` (decimal) → disables TDR completely (riskier, but needed for very long kernels)
4. Reboot.

> Warning: Disabling TDR can cause system instability if a kernel truly hangs. Test responsibly.