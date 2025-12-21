# The UniField Coupling Equation (UFCE) Framework

**A Zero-Memory Streaming Kernel for Trillion-Scale Spatial-Temporal Interactions**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-yellow.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-green.svg)](https://github.com/google/jax)
[![PAPER](https://zenodo.org/records/17984328.svg)](https://zenodo.org/records/17984328) <!-- 

## 🚀 The Breakthrough

Traditional dense matrix algorithms require **1 Terabyte of RAM** to store 125 billion interaction points.

The **UFCE streaming kernel** eliminates the "Memory Wall" entirely — computing aggregate statistics with **zero additional memory overhead** while achieving **real-time performance on consumer hardware**.

### Latest Benchmarks (December 2025 — Ryzen 9 7950X + RTX 4070 Ti)

**Hardware:** AMD Ryzen 9 7950X (CPU) + NVIDIA RTX 4070 Ti (GPU)

| Kernel Implementation | Precision | Throughput | Speedup vs CPU | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| Ryzen 9 7950X (CPU) **Numba CPU** | FP64 | 43.33 Billion ops/s | 1.0× | Baseline Validation |
| RTX 4070 Ti (Scientific) **CUDA C++** | FP32 | 1.50 Trillion ops/s | 34.7× | Legacy Native Kernel |
| **JAX "God Mode"** | **FP32** | **2.02 Trillion ops/s** | **47.0×** | **Theoretical Max / Scanning** |
| **JAX Softmax** | FP32 | **232.6 Billion ops/s** | 5.3× | **LLM Attention (Linear)** |
| **JAX Top-K** | FP32 | 1.40 Billion ops/s | 0.03× | **Deep Security Forensics** |

### Key Achievements
* **The 2 Trillion Barrier Broken:** The JAX Blocked Kernel achieved **2,020 Billion operations per second** on a single consumer GPU.
* **Real-World LLM Attention:** Validated a **Streaming Softmax** kernel running at **232 Billion ops/s**, proving that exact attention statistics can be computed for 100M+ token contexts in real-time.
* **Infinite Context:** Processed a **50 Trillion Interaction** workload (equivalent to a 1 Billion Token Context) in just **24.7 seconds** on RTX 4070 Ti.
**1 Terabyte Challenge**: 125 billion points processed in **2.89 seconds** (CPU) with **0.00 MB memory overhead**.

## 📂 Repository Contents

### 🐍 Next-Gen JAX Kernels (Recommended)
* `ufce_jax_50T_block_kernel_god_mode.py` — **The Record Breaker.** Runs the "God Mode" block-streaming kernel to hit 2.02T ops/s.
* `ufce_jax_real_world_measurements.py` — **The Real World.** Runs the Streaming Softmax (LLM) and Top-K (Security) kernels.
* `ufce_jax_oscillating_hybrid_load_balancing.py` — **Hybrid Physics.** Demonstrates CPU/GPU oscillating load balancing for physical simulations.

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

**1. Run the Record-Breaking "God Mode" Benchmark:**
```bash
python ufce_jax_god_mode_benchmark.py

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

### 🧠 The "Cognitive Tax": Blind vs. Smart Processing
We benchmarked the cost of introducing physics-informed decision logic into the kernel.

| Kernel Type | Logic | Throughput | Insight |
| :--- | :--- | :--- | :--- |
| **Blind "God Mode"** | No decision (Always GPU) | **2.02 Trillion Ops/s** | Pure Tensor Core saturation. |
| **Cognitive Hybrid** | Physics-Check per block | **0.35 Trillion Ops/s** | **The cost of flexibility.** Conditional logic (`if/else`) breaks pure kernel fusion, but enables dynamic energy saving. |

**Conclusion:** For maximum raw power, use the Blind Kernel. For energy-efficient robotics (where you want to idle the GPU during low-flux), use the Cognitive Kernel.