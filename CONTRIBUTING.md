# Contributing to the UFCE Framework

First off, thank you for considering contributing to the UniField Coupling Equation (UFCE) Framework. It's people like you that make the open-source community such an amazing place to learn, inspire, and create.

## ⚖️ The Legal Part (Important)

This project operates under a **Dual-Licensing Model** (GPLv3 + Commercial). To maintain the ability to offer commercial licenses to enterprise partners (supporting the project's sustainability), we require all contributors to agree to the following terms:

### Contributor License Agreement (CLA)
By submitting a Pull Request (PR) to this repository, you acknowledge and agree that:
1.  **You own the copyright** to your contribution or have the authority to submit it.
2.  **You grant the project owner (Kyle Killian)** a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare derivative works of, publicly display, publicly perform, sublicense, and distribute your contribution and such derivative works.
3.  **You specifically grant the right to re-license your contribution** under other licenses, including proprietary commercial licenses, at the project owner's discretion.

**If you do not agree to these terms, please do not submit a Pull Request.**

---

## 🛠️ How to Contribute

### 1. Reporting Bugs
Bugs are tracked as [GitHub Issues](https://github.com/ThoughtTimeMachine/UFCE-Framework/issues).
* Use a clear and descriptive title.
* Describe the exact steps to reproduce the problem.
* **Critical:** Include your hardware specs (CPU, GPU, RAM) as performance varies significantly between architectures.

### 2. Pull Request Process
1.  **Fork** the repo and create your branch from `main`.
2.  **Install Dependencies:** Ensure you are using the exact versions in `requirements.txt`.
3.  **Code Style:**
    * Use Python type hints where possible.
    * Keep JAX kernels pure (no side effects).
    * Ensure all new functions have docstrings explaining inputs and outputs.
4.  **Validate Performance:**
    * If you modify core kernels, you **must** run `ufce_jax_god_mode_benchmark.py` to ensure you haven't introduced performance regressions.
    * Include the benchmark output in your PR description.
5.  **Submit** the Pull Request.

### 3. Development Environment
We recommend using the provided **VS Code Dev Container** to ensure a consistent environment.
1.  Open the folder in VS Code.
2.  Click "Reopen in Container".
3.  The environment will auto-configure with CUDA, JAX, and Numba.

---

## 🧪 Testing Guidelines

Because this is a high-performance scientific framework, "it runs" is not enough. It must run **fast** and **accurately**.

* **Accuracy:** Run `validate_attention_god_mode.py` to verify that your changes still produce mathematically correct Softmax/Top-K results compared to the CPU baseline.
* **Throughput:** Any change to the streaming pipeline (`UFCE_ingestion_pipeline.py` or `UFCE_agent.py`) must maintain the "Zero-Memory" constraint. Do not introduce `list.append()` loops that load full datasets into RAM.

## 🤝 Code of Conduct
We are committed to providing a welcoming and inspiring community for all.
* Be respectful and inclusive.
* Focus on the code and the physics.
* Harassment of any kind will not be tolerated.

Thank you for helping us break the Memory Wall! 🚀