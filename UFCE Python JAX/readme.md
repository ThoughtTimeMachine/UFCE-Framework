Project Setup & Commands

## 1. Prerequisites (Run on Main Host Computer)
Before you can run the full desktop application, you must install these tools on your actual computer (Windows/Mac/Linux), **not** inside the Docker container.

1.  **Node.js:** [Download LTS Version](https://nodejs.org/)
    * *Check:* Open a new terminal and type `node -v`
2.  **Rust:** [Download Rustup](https://rustup.rs/)
    * *Check:* Open a new terminal and type `rustc --version`

---

## 2. Python Dependencies (Run in Dev Container)
Before running the project for the first time, you need to generate the `requirements.txt` with strict hash locking. 

**Run these commands inside the VS Code Terminal (Dev Container):**

before running project for the first time, and any time you add new dependencies to the projects "requirement_list_for_hash_generations.txt" to generate the requirements.txt library list of hashes, run:

```bash
# 1. Setup
pip install pip-tools

# 2. Generates the lockfile with all hashes
pip-compile -v --generate-hashes --extra-index-url https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --output-file=requirements.txt requirement_list_for_hash_generations.txt

# 3. Wipe (Removes the generator and its baggage)
pip uninstall -y pip-tools click build pyproject_hooks

# 4.Rebuild the container and it will install the dependencies now

# 5. run in terminal:
source .venv/bin/activate 