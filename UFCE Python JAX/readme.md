before running project, run:

```bash
# 1. Setup
pip install pip-tools

# 2. Harvest (Generates the lockfile with all hashes)
pip-compile -v --generate-hashes --extra-index-url https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --output-file=requirements.txt requirement_list_for_hash_generations.txt

# 3. Wipe (Removes the generator and its baggage)
pip uninstall -y pip-tools click build pyproject_hooks

# 4.Rebuild the container and it will install the dependencies now
