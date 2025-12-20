before running project, run:

```bash
# 1. Setup
pip install pip-tools

# 2. Harvest (Generates the lockfile with all hashes)
pip-compile -v --generate-hashes requirementList.txt --output-file=requirements.txt

# 3. Wipe (Removes the generator and its baggage)
pip uninstall -y pip-tools click build pyproject_hooks

# 4.Rebuild the container and it will install the dependencies now