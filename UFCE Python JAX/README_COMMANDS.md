# Project Commands & Setup Reference

This document covers how to set up, run, and build the UFCE Recall application.

---

## 1. One-Time Setup (New Machine / First Run)

**Important:** These steps must be done on your main Windows computer (Host), **not** inside the Docker container.

### Step A: Prerequisites
Ensure you have the following installed:
1.  **Node.js (LTS):** [Download here](https://nodejs.org/)
2.  **Rust:** [Download here](https://rustup.rs/) (Run `rustup-init.exe` -> Option 1)

### Step B: Microsoft C++ Build Tools (Crucial for Windows)
If you see errors like `linker link.exe not found`, you are missing these tools.

1.  **Download:** Go to [Microsoft Visual Studio Downloads](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and click **"Download Build Tools"**.
2.  **Run Installer:** Open `vs_buildtools.exe`.
3.  **Select Workload:**
    * Check the box for **"Desktop development with C++"**.
4.  **Verify Optional Components (Right Side):**
    * Ensure **"MSVC ... (Latest)"** (x64/x86 build tools) is checked.
    * Ensure **"Windows 11 SDK"** (or Windows 10 SDK) is checked.
5.  **Install:** Click "Install". This is a large download (1-2 GB).
6.  **Restart:** **You must restart your computer** after installation to update your system path.

### Step C: Install Project Dependencies
1.  Open your Command Prompt (cmd) or PowerShell.
2.  Navigate specifically to the app folder (where `package.json` lives):
```bash
cd "ufce python jax/recall"
```
    *(Note: Adjust the path if your top-level folder name is different)*
3.  Install the JavaScript/Tauri libraries:
```bash
npm install
```

## 2. Python Backend Setup (Run in Docker/DevContainer (I run project in vs code dev containers))
Before running the app, ensure the Python environment is ready.

1.  **Open VS Code Terminal** (inside the container).
```bash
    source .venv/bin/activate
```
2.  **Generate Lockfile:**
```bash
    pip install pip-tools
    pip-compile -v --generate-hashes --extra-index-url [https://storage.googleapis.com/jax-releases/jax_cuda_releases.html](https://storage.googleapis.com/jax-releases/jax_cuda_releases.html) --output-file=requirements.txt requirement_list_for_hash_generations.txt
```
3.  **Rebuild Container:** Press `F1` -> "Dev Containers: Rebuild Container".

###  2.5. Build the Sidecar (Crucial Step)
⚠️ You must do this on the WINDOWS HOST. PyInstaller on Linux creates Linux binaries. For the app to run on Windows, we must compile the Python script into a Windows .exe.

1. **Open Windows CMD (Host Machine).**

2. **Ensure you have pyinstaller installed on Windows:**

```bash
pip install pyinstaller fastapi uvicorn pydantic
```
3. **Navigate to your project root (`recall` folder):**
**Action Required:** Ensure the `python_sidecar` folder has been moved INSIDE the `recall` folder.

4. **Compile the Engine: Build the EXE:**
**Run this command to turn your Python script into a Windows `.exe`:**
```bash
python -m PyInstaller --onefile --clean --name velocity-engine-x86_64-pc-windows-msvc python_sidecar/main.py
```
(Note: This creates a file named velocity-engine-x86_64-pc-windows-msvc.exe inside dist/)

5. **Move the EXE: Move the file from the dist folder to the Tauri binaries folder:**
```bash
mkdir src-tauri\binaries
move dist\velocity-engine-x86_64-pc-windows-msvc.exe src-tauri\binaries\
```

 **Rebuild Container:** Press `F1` -> "Dev Containers: Rebuild Container".

## 3. Running the Application

### Option A: The Real Desktop App (Recommended)
Use this to test the full application window, Python backend integration, and native buttons.

* **Where to run:** Windows Command Prompt / PowerShell (Host Machine).
* **Prerequisite:** Ensure you are in the `recall` folder (`cd "ufce python jax/recall"`).
* **Command:**
```bash
    npm run tauri dev
```
* **Result:** The application window will launch on your desktop.

### Option B: "Headless" Mode (Browser Only)
Use this only for quick CSS/UI tweaks. The Python backend and window controls will **not** work.

* **Where to run:** VS Code Terminal (Inside Container).
* **Command:**
```bash
    npm run dev:headless
```
* **View at:** `http://localhost:5173`
---

## 4. Build for Production
Creates the final standalone installer (e.g., `.exe` for Windows).

* **Command (Host Machine):**
```bash
    npm run tauri build
```
* **Output Location:** `src-tauri/target/release/bundle/msi/` (or `nsis/`)

## 5. Troubleshooting

* **"Linker link.exe not found":** You missed **Step B** (C++ Build Tools). Install them and restart your computer.
* **"Tauri is not recognized as an internal or external command, operable program or batch file":** You missed **Step C** (`npm install`). Run it inside the `recall` folder.
* **Clean Build:** If the app behaves strangely, delete the `src-tauri/target` folder and run `npm run tauri dev` again.

* **"Sidecar not found" or "Permission Denied":** 1. Check that the sidecar is named exactly `velocity-engine-x86_64-pc-windows-msvc.exe` inside `src-tauri/binaries/`.
    2. Ensure `src-tauri/src/lib.rs` includes `.plugin(tauri_plugin_shell::init())`.
    3. Verify that your `src-tauri/capabilities/default.json` (or your main config) explicitly allows the sidecar.




* **"The "Cheat Sheet" for your New Machine:**
When you move to a new machine, ignore the 500 individual commands. Just do these 4 things in order:

1. **Setup the Host (Windows)**
Install the "Big Three" on the actual Windows machine:
Node.js (for npm)
Rust (for cargo)
C++ Build Tools (The giant 2GB download)

2. **Populate the Folders**
In your recall folder on Windows CMD:
```bash
npm install
```
This creates the .bin files Windows needs. If you do this in the container, it creates Linux files that Windows can't read.

3. **Prepare the "Brain" (The Sidecar) ON WINDOWS CMD (Not Docker):**
fromt eh recall folderL: run your pyinstaller command to build the .exe.
Move that .exe to src-tauri/binaries/ and rename it to: velocity-engine-x86_64-pc-windows-msvc.exe

4. **The Magic Launch Command**
Always run this from Windows CMD:

```bash
npm run tauri dev
```
### 3. Add CORS Warning to Troubleshooting
**Where to put it:** Add this to the bottom of the **Troubleshooting** list (Section 5).
*Why:* If you edit the Python code later, you might accidentally delete the CORS lines and get "Timeout" errors again.

```markdown
* **"Engine Timeout" or "Connection Failed":**
  If the logs say `Port Detected` but the status stays Yellow/Red:
  1. Check `python_sidecar/main.py`.
  2. Ensure you have the `CORSMiddleware` block added to the FastAPI app.
  3. **Rebuild the .exe** (See Section 2.5) and overwrite the old one.
  
## 🛑 When do you have to rebuild?

**If you change this file...  Do you need to rebuild the .exe?**
python_sidecar/main.py      YES (Critical)
agents/search_engine.py     YES (If main.py imports it)
requirements.txt            YES (If you install a new library)
src/App.tsx (React)         NO (Just save, it auto-updates)
src-tauri/tauri.conf.json   NO (Restarting npm run is enough)


## What to do if it breaks again:
If you see "'tauri' is not recognized," don't panic. It just means the "link" is broken. Just run: npx tauri dev (The npx prefix tells Windows: "I don't care if you don't know what tauri is, go look in the node_modules folder and find it yourself.")

* **"Access Denied" or "Os { code: 5 }" during build:** This usually means a ghost process is locking the sidecar or the build cache is stuck.
    1. Close the app and any open CMD windows.
    2. Run this command in your project root to clear the cache:
```bash
rd /s /q src-tauri\target
```
    3. Restart the app with `npm run tauri dev`.

* **"Tauri v2 Configuration Rules (CRITICAL):**
If you get a Panic or Deserialization Error mentioning unknown field, it means the configuration is out of sync. Tauri v2 is extremely picky about where settings live.

1. **Rule 1:** 
The "No-Sidecar" Config Rule Never put the word sidecar or scope inside src-tauri/tauri.conf.json. The plugins section must remain "dumb." It should only look like this:

JSON

"plugins": {
  "shell": {
    "open": true
  }
}
2. **Rule 2:** 
The "Capabilities" Brain All sidecar permissions must live in src-tauri/capabilities/default.json. If you move to a new machine, ensure this file exists and explicitly allows the binary:

JSON

{
  "permissions": [
    "core:default",
    "shell:allow-open",
    {
      "identifier": "shell:allow-execute",
      "allow": [{ "name": "binaries/velocity-engine", "sidecar": true }]
    },
    {
      "identifier": "shell:allow-spawn",
      "allow": [{ "name": "binaries/velocity-engine", "sidecar": true }]
    }
  ]
}
3. **Rule 3:** 
Permission to Exist In Tauri v2, the sidecar won't even start unless you've "registered" it in Rust. Ensure src-tauri/src/lib.rs has this line:

Rust

.plugin(tauri_plugin_shell::init())
Why this is a life-saver:
When you move to a new machine or a fresh install, npm install might pull the absolute latest version of the Tauri CLI. If your config files have "old" v1 style code in them, the new CLI will simply refuse to boot the app. These rules ensure your files stay in the "v2-only" lane.

