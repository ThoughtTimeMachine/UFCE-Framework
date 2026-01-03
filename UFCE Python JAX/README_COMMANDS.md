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

---

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

---

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

---

## 5. Troubleshooting

* **"Linker link.exe not found":** You missed **Step B** (C++ Build Tools). Install them and restart your computer.
* **"Tauri is not recognized":** You missed **Step C** (`npm install`). Run it inside the `recall` folder.
* **Clean Build:** If the app behaves strangely, delete the `src-tauri/target` folder and run `npm run tauri dev` again.

* **"Sidecar not found" or "Permission Denied":** 1. Check that the sidecar is named exactly `velocity-engine-x86_64-pc-windows-msvc.exe` inside `src-tauri/binaries/`.
    2. Ensure `src-tauri/src/lib.rs` includes `.plugin(tauri_plugin_shell::init())`.
    3. Verify that your `src-tauri/capabilities/default.json` (or your main config) explicitly allows the sidecar.