# Development Commands

## 1. Run in "Headless" Mode (Browser)
Use this for quick UI/CSS changes or if you are working strictly inside the Docker container.
* **Command:** `npm run dev:headless`
* **View at:** `http://localhost:5173`
* **Note:** Native Tauri features (Python backend, window resizing, system tray) **will not work** here.

## 2. Run the Real Desktop App
Use this to test the actual application window, Python/Rust backend, and native buttons.
* **Command:** `npm run tauri dev`
* **Result:** This launches the actual application window on your desktop.

## 3. Build for Production
Creates the final `.exe` (Windows), `.dmg` (Mac), or `.deb` (Linux) installer.
* **Command:** `npm run tauri build`
* **Output:** Check `src-tauri/target/release/bundle/`

## 4. Troubleshooting
* **Clean Build Assets:** If things look weird, delete the `src-tauri/target` folder and run `npm run tauri dev` again.
* **Check Rust Errors:** Run `npm run tauri info` to see environment details.