use tauri::{Manager, State};
use tauri_plugin_shell::{ShellExt, process::CommandEvent};
use std::sync::Mutex;

// 1. Global State for the Dynamic Port
struct AppState {
    api_port: Mutex<u16>,
}

// 2. Command for Frontend to ask "What port do I use?"
#[tauri::command]
fn get_api_port(state: State<AppState>) -> u16 {
    *state.api_port.lock().unwrap()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(AppState { api_port: Mutex::new(0) }) // Init with 0
        .invoke_handler(tauri::generate_handler![get_api_port])
        .setup(|app| {
            let handle = app.handle().clone();
            
            // 3. Spawn Sidecar asynchronously
            tauri::async_runtime::spawn(async move {
                let (mut rx, mut _child) = handle.shell()
                    .sidecar("velocity-engine")
                    .expect("Failed to setup sidecar")
                    .spawn()
                    .expect("Failed to spawn sidecar");

                // 4. Listen for "PORT: 12345"
                while let Some(event) = rx.recv().await {
                    if let CommandEvent::Stdout(line_bytes) = event {
                        let line = String::from_utf8_lossy(&line_bytes);
                        if line.contains("PORT:") {
                            let port_str = line.trim().replace("PORT:", "").trim().to_string();
                            if let Ok(port) = port_str.parse::<u16>() {
                                println!("✅ Connected to Engine on Port: {}", port);
                                let state = handle.state::<AppState>();
                                *state.api_port.lock().unwrap() = port;
                            }
                        }
                    }
                }
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}