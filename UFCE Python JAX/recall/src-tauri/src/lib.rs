use tauri::{Manager, State};
use tauri_plugin_shell::{ShellExt, process::CommandEvent};
use std::sync::Mutex;

struct AppState {
    api_port: Mutex<u16>,
}

#[tauri::command]
fn get_api_port(state: State<AppState>) -> u16 {
    *state.api_port.lock().unwrap()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .manage(AppState { api_port: Mutex::new(0) })
        .invoke_handler(tauri::generate_handler![get_api_port])
        .setup(|app| {
            let handle = app.handle().clone();
            
            tauri::async_runtime::spawn(async move {
                let (mut rx, mut _child) = handle.shell()
                    .sidecar("velocity-engine")
                    .expect("Failed to setup sidecar")
                    .spawn()
                    .expect("Failed to spawn sidecar");

                while let Some(event) = rx.recv().await {
                    match event {
                        // PRINT EVERYTHING PYTHON SAYS
                        CommandEvent::Stdout(line_bytes) => {
                            let line = String::from_utf8_lossy(&line_bytes);
                            println!("[Python]: {}", line); 
                            
                            if line.contains("VELOCITY_PORT:") {
                                let port_str = line.trim().replace("VELOCITY_PORT:", "").trim().to_string();
                                if let Ok(port) = port_str.parse::<u16>() {
                                    println!("✅ [Rust] Connected to Port: {}", port);
                                    let state = handle.state::<AppState>();
                                    *state.api_port.lock().unwrap() = port;
                                }
                            }
                        }
                        // PRINT CRASH ERRORS
                        CommandEvent::Stderr(line_bytes) => {
                             let line = String::from_utf8_lossy(&line_bytes);
                             eprintln!("[Python Error]: {}", line);
                        }
                        _ => {}
                    }
                }
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}