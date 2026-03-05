// Declare child modules.
pub mod backend;
pub mod feature_flags;
pub mod file_mgmt;
pub mod inference;
pub mod models;
pub mod prediction;
pub mod process_mgmt;
pub mod tasks;
pub mod transport;
pub mod utils;

use crate::backend::BackendState;
use crate::feature_flags::resolve_inference_engine;
use crate::file_mgmt::open_result_in_file_manager;
use crate::prediction::{
    run_fastsurfer_inference, run_fastsurfer_inference_with_progress,
};
use crate::tasks::{cancel_fastsurfer_task, shutdown_backend_for_exit, AppState};
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
#[cfg(mobile)]
use tauri::MobileEntryPoint;
use tauri::Manager;

/// The main entry point for the desktop application.
///
/// This function:
/// 1. Initializes the `BackendState`, starting the python subprocess.
/// 2. Sets up the global `AppState` for Tauri.
/// 3. Registers plugins (dialog, opener).
/// 4. Registers the invoke handlers (commands callable from the frontend).
/// 5. Runs the Tauri application loop.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let inference_engine = resolve_inference_engine();
    eprintln!(
        "[trace][startup] selected inference engine={}",
        inference_engine.as_str()
    );

    // Attempt to initialize the backend (start the python subprocess).
    let result = BackendState::new();

    let app_state = match result {
        Ok(backend) => AppState {
            backend: Some(Arc::new(backend)),
            backend_init_error: None,
            cancelled_tasks: Arc::new(Mutex::new(BTreeSet::new())),
            inference_engine,
        },
        Err(err) => {
            eprintln!("FastSurfer desktop backend initialization failed: {err}");
            AppState {
                backend: None,
                backend_init_error: Some(err),
                cancelled_tasks: Arc::new(Mutex::new(BTreeSet::new())),
                inference_engine,
            }
        }
    };

    tauri::Builder::default()
        .manage(app_state)
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_process::init())
        .invoke_handler(tauri::generate_handler![
            run_fastsurfer_inference,
            run_fastsurfer_inference_with_progress,
            cancel_fastsurfer_task,
            shutdown_backend_for_exit,
            open_result_in_file_manager
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                let app_state = app_handle.state::<AppState>();
                if let Some(backend) = app_state.backend.as_ref() {
                    if let Err(err) = backend.shutdown_for_exit() {
                        eprintln!("Failed to gracefully stop backend on app exit: {err}");
                    }
                }
                app_handle.exit(0);
            }
        });
}

#[cfg(test)]
#[path = "../testing/rust/controller_tests.rs"]
mod controller_tests;
