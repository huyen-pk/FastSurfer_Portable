// Declare child modules.
pub mod backend;
pub mod events;
pub mod feature_flags;
pub mod file_mgmt;
pub mod inference;
pub mod mediator;
pub mod prediction;
pub mod states;
pub mod tasks;
pub mod transport;
pub mod utils;

use crate::backend::BackendManager;
use crate::feature_flags::resolve_inference_engine;
use crate::file_mgmt::open_result_in_file_manager;
use crate::prediction::run_fastsurfer_inference_with_progress;
use crate::states::{AppState, BackendState};
use crate::tasks::{cancel_fastsurfer_task, shutdown_backend_for_exit};
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use tauri::Manager;
#[cfg(mobile)]
use tauri::MobileEntryPoint;

/// The main entry point for the desktop application.
///
/// This function:
/// 1. Initializes the `BackendManager`, starting the python subprocess.
/// 2. Sets up the global `AppState` for Tauri.
/// 3. Registers plugins (dialog, opener).
/// 4. Registers the invoke handlers (commands callable from the frontend).
/// 5. Runs the Tauri application loop.
///
/// # Panics
/// Panics when Tauri fails to build the desktop application context.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let inference_engine = resolve_inference_engine();
    eprintln!(
        "[trace][startup] selected inference engine={}",
        inference_engine.as_str()
    );

    // Initialize non-backend app state.
    let backend_selected =
        inference_engine == crate::feature_flags::InferenceEngine::PythonIpc;
    let app_state = AppState {
        cancelled_tasks: Arc::new(Mutex::new(BTreeSet::new())),
        inference_engine,
        backend_registered: backend_selected,
    };

    // Conditionally build the Tauri builder with BackendState managed only when selected.
    let builder = if backend_selected {
        let backend_state = match BackendManager::new() {
            Ok(backend) => BackendState {
                backend: Some(Arc::new(backend)),
                backend_init_error: None,
            },
            Err(err) => {
                eprintln!(
                    "FastSurfer desktop backend initialization failed: {err}"
                );
                BackendState {
                    backend: None,
                    backend_init_error: Some(err),
                }
            }
        };
        tauri::Builder::default()
            .manage(app_state)
            .manage(backend_state)
    } else {
        tauri::Builder::default().manage(app_state)
    };

    builder
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_process::init())
        .invoke_handler(tauri::generate_handler![
            run_fastsurfer_inference_with_progress,
            cancel_fastsurfer_task,
            shutdown_backend_for_exit,
            open_result_in_file_manager
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                // Gracefully stop backend if present in registered BackendState.
                let backend_state = app_handle.state::<BackendState>();
                if let Some(backend) = backend_state.backend.as_ref() {
                    match backend.shutdown_for_exit() {
                        Ok(()) => {}
                        Err(err) => eprintln!(
                            "Failed to gracefully stop backend on app exit: {err}"
                        ),
                    }
                }
                app_handle.exit(0);
            }
        });
}

#[cfg(test)]
#[path = "../testing/rust/controller_tests.rs"]
mod controller_tests;

#[cfg(test)]
#[path = "../testing/rust/transport_tests.rs"]
mod transport_tests;

#[cfg(test)]
#[path = "../testing/rust/mediator_tests.rs"]
mod mediator_tests;

#[cfg(test)]
#[path = "../testing/rust/unit_tests.rs"]
mod unit_tests;
