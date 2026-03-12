use crate::backend::BackendState;
use crate::feature_flags::InferenceEngine;
use std::collections::BTreeSet;
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex};
use tauri::State;

/// Global application state managed by Tauri.
/// Holds the persistent connection to the backend process.
pub struct AppState {
    /// The shared backend state wrapper. Optional because initialization might fail.
    pub backend: Option<Arc<BackendState>>,
    /// Error message captured if backend initialization failed at startup.
    pub backend_init_error: Option<String>,
    /// Set of task IDs that have been requested to cancel.
    pub cancelled_tasks: Arc<Mutex<BTreeSet<String>>>,
    /// Active inference engine selected for this app run.
    pub inference_engine: InferenceEngine,
}

/// Helper command to flag a specific task ID as cancelled.
/// Also forcefully stops the backend process to interrupt current work immediately.
///
/// # Errors
/// Returns an error when the cancellation registry mutex is poisoned.
#[tauri::command]
pub fn cancel_fastsurfer_task(
    app_state: State<'_, AppState>,
    task_id: &str,
) -> Result<(), String> {
    let app_state = ManuallyDrop::new(app_state);
    let cancelled_tasks = app_state.cancelled_tasks.clone();
    let backend = app_state.backend.clone();

    let mut cancelled = cancelled_tasks
        .lock()
        .map_err(|_| "Cancelled tasks mutex was poisoned".to_string())?;
    cancelled.insert(task_id.to_string());

    if let Some(backend) = backend.as_ref() {
        let _ = backend.force_stop_current_process();
    }

    Ok(())
}

/// Gracefully shuts down the backend process before the application exits.
///
/// # Errors
/// Returns an error when the backend shutdown sequence fails.
#[tauri::command]
pub fn shutdown_backend_for_exit(
    app_state: State<'_, AppState>,
) -> Result<(), String> {
    let app_state = ManuallyDrop::new(app_state);
    let backend = app_state.backend.clone();

    if let Some(backend) = backend.as_ref() {
        backend.shutdown_for_exit()?;
    }
    Ok(())
}
