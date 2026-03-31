use crate::states::{AppState, BackendState};
use std::mem::ManuallyDrop;
use tauri::{AppHandle, Manager, State};

/// Helper command to flag a specific task ID as cancelled.
/// Also forcefully stops the backend process to interrupt current work immediately.
///
/// # Errors
/// Returns an error when the cancellation registry mutex is poisoned.
#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
pub fn cancel_fastsurfer_task(
    app_handle: AppHandle,
    app_state: State<'_, AppState>,
    task_id: &str,
) -> Result<(), String> {
    let app_state = ManuallyDrop::new(app_state);
    let cancelled_tasks = app_state.cancelled_tasks.clone();

    let mut cancelled = cancelled_tasks
        .lock()
        .map_err(|_| "Cancelled tasks mutex was poisoned".to_string())?;
    cancelled.insert(task_id.to_string());

    // Access BackendState only if it was registered at startup.
    if app_state.backend_registered {
        let backend_state = app_handle.state::<BackendState>();
        if let Some(backend) = backend_state.backend.as_ref() {
            let _ = backend.force_stop_current_process();
        }
    }

    Ok(())
}

/// Gracefully shuts down the backend process before the application exits.
///
/// # Errors
/// Returns an error when the backend shutdown sequence fails.
#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
pub fn shutdown_backend_for_exit(
    app_handle: AppHandle,
    app_state: State<'_, AppState>,
) -> Result<(), String> {
    let app_state = ManuallyDrop::new(app_state);
    if app_state.backend_registered {
        let backend_state = app_handle.state::<BackendState>();
        if let Some(backend) = backend_state.backend.as_ref() {
            backend.shutdown_for_exit()?;
        }
    }
    Ok(())
}
