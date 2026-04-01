use crate::backend::BackendManager;
use crate::events::{self, InferenceProgressEvent};
use crate::feature_flags::InferenceEngine;
use crate::inference::entities::ProcessingRunResult;
use crate::inference::pipeline::run::run_native_inference_with_progress;
use crate::mediator::{MediatorSubscription, mediator};
use crate::states::AppState;
use tauri::{AppHandle, Manager, State};

/// Primary command for starting an inference task with full progress tracking.
///
/// # Errors
/// Returns an error when native or Python-backed inference fails, or when the
/// blocking worker task cannot be joined.
#[tauri::command]
pub async fn run_fastsurfer_inference_with_progress(
    app_handle: AppHandle,
    app_state: State<'_, AppState>,
    task_id: String,
    file_paths: Vec<String>,
    folder_paths: Vec<String>,
) -> Result<ProcessingRunResult, String> {
    eprintln!(
        "[trace][tauri-cmd] run_fastsurfer_inference_with_progress called task_id={} file_paths={} folder_paths={}",
        task_id,
        file_paths.len(),
        folder_paths.len()
    );

    if app_state.inference_engine == InferenceEngine::RustOnnx {
        let cancelled_tasks = app_state.cancelled_tasks.clone();
        return tauri::async_runtime::spawn_blocking(move || {
            run_native_inference_with_progress(
                &app_handle,
                &cancelled_tasks,
                &task_id,
                &file_paths,
                &folder_paths,
            )
        })
        .await
        .map_err(|e| format!("Failed to join native inference task: {e}"))?;
    }

    let emit_task_id = task_id.clone();
    let emit_handle = app_handle.clone();
    let observer_id = mediator().subscribe::<InferenceProgressEvent, _>(
        move |event: &InferenceProgressEvent| {
            if event.task_id == emit_task_id {
                events::emit(&emit_handle, event.clone());
            }
        },
    );
    let _subscription =
        MediatorSubscription::new::<InferenceProgressEvent>(observer_id);

    let cancelled_tasks = app_state.cancelled_tasks.clone();
    let backend_task_id = task_id.clone();

    // If backend was registered at startup, retrieve it; otherwise call require_available with a clear error.
    let (backend_opt, backend_init_error_opt) = if app_state.backend_registered
    {
        let backend_state = app_handle.state::<crate::states::BackendState>();
        (
            backend_state.backend.clone(),
            backend_state.backend_init_error.clone(),
        )
    } else {
        (None, Some("Backend not registered in this runtime because Python IPC engine is disabled".to_string()))
    };

    tauri::async_runtime::spawn_blocking(move || {
        let backend = BackendManager::require_available(
            backend_opt.as_deref(),
            backend_init_error_opt.as_deref(),
        )?;

        backend.predict_batch(
            &file_paths,
            &folder_paths,
            Some(&cancelled_tasks),
            Some(&backend_task_id),
        )
    })
    .await
    .map_err(|e| format!("Failed to join inference task: {e}"))?
}
