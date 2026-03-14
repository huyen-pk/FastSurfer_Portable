use crate::backend::{
    legacy_run_fastsurfer_inference_core,
    legacy_run_fastsurfer_inference_with_progress,
};
use crate::feature_flags::InferenceEngine;
use crate::inference::entities::ProcessingRunResult;
use crate::inference::pipeline::run::{
    run_native_inference, run_native_inference_with_progress,
};
use crate::tasks::AppState;
use tauri::{AppHandle, State};

/// Legacy command for simple batch inference (no progress events).
#[tauri::command]
///
/// # Errors
/// Returns an error when native or backend inference fails, or when the
/// blocking worker task cannot be joined.
pub async fn run_fastsurfer_inference(
    app_state: State<'_, AppState>,
    file_paths: Vec<String>,
    folder_paths: Vec<String>,
) -> Result<ProcessingRunResult, String> {
    eprintln!(
        "[trace][tauri-cmd] run_fastsurfer_inference called file_paths={} folder_paths={}",
        file_paths.len(),
        folder_paths.len()
    );
    if app_state.inference_engine == InferenceEngine::RustOnnx {
        return run_native_inference(&file_paths, &folder_paths);
    }

    // legacy: load python IPC backend to handle inference
    let backend = app_state.backend.clone();
    let backend_init_error = app_state.backend_init_error.clone();

    tauri::async_runtime::spawn_blocking(move || {
        legacy_run_fastsurfer_inference_core(
            backend.as_deref(),
            backend_init_error.as_deref(),
            &file_paths,
            &folder_paths,
        )
    })
    .await
    .map_err(|e| format!("Failed to join inference task: {e}"))?
}

/// Primary command for starting an inference task with full progress tracking.
#[tauri::command]
///
/// # Errors
/// Returns an error when progress-aware native or backend inference fails, or
/// when the blocking worker task cannot be joined.
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

    // legacy: python IPC backend to handle inference
    let backend = app_state.backend.clone();
    let backend_init_error = app_state.backend_init_error.clone();
    let cancelled_tasks = app_state.cancelled_tasks.clone();

    tauri::async_runtime::spawn_blocking(move || {
        legacy_run_fastsurfer_inference_with_progress(
            &app_handle,
            backend.as_deref(),
            backend_init_error.as_deref(),
            &cancelled_tasks,
            task_id,
            &file_paths,
            &folder_paths,
        )
    })
    .await
    .map_err(|e| format!("Failed to join inference task: {e}"))?
}
