use crate::backend::BackendState;
use crate::feature_flags::InferenceEngine;
use crate::inference::pipeline::run::{
    run_native_inference, run_native_inference_with_progress,
};
use crate::inference::entities::{
    InferenceOutput, InferenceProgressEvent, ProcessingRunResult,
};
use crate::tasks::AppState;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, State};

/// Helper to emit internal progress events to the frontend.
fn emit_progress_event(app_handle: &AppHandle, event: InferenceProgressEvent) {
    let _ = app_handle.emit("fastsurfer://inference-progress", event);
}

fn ensure_backend<'a>(
    backend: Option<&'a BackendState>,
    backend_init_error: Option<&'a str>,
) -> Result<&'a BackendState, String> {
    backend.ok_or_else(|| {
        let detail = backend_init_error
            .unwrap_or("unknown backend initialization error");
        format!("Backend is unavailable in this desktop runtime: {detail}")
    })
}

fn emit_started_event(
    app_handle: &AppHandle,
    task_id: &str,
    start_ack_message: &str,
    total: usize,
) {
    emit_progress_event(
        app_handle,
        InferenceProgressEvent {
            task_id: task_id.to_string(),
            status: "started".to_string(),
            message: start_ack_message.to_string(),
            total,
            completed: 0,
            progress: 0,
            current_path: None,
            output_path: None,
        },
    );
}

fn emit_item_progress_event(
    app_handle: &AppHandle,
    task_id: &str,
    index: usize,
    total: usize,
    progress: u8,
    input_path: &str,
    message: String,
) {
    emit_progress_event(
        app_handle,
        InferenceProgressEvent {
            task_id: task_id.to_string(),
            status: "item_progress".to_string(),
            message,
            total,
            completed: index,
            progress,
            current_path: Some(input_path.to_string()),
            output_path: None,
        },
    );
}

fn emit_item_completed_event(
    app_handle: &AppHandle,
    task_id: &str,
    index: usize,
    total: usize,
    output_path: &str,
) {
    let completed = index + 1;
    let progress = if total == 0 {
        100
    } else {
        u8::try_from(((completed * 100) / total).min(100)).unwrap_or(100)
    };

    emit_progress_event(
        app_handle,
        InferenceProgressEvent {
            task_id: task_id.to_string(),
            status: "item_completed".to_string(),
            message: format!("Processed {completed}/{total}"),
            total,
            completed,
            progress,
            current_path: None,
            output_path: Some(output_path.to_string()),
        },
    );
}

fn emit_cancelled_event(
    app_handle: &AppHandle,
    task_id: &str,
    total: usize,
    completed: usize,
) {
    let progress = if total == 0 {
        0
    } else {
        u8::try_from(((completed * 100) / total).min(100)).unwrap_or(100)
    };

    emit_progress_event(
        app_handle,
        InferenceProgressEvent {
            task_id: task_id.to_string(),
            status: "cancelled".to_string(),
            message: "Task cancelled by user.".to_string(),
            total,
            completed,
            progress,
            current_path: None,
            output_path: None,
        },
    );
}

fn emit_failed_event(
    app_handle: &AppHandle,
    task_id: String,
    total: usize,
    completed: usize,
    message: String,
) {
    let progress = if total == 0 {
        0
    } else {
        u8::try_from(((completed * 100) / total).min(100)).unwrap_or(100)
    };

    emit_progress_event(
        app_handle,
        InferenceProgressEvent {
            task_id,
            status: "failed".to_string(),
            message,
            total,
            completed,
            progress,
            current_path: None,
            output_path: None,
        },
    );
}

fn add_result_directory(set: &mut BTreeSet<String>, path: &str) {
    if let Some(parent) = Path::new(path).parent() {
        set.insert(parent.to_string_lossy().to_string());
    }
}

fn finalize_ack_message(start_ack: &str, dirs: &[String]) -> String {
    if dirs.is_empty() {
        start_ack.to_string()
    } else {
        format!("{} Results directory: {}", start_ack, dirs.join(", "))
    }
}

fn process_requested_paths(
    backend: &BackendState,
    requested_paths: &[String],
    cancelled_tasks: &Arc<Mutex<BTreeSet<String>>>,
    app_handle: &AppHandle,
    task_id: &str,
) -> Result<(Vec<InferenceOutput>, BTreeSet<String>), String> {
    let total = requested_paths.len();
    let mut results: Vec<InferenceOutput> = Vec::with_capacity(total);
    let mut result_directories: BTreeSet<String> = BTreeSet::new();

    for (index, input_path) in requested_paths.iter().enumerate() {
        let is_cancelled = cancelled_tasks
            .lock()
            .map_err(|_| "Cancelled tasks mutex was poisoned".to_string())?
            .contains(task_id);

        if is_cancelled {
            emit_cancelled_event(app_handle, task_id, total, index);
            if let Ok(mut cancelled) = cancelled_tasks.lock() {
                cancelled.remove(task_id);
            }
            return Err("Task cancelled by user.".to_string());
        }

        let mut intra_file_progress =
            |file_progress: usize, file_message: String| {
                let safe_file_progress = file_progress.min(100);
                let overall_progress = if total == 0 {
                    u8::try_from(safe_file_progress).unwrap_or(100)
                } else {
                    u8::try_from(
                        (((index * 100) + safe_file_progress) / total).min(100),
                    )
                    .unwrap_or(100)
                };

                emit_item_progress_event(
                    app_handle,
                    task_id,
                    index,
                    total,
                    overall_progress,
                    input_path,
                    file_message,
                );
            };

        match backend.predict_single_path(
            input_path,
            task_id,
            Some(&mut intra_file_progress),
        ) {
            Ok(prediction) => {
                add_result_directory(
                    &mut result_directories,
                    &prediction.output_path,
                );
                emit_item_completed_event(
                    app_handle,
                    task_id,
                    index,
                    total,
                    &prediction.output_path,
                );
                results.push(prediction);
            }
            Err(error) => {
                let was_cancelled = cancelled_tasks
                    .lock()
                    .map_err(|_| {
                        "Cancelled tasks mutex was poisoned".to_string()
                    })?
                    .contains(task_id);

                if was_cancelled {
                    emit_cancelled_event(app_handle, task_id, total, index);
                    if let Ok(mut cancelled) = cancelled_tasks.lock() {
                        cancelled.remove(task_id);
                    }
                    let _ = backend.restart_backend_process();
                    return Err("Task cancelled by user.".to_string());
                }

                emit_failed_event(
                    app_handle,
                    task_id.to_string(),
                    total,
                    index,
                    error.clone(),
                );
                return Err(error);
            }
        }
    }

    Ok((results, result_directories))
}

/// Helper that orchestrates the batch prediction flow by calling the backend.
pub(crate) fn run_fastsurfer_inference_with_backend(
    backend: &BackendState,
    file_paths: &[String],
    folder_paths: &[String],
) -> Result<ProcessingRunResult, String> {
    eprintln!(
        "[trace][prediction] run_fastsurfer_inference_with_backend start file_paths={} folder_paths={}",
        file_paths.len(),
        folder_paths.len()
    );
    let (start_ack_message, start_requested_paths) =
        backend.start_predict_batch(file_paths, folder_paths)?;

    eprintln!(
        "[trace][prediction] start_predict_batch ack='{}' requested_paths={}",
        start_ack_message,
        start_requested_paths.len()
    );

    backend.predict_batch(
        file_paths,
        folder_paths,
        &start_ack_message,
        &start_requested_paths,
    )
}

/// Wrapper for inference that checks for backend availability first.
pub(crate) fn run_fastsurfer_inference_with_app_state(
    backend: Option<&BackendState>,
    backend_init_error: Option<&str>,
    file_paths: &[String],
    folder_paths: &[String],
) -> Result<ProcessingRunResult, String> {
    let backend = backend.ok_or_else(|| {
        let detail = backend_init_error
            .unwrap_or("unknown backend initialization error");
        format!("Backend is unavailable in this desktop runtime: {detail}")
    })?;

    run_fastsurfer_inference_with_backend(backend, file_paths, folder_paths)
}

/// Advanced inference runner that supports granular progress tracking and cancellation.
///
/// Iterates through the requested items one by one, updating the UI via events.
/// Handles user-initiated cancellation checks between items.
///
/// # Errors
/// Returns an error when backend inference startup, per-item processing, or
/// event-driven cancellation handling fails.
pub fn run_fastsurfer_inference_with_progress_with_app_state(
    app_handle: &AppHandle,
    backend: Option<&BackendState>,
    backend_init_error: Option<&str>,
    cancelled_tasks: &Arc<Mutex<BTreeSet<String>>>,
    task_id: String,
    file_paths: &[String],
    folder_paths: &[String],
) -> Result<ProcessingRunResult, String> {
    eprintln!(
        "[trace][prediction] run_with_progress task_id={} file_paths={} folder_paths={}",
        task_id,
        file_paths.len(),
        folder_paths.len()
    );
    let backend = ensure_backend(backend, backend_init_error)?;

    let (start_ack_message, requested_paths): (String, Vec<String>) =
        backend.start_predict_batch(file_paths, folder_paths)?;
    let total = requested_paths.len();

    emit_started_event(app_handle, &task_id, &start_ack_message, total);

    let (results, result_directories) = process_requested_paths(
        backend,
        &requested_paths,
        cancelled_tasks,
        app_handle,
        &task_id,
    )?;

    let result_directories =
        result_directories.into_iter().collect::<Vec<String>>();
    let ack_message =
        finalize_ack_message(&start_ack_message, &result_directories);

    emit_progress_event(
        app_handle,
        InferenceProgressEvent {
            task_id,
            status: "completed".to_string(),
            message: ack_message.clone(),
            total,
            completed: total,
            progress: 100,
            current_path: None,
            output_path: None,
        },
    );

    Ok(ProcessingRunResult {
        ack_message,
        requested_paths,
        result_directories,
        qc_summary: None,
        results,
    })
}

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

    let backend = app_state.backend.clone();
    let backend_init_error = app_state.backend_init_error.clone();

    tauri::async_runtime::spawn_blocking(move || {
        run_fastsurfer_inference_with_app_state(
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

    let backend = app_state.backend.clone();
    let backend_init_error = app_state.backend_init_error.clone();
    let cancelled_tasks = app_state.cancelled_tasks.clone();

    tauri::async_runtime::spawn_blocking(move || {
        run_fastsurfer_inference_with_progress_with_app_state(
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
