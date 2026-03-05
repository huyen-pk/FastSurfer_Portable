use crate::backend::BackendState;
use crate::feature_flags::InferenceEngine;
use crate::inference::{run_native_inference, run_native_inference_with_progress};
use crate::models::{InferenceOutput, InferenceProgressEvent, ProcessingRunResult};
use crate::tasks::AppState;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, State};

/// Helper to emit internal progress events to the frontend.
fn emit_progress_event(app_handle: &AppHandle, event: InferenceProgressEvent) {
    let _ = app_handle.emit("fastsurfer://inference-progress", event);
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
    file_paths: Vec<String>,
    folder_paths: Vec<String>,
) -> Result<ProcessingRunResult, String> {
    let backend = backend.ok_or_else(|| {
        let detail = backend_init_error.unwrap_or("unknown backend initialization error");
        format!("Backend is unavailable in this desktop runtime: {detail}")
    })?;

    run_fastsurfer_inference_with_backend(backend, &file_paths, &folder_paths)
}

/// Advanced inference runner that supports granular progress tracking and cancellation.
///
/// Iterates through the requested items one by one, updating the UI via events.
/// Handles user-initiated cancellation checks between items.
pub fn run_fastsurfer_inference_with_progress_with_app_state(
    app_handle: &AppHandle,
    backend: Option<&BackendState>,
    backend_init_error: Option<&str>,
    cancelled_tasks: &Arc<Mutex<BTreeSet<String>>>,
    task_id: String,
    file_paths: Vec<String>,
    folder_paths: Vec<String>,
) -> Result<ProcessingRunResult, String> {
    eprintln!(
        "[trace][prediction] run_with_progress task_id={} file_paths={} folder_paths={}",
        task_id,
        file_paths.len(),
        folder_paths.len()
    );

    let backend = backend.ok_or_else(|| {
        let detail = backend_init_error.unwrap_or("unknown backend initialization error");
        format!("Backend is unavailable in this desktop runtime: {detail}")
    })?;

    let (start_ack_message, requested_paths) =
        backend.start_predict_batch(&file_paths, &folder_paths)?;
    let total = requested_paths.len();

    emit_progress_event(
        app_handle,
        InferenceProgressEvent {
            task_id: task_id.clone(),
            status: "started".to_string(),
            message: start_ack_message.clone(),
            total,
            completed: 0,
            progress: 0,
            current_path: None,
            output_path: None,
        },
    );

    let mut results: Vec<InferenceOutput> = Vec::with_capacity(total);
    let mut result_directories: BTreeSet<String> = BTreeSet::new();

    for (index, input_path) in requested_paths.iter().enumerate() {
        let is_cancelled = cancelled_tasks
            .lock()
            .map_err(|_| "Cancelled tasks mutex was poisoned".to_string())?
            .contains(&task_id);
        if is_cancelled {
            emit_progress_event(
                app_handle,
                InferenceProgressEvent {
                    task_id: task_id.clone(),
                    status: "cancelled".to_string(),
                    message: "Task cancelled by user.".to_string(),
                    total,
                    completed: index,
                    progress: if total == 0 {
                        0
                    } else {
                        (((index * 100) / total).min(100)) as u8
                    },
                    current_path: None,
                    output_path: None,
                },
            );

            if let Ok(mut cancelled) = cancelled_tasks.lock() {
                cancelled.remove(&task_id);
            }

            return Err("Task cancelled by user.".to_string());
        }

        let mut intra_file_progress = |file_progress: usize, file_message: String| {
            let safe_file_progress = file_progress.min(100);
            let overall_progress = if total == 0 {
                safe_file_progress as u8
            } else {
                ((((index * 100) + safe_file_progress) / total).min(100)) as u8
            };

            emit_progress_event(
                app_handle,
                InferenceProgressEvent {
                    task_id: task_id.clone(),
                    status: "item_progress".to_string(),
                    message: file_message,
                    total,
                    completed: index,
                    progress: overall_progress,
                    current_path: Some(input_path.clone()),
                    output_path: None,
                },
            );
        };

        match backend.predict_single_path(input_path, &task_id, Some(&mut intra_file_progress)) {
            Ok(prediction) => {
                if let Some(parent) = Path::new(&prediction.output_path).parent() {
                    result_directories
                        .insert(parent.to_string_lossy().to_string());
                }

                let completed = index + 1;
                let progress = if total == 0 {
                    100
                } else {
                    (((completed * 100) / total).min(100)) as u8
                };

                emit_progress_event(
                    app_handle,
                    InferenceProgressEvent {
                        task_id: task_id.clone(),
                        status: "item_completed".to_string(),
                        message: format!("Processed {completed}/{total}"),
                        total,
                        completed,
                        progress,
                        current_path: Some(input_path.clone()),
                        output_path: Some(prediction.output_path.clone()),
                    },
                );

                results.push(prediction);
            }
            Err(error) => {
                let was_cancelled = cancelled_tasks
                    .lock()
                    .map_err(|_| "Cancelled tasks mutex was poisoned".to_string())?
                    .contains(&task_id);

                if was_cancelled {
                    emit_progress_event(
                        app_handle,
                        InferenceProgressEvent {
                            task_id: task_id.clone(),
                            status: "cancelled".to_string(),
                            message: "Task cancelled by user.".to_string(),
                            total,
                            completed: index,
                            progress: if total == 0 {
                                0
                            } else {
                                (((index * 100) / total).min(100)) as u8
                            },
                            current_path: Some(input_path.clone()),
                            output_path: None,
                        },
                    );

                    if let Ok(mut cancelled) = cancelled_tasks.lock() {
                        cancelled.remove(&task_id);
                    }

                    let _ = backend.restart_backend_process();
                    return Err("Task cancelled by user.".to_string());
                }

                let completed = index;
                let progress = if total == 0 {
                    0
                } else {
                    (((completed * 100) / total).min(100)) as u8
                };

                emit_progress_event(
                    app_handle,
                    InferenceProgressEvent {
                        task_id,
                        status: "failed".to_string(),
                        message: error.clone(),
                        total,
                        completed,
                        progress,
                        current_path: Some(input_path.clone()),
                        output_path: None,
                    },
                );
                return Err(error);
            }
        }
    }

    let result_directories = result_directories.into_iter().collect::<Vec<String>>();
    let ack_message = if result_directories.is_empty() {
        start_ack_message
    } else {
        format!(
            "{} Results directory: {}",
            start_ack_message,
            result_directories.join(", ")
        )
    };

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
            file_paths,
            folder_paths,
        )
    })
    .await
    .map_err(|e| format!("Failed to join inference task: {e}"))?
}

/// Primary command for starting an inference task with full progress tracking.
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
        return run_native_inference_with_progress(
            &app_handle,
            &task_id,
            &file_paths,
            &folder_paths,
        );
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
            file_paths,
            folder_paths,
        )
    })
    .await
    .map_err(|e| format!("Failed to join inference task: {e}"))?
}
