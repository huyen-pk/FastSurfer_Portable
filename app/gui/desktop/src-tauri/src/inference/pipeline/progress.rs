use crate::events::InferenceProgressEvent;
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use tauri::Emitter;

pub(crate) const NATIVE_CANCELLED_MESSAGE: &str = "Task cancelled by user.";

pub(crate) fn cancelled_error() -> String {
    NATIVE_CANCELLED_MESSAGE.to_string()
}

pub(crate) fn is_task_cancelled(
    cancelled_tasks: &Arc<Mutex<BTreeSet<String>>>,
    task_id: &str,
) -> Result<bool, String> {
    cancelled_tasks
        .lock()
        .map_err(|_| "Cancelled tasks mutex was poisoned".to_string())
        .map(|set| set.contains(task_id))
}

pub(crate) struct ProgressUpdate<'a> {
    pub status: &'a str,
    pub message: String,
    pub total: usize,
    pub completed: usize,
    pub progress: u8,
    pub current_path: Option<String>,
    pub output_path: Option<String>,
}

pub(crate) fn progress_for_completed(
    total: usize,
    completed: usize,
    fallback: u8,
) -> u8 {
    if total == 0 {
        fallback
    } else {
        let value = ((completed * 100) / total).min(100);
        u8::try_from(value).unwrap_or(100)
    }
}

pub(crate) fn remove_cancelled_task(
    cancelled_tasks: &Arc<Mutex<BTreeSet<String>>>,
    task_id: &str,
) {
    if let Ok(mut cancelled) = cancelled_tasks.lock() {
        cancelled.remove(task_id);
    }
}

pub(crate) fn emit_single_input_status<R: tauri::Runtime>(
    ctx: &SingleInputContext<'_, R>,
    input_path: &str,
    status: &'static str,
    message: String,
    progress: u8,
) {
    emit_inference_progress(
        ctx.app_handle,
        ctx.task_id,
        ProgressUpdate {
            status,
            message,
            total: ctx.total,
            completed: ctx.index,
            progress,
            current_path: Some(input_path.to_string()),
            output_path: None,
        },
    );
}

pub(crate) fn emit_cancelled_single_input<R: tauri::Runtime>(
    ctx: &SingleInputContext<'_, R>,
    input_path: &str,
) {
    emit_single_input_status(
        ctx,
        input_path,
        "cancelled",
        NATIVE_CANCELLED_MESSAGE.to_string(),
        progress_for_completed(ctx.total, ctx.index, 0),
    );
}

pub(crate) fn emit_failed_single_input<R: tauri::Runtime>(
    ctx: &SingleInputContext<'_, R>,
    input_path: &str,
    message: String,
) {
    emit_single_input_status(
        ctx,
        input_path,
        "failed",
        message,
        progress_for_completed(ctx.total, ctx.index, 0),
    );
}

pub(crate) fn emit_inference_progress<R: tauri::Runtime>(
    app_handle: &tauri::AppHandle<R>,
    task_id: &str,
    update: ProgressUpdate<'_>,
) {
    let ProgressUpdate {
        status,
        message,
        total,
        completed,
        progress,
        current_path,
        output_path,
    } = update;

    let _ = app_handle.emit(
        "fastsurfer://inference-progress",
        InferenceProgressEvent {
            task_id: task_id.to_string(),
            status: status.to_string(),
            message,
            total,
            completed,
            progress,
            current_path,
            output_path,
        },
    );
}

pub(crate) struct SingleInputContext<'a, R: tauri::Runtime> {
    pub app_handle: &'a tauri::AppHandle<R>,
    pub cancelled_tasks: &'a Arc<Mutex<BTreeSet<String>>>,
    pub task_id: &'a str,
    pub sessions: &'a crate::inference::runtime::NativeOnnxSessions,
    pub lut_ids: &'a [u16],
    pub index: usize,
    pub total: usize,
}
// `load_runtime_dependencies` and `process_single_input` intentionally
// live outside of this module to avoid circular dependencies —
// `load_runtime_dependencies` is implemented in `runtime.rs` (no UI emissions)
// and `process_single_input` is implemented in the higher-level `pipeline.rs`.
