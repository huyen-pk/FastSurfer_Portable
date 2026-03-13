use serde::{Deserialize, Serialize};
use tauri::{Emitter, Runtime};

pub fn emit<T: Emitter<R>, R: Runtime, S: Serialize + Clone>(
    event_handler: &T,
    event_payload: S,
) {
    let _ =
        event_handler.emit("fastsurfer://inference-progress", event_payload);
}

/// Event payload for tracking the progress of an inference task.
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceProgressEvent {
    /// Unique identifier for the batch task.
    pub task_id: String,
    /// Current status of the task (e.g., "started", `item_progress`, "completed", "failed", "cancelled").
    pub status: String,
    /// Descriptive message about the current operation.
    pub message: String,
    /// Total number of items to process.
    pub total: usize,
    /// Number of items completed so far.
    pub completed: usize,
    /// Overall progress percentage (0-100).
    pub progress: u8,
    /// The path of the file currently being processed, if any.
    pub current_path: Option<String>,
    /// The path of the output generated, if appropriate for the status.
    pub output_path: Option<String>,
}
