use serde::{Deserialize, Serialize};

/// Additional output artifact paths produced from a segmentation result.
#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct InferenceArtifacts {
    /// Optional brainmask file path.
    pub brainmask_path: Option<String>,
    /// Optional aseg file path.
    pub aseg_path: Option<String>,
}

/// Quality-control metadata produced during inference.
#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct InferenceQc {
    /// Whether volume-based QC passed for this result.
    pub passed: Option<bool>,
    /// Optional QC details.
    pub message: Option<String>,
}

/// Represents the output of a single inference run on an input file.
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct InferenceOutput {
    /// The original path of the input file (MRI).
    pub input_path: String,
    /// The path where the output was saved.
    pub output_path: String,
    /// The specific filename of the output.
    pub output_filename: String,
    /// A raw string result status from the backend.
    pub run_result: String,
    /// Optional artifact outputs generated alongside segmentation.
    pub artifacts: Option<InferenceArtifacts>,
    /// Optional quality-control metadata.
    pub qc: Option<InferenceQc>,
}

/// Represents the aggregated results of a batch processing run.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessingRunResult {
    /// An acknowledgment message from the backend.
    pub ack_message: String,
    /// The list of paths that triggered processing.
    pub requested_paths: Vec<String>,
    /// The directories where results were stored.
    pub result_directories: Vec<String>,
    /// Optional batch-level QC summary.
    pub qc_summary: Option<String>,
    /// The individual results for each processed file.
    pub results: Vec<InferenceOutput>,
}

/// Event payload for tracking the progress of an inference task.
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceProgressEvent {
    /// Unique identifier for the batch task.
    pub task_id: String,
    /// Current status of the task (e.g., "started", "item_progress", "completed", "failed", "cancelled").
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
