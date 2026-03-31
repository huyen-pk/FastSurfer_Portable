use crate::backend::BackendManager;
use crate::feature_flags::InferenceEngine;
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

/// Global application state managed by Tauri (non-backend specific).
pub struct AppState {
    /// Set of task IDs that have been requested to cancel.
    pub cancelled_tasks: Arc<Mutex<BTreeSet<String>>>,
    /// Active inference engine selected for this app run.
    pub inference_engine: InferenceEngine,
    /// Whether a `BackendState` was registered with Tauri state (feature enabled).
    pub backend_registered: bool,
}

/// Optional backend-managed state registered only when the Python IPC
/// backend is enabled at startup.
pub struct BackendState {
    /// The shared backend manager instance, if initialization succeeded.
    pub backend: Option<Arc<BackendManager>>,
    /// Error message captured if backend initialization failed at startup.
    pub backend_init_error: Option<String>,
}
