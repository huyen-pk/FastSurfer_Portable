/// Supported inference execution engines for desktop runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InferenceEngine {
    PythonIpc,
    RustOnnx,
}

impl InferenceEngine {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PythonIpc => "python-ipc",
            Self::RustOnnx => "rust-onnx",
        }
    }
}

/// Resolve the desired inference engine from `FASTSURFER_INFERENCE_ENGINE`.
///
/// Accepted values:
/// - `python`, `python-ipc` (default)
/// - `rust`, `rust-onnx`
#[must_use]
pub fn resolve_inference_engine() -> InferenceEngine {
    let raw = std::env::var("FASTSURFER_INFERENCE_ENGINE")
        .unwrap_or_else(|_| "python-ipc".to_string())
        .trim()
        .to_ascii_lowercase();

    match raw.as_str() {
        "rust" | "rust-onnx" => InferenceEngine::RustOnnx,
        _ => InferenceEngine::PythonIpc,
    }
}
