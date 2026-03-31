use crate::inference::entities::{
    InferenceArtifacts, InferenceOutput, InferenceQc,
};
use serde_json::Value;
use std::path::Path;

/// Extracts the backend `result` field from a backend JSON response envelope.
///
/// # Errors
/// Returns an error when the backend reported an error or omitted the expected
/// result field.
pub fn extract_backend_result(
    method: &str,
    response: &Value,
) -> Result<Value, String> {
    if !response.get("ok").and_then(Value::as_bool).unwrap_or(false) {
        let error_msg = response
            .get("error")
            .and_then(|err| err.get("message"))
            .and_then(Value::as_str)
            .unwrap_or("Unknown backend error");
        return Err(format!("Backend {method} failed: {error_msg}"));
    }

    response
        .get("result")
        .cloned()
        .ok_or_else(|| "Missing result field in IPC response".to_string())
}

fn backend_payload_detail(response: &Value) -> String {
    response
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| response.get("error").and_then(Value::as_str))
        .or_else(|| {
            response
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
        })
        .map_or_else(|| response.to_string(), ToString::to_string)
}

/// Parses the validated input paths returned by the backend.
///
/// # Errors
/// Returns an error when the backend response omits `requested_paths` or
/// provides an empty list.
pub fn parse_requested_paths(response: &Value) -> Result<Vec<String>, String> {
    let requested_paths = response
        .get("requested_paths")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            format!(
                "Missing requested_paths in IPC response: {}",
                backend_payload_detail(response)
            )
        })
        .map(|paths| {
            paths
                .iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect::<Vec<String>>()
        })?;

    if requested_paths.is_empty() {
        return Err(format!(
            "Backend returned no requested paths: {}",
            backend_payload_detail(response)
        ));
    }

    Ok(requested_paths)
}

#[must_use]
pub fn parse_run_result(result: &Value) -> String {
    result.get("run_result").map_or_else(
        || "null".to_string(),
        |value| {
            value
                .as_str()
                .map_or_else(|| value.to_string(), ToString::to_string)
        },
    )
}

/// Parses a single backend inference result entry.
///
/// # Errors
/// Returns an error when required backend output fields are missing.
pub fn parse_inference_output(
    entry: &Value,
    fallback_input_path: Option<&str>,
) -> Result<InferenceOutput, String> {
    let input_path = entry
        .get("input_path")
        .and_then(Value::as_str)
        .map_or_else(
            || {
                fallback_input_path.map(ToString::to_string).ok_or_else(|| {
                    "Missing input_path in IPC result".to_string()
                })
            },
            |value| Ok(value.to_string()),
        )?;

    let output_path = entry
        .get("output_path")
        .and_then(Value::as_str)
        .ok_or_else(|| "Missing output_path in IPC result".to_string())?
        .to_string();

    let output_filename = entry
        .get("output_filename")
        .and_then(Value::as_str)
        .ok_or_else(|| "Missing output_filename in IPC result".to_string())?
        .to_string();

    let artifacts =
        entry
            .get("artifacts")
            .and_then(Value::as_object)
            .map(|obj| InferenceArtifacts {
                brainmask_path: obj
                    .get("brainmask_path")
                    .and_then(Value::as_str)
                    .map(ToString::to_string),
                aseg_path: obj
                    .get("aseg_path")
                    .and_then(Value::as_str)
                    .map(ToString::to_string),
            });

    let qc =
        entry
            .get("qc")
            .and_then(Value::as_object)
            .map(|obj| InferenceQc {
                passed: obj.get("passed").and_then(Value::as_bool),
                message: obj
                    .get("message")
                    .and_then(Value::as_str)
                    .map(ToString::to_string),
            });

    Ok(InferenceOutput {
        input_path,
        output_path,
        output_filename,
        run_result: parse_run_result(entry),
        artifacts,
        qc,
    })
}

#[must_use]
pub fn collect_result_directories(results: &[InferenceOutput]) -> Vec<String> {
    let mut result_directories: Vec<String> = results
        .iter()
        .filter_map(|item| Path::new(&item.output_path).parent())
        .map(|path| path.to_string_lossy().to_string())
        .collect();
    result_directories.sort();
    result_directories.dedup();
    result_directories
}

#[must_use]
pub fn progress_from_value(value: &Value) -> usize {
    value
        .get("progress")
        .and_then(Value::as_u64)
        .and_then(|raw| usize::try_from(raw).ok())
        .map_or(0, |progress| progress.min(100))
}
