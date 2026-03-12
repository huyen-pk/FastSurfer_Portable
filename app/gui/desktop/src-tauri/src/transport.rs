use crate::process_mgmt::BackendProcess;
use serde_json::Value;
use std::io::{BufRead, Write};

/// Writes a JSON RPC request to the backend process via stdin.
///
/// # Arguments
/// * `process` - Reference to the running `BackendProcess` wrapper.
/// * `request` - The JSON-RPC request payload.
///
/// # Errors
/// Returns an error when writing or flushing the request to the backend stdin
/// fails.
pub fn write_ipc_request(
    process: &mut BackendProcess,
    request: &Value,
) -> Result<(), String> {
    let method = request
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("<unknown>");
    eprintln!("[trace][ipc] -> method={method} payload={request}");

    let request_line = format!("{request}\n");
    process
        .stdin
        .write_all(request_line.as_bytes())
        .map_err(|e| {
            format!("Failed to write request to backend process: {e}")
        })?;
    process
        .stdin
        .flush()
        .map_err(|e| format!("Failed to flush request to backend process: {e}"))
}

/// Reads a JSON RPC response from the backend process via stdout.
///
/// Handles asynchronous "progress" events that may arrive before the final response.
///
/// # Arguments
/// * `process` - Reference to the running `BackendProcess` wrapper.
/// * `on_progress` - Optional closure to handle out-of-band progress events.
///
/// # Errors
/// Returns an error when reading stdout fails, the backend exits before a
/// response arrives, or the received JSON payload is invalid.
pub fn read_ipc_response(
    process: &mut BackendProcess,
    on_progress: &mut Option<&mut dyn FnMut(Value)>,
) -> Result<Value, String> {
    let mut response_line = String::new();

    loop {
        response_line.clear();
        let read_bytes = process
            .stdout
            .read_line(&mut response_line)
            .map_err(|e| format!("Failed to read backend response: {e}"))?;

        if read_bytes == 0 {
            return Err(
                "Backend process exited before sending a response".to_string()
            );
        }

        let trimmed = response_line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if !trimmed.starts_with('{') {
            continue;
        }

        match serde_json::from_str::<Value>(trimmed) {
            Ok(response) => {
                if response.get("event").and_then(Value::as_str)
                    == Some("progress")
                {
                    eprintln!("[trace][ipc] <- progress event={response}");
                    if let Some(handler) = on_progress.as_deref_mut() {
                        handler(response);
                    }
                    continue;
                }

                if response.get("ok").is_some() {
                    eprintln!("[trace][ipc] <- response={response}");
                    return Ok(response);
                }
            }
            Err(e) => return Err(format!("Invalid IPC response JSON: {e}")),
        }
    }
}
