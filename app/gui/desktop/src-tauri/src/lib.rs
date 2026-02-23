use serde::Serialize;
use serde_json::{json, Value};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InferenceOutput {
    input_path: String,
    output_path: String,
    output_filename: String,
    run_result: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ProcessingRunResult {
    ack_message: String,
    requested_paths: Vec<String>,
    result_directories: Vec<String>,
    results: Vec<InferenceOutput>,
}

struct BackendProcess {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

struct BackendState {
    process: Mutex<BackendProcess>,
}

fn open_in_file_manager(path: &Path) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        let status = if path.is_file() {
            Command::new("explorer")
                .arg("/select,")
                .arg(path)
                .status()
        } else {
            Command::new("explorer").arg(path).status()
        }
        .map_err(|e| format!("Failed to launch Explorer: {e}"))?;

        if status.success() {
            return Ok(());
        }
        return Err("Explorer returned a non-zero exit code".to_string());
    }

    #[cfg(target_os = "macos")]
    {
        let status = if path.is_file() {
            Command::new("open").arg("-R").arg(path).status()
        } else {
            Command::new("open").arg(path).status()
        }
        .map_err(|e| format!("Failed to launch Finder: {e}"))?;

        if status.success() {
            return Ok(());
        }
        return Err("Finder returned a non-zero exit code".to_string());
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let target = if path.is_file() {
            path.parent()
                .ok_or_else(|| "File has no parent directory to open".to_string())?
        } else {
            path
        };

        let status = Command::new("xdg-open")
            .arg(target)
            .status()
            .map_err(|e| format!("Failed to launch file manager (xdg-open): {e}"))?;

        if status.success() {
            return Ok(());
        }
        return Err("xdg-open returned a non-zero exit code".to_string());
    }

    #[allow(unreachable_code)]
    Err("Unsupported operating system for opening file manager".to_string())
}

fn resolve_backend_binary_path() -> Result<PathBuf, String> {
    let cwd = std::env::current_dir().map_err(|e| format!("Cannot resolve current dir: {e}"))?;
    let exe_path = std::env::current_exe().map_err(|e| format!("Cannot resolve current executable path: {e}"))?;
    let exe_dir = exe_path
        .parent()
        .ok_or_else(|| "Cannot resolve parent directory of executable".to_string())?;

    let workspace_hint = cwd.join("app/gui/desktop/backend/main");
    let candidates = [
        workspace_hint,
        cwd.join("gui/desktop/backend/main"),
        cwd.join("backend/main"),
        cwd.join("../backend/main"),
        cwd.join("../../backend/main"),
        cwd.join("../../../backend/main"),
        exe_dir.join("../backend/main"),
        exe_dir.join("../../backend/main"),
        exe_dir.join("../../../backend/main"),
        exe_dir.join("../../../../backend/main"),
        exe_dir.join("../../../../../backend/main"),
        exe_dir.join("../../../../../../backend/main"),
    ];

    candidates
        .iter()
        .find(|p| p.exists())
        .cloned()
        .ok_or_else(|| "Could not find bundled backend binary app/gui/desktop/backend/main".to_string())
}

impl BackendState {
    fn new() -> Result<Self, String> {
        let backend_binary = resolve_backend_binary_path()?;
        let mut child = Command::new(backend_binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("Failed to spawn bundled backend process: {e}"))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "Failed to capture backend stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to capture backend stdout".to_string())?;

        Ok(Self {
            process: Mutex::new(BackendProcess {
                _child: child,
                stdin,
                stdout: BufReader::new(stdout),
            }),
        })
    }

    fn run_ipc_request(&self, method: &str, params: Value) -> Result<Value, String> {
        let mut process = self
            .process
            .lock()
            .map_err(|_| "Backend process mutex was poisoned".to_string())?;

        let request = json!({
            "id": 1,
            "method": method,
            "params": params,
        });

        let line = format!("{}\n", request);
        process
            .stdin
            .write_all(line.as_bytes())
            .map_err(|e| format!("Failed to write request to backend process: {e}"))?;
        process
            .stdin
            .flush()
            .map_err(|e| format!("Failed to flush request to backend process: {e}"))?;

        let mut response_line = String::new();
        loop {
            response_line.clear();
            let read_bytes = process
                .stdout
                .read_line(&mut response_line)
                .map_err(|e| format!("Failed to read backend response: {e}"))?;
            if read_bytes == 0 {
                return Err("Backend process exited before sending a response".to_string());
            }
            if !response_line.trim().is_empty() {
                break;
            }
        }

        let response: Value = serde_json::from_str(response_line.trim())
            .map_err(|e| format!("Invalid IPC response JSON: {e}"))?;

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

    fn start_predict_batch(&self, file_paths: &[String], folder_paths: &[String]) -> Result<(String, Vec<String>), String> {
        let result = self.run_ipc_request(
            "start_predict_batch",
            json!({
                "file_paths": file_paths,
                "folder_paths": folder_paths,
            }),
        )?;

        let ack_message = result
            .get("ack_message")
            .and_then(Value::as_str)
            .ok_or_else(|| "Missing ack_message in IPC response".to_string())?
            .to_string();

        let requested_paths = result
            .get("requested_paths")
            .and_then(Value::as_array)
            .ok_or_else(|| "Missing requested_paths in IPC response".to_string())?
            .iter()
            .filter_map(Value::as_str)
            .map(ToString::to_string)
            .collect::<Vec<String>>();

        Ok((ack_message, requested_paths))
    }

    fn predict_batch(
        &self,
        file_paths: &[String],
        folder_paths: &[String],
        fallback_ack_message: &str,
        fallback_requested_paths: &[String],
    ) -> Result<ProcessingRunResult, String> {
        let result = self.run_ipc_request(
            "predict_batch",
            json!({
                "file_paths": file_paths,
                "folder_paths": folder_paths,
            }),
        )?;

        let ack_message_from_backend = result
            .get("ack_message")
            .and_then(Value::as_str)
            .unwrap_or(fallback_ack_message)
            .to_string();

        let requested_paths = result
            .get("requested_paths")
            .and_then(Value::as_array)
            .map(|paths| {
                paths
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToString::to_string)
                    .collect::<Vec<String>>()
            })
            .unwrap_or_else(|| fallback_requested_paths.to_vec());

        let results_array = result
            .get("results")
            .and_then(Value::as_array)
            .ok_or_else(|| "Missing results in IPC response".to_string())?;

        let mut results = Vec::with_capacity(results_array.len());
        for entry in results_array {
            let input_path = entry
                .get("input_path")
                .and_then(Value::as_str)
                .ok_or_else(|| "Missing input_path in IPC result".to_string())?
                .to_string();

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

            let run_result = entry
                .get("run_result")
                .map(|v| {
                    if let Some(s) = v.as_str() {
                        s.to_string()
                    } else {
                        v.to_string()
                    }
                })
                .unwrap_or_else(|| "null".to_string());

            results.push(InferenceOutput {
                input_path,
                output_path,
                output_filename,
                run_result,
            });
        }

        let mut result_directories: Vec<String> = results
            .iter()
            .filter_map(|item| Path::new(&item.output_path).parent())
            .map(|path| path.to_string_lossy().to_string())
            .collect();
        result_directories.sort();
        result_directories.dedup();

        let ack_message = if result_directories.is_empty() {
            ack_message_from_backend
        } else {
            format!(
                "{} Results directory: {}",
                ack_message_from_backend,
                result_directories.join(", ")
            )
        };

        Ok(ProcessingRunResult {
            ack_message,
            requested_paths,
            result_directories,
            results,
        })
    }
}

#[tauri::command]
fn run_fastsurfer_inference(
    backend: tauri::State<'_, BackendState>,
    file_paths: Vec<String>,
    folder_paths: Vec<String>,
) -> Result<ProcessingRunResult, String> {
    let (start_ack_message, start_requested_paths) =
        backend.start_predict_batch(&file_paths, &folder_paths)?;

    backend.predict_batch(
        &file_paths,
        &folder_paths,
        &start_ack_message,
        &start_requested_paths,
    )
}

#[tauri::command]
fn open_result_in_file_manager(result_path: String) -> Result<(), String> {
    if result_path.trim().is_empty() {
        return Err("Path is empty".to_string());
    }

    let raw_path = PathBuf::from(result_path);
    if !raw_path.exists() {
        return Err(format!("Path does not exist: {}", raw_path.display()));
    }

    let canonical_path = raw_path.canonicalize().map_err(|e| {
        if e.kind() == std::io::ErrorKind::PermissionDenied {
            format!("Permission denied when resolving path: {}", raw_path.display())
        } else {
            format!("Failed to resolve path {}: {e}", raw_path.display())
        }
    })?;

    if canonical_path.is_file() {
        fs::File::open(&canonical_path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                format!("Permission denied when accessing file: {}", canonical_path.display())
            } else {
                format!("Cannot access file {}: {e}", canonical_path.display())
            }
        })?;
    } else if canonical_path.is_dir() {
        fs::read_dir(&canonical_path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                format!("Permission denied when accessing directory: {}", canonical_path.display())
            } else {
                format!("Cannot access directory {}: {e}", canonical_path.display())
            }
        })?;
    } else {
        return Err(format!("Path is neither a regular file nor directory: {}", canonical_path.display()));
    }

    open_in_file_manager(&canonical_path)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let backend_state = BackendState::new().expect("Failed to initialize bundled backend process");

    tauri::Builder::default()
        .manage(backend_state)
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            run_fastsurfer_inference,
            open_result_in_file_manager
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
