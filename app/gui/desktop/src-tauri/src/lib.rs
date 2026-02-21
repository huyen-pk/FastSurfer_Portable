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

struct BackendProcess {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

struct BackendState {
    process: Mutex<BackendProcess>,
}

fn is_supported_image(path: &Path) -> bool {
    let lower = path.to_string_lossy().to_lowercase();
    lower.ends_with(".nii")
        || lower.ends_with(".nii.gz")
        || lower.ends_with(".mgz")
        || lower.ends_with(".mgh")
}

fn collect_images_recursive(dir: &Path, output: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("Cannot read directory {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("Cannot read directory entry in {}: {e}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_images_recursive(&path, output)?;
        } else if is_supported_image(&path) {
            output.push(path);
        }
    }
    Ok(())
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

    let candidates = [
        cwd.join("../backend/main"),
        cwd.join("../../backend/main"),
        cwd.join("../../../backend/main"),
        exe_dir.join("../backend/main"),
        exe_dir.join("../../backend/main"),
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

    fn run_ipc_predict(&self, input_path: &Path) -> Result<InferenceOutput, String> {
        let mut process = self
            .process
            .lock()
            .map_err(|_| "Backend process mutex was poisoned".to_string())?;

        let request = json!({
            "id": 1,
            "method": "predict",
            "params": {
                "input_path": input_path,
            }
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
            return Err(format!("Backend predict failed for {}: {error_msg}", input_path.display()));
        }

        let result = response
            .get("result")
            .ok_or_else(|| "Missing result field in IPC response".to_string())?;

        let output_path = result
            .get("output_path")
            .and_then(Value::as_str)
            .ok_or_else(|| "Missing output_path in IPC result".to_string())?
            .to_string();
        let output_filename = result
            .get("output_filename")
            .and_then(Value::as_str)
            .ok_or_else(|| "Missing output_filename in IPC result".to_string())?
            .to_string();
        let run_result = result
            .get("run_result")
            .map(|v| {
                if let Some(s) = v.as_str() {
                    s.to_string()
                } else {
                    v.to_string()
                }
            })
            .unwrap_or_else(|| "null".to_string());

        Ok(InferenceOutput {
            input_path: input_path.to_string_lossy().to_string(),
            output_path,
            output_filename,
            run_result,
        })
    }
}

#[tauri::command]
fn run_fastsurfer_inference(
    backend: tauri::State<'_, BackendState>,
    file_paths: Vec<String>,
    folder_paths: Vec<String>,
) -> Result<Vec<InferenceOutput>, String> {
    let mut inputs: Vec<PathBuf> = vec![];

    for file in file_paths {
        let path = PathBuf::from(file);
        if path.exists() && path.is_file() && is_supported_image(&path) {
            inputs.push(path);
        }
    }

    for folder in folder_paths {
        let path = PathBuf::from(folder);
        if path.exists() && path.is_dir() {
            collect_images_recursive(&path, &mut inputs)?;
        }
    }

    if inputs.is_empty() {
        return Err("No valid input image files selected (.nii, .nii.gz, .mgz, .mgh).".to_string());
    }

    inputs.sort();
    inputs.dedup();

    let mut outputs: Vec<InferenceOutput> = vec![];

    for input in inputs {
        outputs.push(backend.run_ipc_predict(&input)?);
    }

    Ok(outputs)
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
        .invoke_handler(tauri::generate_handler![run_fastsurfer_inference, open_result_in_file_manager])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
