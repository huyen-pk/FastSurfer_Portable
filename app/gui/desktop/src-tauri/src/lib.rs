use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::thread;
use std::time::Duration;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use tauri::Emitter;

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

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct InferenceProgressEvent {
    task_id: String,
    status: String,
    message: String,
    total: usize,
    completed: usize,
    progress: u8,
    current_path: Option<String>,
    output_path: Option<String>,
}

struct BackendProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

struct BackendState {
    process: Mutex<BackendProcess>,
    launch: BackendLaunchCommand,
    repo_root: Option<PathBuf>,
    current_pid: AtomicU32,
}

struct AppState {
    backend: Option<Arc<BackendState>>,
    backend_init_error: Option<String>,
    cancelled_tasks: Arc<Mutex<BTreeSet<String>>>,
}

#[derive(Clone)]
struct BackendLaunchCommand {
    program: String,
    args: Vec<String>,
}

fn spawn_backend_process(
    launch: &BackendLaunchCommand,
    repo_root: Option<&PathBuf>,
) -> Result<BackendProcess, String> {
    let mut launch_cmd = Command::new(&launch.program);
    launch_cmd
        .args(&launch.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());

    if launch.args.is_empty() {
        let sidecar_dir = Path::new(&launch.program)
            .parent()
            .ok_or_else(|| "Cannot resolve sidecar directory".to_string())?;
        launch_cmd.current_dir(sidecar_dir);
    } else if let Some(root) = repo_root {
        launch_cmd.current_dir(root);
        launch_cmd.env("PYTHONPATH", with_repo_on_pythonpath(root));
    }

    let mut child = launch_cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn backend process ({}): {e}", launch.program))?;

    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "Failed to capture backend stdin".to_string())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Failed to capture backend stdout".to_string())?;

    Ok(BackendProcess {
        child,
        stdin,
        stdout: BufReader::new(stdout),
    })
}

fn resolve_repo_root_from_python_script(script_path: &Path) -> Option<PathBuf> {
    script_path
        .ancestors()
        .find(|ancestor| ancestor.join("FastSurferCNN").exists())
        .map(Path::to_path_buf)
}

fn with_repo_on_pythonpath(repo_root: &Path) -> String {
    let repo_root_str = repo_root.to_string_lossy().to_string();
    if let Ok(existing) = std::env::var("PYTHONPATH") {
        if existing
            .split(':')
            .any(|entry| entry.trim() == repo_root_str)
        {
            return existing;
        }
        if existing.trim().is_empty() {
            return repo_root_str;
        }
        return format!("{repo_root_str}:{existing}");
    }
    repo_root_str
}

fn first_existing_path(base: &Path, suffixes: &[&str]) -> Option<PathBuf> {
    suffixes
        .iter()
        .map(|suffix| base.join(suffix))
        .find(|candidate| candidate.exists())
}

fn load_desktop_env(cwd: &Path, exe_dir: &Path) {
    let env_suffixes = [
        ".env",
        "app/gui/desktop/.env",
        "../app/gui/desktop/.env",
        "../../app/gui/desktop/.env",
        "../../../app/gui/desktop/.env",
    ];

    let env_path = first_existing_path(cwd, &env_suffixes)
        .or_else(|| first_existing_path(exe_dir, &["../.env", "../../.env", "../../../.env"]));

    let Some(path) = env_path else {
        return;
    };

    let Ok(contents) = fs::read_to_string(path) else {
        return;
    };

    for raw_line in contents.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let Some((key, value)) = line.split_once('=') else {
            continue;
        };

        let env_key = key.trim();
        if env_key.is_empty() || std::env::var(env_key).is_ok() {
            continue;
        }

        let env_value = value.trim().trim_matches('"').trim_matches('\'');
        if env_value.is_empty() {
            continue;
        }

        unsafe {
            std::env::set_var(env_key, env_value);
        }
    }
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

fn validate_result_path(result_path: &str) -> Result<PathBuf, String> {
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

    Ok(canonical_path)
}

fn resolve_backend_binary_path_from(cwd: &Path, exe_dir: &Path) -> Result<PathBuf, String> {
    let cwd_suffixes = [
        "app/gui/desktop/backend/main",
        "gui/desktop/backend/main",
        "backend/main",
        "../backend/main",
        "../../backend/main",
        "../../../backend/main",
    ];

    let exe_suffixes = [
        "../backend/main",
        "../../backend/main",
        "../../../backend/main",
        "../../../../backend/main",
        "../../../../../backend/main",
        "../../../../../../backend/main",
    ];

    first_existing_path(cwd, &cwd_suffixes)
        .or_else(|| first_existing_path(exe_dir, &exe_suffixes))
        .ok_or_else(|| "Could not find bundled backend binary app/gui/desktop/backend/main".to_string())
}

fn resolve_python_backend_script_path_from(cwd: &Path, exe_dir: &Path) -> Option<PathBuf> {
    let cwd_suffixes = [
        "app/backend/ipc_server.py",
        "../app/backend/ipc_server.py",
        "../../app/backend/ipc_server.py",
        "../../../app/backend/ipc_server.py",
        "backend/ipc_server.py",
        "../backend/ipc_server.py",
        "../../backend/ipc_server.py",
        "../../../backend/ipc_server.py",
    ];

    let exe_suffixes = [
        "../app/backend/ipc_server.py",
        "../../app/backend/ipc_server.py",
        "../../../app/backend/ipc_server.py",
        "../../../../app/backend/ipc_server.py",
    ];

    first_existing_path(cwd, &cwd_suffixes)
        .or_else(|| first_existing_path(exe_dir, &exe_suffixes))
}

fn resolve_python_executable() -> Option<String> {
    let mut candidates: Vec<String> = Vec::new();

    if let Ok(py_bin) = std::env::var("FASTSURFER_PYTHON_BIN") {
        if !py_bin.trim().is_empty() {
            candidates.push(py_bin);
        }
    }

    if let Ok(py_bin) = std::env::var("PYTHON_BIN") {
        if !py_bin.trim().is_empty() {
            candidates.push(py_bin);
        }
    }

    candidates.push("python3".to_string());
    candidates.push("python".to_string());

    candidates.into_iter().find(|candidate| {
        Command::new(candidate)
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
    })
}

fn resolve_backend_launch_command_from(cwd: &Path, exe_dir: &Path) -> Result<BackendLaunchCommand, String> {
    if let Ok(binary_path) = resolve_backend_binary_path_from(cwd, exe_dir) {
        return Ok(BackendLaunchCommand {
            program: binary_path.to_string_lossy().to_string(),
            args: Vec::new(),
        });
    }

    let python_script = resolve_python_backend_script_path_from(cwd, exe_dir);
    let python_exec = resolve_python_executable();

    if let (Some(script), Some(py)) = (python_script, python_exec) {
        return Ok(BackendLaunchCommand {
            program: py,
            args: vec![script.to_string_lossy().to_string()],
        });
    }

    Err(
        "Could not resolve backend launch command: bundled backend binary not found and no Python interpreter available. Install Python or set FASTSURFER_PYTHON_BIN, or provide bundled backend binary app/gui/desktop/backend/main.".to_string(),
    )
}

fn write_ipc_request(process: &mut BackendProcess, request: &Value) -> Result<(), String> {
    let request_line = format!("{}\n", request);
    process
        .stdin
        .write_all(request_line.as_bytes())
        .map_err(|e| format!("Failed to write request to backend process: {e}"))?;
    process
        .stdin
        .flush()
        .map_err(|e| format!("Failed to flush request to backend process: {e}"))
}

fn read_ipc_response(
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
            return Err("Backend process exited before sending a response".to_string());
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
                if response
                    .get("event")
                    .and_then(Value::as_str)
                    .map(|name| name == "progress")
                    .unwrap_or(false)
                {
                    if let Some(handler) = on_progress.as_deref_mut() {
                        handler(response);
                    }
                    continue;
                }

                if response.get("ok").is_some() {
                    return Ok(response);
                }

                continue;
            }
            Err(e) => return Err(format!("Invalid IPC response JSON: {e}")),
        }
    }
}

impl BackendState {
    fn new() -> Result<Self, String> {
        let cwd = std::env::current_dir().map_err(|e| format!("Cannot resolve current dir: {e}"))?;
        let exe_path = std::env::current_exe().map_err(|e| format!("Cannot resolve current executable path: {e}"))?;
        let exe_dir = exe_path
            .parent()
            .ok_or_else(|| "Cannot resolve parent directory of executable".to_string())?;

        load_desktop_env(&cwd, exe_dir);

        let python_script_path = resolve_python_backend_script_path_from(&cwd, exe_dir);
        let repo_root = python_script_path
            .as_deref()
            .and_then(resolve_repo_root_from_python_script);

        let launch = resolve_backend_launch_command_from(&cwd, exe_dir)?;
        let process = spawn_backend_process(&launch, repo_root.as_ref())?;
        let current_pid = process.child.id();

        let backend = Self {
            process: Mutex::new(process),
            launch,
            repo_root,
            current_pid: AtomicU32::new(current_pid),
        };

        backend
            .run_ipc_request("health", json!({}))
            .map_err(|e| format!("Backend process failed health check on startup: {e}"))?;

        Ok(backend)
    }

    fn run_ipc_request_with_progress(
        &self,
        method: &str,
        params: Value,
        mut on_progress: Option<&mut dyn FnMut(Value)>,
    ) -> Result<Value, String> {
        let mut process = self
            .process
            .lock()
            .map_err(|_| "Backend process mutex was poisoned".to_string())?;

        let request = json!({
            "id": 1,
            "method": method,
            "params": params,
        });

        write_ipc_request(&mut process, &request)?;
        let response = read_ipc_response(&mut process, &mut on_progress)?;

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

    fn run_ipc_request(&self, method: &str, params: Value) -> Result<Value, String> {
        self.run_ipc_request_with_progress(method, params, None)
    }

    fn restart_backend_process(&self) -> Result<(), String> {
        let mut process_guard = self
            .process
            .lock()
            .map_err(|_| "Backend process mutex was poisoned".to_string())?;

        let mut new_process = spawn_backend_process(&self.launch, self.repo_root.as_ref())?;
        write_ipc_request(
            &mut new_process,
            &json!({
                "id": 1,
                "method": "health",
                "params": {},
            }),
        )?;

        let mut no_progress: Option<&mut dyn FnMut(Value)> = None;
        let response = read_ipc_response(&mut new_process, &mut no_progress)?;
        if !response.get("ok").and_then(Value::as_bool).unwrap_or(false) {
            let error_msg = response
                .get("error")
                .and_then(|err| err.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("Unknown backend error");
            return Err(format!("Backend health failed after restart: {error_msg}"));
        }

        let old_pid = process_guard.child.id();
        let _ = process_guard.child.kill();
        let _ = process_guard.child.wait();

        self.current_pid.store(new_process.child.id(), Ordering::SeqCst);
        *process_guard = new_process;

        if old_pid != 0 {
            let _ = old_pid;
        }

        Ok(())
    }

    fn force_stop_current_process(&self) -> Result<(), String> {
        let pid = self.current_pid.load(Ordering::SeqCst);
        if pid == 0 {
            return Ok(());
        }

        #[cfg(target_os = "windows")]
        {
            let gentle = Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T"])
                .status();

            match gentle {
                Ok(status) if status.success() => {
                    self.current_pid.store(0, Ordering::SeqCst);
                    return Ok(());
                }
                _ => {
                    let forced = Command::new("taskkill")
                        .args(["/PID", &pid.to_string(), "/F", "/T"])
                        .status()
                        .map_err(|e| format!("Failed to invoke taskkill for backend process: {e}"))?;
                    if !forced.success() {
                        return Err("taskkill returned non-zero exit code".to_string());
                    }
                }
            }
        }

        #[cfg(all(unix, not(target_os = "windows")))]
        {
            let gentle = Command::new("kill")
                .args(["-15", &pid.to_string()])
                .status();

            if let Ok(status) = gentle {
                if status.success() {
                    thread::sleep(Duration::from_millis(250));
                    self.current_pid.store(0, Ordering::SeqCst);
                    return Ok(());
                }
            }

            let forced = Command::new("kill")
                .args(["-9", &pid.to_string()])
                .status()
                .map_err(|e| format!("Failed to invoke kill for backend process: {e}"))?;
            if !forced.success() {
                return Err("kill returned non-zero exit code".to_string());
            }
        }

        self.current_pid.store(0, Ordering::SeqCst);

        Ok(())
    }

    fn shutdown_for_exit(&self) -> Result<(), String> {
        if let Ok(mut process) = self.process.try_lock() {
            let request = json!({
                "id": 1,
                "method": "shutdown",
                "params": {},
            });
            let _ = write_ipc_request(&mut process, &request);
        }

        self.force_stop_current_process()
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

    fn predict_single_path(
        &self,
        input_path: &str,
        task_id: &str,
        on_progress: Option<&mut dyn FnMut(usize, String)>,
    ) -> Result<InferenceOutput, String> {
        let mut on_progress = on_progress;
        let mut progress_adapter = |value: Value| {
            let progress = value
                .get("progress")
                .and_then(Value::as_u64)
                .map(|raw| (raw as usize).min(100))
                .unwrap_or(0);

            let message = value
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("Processing MRI")
                .to_string();

            if let Some(handler) = on_progress.as_deref_mut() {
                handler(progress, message);
            }
        };

        let result = self.run_ipc_request_with_progress(
            "predict",
            json!({
                "input_path": input_path,
                "task_id": task_id,
            }),
            Some(&mut progress_adapter),
        )?;

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
            input_path: input_path.to_string(),
            output_path,
            output_filename,
            run_result,
        })
    }
}

fn emit_progress_event(app_handle: &tauri::AppHandle, event: InferenceProgressEvent) {
    let _ = app_handle.emit("fastsurfer://inference-progress", event);
}

fn run_fastsurfer_inference_with_backend(
    backend: &BackendState,
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

fn run_fastsurfer_inference_with_app_state(
    backend: Option<&BackendState>,
    backend_init_error: Option<&str>,
    file_paths: Vec<String>,
    folder_paths: Vec<String>,
) -> Result<ProcessingRunResult, String> {
    let backend = backend.ok_or_else(|| {
        let detail = backend_init_error.unwrap_or("unknown backend initialization error");
        format!("Backend is unavailable in this desktop runtime: {detail}")
    })?;

    run_fastsurfer_inference_with_backend(backend, file_paths, folder_paths)
}

fn run_fastsurfer_inference_with_progress_with_app_state(
    app_handle: &tauri::AppHandle,
    backend: Option<&BackendState>,
    backend_init_error: Option<&str>,
    cancelled_tasks: &Arc<Mutex<BTreeSet<String>>>,
    task_id: String,
    file_paths: Vec<String>,
    folder_paths: Vec<String>,
) -> Result<ProcessingRunResult, String> {
    let backend = backend.ok_or_else(|| {
        let detail = backend_init_error.unwrap_or("unknown backend initialization error");
        format!("Backend is unavailable in this desktop runtime: {detail}")
    })?;

    let (start_ack_message, requested_paths) = backend.start_predict_batch(&file_paths, &folder_paths)?;
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
                    result_directories.insert(parent.to_string_lossy().to_string());
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
        results,
    })
}

#[tauri::command]
async fn run_fastsurfer_inference(
    app_state: tauri::State<'_, AppState>,
    file_paths: Vec<String>,
    folder_paths: Vec<String>,
) -> Result<ProcessingRunResult, String> {
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

#[tauri::command]
async fn run_fastsurfer_inference_with_progress(
    app_handle: tauri::AppHandle,
    app_state: tauri::State<'_, AppState>,
    task_id: String,
    file_paths: Vec<String>,
    folder_paths: Vec<String>,
) -> Result<ProcessingRunResult, String> {
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

#[tauri::command]
fn cancel_fastsurfer_task(
    app_state: tauri::State<'_, AppState>,
    task_id: String,
) -> Result<(), String> {
    let mut cancelled = app_state
        .cancelled_tasks
        .lock()
        .map_err(|_| "Cancelled tasks mutex was poisoned".to_string())?;
    cancelled.insert(task_id);

    if let Some(backend) = app_state.backend.as_ref() {
        let _ = backend.force_stop_current_process();
    }

    Ok(())
}

#[tauri::command]
fn shutdown_backend_for_exit(app_state: tauri::State<'_, AppState>) -> Result<(), String> {
    if let Some(backend) = app_state.backend.as_ref() {
        backend.shutdown_for_exit()?;
    }
    Ok(())
}

#[tauri::command]
fn open_result_in_file_manager(result_path: String) -> Result<(), String> {
    let canonical_path = validate_result_path(&result_path)?;

    open_in_file_manager(&canonical_path)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app_state = match BackendState::new() {
        Ok(backend) => AppState {
            backend: Some(Arc::new(backend)),
            backend_init_error: None,
            cancelled_tasks: Arc::new(Mutex::new(BTreeSet::new())),
        },
        Err(err) => {
            eprintln!("FastSurfer desktop backend initialization failed: {err}");
            AppState {
                backend: None,
                backend_init_error: Some(err),
                cancelled_tasks: Arc::new(Mutex::new(BTreeSet::new())),
            }
        }
    };

    tauri::Builder::default()
        .manage(app_state)
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            run_fastsurfer_inference,
            run_fastsurfer_inference_with_progress,
            cancel_fastsurfer_task,
            shutdown_backend_for_exit,
            open_result_in_file_manager
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
#[path = "../testing/rust/controller_tests.rs"]
mod controller_tests;
