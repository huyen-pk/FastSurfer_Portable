use crate::models::{InferenceArtifacts, InferenceOutput, InferenceQc, ProcessingRunResult};
use crate::process_mgmt::{
    load_desktop_env, resolve_backend_launch_command_from, resolve_python_backend_script_path_from,
    resolve_repo_root_from_python_script, spawn_backend_process, BackendLaunchCommand,
    BackendProcess,
};
use crate::transport::{read_ipc_response, write_ipc_request};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

/// Main state wrapper for the persistent backend process.
/// Handles lifecycle management (spawn, restart, kill) and High-level RPC calls.
pub struct BackendState {
    /// Mutex for thread-safe access to the single backend process instance.
    pub process: Mutex<BackendProcess>,
    /// Configuration used to launch (or relaunch) the backend.
    pub launch: BackendLaunchCommand,
    /// Detected repository root path (useful for development mode python path resolution).
    pub repo_root: Option<PathBuf>,
    /// Atomic tracker for the current process ID, used for force-killing.
    pub current_pid: AtomicU32,
}

impl BackendState {
    /// Initializes the backend state by resolving paths and spawning the initial process.
    /// Performs an immediate "health" check RPC call to verify readiness.
    pub fn new() -> Result<Self, String> {
        let cwd = std::env::current_dir().map_err(|e| format!("Cannot resolve current dir: {e}"))?;
        let exe_path = std::env::current_exe()
            .map_err(|e| format!("Cannot resolve current executable path: {e}"))?;
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

    /// Executes a JSON-RPC method that may return intermediate progress updates.
    ///
    /// # Arguments
    /// * `method` - The RPC method name (e.g., "predict").
    /// * `params` - The JSON parameters for the method.
    /// * `on_progress` - Optional callback for handling "progress" events.
    pub fn run_ipc_request_with_progress(
        &self,
        method: &str,
        params: Value,
        mut on_progress: Option<&mut dyn FnMut(Value)>,
    ) -> Result<Value, String> {
        eprintln!("[trace][backend] run_ipc_request_with_progress method={method}");
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

    /// Executes a simple JSON-RPC method without progress tracking.
    pub fn run_ipc_request(&self, method: &str, params: Value) -> Result<Value, String> {
        self.run_ipc_request_with_progress(method, params, None)
    }

    /// Restarts the backend subprocess.
    /// Used when the previous process has crashed, been killed, or is unresponsive.
    pub fn restart_backend_process(&self) -> Result<(), String> {
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

        self.current_pid
            .store(new_process.child.id(), Ordering::SeqCst);
        *process_guard = new_process;

        if old_pid != 0 {
            let _ = old_pid;
        }

        Ok(())
    }

    /// Terminates the current backend process explicitly.
    /// Tries a gentle SIGTERM first, then falls back to forceful SIGKILL (or TaskKill /F).
    pub fn force_stop_current_process(&self) -> Result<(), String> {
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

    /// Sends a shutdown command to the backend politely before killing it.
    pub fn shutdown_for_exit(&self) -> Result<(), String> {
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

    /// Initiates a batch prediction request via IPC.
    pub fn start_predict_batch(
        &self,
        file_paths: &[String],
        folder_paths: &[String],
    ) -> Result<(String, Vec<String>), String> {
        eprintln!(
            "[trace][backend] start_predict_batch file_paths={} folder_paths={}",
            file_paths.len(),
            folder_paths.len()
        );
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

        eprintln!(
            "[trace][backend] start_predict_batch ack='{}' requested_paths={}",
            ack_message,
            requested_paths.len()
        );

        Ok((ack_message, requested_paths))
    }

    /// Runs a full batch prediction call (blocking until complete).
    pub fn predict_batch(
        &self,
        file_paths: &[String],
        folder_paths: &[String],
        fallback_ack_message: &str,
        fallback_requested_paths: &[String],
    ) -> Result<ProcessingRunResult, String> {
        eprintln!(
            "[trace][backend] predict_batch file_paths={} folder_paths={}",
            file_paths.len(),
            folder_paths.len()
        );
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

            let artifacts = entry
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

            let qc = entry
                .get("qc")
                .and_then(Value::as_object)
                .map(|obj| InferenceQc {
                    passed: obj.get("passed").and_then(Value::as_bool),
                    message: obj
                        .get("message")
                        .and_then(Value::as_str)
                        .map(ToString::to_string),
                });

            results.push(InferenceOutput {
                input_path,
                output_path,
                output_filename,
                run_result,
                artifacts,
                qc,
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
            qc_summary: result
                .get("qc_summary")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            results,
        })
    }

    /// Predicts a single file, streaming progress events via the provided closure.
    pub fn predict_single_path(
        &self,
        input_path: &str,
        task_id: &str,
        on_progress: Option<&mut dyn FnMut(usize, String)>,
    ) -> Result<InferenceOutput, String> {
        eprintln!(
            "[trace][backend] predict_single_path task_id={} input_path={}",
            task_id,
            input_path
        );
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

        let artifacts = result
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

        let qc = result
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
            input_path: input_path.to_string(),
            output_path,
            output_filename,
            run_result,
            artifacts,
            qc,
        })
    }
}
