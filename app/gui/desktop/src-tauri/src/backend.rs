use crate::events::InferenceProgressEvent;
use crate::inference::entities::{InferenceArtifacts, InferenceQc};
use crate::inference::entities::{InferenceOutput, ProcessingRunResult};
use crate::process_mgmt::{
    BackendLaunchCommand, BackendProcess, load_desktop_env,
    resolve_backend_launch_command_from,
    resolve_python_backend_script_path_from,
    resolve_repo_root_from_python_script, spawn_backend_process,
};
use crate::transport::Transport;
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::path::Path;
use std::path::PathBuf;
use std::process::Child;
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Runtime};

/// Main state wrapper for the persistent backend process.
/// Handles lifecycle management (spawn, restart, kill) and High-level IPC calls.
pub struct BackendState {
    /// Shared transport wrapper for request/response IPC.
    pub transport: Mutex<Arc<Transport>>,
    /// Child handle retained for process lifecycle operations.
    pub child: Mutex<Option<Child>>,
    /// Configuration used to launch (or relaunch) the backend.
    pub launch: BackendLaunchCommand,
    /// Detected repository root path (useful for development mode python path resolution).
    pub repo_root: Option<PathBuf>,
    /// Atomic tracker for the current process ID, used for force-killing.
    pub current_pid: AtomicU32,
}

fn parse_requested_paths(
    response: &Value,
    fallback_requested_paths: &[String],
) -> Vec<String> {
    response
        .get("requested_paths")
        .and_then(Value::as_array)
        .map_or_else(
            || fallback_requested_paths.to_vec(),
            |paths| {
                paths
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToString::to_string)
                    .collect::<Vec<String>>()
            },
        )
}

fn parse_run_result(result: &Value) -> String {
    result.get("run_result").map_or_else(
        || "null".to_string(),
        |value| {
            value
                .as_str()
                .map_or_else(|| value.to_string(), ToString::to_string)
        },
    )
}

fn parse_inference_output(
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

fn collect_result_directories(results: &[InferenceOutput]) -> Vec<String> {
    let mut result_directories: Vec<String> = results
        .iter()
        .filter_map(|item| Path::new(&item.output_path).parent())
        .map(|path| path.to_string_lossy().to_string())
        .collect();
    result_directories.sort();
    result_directories.dedup();
    result_directories
}

fn progress_from_event(value: &Value) -> usize {
    value
        .get("progress")
        .and_then(Value::as_u64)
        .and_then(|raw| usize::try_from(raw).ok())
        .map_or(0, |progress| progress.min(100))
}

impl BackendState {
    #[must_use]
    fn from_backend_process(
        process: BackendProcess,
        launch: BackendLaunchCommand,
        repo_root: Option<PathBuf>,
    ) -> Self {
        let (child, stdin, stdout) = process.into_parts();
        let current_pid = child.id();

        Self {
            transport: Mutex::new(Arc::new(Transport::new(stdin, stdout))),
            child: Mutex::new(Some(child)),
            launch,
            repo_root,
            current_pid: AtomicU32::new(current_pid),
        }
    }

    #[must_use]
    pub fn from_process(
        process: BackendProcess,
        launch: BackendLaunchCommand,
        repo_root: Option<PathBuf>,
    ) -> Self {
        Self::from_backend_process(process, launch, repo_root)
    }

    fn active_transport(&self) -> Result<Arc<Transport>, String> {
        self.transport
            .lock()
            .map(|transport| Arc::clone(&*transport))
            .map_err(|_| "Backend transport mutex was poisoned".to_string())
    }

    fn clear_child_handle(&self) {
        if let Ok(mut child_guard) = self.child.lock()
            && let Some(mut child) = child_guard.take()
        {
            let _ = child.wait();
        }
    }

    fn close_active_transport(&self) {
        if let Ok(transport) = self.active_transport() {
            transport.close();
        }
    }

    /// Drains any non-IPC stdout lines captured by the transport log channel.
    ///
    /// # Errors
    /// Returns an error when the active transport cannot be accessed.
    pub fn drain_output_lines(&self) -> Result<Vec<String>, String> {
        Ok(self.active_transport()?.drain_output_lines())
    }

    /// Initializes the backend state by resolving paths and spawning the initial process.
    /// Performs an immediate "health" check IPC call to verify readiness.
    ///
    /// # Errors
    /// Returns an error when process launch, backend path resolution, or the
    /// initial health IPC check fails.
    pub fn new() -> Result<Self, String> {
        let cwd = std::env::current_dir()
            .map_err(|e| format!("Cannot resolve current dir: {e}"))?;
        let exe_path = std::env::current_exe().map_err(|e| {
            format!("Cannot resolve current executable path: {e}")
        })?;
        let exe_dir = exe_path.parent().ok_or_else(|| {
            "Cannot resolve parent directory of executable".to_string()
        })?;

        load_desktop_env(&cwd, exe_dir);

        let python_script_path =
            resolve_python_backend_script_path_from(&cwd, exe_dir);
        let repo_root = python_script_path
            .as_deref()
            .and_then(resolve_repo_root_from_python_script);

        let launch = resolve_backend_launch_command_from(&cwd, exe_dir)?;
        let process = spawn_backend_process(&launch, repo_root.as_ref())?;
        let backend = Self::from_backend_process(process, launch, repo_root);

        let request = json!({});
        backend.run_ipc_request("health", &request).map_err(|e| {
            format!("Backend process failed health check on startup: {e}")
        })?;

        Ok(backend)
    }

    /// Sends a single IPC request and emits correlated progress updates.
    ///
    /// # Errors
    /// Returns an error when the transport cannot queue the request, the backend
    /// reports an error response, or the response stream terminates unexpectedly.
    pub fn run_ipc_request_with_progress(
        &self,
        method: &str,
        params: &Value,
        mut on_progress: Option<&mut dyn FnMut(Value)>,
    ) -> Result<Value, String> {
        eprintln!(
            "[trace][backend] run_ipc_request_with_progress method={method}"
        );
        let transport = self.active_transport()?;
        let pending = transport.send_request(method, params)?;
        let response =
            transport.read_ipc_response(pending, &mut on_progress)?;

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

    /// Sends a single IPC request and waits for the final response.
    ///
    /// # Errors
    /// Returns an error when the transport request fails or the backend returns
    /// an error payload.
    pub fn run_ipc_request(
        &self,
        method: &str,
        params: &Value,
    ) -> Result<Value, String> {
        self.run_ipc_request_with_progress(method, params, None)
    }

    /// Restarts the active backend process after verifying the replacement health check.
    ///
    /// # Errors
    /// Returns an error when the replacement backend cannot be spawned or does
    /// not pass the health check.
    pub fn restart_backend_process(&self) -> Result<(), String> {
        let new_process =
            spawn_backend_process(&self.launch, self.repo_root.as_ref())?;
        let (new_child, new_stdin, new_stdout) = new_process.into_parts();
        let new_pid = new_child.id();
        let new_transport = Arc::new(Transport::new(new_stdin, new_stdout));

        let pending = new_transport.send_request("health", &json!({}))?;
        let mut no_progress: Option<&mut dyn FnMut(Value)> = None;
        let response =
            new_transport.read_ipc_response(pending, &mut no_progress)?;
        if !response.get("ok").and_then(Value::as_bool).unwrap_or(false) {
            let error_msg = response
                .get("error")
                .and_then(|err| err.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("Unknown backend error");
            return Err(format!(
                "Backend health failed after restart: {error_msg}"
            ));
        }

        let old_transport = {
            let mut transport_guard = self.transport.lock().map_err(|_| {
                "Backend transport mutex was poisoned".to_string()
            })?;
            std::mem::replace(&mut *transport_guard, new_transport)
        };
        let old_child = {
            let mut child_guard = self
                .child
                .lock()
                .map_err(|_| "Backend child mutex was poisoned".to_string())?;
            child_guard.replace(new_child)
        };

        self.current_pid.store(new_pid, Ordering::SeqCst);

        if let Some(mut child) = old_child {
            let _ = child.kill();
            let _ = child.wait();
        }
        old_transport.close();

        Ok(())
    }

    /// Terminates the current backend process explicitly.
    /// Tries a gentle `SIGTERM` first, then falls back to forceful `SIGKILL`
    /// or `taskkill /F`.
    ///
    /// # Errors
    /// Returns an error when the force-stop command cannot be invoked or exits
    /// unsuccessfully.
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
                    self.clear_child_handle();
                    self.close_active_transport();
                    return Ok(());
                }
                _ => {
                    let forced = Command::new("taskkill")
                        .args(["/PID", &pid.to_string(), "/F", "/T"])
                        .status()
                        .map_err(|e| {
                            format!("Failed to invoke taskkill for backend process: {e}")
                        })?;
                    if !forced.success() {
                        return Err(
                            "taskkill returned non-zero exit code".to_string()
                        );
                    }
                }
            }
        }

        #[cfg(all(unix, not(target_os = "windows")))]
        {
            let gentle = Command::new("kill")
                .args(["-15", &pid.to_string()])
                .status();

            if let Ok(status) = gentle
                && status.success()
            {
                thread::sleep(Duration::from_millis(250));
                self.current_pid.store(0, Ordering::SeqCst);
                self.clear_child_handle();
                self.close_active_transport();
                return Ok(());
            }

            let forced = Command::new("kill")
                .args(["-9", &pid.to_string()])
                .status()
                .map_err(|e| {
                    format!("Failed to invoke kill for backend process: {e}")
                })?;
            if !forced.success() {
                return Err("kill returned non-zero exit code".to_string());
            }
        }

        self.current_pid.store(0, Ordering::SeqCst);
        self.clear_child_handle();
        self.close_active_transport();

        Ok(())
    }

    /// Sends a shutdown command to the backend politely before killing it.
    ///
    /// # Errors
    /// Returns an error when the backend cannot be terminated after the best
    /// effort shutdown request.
    pub fn shutdown_for_exit(&self) -> Result<(), String> {
        if let Ok(transport) = self.active_transport()
            && let Ok(pending) = transport.send_request("shutdown", &json!({}))
        {
            let mut no_progress: Option<&mut dyn FnMut(Value)> = None;
            let _ = transport.read_ipc_response(pending, &mut no_progress);
        }

        self.force_stop_current_process()
    }

    /// Initiates a batch prediction request via IPC.
    ///
    /// # Errors
    /// Returns an error when the request fails or the backend response omits
    /// the expected acknowledgement fields.
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
        let request = json!({
            "file_paths": file_paths,
            "folder_paths": folder_paths,
        });
        let result = self.run_ipc_request("start_predict_batch", &request)?;

        let ack_message = result
            .get("ack_message")
            .and_then(Value::as_str)
            .ok_or_else(|| "Missing ack_message in IPC response".to_string())?
            .to_string();

        let requested_paths = result
            .get("requested_paths")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                "Missing requested_paths in IPC response".to_string()
            })?
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
    ///
    /// # Errors
    /// Returns an error when the backend request fails or the response is
    /// missing required result fields.
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
        let request = json!({
            "file_paths": file_paths,
            "folder_paths": folder_paths,
        });
        let result = self.run_ipc_request("predict_batch", &request)?;

        let ack_message_from_backend = result
            .get("ack_message")
            .and_then(Value::as_str)
            .unwrap_or(fallback_ack_message)
            .to_string();

        let requested_paths =
            parse_requested_paths(&result, fallback_requested_paths);

        let results_array = result
            .get("results")
            .and_then(Value::as_array)
            .ok_or_else(|| "Missing results in IPC response".to_string())?;

        let mut results = Vec::with_capacity(results_array.len());
        for entry in results_array {
            results.push(parse_inference_output(entry, None)?);
        }

        let result_directories = collect_result_directories(&results);

        let ack_message = finalize_ack_message(
            &ack_message_from_backend,
            &result_directories,
        );

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
    ///
    /// # Errors
    /// Returns an error when the prediction IPC fails or the response omits
    /// required inference output fields.
    pub fn predict_single_path(
        &self,
        input_path: &str,
        task_id: &str,
        on_progress: Option<&mut dyn FnMut(usize, String)>,
    ) -> Result<InferenceOutput, String> {
        eprintln!(
            "[trace][backend] predict_single_path task_id={task_id} input_path={input_path}"
        );
        let mut on_progress = on_progress;
        let mut progress_adapter = |value: Value| {
            let progress = progress_from_event(&value);

            let message = value
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("Processing MRI")
                .to_string();

            if let Some(handler) = on_progress.as_deref_mut() {
                handler(progress, message);
            }
        };

        let request = json!({
            "input_path": input_path,
            "task_id": task_id,
        });
        let result = self.run_ipc_request_with_progress(
            "predict",
            &request,
            Some(&mut progress_adapter),
        )?;

        parse_inference_output(&result, Some(input_path))
    }
}

/// Helper to emit internal progress events to the frontend.
fn emit_progress_event<T: Emitter<R>, R: Runtime>(
    event_handler: &T,
    event: InferenceProgressEvent,
) {
    let _ = event_handler.emit("fastsurfer://inference-progress", event);
}

fn ensure_backend<'a>(
    backend: Option<&'a BackendState>,
    backend_init_error: Option<&'a str>,
) -> Result<&'a BackendState, String> {
    backend.ok_or_else(|| {
        let detail = backend_init_error
            .unwrap_or("unknown backend initialization error");
        format!("Backend is unavailable in this desktop runtime: {detail}")
    })
}

fn emit_started_event(
    app_handle: &AppHandle,
    task_id: &str,
    start_ack_message: &str,
    total: usize,
) {
    emit_progress_event(
        app_handle,
        InferenceProgressEvent {
            task_id: task_id.to_string(),
            status: "started".to_string(),
            message: start_ack_message.to_string(),
            total,
            completed: 0,
            progress: 0,
            current_path: None,
            output_path: None,
        },
    );
}

fn emit_item_progress_event(
    app_handle: &AppHandle,
    task_id: &str,
    index: usize,
    total: usize,
    progress: u8,
    input_path: &str,
    message: String,
) {
    emit_progress_event(
        app_handle,
        InferenceProgressEvent {
            task_id: task_id.to_string(),
            status: "item_progress".to_string(),
            message,
            total,
            completed: index,
            progress,
            current_path: Some(input_path.to_string()),
            output_path: None,
        },
    );
}

fn emit_item_completed_event(
    app_handle: &AppHandle,
    task_id: &str,
    index: usize,
    total: usize,
    output_path: &str,
) {
    let completed = index + 1;
    let progress = if total == 0 {
        100
    } else {
        u8::try_from(((completed * 100) / total).min(100)).unwrap_or(100)
    };

    emit_progress_event(
        app_handle,
        InferenceProgressEvent {
            task_id: task_id.to_string(),
            status: "item_completed".to_string(),
            message: format!("Processed {completed}/{total}"),
            total,
            completed,
            progress,
            current_path: None,
            output_path: Some(output_path.to_string()),
        },
    );
}

fn emit_cancelled_event(
    app_handle: &AppHandle,
    task_id: &str,
    total: usize,
    completed: usize,
) {
    let progress = if total == 0 {
        0
    } else {
        u8::try_from(((completed * 100) / total).min(100)).unwrap_or(100)
    };

    emit_progress_event(
        app_handle,
        InferenceProgressEvent {
            task_id: task_id.to_string(),
            status: "cancelled".to_string(),
            message: "Task cancelled by user.".to_string(),
            total,
            completed,
            progress,
            current_path: None,
            output_path: None,
        },
    );
}

fn emit_failed_event(
    app_handle: &AppHandle,
    task_id: String,
    total: usize,
    completed: usize,
    message: String,
) {
    let progress = if total == 0 {
        0
    } else {
        u8::try_from(((completed * 100) / total).min(100)).unwrap_or(100)
    };

    emit_progress_event(
        app_handle,
        InferenceProgressEvent {
            task_id,
            status: "failed".to_string(),
            message,
            total,
            completed,
            progress,
            current_path: None,
            output_path: None,
        },
    );
}

fn add_result_directory(set: &mut BTreeSet<String>, path: &str) {
    if let Some(parent) = Path::new(path).parent() {
        set.insert(parent.to_string_lossy().to_string());
    }
}

fn finalize_ack_message(start_ack: &str, dirs: &[String]) -> String {
    if dirs.is_empty() {
        start_ack.to_string()
    } else {
        format!("{} Results directory: {}", start_ack, dirs.join(", "))
    }
}

fn legacy_process_requested_paths(
    backend: &BackendState,
    requested_paths: &[String],
    cancelled_tasks: &Arc<Mutex<BTreeSet<String>>>,
    app_handle: &AppHandle,
    task_id: &str,
) -> Result<(Vec<InferenceOutput>, BTreeSet<String>), String> {
    let total = requested_paths.len();
    let mut results: Vec<InferenceOutput> = Vec::with_capacity(total);
    let mut result_directories: BTreeSet<String> = BTreeSet::new();

    for (index, input_path) in requested_paths.iter().enumerate() {
        let is_cancelled = cancelled_tasks
            .lock()
            .map_err(|_| "Cancelled tasks mutex was poisoned".to_string())?
            .contains(task_id);

        if is_cancelled {
            emit_cancelled_event(app_handle, task_id, total, index);
            if let Ok(mut cancelled) = cancelled_tasks.lock() {
                cancelled.remove(task_id);
            }
            return Err("Task cancelled by user.".to_string());
        }

        let mut intra_file_progress =
            |file_progress: usize, file_message: String| {
                let safe_file_progress = file_progress.min(100);
                let overall_progress = if total == 0 {
                    u8::try_from(safe_file_progress).unwrap_or(100)
                } else {
                    u8::try_from(
                        (((index * 100) + safe_file_progress) / total).min(100),
                    )
                    .unwrap_or(100)
                };

                emit_item_progress_event(
                    app_handle,
                    task_id,
                    index,
                    total,
                    overall_progress,
                    input_path,
                    file_message,
                );
            };

        match backend.predict_single_path(
            input_path,
            task_id,
            Some(&mut intra_file_progress),
        ) {
            Ok(prediction) => {
                add_result_directory(
                    &mut result_directories,
                    &prediction.output_path,
                );
                emit_item_completed_event(
                    app_handle,
                    task_id,
                    index,
                    total,
                    &prediction.output_path,
                );
                results.push(prediction);
            }
            Err(error) => {
                let was_cancelled = cancelled_tasks
                    .lock()
                    .map_err(|_| {
                        "Cancelled tasks mutex was poisoned".to_string()
                    })?
                    .contains(task_id);

                if was_cancelled {
                    emit_cancelled_event(app_handle, task_id, total, index);
                    if let Ok(mut cancelled) = cancelled_tasks.lock() {
                        cancelled.remove(task_id);
                    }
                    let _ = backend.restart_backend_process();
                    return Err("Task cancelled by user.".to_string());
                }

                emit_failed_event(
                    app_handle,
                    task_id.to_string(),
                    total,
                    index,
                    error.clone(),
                );
                return Err(error);
            }
        }
    }

    Ok((results, result_directories))
}

/// Wrapper for inference that checks for backend availability first.
pub(crate) fn legacy_run_fastsurfer_inference_core(
    backend: Option<&BackendState>,
    backend_init_error: Option<&str>,
    file_paths: &[String],
    folder_paths: &[String],
) -> Result<ProcessingRunResult, String> {
    let backend = ensure_backend(backend, backend_init_error)?;

    eprintln!(
        "[trace][prediction] run_fastsurfer_inference_with_backend start file_paths={} folder_paths={}",
        file_paths.len(),
        folder_paths.len()
    );
    let (start_ack_message, start_requested_paths) =
        backend.start_predict_batch(file_paths, folder_paths)?;

    eprintln!(
        "[trace][prediction] start_predict_batch ack='{}' requested_paths={}",
        start_ack_message,
        start_requested_paths.len()
    );

    backend.predict_batch(
        file_paths,
        folder_paths,
        &start_ack_message,
        &start_requested_paths,
    )
}

/// Advanced inference runner that supports granular progress tracking and cancellation.
///
/// Iterates through the requested items one by one, updating the UI via events.
/// Handles user-initiated cancellation checks between items.
///
/// # Errors
/// Returns an error when backend inference startup, per-item processing, or
/// event-driven cancellation handling fails.
pub fn legacy_run_fastsurfer_inference_with_progress(
    app_handle: &AppHandle,
    backend: Option<&BackendState>,
    backend_init_error: Option<&str>,
    cancelled_tasks: &Arc<Mutex<BTreeSet<String>>>,
    task_id: String,
    file_paths: &[String],
    folder_paths: &[String],
) -> Result<ProcessingRunResult, String> {
    eprintln!(
        "[trace][prediction] run_with_progress task_id={} file_paths={} folder_paths={}",
        task_id,
        file_paths.len(),
        folder_paths.len()
    );
    let backend = ensure_backend(backend, backend_init_error)?;

    let (start_ack_message, requested_paths): (String, Vec<String>) =
        backend.start_predict_batch(file_paths, folder_paths)?;
    let total = requested_paths.len();

    emit_started_event(app_handle, &task_id, &start_ack_message, total);

    let (results, result_directories) = legacy_process_requested_paths(
        backend,
        &requested_paths,
        cancelled_tasks,
        app_handle,
        &task_id,
    )?;

    let result_directories =
        result_directories.into_iter().collect::<Vec<String>>();
    let ack_message =
        finalize_ack_message(&start_ack_message, &result_directories);

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
        qc_summary: None,
        results,
    })
}
