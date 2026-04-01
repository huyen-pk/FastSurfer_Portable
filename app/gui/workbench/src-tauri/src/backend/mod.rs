pub mod protocol;
pub mod subprocess;

use self::protocol::{
    collect_result_directories, extract_backend_result, parse_inference_output,
    parse_requested_paths, progress_from_value,
};
use crate::backend::subprocess::{
    BackendLaunchCommand, BackendProcess, load_desktop_env,
    resolve_backend_launch_command_from, spawn_backend_process,
};
use crate::events::InferenceProgressEvent;
use crate::inference::entities::{InferenceOutput, ProcessingRunResult};
use crate::transport::Transport;
use crate::transport::factory::create_fastsurfer_stdio_transport;
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Child;
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const TASK_CANCELLED_MESSAGE: &str = "Task cancelled by user.";

/// Main state wrapper for the persistent backend process.
/// Handles lifecycle management (spawn, restart, kill) and High-level IPC calls.
pub struct BackendManager {
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

impl BackendManager {
    /// Builds backend state from an already spawned process.
    ///
    /// # Panics
    /// Panics when the process cannot be wrapped in the configured transport.
    #[must_use]
    pub fn from_process(
        process: BackendProcess,
        launch: BackendLaunchCommand,
        repo_root: Option<PathBuf>,
    ) -> Self {
        let (child, transport) = create_fastsurfer_stdio_transport(process)
            .unwrap_or_else(|error| {
                panic!("Failed to build backend state: {error}")
            });
        let current_pid = child.id();

        Self {
            transport: Mutex::new(Arc::new(transport)),
            child: Mutex::new(Some(child)),
            launch,
            repo_root,
            current_pid: AtomicU32::new(current_pid),
        }
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

        // No development-time Python script probing: always prefer the
        // bundled backend binary. Repository root is not required.
        let repo_root: Option<PathBuf> = None;

        let launch = resolve_backend_launch_command_from(&cwd, exe_dir)?;
        let process = spawn_backend_process(&launch, repo_root.as_ref())?;
        let backend = Self::from_process(process, launch, repo_root);

        let request = json!({});
        backend
            .request_backend_response("health", &request, None)
            .map_err(|e| {
                format!("Backend process failed health check on startup: {e}")
            })?;

        Ok(backend)
    }

    /// Sends a backend request and returns the backend result payload.
    ///
    /// # Errors
    /// Returns an error when the transport cannot queue the request, the backend
    /// reports an error response, or the response stream terminates unexpectedly.
    fn request_backend_response(
        &self,
        method: &str,
        params: &Value,
        on_progress: Option<&mut dyn FnMut(Value)>,
    ) -> Result<Value, String> {
        eprintln!("[trace][backend] request_backend_response method={method}");
        let transport = self.active_transport()?;
        let response =
            transport.request_json(method, params.clone(), on_progress)?;

        extract_backend_result(method, &response)
    }

    /// Restarts the active backend process after verifying the replacement health check.
    ///
    /// # Errors
    /// Returns an error when the replacement backend cannot be spawned or does
    /// not pass the health check.
    pub fn restart_backend_process(&self) -> Result<(), String> {
        let new_process =
            spawn_backend_process(&self.launch, self.repo_root.as_ref())?;
        let (new_child, new_transport) =
            create_fastsurfer_stdio_transport(new_process)?;
        let new_pid = new_child.id();
        let new_transport = Arc::new(new_transport);

        let response = new_transport.request_json("health", json!({}), None)?;
        extract_backend_result("health", &response).map_err(|error| {
            format!("Backend health failed after restart: {error}")
        })?;

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
        if let Ok(transport) = self.active_transport() {
            let _ = transport.request_json("shutdown", json!({}), None);
        }

        self.force_stop_current_process()
    }

    fn ensure_ready_for_request(&self) -> Result<(), String> {
        if self.current_pid.load(Ordering::SeqCst) == 0 {
            self.restart_backend_process()?;
        }

        Ok(())
    }

    /// Resolves an optional backend handle into a usable backend reference.
    ///
    /// # Errors
    /// Returns an error when the desktop runtime did not initialize the
    /// backend process successfully.
    pub(crate) fn require_available<'a>(
        backend: Option<&'a Self>,
        backend_init_error: Option<&'a str>,
    ) -> Result<&'a Self, String> {
        backend.ok_or_else(|| {
            let detail = backend_init_error
                .unwrap_or("unknown backend initialization error");
            format!("Backend is unavailable in this desktop runtime: {detail}")
        })
    }

    /// Discovers the canonical set of input paths for a batch request.
    ///
    /// # Errors
    /// Returns an error when the backend is unavailable or rejects the input
    /// discovery request.
    pub fn resolve_requested_paths(
        &self,
        file_paths: &[String],
        folder_paths: &[String],
    ) -> Result<(String, Vec<String>), String> {
        self.ensure_ready_for_request()?;

        let request = json!({
            "file_paths": file_paths,
            "folder_paths": folder_paths,
        });
        let result = self.request_backend_response(
            "resolve_requested_paths",
            &request,
            None,
        )?;

        let ack_message = result
            .get("ack_message")
            .and_then(Value::as_str)
            .ok_or_else(|| "Missing ack_message in IPC response".to_string())?
            .to_string();
        let requested_paths = parse_requested_paths(&result)?;

        Ok((ack_message, requested_paths))
    }

    /// Probes the backend health endpoint and returns the extracted payload.
    ///
    /// # Errors
    /// Returns an error when the backend is unavailable, the health request
    /// fails, or the response status is not `ok`.
    pub fn health_probe(&self) -> Result<Value, String> {
        self.ensure_ready_for_request()?;
        let request = json!({});
        let result = self.request_backend_response("health", &request, None)?;
        if result.get("status").and_then(Value::as_str) != Some("ok") {
            return Err("Backend health check failed".to_string());
        }
        Ok(result)
    }

    /// Predicts a single validated input path through the Python child process.
    ///
    /// # Errors
    /// Returns an error when input validation fails, the backend request fails,
    /// or the response omits required inference output fields.
    pub fn predict_single_path(
        &self,
        input_path: &str,
        task_id: Option<&str>,
        on_progress: Option<&mut dyn FnMut(usize, String)>,
    ) -> Result<InferenceOutput, String> {
        let mut on_progress = on_progress;
        let mut progress_adapter = |value: Value| {
            let progress = progress_from_value(&value);
            let message = value
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("Processing MRI")
                .to_string();

            if let Some(handler) = on_progress.as_deref_mut() {
                handler(progress, message.clone());
            }
        };

        let request = json!({
            "input_path": input_path,
            "task_id": task_id.unwrap_or_default(),
        });
        let result = self.request_backend_response(
            "predict",
            &request,
            Some(&mut progress_adapter),
        )?;

        parse_inference_output(&result, Some(input_path))
    }

    /// Runs Python-backed batch inference through the persistent child process.
    ///
    /// # Errors
    /// Returns an error when input discovery fails, a prediction fails, or a
    /// task is cancelled.
    pub fn predict_batch(
        &self,
        file_paths: &[String],
        folder_paths: &[String],
        cancelled_tasks: Option<&Arc<Mutex<BTreeSet<String>>>>,
        task_id: Option<&str>,
    ) -> Result<ProcessingRunResult, String> {
        let (start_ack_message, requested_paths) =
            self.resolve_requested_paths(file_paths, folder_paths)?;
        let total = requested_paths.len();

        let is_cancelled = || {
            task_id.zip(cancelled_tasks).map_or(
                Ok(false),
                |(task_id, cancelled_tasks)| {
                    Self::task_is_cancelled(cancelled_tasks, task_id)
                },
            )
        };

        if let Some(task_id) = task_id {
            Self::publish_progress(Self::started_event(
                task_id,
                &start_ack_message,
                total,
            ));
        }
        let mut results = Vec::with_capacity(total);
        for (index, input_path) in requested_paths.iter().enumerate() {
            if is_cancelled()? {
                return self.handle_batch_cancellation(
                    cancelled_tasks,
                    task_id,
                    total,
                    index,
                );
            }

            let mut on_progress =
                Self::on_progress_handler(task_id, total, index, input_path);
            let prediction = match self.predict_single_path(
                input_path,
                task_id,
                Some(&mut on_progress),
            ) {
                Ok(prediction) => prediction,
                Err(error) => {
                    if is_cancelled()? {
                        return self.handle_batch_cancellation(
                            cancelled_tasks,
                            task_id,
                            total,
                            index,
                        );
                    }
                    if let Some(task_id) = task_id {
                        Self::publish_progress(Self::failed_event(
                            task_id,
                            input_path,
                            total,
                            index,
                            error.clone(),
                        ));
                    }
                    return Err(error);
                }
            };
            if let Some(task_id) = task_id {
                Self::publish_progress(Self::item_completed_event(
                    task_id,
                    input_path,
                    &prediction.output_path,
                    total,
                    index + 1,
                ));
            }
            results.push(prediction);
        }
        if let (Some(cancelled_tasks), Some(task_id)) =
            (cancelled_tasks, task_id)
        {
            Self::clear_cancelled_task(cancelled_tasks, task_id);
        }
        let result_directories = collect_result_directories(&results);
        let ack_message = Self::result_ack_message(
            requested_paths.len(),
            &result_directories,
        );
        if let Some(task_id) = task_id {
            Self::publish_progress(Self::completed_event(
                task_id,
                ack_message.clone(),
                total,
            ));
        }
        Ok(ProcessingRunResult {
            ack_message,
            requested_paths,
            result_directories,
            qc_summary: None,
            results,
        })
    }

    fn result_ack_message(
        requested_count: usize,
        result_directories: &[String],
    ) -> String {
        if result_directories.is_empty() {
            format!("Processing started for {requested_count} path(s).")
        } else {
            format!(
                "Processing started for {requested_count} path(s). Results directory: {}",
                result_directories.join(", ")
            )
        }
    }

    fn task_is_cancelled(
        cancelled_tasks: &Arc<Mutex<BTreeSet<String>>>,
        task_id: &str,
    ) -> Result<bool, String> {
        cancelled_tasks
            .lock()
            .map(|cancelled| cancelled.contains(task_id))
            .map_err(|_| "Cancelled tasks mutex was poisoned".to_string())
    }

    fn clear_cancelled_task(
        cancelled_tasks: &Arc<Mutex<BTreeSet<String>>>,
        task_id: &str,
    ) {
        if let Ok(mut cancelled) = cancelled_tasks.lock() {
            cancelled.remove(task_id);
        }
    }

    fn publish_progress(event: InferenceProgressEvent) {
        crate::mediator::mediator().publish(Arc::new(event));
    }

    fn started_event(
        task_id: &str,
        ack_message: &str,
        total: usize,
    ) -> InferenceProgressEvent {
        InferenceProgressEvent {
            task_id: task_id.to_string(),
            status: "started".to_string(),
            message: ack_message.to_string(),
            total,
            completed: 0,
            progress: 0,
            current_path: None,
            output_path: None,
        }
    }

    fn item_progress_event(
        task_id: &str,
        input_path: &str,
        total: usize,
        completed: usize,
        progress: u8,
        message: String,
    ) -> InferenceProgressEvent {
        InferenceProgressEvent {
            task_id: task_id.to_string(),
            status: "item_progress".to_string(),
            message,
            total,
            completed,
            progress,
            current_path: Some(input_path.to_string()),
            output_path: None,
        }
    }

    fn on_progress_handler<'a>(
        task_id: Option<&'a str>,
        total: usize,
        index: usize,
        input_path: &'a str,
    ) -> Box<dyn FnMut(usize, String) + 'a> {
        Box::new(move |file_progress: usize, file_message: String| {
            if let Some(task_id) = task_id {
                let overall_progress = if total == 0 {
                    u8::try_from(file_progress.min(100)).unwrap_or(100)
                } else {
                    u8::try_from(
                        (((index * 100) + file_progress.min(100)) / total)
                            .min(99),
                    )
                    .unwrap_or(99)
                };
                Self::publish_progress(Self::item_progress_event(
                    task_id,
                    input_path,
                    total,
                    index,
                    overall_progress,
                    file_message,
                ));
            }
        })
    }

    fn item_completed_event(
        task_id: &str,
        input_path: &str,
        output_path: &str,
        total: usize,
        completed: usize,
    ) -> InferenceProgressEvent {
        let progress = if total == 0 {
            100
        } else {
            u8::try_from(((completed * 100) / total).min(100)).unwrap_or(100)
        };

        InferenceProgressEvent {
            task_id: task_id.to_string(),
            status: "item_completed".to_string(),
            message: format!("Processed {completed}/{total}"),
            total,
            completed,
            progress,
            current_path: Some(input_path.to_string()),
            output_path: Some(output_path.to_string()),
        }
    }

    fn cancelled_event(
        task_id: &str,
        total: usize,
        completed: usize,
    ) -> InferenceProgressEvent {
        let progress = if total == 0 {
            0
        } else {
            u8::try_from(((completed * 100) / total).min(100)).unwrap_or(100)
        };

        InferenceProgressEvent {
            task_id: task_id.to_string(),
            status: "cancelled".to_string(),
            message: "Task cancelled by user.".to_string(),
            total,
            completed,
            progress,
            current_path: None,
            output_path: None,
        }
    }

    fn failed_event(
        task_id: &str,
        input_path: &str,
        total: usize,
        completed: usize,
        message: String,
    ) -> InferenceProgressEvent {
        let progress = if total == 0 {
            0
        } else {
            u8::try_from(((completed * 100) / total).min(100)).unwrap_or(100)
        };

        InferenceProgressEvent {
            task_id: task_id.to_string(),
            status: "failed".to_string(),
            message,
            total,
            completed,
            progress,
            current_path: Some(input_path.to_string()),
            output_path: None,
        }
    }

    fn completed_event(
        task_id: &str,
        message: String,
        total: usize,
    ) -> InferenceProgressEvent {
        InferenceProgressEvent {
            task_id: task_id.to_string(),
            status: "completed".to_string(),
            message,
            total,
            completed: total,
            progress: 100,
            current_path: None,
            output_path: None,
        }
    }

    fn handle_batch_cancellation<T>(
        &self,
        cancelled_tasks: Option<&Arc<Mutex<BTreeSet<String>>>>,
        task_id: Option<&str>,
        total: usize,
        completed: usize,
    ) -> Result<T, String> {
        if let (Some(cancelled_tasks), Some(task_id)) =
            (cancelled_tasks, task_id)
        {
            Self::publish_progress(Self::cancelled_event(
                task_id, total, completed,
            ));
            Self::clear_cancelled_task(cancelled_tasks, task_id);
        }
        let _ = self.restart_backend_process();
        Err(TASK_CANCELLED_MESSAGE.to_string())
    }
}
