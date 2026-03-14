use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, ChildStdout};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::runtime::{Builder, Runtime};
use tokio::sync::{mpsc, oneshot};

const REQUEST_CHANNEL_CAPACITY: usize = 32;
const RESPONSE_CHANNEL_CAPACITY: usize = 128;
const OUTPUT_CHANNEL_CAPACITY: usize = 128;
const PER_REQUEST_CHANNEL_CAPACITY: usize = 32;

struct OutboundRequest {
    request_id: u64,
    method: String,
    payload: String,
    write_result_tx: oneshot::Sender<Result<(), String>>,
}

#[derive(Debug)]
enum ParsedStdoutEvent {
    Progress {
        request_id: Option<u64>,
        value: Value,
    },
    Response {
        request_id: Option<u64>,
        value: Value,
    },
    Fatal(String),
}

#[derive(Debug)]
enum InflightEvent {
    Progress(Value),
    Final(Value),
    TerminalError(String),
}

pub struct PendingIpcRequest {
    write_result_rx: oneshot::Receiver<Result<(), String>>,
    event_rx: mpsc::Receiver<InflightEvent>,
}

struct TransportInner {
    next_request_id: AtomicU64,
    request_tx: Mutex<Option<mpsc::Sender<OutboundRequest>>>,
    inflight: Mutex<HashMap<u64, mpsc::Sender<InflightEvent>>>,
    output_rx: Mutex<mpsc::Receiver<String>>,
    closed: AtomicBool,
    runtime: Mutex<Option<Runtime>>,
}

#[derive(Clone)]
pub struct Transport {
    inner: Arc<TransportInner>,
}

impl Transport {
    #[must_use]
    pub fn new(stdin: ChildStdin, stdout: BufReader<ChildStdout>) -> Self {
        let (request_tx, request_rx) = mpsc::channel(REQUEST_CHANNEL_CAPACITY);
        let (response_tx, response_rx) =
            mpsc::channel(RESPONSE_CHANNEL_CAPACITY);
        let (output_tx, output_rx) = mpsc::channel(OUTPUT_CHANNEL_CAPACITY);

        let inner = Arc::new(TransportInner {
            next_request_id: AtomicU64::new(1),
            request_tx: Mutex::new(Some(request_tx)),
            inflight: Mutex::new(HashMap::new()),
            output_rx: Mutex::new(output_rx),
            closed: AtomicBool::new(false),
            runtime: Mutex::new(Some(build_transport_runtime())),
        });

        let writer_inner = Arc::clone(&inner);
        let writer_response_tx = response_tx.clone();
        let dispatcher_inner = Arc::clone(&inner);
        if let Ok(runtime_guard) = inner.runtime.lock()
            && let Some(runtime) = runtime_guard.as_ref()
        {
            runtime.spawn_blocking(move || {
                writer_loop(
                    stdin,
                    request_rx,
                    writer_response_tx,
                    writer_inner,
                );
            });
            runtime.spawn_blocking(move || {
                reader_loop(stdout, response_tx, output_tx);
            });
            runtime.spawn(async move {
                dispatcher_loop(response_rx, dispatcher_inner).await;
            });
        }

        Self { inner }
    }

    /// Queues an IPC request for the writer actor and returns a handle for its response stream.
    ///
    /// # Errors
    /// Returns an error when the transport is already closed, the inflight registry
    /// is unavailable, or the request cannot be queued to the writer thread.
    pub fn send_request(
        &self,
        method: &str,
        params: &Value,
    ) -> Result<PendingIpcRequest, String> {
        if self.inner.closed.load(Ordering::SeqCst) {
            return Err("Transport is closed".to_string());
        }

        let request_id =
            self.inner.next_request_id.fetch_add(1, Ordering::SeqCst);
        let request = json!({
            "id": request_id,
            "method": method,
            "params": params,
        });

        let (event_tx, event_rx) = mpsc::channel(PER_REQUEST_CHANNEL_CAPACITY);
        let mut inflight = self.inner.inflight.lock().map_err(|_| {
            "Transport inflight registry was poisoned".to_string()
        })?;
        inflight.insert(request_id, event_tx);
        drop(inflight);

        let request_tx = self
            .inner
            .request_tx
            .lock()
            .map_err(|_| {
                "Transport request channel mutex was poisoned".to_string()
            })?
            .clone()
            .ok_or_else(|| "Transport request channel is closed".to_string())?;

        let (write_result_tx, write_result_rx) = oneshot::channel();
        let outbound = OutboundRequest {
            request_id,
            method: method.to_string(),
            payload: format!("{request}\n"),
            write_result_tx,
        };

        request_tx.blocking_send(outbound).map_err(|_| {
            let _ = self.remove_inflight_request(request_id);
            "Failed to queue IPC request: writer channel closed".to_string()
        })?;

        Ok(PendingIpcRequest {
            write_result_rx,
            event_rx,
        })
    }

    /// Consumes the correlated response stream for a previously queued request.
    ///
    /// # Errors
    /// Returns an error when the async write fails, the response stream closes,
    /// or the reader/dispatcher reports a terminal transport failure.
    pub fn read_ipc_response(
        &self,
        pending: PendingIpcRequest,
        on_progress: &mut Option<&mut dyn FnMut(Value)>,
    ) -> Result<Value, String> {
        let PendingIpcRequest {
            write_result_rx,
            event_rx,
        } = pending;
        let write_result = write_result_rx.blocking_recv().map_err(|_| {
            "Failed to receive IPC write result from writer task".to_string()
        })?;
        write_result?;

        let mut event_rx = event_rx;
        loop {
            let Some(event) = event_rx.blocking_recv() else {
                return Err(
                    "Transport response stream closed before a final IPC response arrived"
                        .to_string(),
                );
            };

            match event {
                InflightEvent::Progress(value) => {
                    eprintln!("[trace][ipc] <- progress event={value}");
                    if let Some(handler) = on_progress.as_deref_mut() {
                        handler(value);
                    }
                }
                InflightEvent::Final(value) => {
                    eprintln!("[trace][ipc] <- response={value}");
                    return Ok(value);
                }
                InflightEvent::TerminalError(error) => return Err(error),
            }
        }
    }

    pub fn close(&self) {
        if self.inner.closed.swap(true, Ordering::SeqCst) {
            return;
        }

        self.fail_inflight_requests("Transport closed".to_string());

        if let Ok(mut request_tx) = self.inner.request_tx.lock() {
            let _ = request_tx.take();
        }

        if let Ok(mut runtime_guard) = self.inner.runtime.lock()
            && let Some(runtime) = runtime_guard.take()
        {
            runtime.shutdown_timeout(Duration::from_secs(2));
        }
    }

    #[must_use]
    pub fn drain_output_lines(&self) -> Vec<String> {
        let Ok(mut output_rx) = self.inner.output_rx.lock() else {
            return Vec::new();
        };

        let mut lines = Vec::new();
        while let Ok(line) = output_rx.try_recv() {
            lines.push(line);
        }

        lines
    }

    #[must_use]
    fn remove_inflight_request(
        &self,
        request_id: u64,
    ) -> Option<mpsc::Sender<InflightEvent>> {
        self.inner
            .inflight
            .lock()
            .ok()
            .and_then(|mut inflight| inflight.remove(&request_id))
    }

    fn fail_inflight_requests(&self, error: String) {
        fail_inflight_requests(&self.inner, error);
    }
}

fn build_transport_runtime() -> Runtime {
    Builder::new_multi_thread()
        .worker_threads(1)
        .max_blocking_threads(2)
        .enable_all()
        .thread_name("fastsurfer-ipc")
        .build()
        .unwrap_or_else(|error| {
            panic!("Failed to initialize transport runtime: {error}")
        })
}

#[allow(clippy::needless_pass_by_value)]
fn writer_loop(
    mut stdin: ChildStdin,
    mut request_rx: mpsc::Receiver<OutboundRequest>,
    response_tx: mpsc::Sender<ParsedStdoutEvent>,
    inner: Arc<TransportInner>,
) {
    while let Some(request) = request_rx.blocking_recv() {
        eprintln!(
            "[trace][ipc] -> method={} request_id={} payload={}",
            request.method,
            request.request_id,
            request.payload.trim()
        );

        let write_result = stdin
            .write_all(request.payload.as_bytes())
            .map_err(|error| {
                format!("Failed to write request to backend process: {error}")
            })
            .and_then(|()| {
                stdin.flush().map_err(|error| {
                    format!(
                        "Failed to flush request to backend process: {error}"
                    )
                })
            });

        let _ = request.write_result_tx.send(write_result.clone());

        if let Err(error) = write_result {
            let _ = response_tx.blocking_send(ParsedStdoutEvent::Fatal(error));
            inner.closed.store(true, Ordering::SeqCst);
            break;
        }
    }
}

#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::match_wildcard_for_single_variants)]
fn reader_loop(
    mut stdout: BufReader<ChildStdout>,
    response_tx: mpsc::Sender<ParsedStdoutEvent>,
    output_tx: mpsc::Sender<String>,
) {
    let mut response_line = String::new();

    loop {
        response_line.clear();
        let read_bytes = match stdout.read_line(&mut response_line) {
            Ok(read_bytes) => read_bytes,
            Err(error) => {
                let _ = response_tx.blocking_send(ParsedStdoutEvent::Fatal(
                    format!("Failed to read backend response: {error}"),
                ));
                break;
            }
        };

        if read_bytes == 0 {
            let _ = response_tx.blocking_send(ParsedStdoutEvent::Fatal(
                "Backend process exited before sending a response".to_string(),
            ));
            break;
        }

        let trimmed = response_line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if !trimmed.starts_with('{') {
            let _ = output_tx.try_send(trimmed.to_string());
            continue;
        }

        match serde_json::from_str::<Value>(trimmed) {
            Ok(value) => {
                if value.get("event").and_then(Value::as_str)
                    == Some("progress")
                {
                    let request_id = value.get("id").and_then(Value::as_u64);
                    if response_tx
                        .blocking_send(ParsedStdoutEvent::Progress {
                            request_id,
                            value,
                        })
                        .is_err()
                    {
                        break;
                    }
                } else if value.get("ok").is_some() {
                    if response_tx
                        .blocking_send(ParsedStdoutEvent::Response {
                            request_id: value.get("id").and_then(Value::as_u64),
                            value,
                        })
                        .is_err()
                    {
                        break;
                    }
                } else {
                    let _ = output_tx.try_send(trimmed.to_string());
                }
            }
            Err(error) => {
                let _ = response_tx.blocking_send(ParsedStdoutEvent::Fatal(
                    format!("Invalid IPC response JSON: {error}"),
                ));
                break;
            }
        }
    }
}

#[allow(clippy::needless_pass_by_value)]
async fn dispatcher_loop(
    mut response_rx: mpsc::Receiver<ParsedStdoutEvent>,
    inner: Arc<TransportInner>,
) {
    while let Some(event) = response_rx.recv().await {
        match event {
            ParsedStdoutEvent::Progress { request_id, value } => {
                if let Some(sender) = find_progress_sender(&inner, request_id)
                    && sender
                        .send(InflightEvent::Progress(value))
                        .await
                        .is_err()
                {
                    remove_sender(&inner, request_id);
                }
            }
            ParsedStdoutEvent::Response { request_id, value } => {
                let Some(sender) = take_response_sender(&inner, request_id)
                else {
                    let error = match request_id {
                        Some(id) => format!(
                            "Received IPC response for unknown request id {id}"
                        ),
                        None => "Received IPC response without a request id while multiple requests may be in flight"
                            .to_string(),
                    };
                    fail_inflight_requests(&inner, error);
                    break;
                };

                let _ = sender.send(InflightEvent::Final(value)).await;
            }
            ParsedStdoutEvent::Fatal(error) => {
                fail_inflight_requests(&inner, error);
                inner.closed.store(true, Ordering::SeqCst);
                break;
            }
        }
    }

    fail_inflight_requests(
        &inner,
        "Transport response dispatcher stopped".to_string(),
    );
    inner.closed.store(true, Ordering::SeqCst);
}

#[allow(clippy::needless_pass_by_value)]
fn find_progress_sender(
    inner: &Arc<TransportInner>,
    request_id: Option<u64>,
) -> Option<mpsc::Sender<InflightEvent>> {
    let inflight = inner.inflight.lock().ok()?;
    match request_id {
        Some(id) => inflight.get(&id).cloned(),
        None if inflight.len() == 1 => inflight.values().next().cloned(),
        None => None,
    }
}

#[allow(clippy::needless_pass_by_value)]
fn take_response_sender(
    inner: &Arc<TransportInner>,
    request_id: Option<u64>,
) -> Option<mpsc::Sender<InflightEvent>> {
    let mut inflight = inner.inflight.lock().ok()?;
    match request_id {
        Some(id) => inflight.remove(&id),
        None if inflight.len() == 1 => {
            let sole_id = *inflight.keys().next()?;
            inflight.remove(&sole_id)
        }
        None => None,
    }
}

#[allow(clippy::needless_pass_by_value)]
fn remove_sender(inner: &Arc<TransportInner>, request_id: Option<u64>) {
    if let Ok(mut inflight) = inner.inflight.lock() {
        match request_id {
            Some(id) => {
                let _ = inflight.remove(&id);
            }
            None if inflight.len() == 1 => {
                if let Some(id) = inflight.keys().next().copied() {
                    let _ = inflight.remove(&id);
                }
            }
            None => {}
        }
    }
}

#[allow(clippy::needless_pass_by_value)]
fn fail_inflight_requests(inner: &Arc<TransportInner>, error: String) {
    let senders = if let Ok(mut inflight) = inner.inflight.lock() {
        inflight
            .drain()
            .map(|(_, sender)| sender)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    for sender in senders {
        let _ = sender.try_send(InflightEvent::TerminalError(error.clone()));
    }
}
