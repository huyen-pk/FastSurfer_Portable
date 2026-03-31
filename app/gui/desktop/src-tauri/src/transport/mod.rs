pub mod driver;
pub mod factory;
pub mod ipc_protocol;
pub mod message;
pub mod protocol;
pub mod stdio;

use self::driver::{
    DRIVER_EVENT_CHANNEL_CAPACITY, DriverEvent, DriverHandle,
    OutboundDriverFrame, TransportDriver,
};
use self::message::{MessagePayload, TransportMessage, TransportMessageKind};
use self::protocol::TransportProtocol;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::runtime::{Builder, Runtime};
use tokio::sync::{mpsc, oneshot};

const OUTPUT_CHANNEL_CAPACITY: usize = 128;
const PER_REQUEST_CHANNEL_CAPACITY: usize = 32;

#[derive(Debug)]
enum InflightEvent {
    Progress(TransportMessage),
    Final(TransportMessage),
    TerminalError(String),
}

pub struct PendingTransportRequest {
    write_result_rx: oneshot::Receiver<Result<(), String>>,
    event_rx: mpsc::Receiver<InflightEvent>,
}

struct TransportInner {
    next_request_id: AtomicU64,
    driver: DriverHandle,
    protocol: Arc<dyn TransportProtocol>,
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
    /// Creates a transport coordinator around a driver/protocol pair.
    ///
    /// # Errors
    /// Returns an error when the driver cannot be started.
    pub fn new(
        driver: Box<dyn TransportDriver>,
        protocol: Arc<dyn TransportProtocol>,
    ) -> Result<Self, String> {
        let (driver_event_tx, driver_event_rx) =
            mpsc::channel(DRIVER_EVENT_CHANNEL_CAPACITY);
        let (message_tx, message_rx) =
            mpsc::channel(DRIVER_EVENT_CHANNEL_CAPACITY);
        let (output_tx, output_rx) = mpsc::channel(OUTPUT_CHANNEL_CAPACITY);
        let runtime = build_transport_runtime();
        let driver_handle = driver.start(&runtime, driver_event_tx)?;

        let inner = Arc::new(TransportInner {
            next_request_id: AtomicU64::new(1),
            driver: driver_handle,
            protocol,
            inflight: Mutex::new(HashMap::new()),
            output_rx: Mutex::new(output_rx),
            closed: AtomicBool::new(false),
            runtime: Mutex::new(Some(runtime)),
        });

        let protocol_inner = Arc::clone(&inner);
        let dispatcher_inner = Arc::clone(&inner);

        if let Ok(runtime_guard) = inner.runtime.lock()
            && let Some(runtime) = runtime_guard.as_ref()
        {
            runtime.spawn(protocol_loop(
                driver_event_rx,
                message_tx,
                output_tx,
                protocol_inner,
            ));
            runtime.spawn(async move {
                dispatcher_loop(message_rx, dispatcher_inner).await;
            });
        }

        Ok(Self { inner })
    }

    /// Encodes and queues a generic request on the underlying transport.
    ///
    /// # Errors
    /// Returns an error when the transport is closed, the inflight registry is
    /// unavailable, the protocol cannot encode the request, or the driver queue
    /// is unavailable.
    pub fn send_request(
        &self,
        method: &str,
        payload: MessagePayload,
    ) -> Result<PendingTransportRequest, String> {
        if self.inner.closed.load(Ordering::SeqCst) {
            return Err("Transport is closed".to_string());
        }

        let request_id =
            self.inner.next_request_id.fetch_add(1, Ordering::SeqCst);
        let request = TransportMessage::request(request_id, method, payload);

        let (event_tx, event_rx) = mpsc::channel(PER_REQUEST_CHANNEL_CAPACITY);
        let mut inflight = self.inner.inflight.lock().map_err(|_| {
            "Transport inflight registry was poisoned".to_string()
        })?;
        inflight.insert(request_id, event_tx);
        drop(inflight);

        let frame = self.inner.protocol.encode_request(&request)?;

        let (write_result_tx, write_result_rx) = oneshot::channel();
        let outbound = OutboundDriverFrame {
            correlation_id: request_id,
            frame,
            write_result_tx,
        };

        self.inner.driver.send_frame(outbound).map_err(|_| {
            let _ = self.remove_inflight_request(request_id);
            "Failed to queue transport request".to_string()
        })?;

        Ok(PendingTransportRequest {
            write_result_rx,
            event_rx,
        })
    }

    /// Waits for the correlated final response while streaming progress events.
    ///
    /// # Errors
    /// Returns an error when the write fails, when the response stream closes
    /// early, or when the transport reports a terminal error.
    pub fn read_response(
        &self,
        pending: PendingTransportRequest,
        on_progress: &mut Option<&mut dyn FnMut(TransportMessage)>,
    ) -> Result<TransportMessage, String> {
        let PendingTransportRequest {
            write_result_rx,
            event_rx,
        } = pending;
        let write_result = write_result_rx.blocking_recv().map_err(|_| {
            "Failed to receive transport write result from driver task"
                .to_string()
        })?;
        write_result?;

        let mut event_rx = event_rx;
        loop {
            let Some(event) = event_rx.blocking_recv() else {
                return Err(
                    "Transport response stream closed before a final response arrived"
                        .to_string(),
                );
            };

            match event {
                InflightEvent::Progress(message) => {
                    if let Some(handler) = on_progress.as_deref_mut() {
                        handler(message);
                    }
                }
                InflightEvent::Final(message) => return Ok(message),
                InflightEvent::TerminalError(error) => return Err(error),
            }
        }
    }

    /// Sends a JSON request and returns the final JSON response payload.
    ///
    /// # Errors
    /// Returns an error when the request cannot be queued, when the response
    /// stream terminates unexpectedly, or when the final transport payload is
    /// not JSON.
    pub fn request_json(
        &self,
        method: &str,
        payload: serde_json::Value,
        mut on_progress: Option<&mut dyn FnMut(serde_json::Value)>,
    ) -> Result<serde_json::Value, String> {
        let pending =
            self.send_request(method, MessagePayload::Json(payload))?;
        let has_progress_handler = on_progress.is_some();
        let mut progress_adapter = |message: TransportMessage| {
            if let Some(handler) = on_progress.as_deref_mut()
                && let Some(value) = message.payload.as_json().cloned()
            {
                handler(value);
            }
        };
        let mut progress_handler: Option<&mut dyn FnMut(TransportMessage)> =
            if has_progress_handler {
                Some(&mut progress_adapter)
            } else {
                None
            };

        let response = self.read_response(pending, &mut progress_handler)?;
        final_response_into_json(response)
    }

    pub fn close(&self) {
        if self.inner.closed.swap(true, Ordering::SeqCst) {
            return;
        }

        self.fail_inflight_requests("Transport closed".to_string());
        self.inner.driver.close();

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
        .thread_name("fastsurfer-transport")
        .build()
        .unwrap_or_else(|error| {
            panic!("Failed to initialize transport runtime: {error}")
        })
}

fn final_response_into_json(
    response: TransportMessage,
) -> Result<serde_json::Value, String> {
    if response.kind != TransportMessageKind::Final {
        return Err(
            "Transport returned unexpected message kind for final response"
                .to_string(),
        );
    }

    response.payload.into_json()
}

#[allow(clippy::needless_pass_by_value)]
async fn protocol_loop(
    mut driver_event_rx: mpsc::Receiver<DriverEvent>,
    message_tx: mpsc::Sender<TransportMessage>,
    output_tx: mpsc::Sender<String>,
    inner: Arc<TransportInner>,
) {
    while let Some(event) = driver_event_rx.recv().await {
        match event {
            DriverEvent::Frame(frame) => {
                match inner.protocol.decode_inbound(frame) {
                    Ok(messages) => {
                        for message in messages {
                            if message.kind == TransportMessageKind::Log {
                                let _ = output_tx
                                    .try_send(message.payload.into_log_line());
                                continue;
                            }

                            if message_tx.send(message).await.is_err() {
                                inner.closed.store(true, Ordering::SeqCst);
                                return;
                            }
                        }
                    }
                    Err(error) => {
                        let _ = message_tx
                            .send(TransportMessage::terminal_error(error))
                            .await;
                        inner.closed.store(true, Ordering::SeqCst);
                        return;
                    }
                }
            }
            DriverEvent::Fatal(error) => {
                let _ = message_tx
                    .send(TransportMessage::terminal_error(error))
                    .await;
                inner.closed.store(true, Ordering::SeqCst);
                return;
            }
        }
    }

    let _ = message_tx
        .send(TransportMessage::terminal_error(
            "Transport protocol loop stopped".to_string(),
        ))
        .await;
    inner.closed.store(true, Ordering::SeqCst);
}

#[allow(clippy::needless_pass_by_value)]
async fn dispatcher_loop(
    mut response_rx: mpsc::Receiver<TransportMessage>,
    inner: Arc<TransportInner>,
) {
    while let Some(event) = response_rx.recv().await {
        match event.kind {
            TransportMessageKind::Progress => {
                let request_id = event.correlation_id;
                if let Some(sender) = find_progress_sender(&inner, request_id)
                    && sender
                        .send(InflightEvent::Progress(event))
                        .await
                        .is_err()
                {
                    remove_sender(&inner, request_id);
                }
            }
            TransportMessageKind::Final => {
                let request_id = event.correlation_id;
                let Some(sender) = take_response_sender(&inner, request_id)
                else {
                    let error = match request_id {
                        Some(id) => {
                            format!(
                                "Received transport response for unknown request id {id}"
                            )
                        }
                        None => "Received transport response without a request id while multiple requests may be in flight"
                            .to_string(),
                    };
                    fail_inflight_requests(&inner, error);
                    break;
                };

                let _ = sender.send(InflightEvent::Final(event)).await;
            }
            TransportMessageKind::TerminalError => {
                let error = match event.payload {
                    MessagePayload::Text(text) => text,
                    MessagePayload::Json(value) => value.to_string(),
                    MessagePayload::Binary(bytes) => {
                        format!(
                            "Terminal transport error with {} bytes",
                            bytes.len()
                        )
                    }
                };
                fail_inflight_requests(&inner, error);
                inner.closed.store(true, Ordering::SeqCst);
                break;
            }
            TransportMessageKind::Request | TransportMessageKind::Log => {}
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
