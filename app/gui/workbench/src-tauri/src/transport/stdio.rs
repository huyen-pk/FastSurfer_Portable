use super::driver::{
    DriverEvent, DriverFrame, DriverHandle, OutboundDriverFrame,
    REQUEST_CHANNEL_CAPACITY, TransportDriver,
};
use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, ChildStdout};
use tokio::runtime::Runtime;
use tokio::sync::mpsc;

pub struct StdioTransportDriver {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl StdioTransportDriver {
    #[must_use]
    pub fn new(stdin: ChildStdin, stdout: BufReader<ChildStdout>) -> Self {
        Self { stdin, stdout }
    }
}

impl TransportDriver for StdioTransportDriver {
    fn start(
        self: Box<Self>,
        runtime: &Runtime,
        event_tx: mpsc::Sender<DriverEvent>,
    ) -> Result<DriverHandle, String> {
        let Self { stdin, stdout } = *self;
        let (request_tx, request_rx) = mpsc::channel(REQUEST_CHANNEL_CAPACITY);

        let writer_event_tx = event_tx.clone();
        runtime.spawn_blocking(move || {
            writer_loop(stdin, request_rx, writer_event_tx);
        });
        runtime.spawn_blocking(move || {
            reader_loop(stdout, event_tx);
        });

        Ok(DriverHandle::new(request_tx))
    }
}

#[allow(clippy::needless_pass_by_value)]
fn writer_loop(
    mut stdin: ChildStdin,
    mut request_rx: mpsc::Receiver<OutboundDriverFrame>,
    event_tx: mpsc::Sender<DriverEvent>,
) {
    while let Some(request) = request_rx.blocking_recv() {
        let DriverFrame::Text(payload) = request.frame else {
            let error = "Stdio driver only supports text frames".to_string();
            let _ = request.write_result_tx.send(Err(error.clone()));
            let _ = event_tx.blocking_send(DriverEvent::Fatal(error));
            break;
        };

        eprintln!(
            "[trace][transport-stdio] -> request_id={} payload={}",
            request.correlation_id,
            payload.trim()
        );

        let write_result = stdin
            .write_all(payload.as_bytes())
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
            let _ = event_tx.blocking_send(DriverEvent::Fatal(error));
            break;
        }
    }
}

#[allow(clippy::needless_pass_by_value)]
fn reader_loop(
    mut stdout: BufReader<ChildStdout>,
    event_tx: mpsc::Sender<DriverEvent>,
) {
    let mut line_buffer = String::new();

    loop {
        line_buffer.clear();
        let read_bytes = match stdout.read_line(&mut line_buffer) {
            Ok(read_bytes) => read_bytes,
            Err(error) => {
                let _ = event_tx.blocking_send(DriverEvent::Fatal(format!(
                    "Failed to read backend response: {error}"
                )));
                break;
            }
        };

        if read_bytes == 0 {
            let _ = event_tx.blocking_send(DriverEvent::Fatal(
                "Backend process exited before sending a response".to_string(),
            ));
            break;
        }

        let trimmed = line_buffer.trim();
        if trimmed.is_empty() {
            continue;
        }

        if event_tx
            .blocking_send(DriverEvent::Frame(DriverFrame::Text(
                trimmed.to_string(),
            )))
            .is_err()
        {
            break;
        }
    }
}
