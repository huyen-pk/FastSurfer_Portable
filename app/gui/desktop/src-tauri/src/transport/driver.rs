use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;
use tokio::sync::{mpsc, oneshot};

pub const REQUEST_CHANNEL_CAPACITY: usize = 32;
pub const DRIVER_EVENT_CHANNEL_CAPACITY: usize = 128;

#[derive(Debug, Clone)]
pub enum DriverFrame {
    Text(String),
    Binary(Vec<u8>),
}

pub struct OutboundDriverFrame {
    pub correlation_id: u64,
    pub frame: DriverFrame,
    pub write_result_tx: oneshot::Sender<Result<(), String>>,
}

#[derive(Debug)]
pub enum DriverEvent {
    Frame(DriverFrame),
    Fatal(String),
}

pub trait TransportDriver: Send + 'static {
    /// Starts the concrete transport driver and returns a handle for outbound writes.
    ///
    /// # Errors
    /// Returns an error when the driver cannot initialize its runtime tasks.
    fn start(
        self: Box<Self>,
        runtime: &Runtime,
        event_tx: mpsc::Sender<DriverEvent>,
    ) -> Result<DriverHandle, String>;
}

#[derive(Clone)]
pub struct DriverHandle {
    sender: Arc<Mutex<Option<mpsc::Sender<OutboundDriverFrame>>>>,
}

impl DriverHandle {
    #[must_use]
    pub fn new(sender: mpsc::Sender<OutboundDriverFrame>) -> Self {
        Self {
            sender: Arc::new(Mutex::new(Some(sender))),
        }
    }

    /// Queues an outbound frame for the driver writer loop.
    ///
    /// # Errors
    /// Returns an error when the driver request channel is unavailable.
    pub fn send_frame(&self, frame: OutboundDriverFrame) -> Result<(), String> {
        let sender = self
            .sender
            .lock()
            .map_err(|_| "Driver sender mutex was poisoned".to_string())?
            .clone()
            .ok_or_else(|| "Driver request channel is closed".to_string())?;

        sender
            .blocking_send(frame)
            .map_err(|_| "Failed to queue outbound driver frame".to_string())
    }

    pub fn close(&self) {
        if let Ok(mut sender) = self.sender.lock() {
            let _ = sender.take();
        }
    }
}
