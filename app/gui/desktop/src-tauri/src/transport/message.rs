use serde_json::Value;
use std::collections::BTreeMap;

pub type TransportMetadata = BTreeMap<String, String>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportMessageKind {
    Request,
    Progress,
    Final,
    Log,
    TerminalError,
}

#[derive(Debug, Clone)]
pub enum MessagePayload {
    Json(Value),
    Text(String),
    Binary(Vec<u8>),
}

impl MessagePayload {
    #[must_use]
    pub fn as_json(&self) -> Option<&Value> {
        match self {
            Self::Json(value) => Some(value),
            Self::Text(_) | Self::Binary(_) => None,
        }
    }

    /// Consumes the payload as JSON.
    ///
    /// # Errors
    /// Returns an error when the payload is not JSON.
    pub fn into_json(self) -> Result<Value, String> {
        match self {
            Self::Json(value) => Ok(value),
            Self::Text(_) | Self::Binary(_) => {
                Err("Expected JSON transport payload".to_string())
            }
        }
    }

    #[must_use]
    pub fn into_log_line(self) -> String {
        match self {
            Self::Json(value) => value.to_string(),
            Self::Text(text) => text,
            Self::Binary(bytes) => {
                format!("<binary payload: {} bytes>", bytes.len())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransportMessage {
    pub correlation_id: Option<u64>,
    pub kind: TransportMessageKind,
    pub metadata: TransportMetadata,
    pub payload: MessagePayload,
}

impl TransportMessage {
    #[must_use]
    pub fn request(
        correlation_id: u64,
        method: &str,
        payload: MessagePayload,
    ) -> Self {
        let mut metadata = TransportMetadata::new();
        metadata.insert("method".to_string(), method.to_string());

        Self {
            correlation_id: Some(correlation_id),
            kind: TransportMessageKind::Request,
            metadata,
            payload,
        }
    }

    #[must_use]
    pub fn terminal_error(error: String) -> Self {
        Self {
            correlation_id: None,
            kind: TransportMessageKind::TerminalError,
            metadata: TransportMetadata::new(),
            payload: MessagePayload::Text(error),
        }
    }
}
