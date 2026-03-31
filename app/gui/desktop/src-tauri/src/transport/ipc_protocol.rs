use super::driver::DriverFrame;
use super::message::{MessagePayload, TransportMessage, TransportMessageKind};
use super::protocol::TransportProtocol;
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Default)]
pub struct FastSurferJsonLineProtocol;

impl FastSurferJsonLineProtocol {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl TransportProtocol for FastSurferJsonLineProtocol {
    fn encode_request(
        &self,
        message: &TransportMessage,
    ) -> Result<DriverFrame, String> {
        let method = message.metadata.get("method").ok_or_else(|| {
            "Transport request missing method metadata".to_string()
        })?;
        let correlation_id = message.correlation_id.ok_or_else(|| {
            "Transport request missing correlation id".to_string()
        })?;
        let params = message.payload.as_json().cloned().ok_or_else(|| {
            "FastSurfer JSON-line protocol requires JSON request payloads"
                .to_string()
        })?;

        let request = json!({
            "id": correlation_id,
            "method": method,
            "params": params,
        });

        Ok(DriverFrame::Text(format!("{request}\n")))
    }

    fn decode_inbound(
        &self,
        frame: DriverFrame,
    ) -> Result<Vec<TransportMessage>, String> {
        let DriverFrame::Text(line) = frame else {
            return Err(
                "FastSurfer JSON-line protocol only supports text frames"
                    .to_string(),
            );
        };

        if !line.starts_with('{') {
            return Ok(vec![TransportMessage {
                correlation_id: None,
                kind: TransportMessageKind::Log,
                metadata: BTreeMap::default(),
                payload: MessagePayload::Text(line),
            }]);
        }

        let value = serde_json::from_str::<Value>(&line)
            .map_err(|error| format!("Invalid IPC response JSON: {error}"))?;
        let correlation_id = value.get("id").and_then(Value::as_u64);

        if value.get("event").and_then(Value::as_str) == Some("progress") {
            return Ok(vec![TransportMessage {
                correlation_id,
                kind: TransportMessageKind::Progress,
                metadata: BTreeMap::default(),
                payload: MessagePayload::Json(value),
            }]);
        }

        if value.get("ok").is_some() {
            return Ok(vec![TransportMessage {
                correlation_id,
                kind: TransportMessageKind::Final,
                metadata: BTreeMap::default(),
                payload: MessagePayload::Json(value),
            }]);
        }

        Ok(vec![TransportMessage {
            correlation_id,
            kind: TransportMessageKind::Log,
            metadata: BTreeMap::default(),
            payload: MessagePayload::Json(value),
        }])
    }
}
