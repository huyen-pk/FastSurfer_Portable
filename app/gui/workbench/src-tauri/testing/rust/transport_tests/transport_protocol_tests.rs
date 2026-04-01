use crate::transport::driver::DriverFrame;
use crate::transport::ipc_protocol::FastSurferJsonLineProtocol;
use crate::transport::message::{
    MessagePayload, TransportMessage, TransportMessageKind,
};
use crate::transport::protocol::TransportProtocol;
use serde_json::json;

#[test]
fn fastsurfer_protocol_should_classify_progress_final_and_log_frames() {
    let protocol = FastSurferJsonLineProtocol::new();

    let progress = protocol
        .decode_inbound(DriverFrame::Text(
            r#"{"id":9,"event":"progress","progress":33,"message":"Slice 1"}"#
                .to_string(),
        ))
        .expect("progress frame should decode");
    assert_eq!(progress[0].kind, TransportMessageKind::Progress);

    let final_response = protocol
        .decode_inbound(DriverFrame::Text(
            r#"{"id":9,"ok":true,"result":{"ack_message":"ok"}}"#.to_string(),
        ))
        .expect("final frame should decode");
    assert_eq!(final_response[0].kind, TransportMessageKind::Final);

    let log = protocol
        .decode_inbound(DriverFrame::Text("startup banner".to_string()))
        .expect("log frame should decode");
    assert_eq!(log[0].kind, TransportMessageKind::Log);
}

#[test]
fn fastsurfer_protocol_should_encode_json_request_frames() {
    let protocol = FastSurferJsonLineProtocol::new();
    let request =
        TransportMessage::request(4, "health", MessagePayload::Json(json!({})));

    let encoded = protocol
        .encode_request(&request)
        .expect("request should encode");

    let DriverFrame::Text(payload) = encoded else {
        panic!("expected text payload");
    };
    assert!(payload.contains("\"id\":4"));
    assert!(payload.contains("\"method\":\"health\""));
    assert!(payload.ends_with('\n'));
}
