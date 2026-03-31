use crate::backend::subprocess::BackendProcess;
use crate::transport::factory::create_fastsurfer_stdio_transport;
use crate::transport::message::{MessagePayload, TransportMessage};
use serde_json::json;
use std::io::BufReader;
use std::process::{Child, Command, Stdio};

fn spawn_transport_via_shell_script(
    script: &str,
) -> (Child, crate::transport::Transport) {
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("failed to spawn scripted transport process");

    let stdin = child
        .stdin
        .take()
        .expect("failed to capture scripted transport stdin");
    let stdout = child
        .stdout
        .take()
        .expect("failed to capture scripted transport stdout");

    create_fastsurfer_stdio_transport(BackendProcess {
        child,
        stdin,
        stdout: BufReader::new(stdout),
    })
    .expect("failed to create transport from scripted process")
}

#[test]
fn transport_stdio_driver_should_forward_frames_and_drain_logs() {
    let (mut child, transport) = spawn_transport_via_shell_script(
        "while IFS= read -r _line; do printf '%s\\n' 'backend log line'; printf '%s\\n' '{\"id\":1,\"ok\":true,\"result\":{\"ack_message\":\"Accepted\"}}'; break; done",
    );

    let pending = transport
        .send_request("ping", MessagePayload::Json(json!({"hello":"world"})))
        .expect("expected request to be queued");
    let mut no_progress: Option<&mut dyn FnMut(TransportMessage)> = None;
    let response = transport
        .read_response(pending, &mut no_progress)
        .expect("expected final response");
    let value = response
        .payload
        .into_json()
        .expect("expected JSON final response payload");

    assert_eq!(value["result"]["ack_message"], json!("Accepted"));
    assert_eq!(
        transport.drain_output_lines(),
        vec!["backend log line".to_string()]
    );

    transport.close();
    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn transport_stdio_driver_should_wake_waiters_when_backend_exits_mid_stream() {
    let (mut child, transport) = spawn_transport_via_shell_script(
        "while IFS= read -r _line; do printf '%s\\n' '{\"id\":1,\"event\":\"progress\",\"progress\":25,\"message\":\"Starting\"}'; exit 0; done",
    );

    let pending = transport
        .send_request(
            "predict",
            MessagePayload::Json(json!({"input_path":"/tmp/in.mgz"})),
        )
        .expect("expected request to be queued");

    let mut seen_progress = Vec::<usize>::new();
    let mut on_progress = |message: TransportMessage| {
        let progress = message
            .payload
            .as_json()
            .and_then(|value| value.get("progress"))
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .expect("expected progress payload");
        seen_progress.push(progress);
    };
    let mut progress_handler: Option<&mut dyn FnMut(TransportMessage)> =
        Some(&mut on_progress);

    let error = transport
        .read_response(pending, &mut progress_handler)
        .expect_err("expected transport error after backend exit");

    assert_eq!(seen_progress, vec![25]);
    assert_eq!(error, "Backend process exited before sending a response");

    transport.close();
    let _ = child.kill();
    let _ = child.wait();
}
