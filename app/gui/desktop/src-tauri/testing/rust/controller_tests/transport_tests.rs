// This test suite integrates with test-containers for environment isolation.
use super::support::{
    find_repo_root, fixture_native_input, spawn_bundled_backend_state,
};
use crate::backend::BackendManager;
use serde_json::{Value, json};
use std::sync::Arc;
use std::thread;

fn request_json(
    backend: &BackendManager,
    method: &str,
    params: Value,
) -> Result<Value, String> {
    backend
        .transport
        .lock()
        .map_err(|_| "Backend transport mutex was poisoned".to_string())?
        .request_json(method, params, None)
}

#[test]
fn transport_request_json_with_ping_should_return_backend_error_envelope() {
    let backend = spawn_bundled_backend_state()
        .expect("expected bundled backend to start");

    let result = request_json(&backend, "ping", json!({ "hello": "world" }))
        .expect("expected unsupported ping request to return a backend error envelope");

    assert_eq!(result.get("ok"), Some(&Value::Bool(false)));
    assert_eq!(
        result.get("error").and_then(|value| value.get("message")),
        Some(&Value::String("Unknown method: ping".to_string()))
    );
}

#[test]
fn transport_request_json_with_health_should_return_status_ok() {
    let backend = spawn_bundled_backend_state()
        .expect("expected bundled backend to start");

    let result = request_json(&backend, "health", json!({}))
        .expect("expected transport request to succeed");

    assert_eq!(
        result.get("result").and_then(|value| value.get("status")),
        Some(&Value::String("ok".to_string()))
    );
}

#[test]
fn transport_request_json_with_valid_resolve_requested_paths_should_return_requested_paths()
 {
    let repo_root = find_repo_root().expect("failed to locate repo root");
    let input_nii = fixture_native_input(&repo_root);
    let backend = spawn_bundled_backend_state()
        .expect("expected bundled backend to start");

    let result = request_json(
        &backend,
        "resolve_requested_paths",
        json!({ "file_paths": [input_nii.to_string_lossy().to_string()], "folder_paths": [] }),
    )
    .expect("expected resolve_requested_paths to succeed with the real fixture path");

    assert_eq!(
        result
            .get("result")
            .and_then(|value| value.get("requested_paths"))
            .and_then(Value::as_array)
            .map(Vec::len),
        Some(1)
    );
}

#[test]
fn transport_request_json_with_concurrent_interleaving_should_route_by_request_id()
 {
    let repo_root = find_repo_root().expect("failed to locate repo root");
    let input_nii = fixture_native_input(&repo_root)
        .to_string_lossy()
        .to_string();
    let backend = Arc::new(
        spawn_bundled_backend_state()
            .expect("expected bundled backend to start"),
    );

    let first_backend = Arc::clone(&backend);
    let first = thread::spawn(move || {
        request_json(&first_backend, "health", json!({}))
            .expect("expected first request to succeed")
    });

    let second_backend = Arc::clone(&backend);
    let second = thread::spawn(move || {
        request_json(
            &second_backend,
            "resolve_requested_paths",
            json!({ "file_paths": [input_nii], "folder_paths": [] }),
        )
        .expect("expected second request to succeed")
    });

    let first_result = first.join().expect("first request thread should join");
    let second_result =
        second.join().expect("second request thread should join");

    assert_eq!(
        first_result
            .get("result")
            .and_then(|value| value.get("status")),
        Some(&Value::String("ok".to_string()))
    );
    assert_eq!(
        second_result
            .get("result")
            .and_then(|value| value.get("requested_paths"))
            .and_then(Value::as_array)
            .map(Vec::len),
        Some(1)
    );
}

#[test]
fn transport_request_json_with_unknown_method_should_return_backend_error_envelope()
 {
    let backend = spawn_bundled_backend_state()
        .expect("expected bundled backend to start");

    let result = request_json(&backend, "missing_method", json!({}))
        .expect("expected backend error envelope for unknown method");

    assert_eq!(
        result.get("error").and_then(|value| value.get("message")),
        Some(&Value::String("Unknown method: missing_method".to_string()))
    );
}

#[test]
fn transport_request_json_with_backend_error_should_return_backend_envelope() {
    let backend = spawn_bundled_backend_state()
        .expect("expected bundled backend to start");

    let result = request_json(
        &backend,
        "predict_batch",
        json!({ "file_paths": [], "folder_paths": [] }),
    )
    .expect("expected transport to return backend error envelope");

    assert_eq!(result.get("ok"), Some(&Value::Bool(false)));
    assert_eq!(
        result.get("error").and_then(|value| value.get("message")),
        Some(&Value::String(
            "No valid input image files selected (.nii, .nii.gz, .mgz, .mgh)."
                .to_string()
        ))
    );
}
