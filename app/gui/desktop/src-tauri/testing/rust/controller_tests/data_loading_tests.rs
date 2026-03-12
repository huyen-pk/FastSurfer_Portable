// This test suite integrates with test-containers for environment isolation.
use super::support::{
    create_backend_state_via_shell_script,
    create_backend_state_with_fake_responses, find_repo_root,
    fixture_native_input, is_ci, resolve_python_with_component_runtime,
    spawn_python_backend_inline,
};
use crate::inference::preprocess::load_input_volume;
use serde_json::{Value, json};
use std::process::Command;

#[test]
fn run_ipc_request_with_ok_response_should_return_result_value() {
    let backend = create_backend_state_with_fake_responses(&[
        r#"{"ok":true,"result":{"ack_message":"Accepted"}}"#,
    ]);

    let result = backend
        .run_ipc_request("ping", json!({ "hello": "world" }))
        .expect("expected ipc request to succeed");

    assert_eq!(
        result.get("ack_message"),
        Some(&Value::String("Accepted".to_string()))
    );
}

#[test]
fn run_ipc_request_with_non_json_prelude_should_wait_for_json_response() {
    let backend = create_backend_state_via_shell_script(
        "while IFS= read -r _line; do printf '%s\\n' 'backend log line'; printf '%s\\n' '{\"ok\":true,\"result\":{\"ack_message\":\"Accepted\"}}'; done",
    );

    let result = backend
        .run_ipc_request("ping", json!({ "hello": "world" }))
        .expect(
            "expected ipc request to ignore prelude and parse json response",
        );

    assert_eq!(
        result.get("ack_message"),
        Some(&Value::String("Accepted".to_string()))
    );
}

#[test]
fn run_ipc_request_with_real_python_process_should_pipe_request_and_receive_response()
 {
    let backend = spawn_python_backend_inline(
        "import json,sys\nfor line in sys.stdin:\n line=line.strip()\n if not line: continue\n req=json.loads(line)\n sys.stdout.write(json.dumps({'ok': True, 'result': {'echo_method': req.get('method')}}) + '\\n')\n sys.stdout.flush()",
    );

    let Some(backend) = backend else {
        if is_ci() {
            eprintln!(
                "python executable unavailable in CI; skipping real python subprocess test"
            );
            return;
        }
        panic!(
            "python executable unavailable in local environment; set FASTSURFER_PYTHON_BIN or install python3"
        );
    };

    let result = backend.run_ipc_request("health", json!({})).expect(
        "expected run_ipc_request to communicate with real python process",
    );

    assert_eq!(
        result.get("echo_method"),
        Some(&Value::String("health".to_string()))
    );
}

#[test]
fn run_ipc_request_with_backend_error_should_return_formatted_error() {
    let backend = create_backend_state_with_fake_responses(&[
        r#"{"ok":false,"error":{"message":"boom"}}"#,
    ]);

    let result = backend.run_ipc_request("predict_batch", json!({}));

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Backend predict_batch failed: boom");
}

#[test]
fn run_ipc_request_with_invalid_json_should_return_invalid_json_error() {
    let backend = create_backend_state_with_fake_responses(&["{\"ok\":true"]);

    let result = backend.run_ipc_request("predict_batch", json!({}));

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .starts_with("Invalid IPC response JSON:")
    );
}

#[test]
fn parity_data_loading_should_match_python_load_and_conform_image_component() {
    let Some(repo_root) = find_repo_root() else {
        panic!("failed to locate repo root for data loading parity test");
    };

    let input_nii = fixture_native_input(&repo_root);
    if !input_nii.exists() {
        eprintln!(
            "missing fixture input '{}'; skipping data loading parity test",
            input_nii.display()
        );
        return;
    }

    let Some(python_bin) = resolve_python_with_component_runtime(&repo_root)
    else {
        eprintln!(
            "python runtime missing required FastSurfer component deps; skipping data loading parity test"
        );
        return;
    };

    let rust_volume = load_input_volume(&input_nii.to_string_lossy())
        .expect("failed to load fixture input with Rust data loader");

    let shape = rust_volume.shape_xyz;
    let coords = vec![
        vec![0usize, 0usize, 0usize],
        vec![shape[0] / 2, shape[1] / 2, shape[2] / 2],
        vec![
            shape[0].saturating_sub(1),
            shape[1].saturating_sub(1),
            shape[2].saturating_sub(1),
        ],
        vec![shape[0] / 4, shape[1] / 3, shape[2] / 5],
        vec![shape[0] / 3, shape[1] / 5, shape[2] / 7],
    ];

    let script = r#"
import json
import sys

from FastSurferCNN.data_loader.data_utils import load_and_conform_image

in_path = sys.argv[1]
coords = json.loads(sys.argv[2])
header, _affine, data = load_and_conform_image(in_path, vox_size=1.0)
shape = list(data.shape[:3])
zoom = [float(v) for v in header.get_zooms()[:3]]
samples = [float(data[x, y, z]) for x, y, z in coords]
print(json.dumps({"shape": shape, "zoom": zoom, "samples": samples}))
"#;

    let output = Command::new(&python_bin)
        .arg("-c")
        .arg(script)
        .arg(&input_nii)
        .arg(
            serde_json::to_string(&coords)
                .expect("failed to serialize coordinate list"),
        )
        .current_dir(&repo_root)
        .env("PYTHONPATH", repo_root.to_string_lossy().to_string())
        .output()
        .expect("failed to execute python data-loading parity script");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("python data-loading parity failed: {stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: Value = serde_json::from_str(stdout.trim())
        .expect("failed to parse python data-loading parity json output");

    let py_shape = parsed["shape"]
        .as_array()
        .expect("python output missing shape")
        .iter()
        .map(|v| v.as_u64().expect("invalid shape value") as usize)
        .collect::<Vec<usize>>();
    assert_eq!(py_shape, rust_volume.shape_xyz.to_vec(), "shape mismatch");

    let py_zoom = parsed["zoom"]
        .as_array()
        .expect("python output missing zoom")
        .iter()
        .map(|v| v.as_f64().expect("invalid zoom value") as f32)
        .collect::<Vec<f32>>();

    for (idx, py) in py_zoom.iter().enumerate() {
        let rust = rust_volume.zoom_xyz[idx];
        assert!(
            (rust - *py).abs() <= 1e-5,
            "zoom mismatch at axis {}: rust={} python={}",
            idx,
            rust,
            py
        );
    }

    let py_samples = parsed["samples"]
        .as_array()
        .expect("python output missing samples")
        .iter()
        .map(|v| v.as_f64().expect("invalid sample value") as f32)
        .collect::<Vec<f32>>();

    for (i, coord) in coords.iter().enumerate() {
        let [x, y, z] = [coord[0], coord[1], coord[2]];
        let rust_offset = (x * shape[1] * shape[2]) + (y * shape[2]) + z;
        let rust_value = rust_volume.data_xyz[rust_offset];
        let py_value = py_samples[i];
        assert!(
            (rust_value - py_value).abs() <= 1e-5,
            "sample mismatch at {:?}: rust={} python={}",
            coord,
            rust_value,
            py_value
        );
    }
}
