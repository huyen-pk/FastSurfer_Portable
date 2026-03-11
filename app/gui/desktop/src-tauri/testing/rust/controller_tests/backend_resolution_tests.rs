// This test suite integrates with test-containers for environment isolation.
use crate::prediction::run_fastsurfer_inference_with_app_state;
use crate::process_mgmt::{resolve_backend_binary_path_from, resolve_backend_launch_command_from};
use std::fs;
use std::path::PathBuf;

use super::support::next_test_id;

#[test]
fn resolve_backend_binary_from_paths_should_return_first_existing_candidate() {
    let root = std::env::temp_dir().join(format!("desktop_test_root_{}", next_test_id()));
    let exe_dir = root.join("bin/app");
    let candidate_dir = root.join("backend");
    let candidate_path = candidate_dir.join("main");

    fs::create_dir_all(&exe_dir).expect("failed to create mock exe dir");
    fs::create_dir_all(&candidate_dir).expect("failed to create mock backend dir");
    fs::write(&candidate_path, b"mock").expect("failed to create mock backend binary");

    let result = resolve_backend_binary_path_from(&root, &exe_dir)
        .expect("expected backend path resolution to succeed");

    assert_eq!(result, candidate_path);

    fs::remove_dir_all(&root).expect("failed to cleanup mock root");
}

#[test]
fn resolve_backend_binary_from_paths_should_return_error_when_no_candidate_exists() {
    let root = std::env::temp_dir().join(format!("desktop_test_root_{}", next_test_id()));
    let exe_dir = root.join("bin/app");

    fs::create_dir_all(&exe_dir).expect("failed to create mock exe dir");

    let result = resolve_backend_binary_path_from(&root, &exe_dir);

    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        "Could not find bundled backend binary app/gui/desktop/backend/main"
    );

    fs::remove_dir_all(&root).expect("failed to cleanup mock root");
}

#[test]
fn resolve_backend_launch_command_should_prefer_bundled_binary_when_available() {
    let root = std::env::temp_dir().join(format!("desktop_test_launch_root_{}", next_test_id()));
    let exe_dir = root.join("target/debug");
    let backend_dir = root.join("backend");

    fs::create_dir_all(&exe_dir).expect("failed to create mock exe dir");
    fs::create_dir_all(&backend_dir).expect("failed to create mock backend dir");

    let binary_path = backend_dir.join("main");
    fs::write(&binary_path, b"mock-binary").expect("failed to create mock backend binary");

    let script_path = root.join("backend/ipc_server.py");
    fs::write(&script_path, b"print('mock')").expect("failed to create mock python backend script");

    let launch = resolve_backend_launch_command_from(&root, &exe_dir)
        .expect("expected launch command resolution to succeed");

    assert_eq!(PathBuf::from(launch.program), binary_path);
    assert!(launch.args.is_empty());

    fs::remove_dir_all(&root).expect("failed to cleanup mock root");
}

#[test]
fn run_fastsurfer_inference_with_unavailable_backend_should_return_clear_error() {
    let result = run_fastsurfer_inference_with_app_state(
        None,
        Some("Could not resolve backend launch command"),
        vec!["/in/a.nii.gz".to_string()],
        vec![],
    );

    let error = match result {
        Ok(_) => panic!("expected unavailable backend to return an error"),
        Err(error) => error,
    };
    assert!(error.contains("Backend is unavailable in this desktop runtime"));
    assert!(error.contains("Could not resolve backend launch command"));
}
