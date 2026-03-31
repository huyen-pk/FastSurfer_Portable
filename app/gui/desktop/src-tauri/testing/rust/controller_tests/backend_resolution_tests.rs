// This test suite integrates with test-containers for environment isolation.
use crate::backend::BackendManager;
use crate::backend::subprocess::{
    resolve_backend_binary_path_from, resolve_backend_launch_command_from,
};
use serde_json::Value;
use std::path::PathBuf;

fn setup_repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("failed to compute repo root from manifest dir")
        .to_path_buf()
}

#[test]
fn resolve_backend_binary_from_repo_paths_should_find_bundled_binary() {
    // Avoid relying on `find_repo_root` (which probes for Python scripts).
    // Use the compile-time manifest directory and walk up to the repository root.
    let repo_root = setup_repo_root();

    let exe_dir = repo_root.join("app/gui/desktop/src-tauri");
    let candidate_path = repo_root.join("app/gui/desktop/backend/main");

    let result = resolve_backend_binary_path_from(&repo_root, &exe_dir)
        .expect("expected backend path resolution to succeed");

    assert_eq!(result, candidate_path);
}

#[test]
fn bundled_backend_binary_should_respond_to_health_request() {
    let repo_root = setup_repo_root();
    let exe_dir = repo_root.join("app/gui/desktop/src-tauri");
    let launch = resolve_backend_launch_command_from(&repo_root, &exe_dir)
        .expect("expected launch command resolution to succeed");
    let backend_proc = crate::backend::subprocess::spawn_backend_process(
        &launch,
        Some(&repo_root),
    )
    .expect("expected bundled backend to start");

    let backend =
        BackendManager::from_process(backend_proc, launch, Some(repo_root));

    let result = backend
        .health_probe()
        .expect("expected bundled backend binary to respond to health probe");

    assert_eq!(result.get("status"), Some(&Value::String("ok".to_string())));
}

#[test]
fn require_available_with_unavailable_backend_should_return_clear_error() {
    let Err(error) = BackendManager::require_available(
        None,
        Some("Could not resolve backend launch command"),
    ) else {
        panic!("expected unavailable backend to return an error");
    };

    assert!(error.contains("Backend is unavailable in this desktop runtime"));
    assert!(error.contains("Could not resolve backend launch command"));
}
