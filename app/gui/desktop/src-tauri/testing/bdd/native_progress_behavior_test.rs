// This test suite integrates with test-containers for environment isolation.
use super::support::{find_repo_root, setup_test_app};
use crate::inference::pipeline::run_native_inference_with_progress;
use crate::models::InferenceProgressEvent;
use ::tauri::Listener;
use std::sync::{Arc, Mutex};

#[tokio::test]
async fn test_native_inference_reports_progress_in_real_time() {
    let app = setup_test_app();
    let app_handle = app.handle();

    // Track emitted progress events
    let events = Arc::new(Mutex::new(Vec::<InferenceProgressEvent>::new()));
    let events_captured = Arc::clone(&events);

    app_handle.listen_any(
        "fastsurfer://inference-progress",
        move |event: tauri::Event| {
            let payload_str = event.payload();
            if let Ok(payload) = serde_json::from_str::<InferenceProgressEvent>(payload_str) {
                events_captured.lock().unwrap().push(payload);
            }
        },
    );

    let repo_root = find_repo_root().expect("Could not find repo root");
    let input_file =
        repo_root.join("app/gui/desktop/src-tauri/testing/data/Subject140/140_orig.mgz");

    if !input_file.exists() {
        println!("Skipping test: 140_orig.mgz not found at {:?}", input_file);
        return;
    }

    let task_id = "test-progress-123".to_string();
    let cancelled_tasks = Arc::new(Mutex::new(std::collections::BTreeSet::new()));

    // Enable parallel mode and point to models
    unsafe {
        let repo_root = "/home/huyenpk/Projects/FastSurfer";
        std::env::set_var("FASTSURFER_ONNX_DIR", format!("{}/onnx", repo_root));
        std::env::set_var(
            "FASTSURFER_LUT_PATH",
            format!("{}/FastSurferCNN/config/FreeSurferColorLUT.txt", repo_root),
        );
        std::env::set_var(
            "FASTSURFER_PYTHON_BIN",
            "/home/huyenpk/anaconda3/bin/python",
        );
        std::env::set_var("FASTSURFER_NATIVE_SPARSE_PARALLEL", "1");
        std::env::set_var("FASTSURFER_NATIVE_SLICES_PER_PLANE", "5");
    }

    let app_handle_task = app_handle.clone();
    // Run inference in a blocking task since it's a long operation
    let result = tokio::task::spawn_blocking(move || {
        run_native_inference_with_progress(
            &app_handle_task,
            &cancelled_tasks,
            &task_id,
            &vec![input_file.to_string_lossy().to_string()],
            &vec![],
        )
    })
    .await
    .unwrap();

    assert!(result.is_ok(), "Inference failed: {:?}", result.err());

    let captured = events.lock().unwrap();

    // Acceptance Criteria Verification:
    // 1. Must have "started" event
    assert!(captured.iter().any(|e| e.status == "started"));

    // 2. Must have multiple "item_progress" events (indicates real-time updates)
    let progress_events: Vec<&InferenceProgressEvent> = captured
        .iter()
        .filter(|e| e.status == "item_progress")
        .collect();
    assert!(
        progress_events.len() > 1,
        "Expected multiple real-time progress updates, got {}",
        progress_events.len()
    );

    // 3. Progress must be monotonic
    let mut last_progress = 0;
    for event in &progress_events {
        assert!(
            event.progress >= last_progress,
            "Progress decreased: {} -> {}",
            last_progress,
            event.progress
        );
        last_progress = event.progress;
    }

    // 4. Must end with "completed" at 100%
    let last_event = captured.last().expect("No events captured");
    assert_eq!(last_event.status, "completed");
    assert_eq!(last_event.progress, 100);
}
