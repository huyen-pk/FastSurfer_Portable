use super::support::{find_repo_root, run_native_inference_with_timeout};
use crate::models::{InferenceProgressEvent, ProcessingRunResult};
use crate::inference::pipeline::run_native_inference_with_progress;
use std::time::Duration;
use std::sync::{Arc, Mutex};
use std::collections::{BTreeSet, VecDeque};
use tauri::test::mock_app;

#[tokio::test]
async fn native_inference_should_report_real_time_progress_during_parallel_execution() {
    let Some(repo_root) = find_repo_root() else {
        panic!("failed to locate repo root for native bdd test");
    };

    // SETUP: Set environment for sparse parallel execution to speed up test but involve parallel logic
    unsafe {
        std::env::set_var("FASTSURFER_NATIVE_SPARSE_PARALLEL", "1");
        std::env::set_var("FASTSURFER_NATIVE_SLICES_PER_PLANE", "5"); // Total 15 slices
    }

    let input_path = repo_root.join("app/gui/desktop/src-tauri/testing/data/Subject140/140_orig.mgz");
    assert!(input_path.exists());

    let app = mock_app();
    let app_handle = app.handle();
    let cancelled_tasks = Arc::new(Mutex::new(BTreeSet::new()));
    let task_id = "bdd-progress-task-001".to_string();
    
    // We'll use a thread-safe queue to capture events in real-time
    let events = Arc::new(Mutex::new(VecDeque::<InferenceProgressEvent>::new()));
    
    // We'll hook into tauri events if possible, but since app_handle.emit is used internally,
    // we might need to verify the events arrive via the callback if we were using BackendState.
    // However, run_native_inference_with_progress emits directly to tauri.
    // For this BDD test, we want to see events arriving WHILE run_native_inference_with_progress is running.
    
    let events_clone = events.clone();
    let task_id_clone = task_id.clone();
    
    // START: Run inference in a separate task so we can monitor events (simulated by checking the queue)
    // Actually, in our current architecture, the events are emitted. We need a way to listen to them.
    // In Tauri mock_app, we can listen to events.
    
    let (tx, mut rx) = tokio::sync::mpsc::channel::<InferenceProgressEvent>(100);
    
    app_handle.listen_any("fastsurfer://inference-progress", move |event| {
        if let Ok(progress_event) = serde_json::from_str::<InferenceProgressEvent>(event.payload()) {
            if progress_event.task_id == task_id_clone {
                let _ = tx.blocking_send(progress_event);
            }
        }
    });

    let handle = tokio::spawn(async move {
        run_native_inference_with_progress(
            &app_handle,
            &cancelled_tasks,
            &task_id,
            &[input_path.to_string_lossy().to_string()],
            &[],
        )
    });

    let mut captured = Vec::new();
    let mut finished = false;
    
    // Monitor events for a reasonable timeout
    let timeout = tokio::time::sleep(Duration::from_secs(30));
    tokio::pin!(timeout);

    loop {
        tokio::select! {
            Some(event) = rx.recv() => {
                captured.push(event.clone());
                if event.status == "completed" || event.status == "failed" {
                    finished = true;
                }
            }
            _ = &mut timeout => {
                break;
            }
            _ = &mut handle, if !finished => {
                // Task finished but we might still have events in rx
            }
        }
        if finished && rx.is_empty() { break; }
    }

    // ASSERTIONS
    assert!(!captured.is_empty(), "No progress events captured!");
    
    let started = captured.iter().find(|e| e.status == "started").expect("Missing started event");
    assert_eq!(started.completed, 0);
    assert!(started.total > 0, "Total slices should be > 0");

    let items = captured.iter().filter(|e| e.status == "item_progress").collect::<Vec<_>>();
    assert!(!items.is_empty(), "Missing item_progress events DURING execution");

    let completed = captured.iter().find(|e| e.status == "completed").expect("Missing completed event");
    assert_eq!(completed.progress, 100);
    assert_eq!(completed.completed, completed.total);

    // Monotonicity check
    let mut last_progress = 0;
    for event in &captured {
        if event.status == "item_progress" || event.status == "item_completed" || event.status == "completed" {
            assert!(event.progress >= last_progress, "Progress decreased from {} to {}!", last_progress, event.progress);
            last_progress = event.progress;
        }
    }
}
