// This test suite integrates with test-containers for environment isolation.
use super::support::{find_repo_root, setup_test_app};
use crate::inference::pipeline::run_native_inference_with_progress;
use crate::models::InferenceProgressEvent;
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::Listener;

#[tokio::test]
async fn test_native_inference_reports_progress_in_real_time() {
    let app = setup_test_app();
    let app_handle = app.handle();
    let repo_root = find_repo_root()
        .expect("failed to locate repo root for native bdd test");

    // REQUIREMENT: The following environment variables must be set externally for this BDD test to run.
    // We skip if not set to avoid unsafe { std::env::set_var(...) } which is denied in this crate.
    let required_vars = ["FASTSURFER_PYTHON_BIN"];

    for var in required_vars {
        if std::env::var(var).is_err() {
            println!("Skipping BDD test: {} is not set.", var);
            return;
        }
    }

    let input_path = repo_root
        .join("app/gui/desktop/src-tauri/testing/data/Subject140/140_orig.mgz");
    if !input_path.exists() {
        println!("Skipping test: input file not found");
        return;
    }

    let cancelled_tasks = Arc::new(Mutex::new(BTreeSet::new()));
    let task_id = "bdd-progress-task-001".to_string();

    let (tx, mut rx) =
        tokio::sync::mpsc::channel::<InferenceProgressEvent>(100);

    let task_id_clone = task_id.clone();
    app_handle.listen_any("fastsurfer://inference-progress", move |event| {
        if let Ok(progress_event) =
            serde_json::from_str::<InferenceProgressEvent>(event.payload())
        {
            if progress_event.task_id == task_id_clone {
                let _ = tx.blocking_send(progress_event);
            }
        }
    });

    let app_handle_task = app_handle.clone();
    let task_id_task = task_id.clone();
    let input_task = input_path.to_string_lossy().to_string();

    let handle = tokio::task::spawn_blocking(move || {
        run_native_inference_with_progress(
            &app_handle_task,
            &cancelled_tasks,
            &task_id_task,
            &[input_task],
            &[],
        )
    });

    let mut captured = Vec::new();
    let mut finished = false;

    let timeout = tokio::time::sleep(Duration::from_secs(60));
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
        }
        if finished && rx.is_empty() {
            break;
        }
    }

    // ASSERTIONS
    assert!(!captured.is_empty(), "No progress events captured!");

    let started = captured
        .iter()
        .find(|e| e.status == "started")
        .expect("Missing started event");
    assert_eq!(started.completed, 0);

    let items = captured
        .iter()
        .filter(|e| e.status == "item_progress")
        .collect::<Vec<_>>();
    assert!(
        !items.is_empty(),
        "Missing item_progress events DURING execution"
    );

    let completed = captured
        .iter()
        .find(|e| e.status == "completed")
        .expect("Missing completed event");
    assert_eq!(completed.progress, 100);

    // Monotonicity check
    let mut last_progress = 0;
    for event in &captured {
        if event.status == "item_progress" || event.status == "completed" {
            assert!(
                event.progress >= last_progress,
                "Progress decreased from {} to {}!",
                last_progress,
                event.progress
            );
            last_progress = event.progress;
        }
    }

    let _ = handle.await.unwrap();
}
