// This test suite integrates with test-containers for environment isolation.
use super::support::create_backend_state_via_shell_script;

#[test]
fn progress_callback_should_capture_sequential_progress_events() {
    let backend = create_backend_state_via_shell_script(
        "while IFS= read -r _line; do printf '%s\\n' '{\"event\":\"progress\",\"progress\":10,\"message\":\"Step 1\"}'; printf '%s\\n' '{\"event\":\"progress\",\"progress\":30,\"message\":\"Step 2\"}'; printf '%s\\n' '{\"event\":\"progress\",\"progress\":50,\"message\":\"Step 3\"}'; printf '%s\\n' '{\"ok\":true,\"result\":{\"output_path\":\"/tmp/out.mgz\",\"output_filename\":\"out.mgz\",\"run_result\":\"ok\"}}'; done",
    );

    let mut captured: Vec<(usize, String)> = Vec::new();
    let prediction = backend
        .predict_single_path(
            "/in/test.mgz",
            "progress-seq-test",
            Some(&mut |progress, message| {
                captured.push((progress, message));
            }),
        )
        .expect("expected predict_single_path to succeed");

    assert_eq!(captured.len(), 3);
    assert_eq!(captured[0].0, 10);
    assert_eq!(captured[1].0, 30);
    assert_eq!(captured[2].0, 50);
    assert_eq!(captured[0].1, "Step 1");
    assert_eq!(captured[1].1, "Step 2");
    assert_eq!(captured[2].1, "Step 3");
    assert_eq!(prediction.output_filename, "out.mgz");
}

#[test]
fn progress_callback_should_handle_progress_at_boundaries_0_and_100() {
    let backend = create_backend_state_via_shell_script(
        "while IFS= read -r _line; do printf '%s\\n' '{\"event\":\"progress\",\"progress\":0,\"message\":\"Starting\"}'; printf '%s\\n' '{\"event\":\"progress\",\"progress\":100,\"message\":\"Complete\"}'; printf '%s\\n' '{\"ok\":true,\"result\":{\"output_path\":\"/tmp/out.mgz\",\"output_filename\":\"out.mgz\",\"run_result\":\"ok\"}}'; done",
    );

    let mut captured: Vec<(usize, String)> = Vec::new();
    backend
        .predict_single_path(
            "/in/test.mgz",
            "boundary-test",
            Some(&mut |progress, message| {
                captured.push((progress, message));
            }),
        )
        .expect("expected predict_single_path to succeed");

    assert_eq!(captured.len(), 2);
    assert_eq!(captured[0].0, 0);
    assert_eq!(captured[1].0, 100);
}

#[test]
fn progress_callback_should_ignore_non_progress_events() {
    let backend = create_backend_state_via_shell_script(
        "while IFS= read -r _line; do printf '%s\\n' '{\"event\":\"log\",\"level\":\"info\",\"message\":\"Not a progress event\"}'; printf '%s\\n' '{\"event\":\"progress\",\"progress\":42,\"message\":\"Real progress\"}'; printf '%s\\n' '{\"ok\":true,\"result\":{\"output_path\":\"/tmp/out.mgz\",\"output_filename\":\"out.mgz\",\"run_result\":\"ok\"}}'; done",
    );

    let mut captured: Vec<(usize, String)> = Vec::new();
    backend
        .predict_single_path(
            "/in/test.mgz",
            "event-filter-test",
            Some(&mut |progress, message| {
                captured.push((progress, message));
            }),
        )
        .expect("expected predict_single_path to succeed");

    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].0, 42);
    assert_eq!(captured[0].1, "Real progress");
}

#[test]
fn progress_callback_should_handle_rapid_progress_updates() {
    use std::fmt::Write as _;

    let mut rapid_updates = String::new();
    for i in 1..=10 {
        let progress = i * 10;
        write!(
            rapid_updates,
            "printf '%s\\n' '{{\"event\":\"progress\",\"progress\":{progress},\"message\":\"Update {i}\"}}'; "
        )
        .expect("writing progress update should succeed");
    }
    rapid_updates.push_str("printf '%s\\n' '{\"ok\":true,\"result\":{\"output_path\":\"/tmp/out.mgz\",\"output_filename\":\"out.mgz\",\"run_result\":\"ok\"}}'; ");

    let script = format!("while IFS= read -r _line; do {rapid_updates}done");

    let backend = create_backend_state_via_shell_script(&script);

    let mut captured: Vec<(usize, String)> = Vec::new();
    backend
        .predict_single_path(
            "/in/test.mgz",
            "rapid-test",
            Some(&mut |progress, message| {
                captured.push((progress, message));
            }),
        )
        .expect("expected predict_single_path to succeed");

    assert_eq!(captured.len(), 10);
    for (i, (progress, message)) in captured.iter().enumerate() {
        let expected_progress = (i + 1) * 10;
        assert_eq!(*progress, expected_progress);
        assert_eq!(*message, format!("Update {}", i + 1));
    }
}

#[test]
fn progress_callback_should_extract_message_correctly() {
    let backend = create_backend_state_via_shell_script(
        "while IFS= read -r _line; do printf '%s\\n' '{\"event\":\"progress\",\"progress\":45,\"message\":\"Processing sagittal plane: 181/256\"}'; printf '%s\\n' '{\"ok\":true,\"result\":{\"output_path\":\"/tmp/out.mgz\",\"output_filename\":\"out.mgz\",\"run_result\":\"ok\"}}'; done",
    );

    let mut captured: Vec<(usize, String)> = Vec::new();
    backend
        .predict_single_path(
            "/in/test.mgz",
            "message-test",
            Some(&mut |progress, message| {
                captured.push((progress, message));
            }),
        )
        .expect("expected predict_single_path to succeed");

    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].1, "Processing sagittal plane: 181/256");
}

#[test]
fn progress_callback_should_not_fail_if_none_callback_provided() {
    let backend = create_backend_state_via_shell_script(
        "while IFS= read -r _line; do printf '%s\\n' '{\"event\":\"progress\",\"progress\":50,\"message\":\"Processing\"}'; printf '%s\\n' '{\"ok\":true,\"result\":{\"output_path\":\"/tmp/out.mgz\",\"output_filename\":\"out.mgz\",\"run_result\":\"ok\"}}'; done",
    );

    let prediction = backend
        .predict_single_path("/in/test.mgz", "no-callback-test", None)
        .expect("expected predict_single_path to succeed");

    assert_eq!(prediction.output_filename, "out.mgz");
}

#[test]
fn progress_events_with_all_required_fields_should_be_processable_by_gui() {
    let backend = create_backend_state_via_shell_script(
        "while IFS= read -r _line; do printf '%s\\n' '{\"event\":\"progress\",\"progress\":33,\"message\":\"Sagittal: 95/256\",\"task_id\":\"segmentation-001\"}'; printf '%s\\n' '{\"event\":\"progress\",\"progress\":67,\"message\":\"Coronal: 180/256\",\"task_id\":\"segmentation-001\"}'; printf '%s\\n' '{\"ok\":true,\"result\":{\"output_path\":\"/tmp/out.mgz\",\"output_filename\":\"out.mgz\",\"run_result\":\"ok\"}}'; done",
    );

    let mut captured: Vec<(usize, String)> = Vec::new();
    backend
        .predict_single_path(
            "/in/test.mgz",
            "gui-display-test",
            Some(&mut |progress, message| {
                captured.push((progress, message));
            }),
        )
        .expect("expected predict_single_path to succeed");

    assert_eq!(captured.len(), 2);
    assert_eq!(captured[0].0, 33);
    assert!(captured[0].1.contains("Sagittal"));
    assert_eq!(captured[1].0, 67);
    assert!(captured[1].1.contains("Coronal"));
}

#[test]
fn progress_events_should_be_monotonically_increasing_in_typical_workflow() {
    let backend = create_backend_state_via_shell_script(
        "while IFS= read -r _line; do printf '%s\\n' '{\"event\":\"progress\",\"progress\":5,\"message\":\"Started\"}'; printf '%s\\n' '{\"event\":\"progress\",\"progress\":20,\"message\":\"One third\"}'; printf '%s\\n' '{\"event\":\"progress\",\"progress\":50,\"message\":\"Halfway\"}'; printf '%s\\n' '{\"event\":\"progress\",\"progress\":85,\"message\":\"Almost done\"}'; printf '%s\\n' '{\"event\":\"progress\",\"progress\":100,\"message\":\"Complete\"}'; printf '%s\\n' '{\"ok\":true,\"result\":{\"output_path\":\"/tmp/out.mgz\",\"output_filename\":\"out.mgz\",\"run_result\":\"ok\"}}'; done",
    );

    let mut captured: Vec<(usize, String)> = Vec::new();
    backend
        .predict_single_path(
            "/in/test.mgz",
            "monotonic-test",
            Some(&mut |progress, message| {
                captured.push((progress, message));
            }),
        )
        .expect("expected predict_single_path to succeed");

    assert_eq!(captured.len(), 5);
    let progress_values: Vec<usize> =
        captured.iter().map(|(p, _)| *p).collect();
    assert_eq!(progress_values, vec![5, 20, 50, 85, 100]);

    for i in 1..progress_values.len() {
        assert!(
            progress_values[i] >= progress_values[i - 1],
            "progress should be monotonically increasing"
        );
    }
}

#[test]
fn progress_callback_should_return_error_when_backend_exits_mid_stream() {
    let backend = create_backend_state_via_shell_script(
        "while IFS= read -r _line; do printf '%s\\n' '{\"event\":\"progress\",\"progress\":25,\"message\":\"Starting\"}'; exit 0; done",
    );

    let mut captured: Vec<(usize, String)> = Vec::new();
    let result = backend.predict_single_path(
        "/in/test.mgz",
        "mid-stream-exit-test",
        Some(&mut |progress, message| {
            captured.push((progress, message));
        }),
    );

    assert!(result.is_err());
    assert_eq!(captured, vec![(25, "Starting".to_string())]);
}
