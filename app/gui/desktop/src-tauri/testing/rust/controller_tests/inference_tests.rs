// This test suite integrates with test-containers for environment isolation.
use super::support::{
    create_backend_state_via_shell_script, create_backend_state_with_fake_responses,
};
use crate::prediction::run_fastsurfer_inference_with_backend;

#[test]
fn predict_single_path_should_forward_progress_events_from_backend() {
    let backend = create_backend_state_via_shell_script(
        "while IFS= read -r _line; do printf '%s\\n' '{\"event\":\"progress\",\"progress\":2,\"message\":\"Processing MRI: 2%\"}'; printf '%s\\n' '{\"event\":\"progress\",\"progress\":71,\"message\":\"Processing MRI: 71%\"}'; printf '%s\\n' '{\"ok\":true,\"result\":{\"output_path\":\"/tmp/out/pred.mgz\",\"output_filename\":\"pred.mgz\",\"run_result\":\"ok\"}}'; done",
    );

    let mut captured: Vec<(usize, String)> = Vec::new();
    let prediction = backend
        .predict_single_path(
            "/in/subject.mgz",
            "task-progress-test",
            Some(&mut |progress, message| {
                captured.push((progress, message));
            }),
        )
        .expect("expected predict_single_path to succeed");

    assert_eq!(prediction.output_filename, "pred.mgz");
    assert_eq!(prediction.output_path, "/tmp/out/pred.mgz");
    assert_eq!(captured.len(), 2);
    assert_eq!(captured[0].0, 2);
    assert_eq!(captured[1].0, 71);
}

#[test]
fn run_fastsurfer_inference_with_backend_should_merge_start_and_predict_results() {
    let backend = create_backend_state_with_fake_responses(&[
        r#"{"ok":true,"result":{"ack_message":"queued","requested_paths":["/in/a.nii.gz"]}}"#,
        r#"{"ok":true,"result":{"ack_message":"done","requested_paths":["/in/a.nii.gz"],"results":[{"input_path":"/in/a.nii.gz","output_path":"/tmp/out/a.mgz","output_filename":"a.mgz","run_result":"ok"}]}}"#,
    ]);

    let file_paths = vec!["/in/a.nii.gz".to_string()];
    let folder_paths: Vec<String> = vec![];
    let result = run_fastsurfer_inference_with_backend(&backend, &file_paths, &folder_paths)
        .expect("expected run_fastsurfer_inference_with_backend to succeed");

    assert_eq!(result.requested_paths, vec!["/in/a.nii.gz".to_string()]);
    assert_eq!(result.results.len(), 1);
    assert!(result.ack_message.starts_with("done"));
}
