use super::{
    load_desktop_env, open_result_in_file_manager, resolve_backend_binary_path_from,
    resolve_backend_launch_command_from,
    resolve_python_executable, run_fastsurfer_inference_with_app_state,
    run_fastsurfer_inference_with_backend, validate_result_path, BackendProcess, BackendState,
};
use serde_json::{json, Value};
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

fn next_test_id() -> usize {
    static NEXT_ID: AtomicUsize = AtomicUsize::new(1);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

fn create_backend_state_with_mocked_responses(responses: &[&str]) -> BackendState {
    let mut branches = String::new();
    for (index, response) in responses.iter().enumerate() {
        branches.push_str(&format!("{index}) printf '%s\\n' '{response}' ;;") );
    }

    let script = format!(
        "i=0; while IFS= read -r _line; do case \"$i\" in {branches} *) printf '%s\\n' '{{\"ok\":false,\"error\":{{\"message\":\"unexpected request\"}}}}' ;; esac; i=$((i+1)); done"
    );

    let mut child = Command::new("sh")
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("failed to spawn mock backend process");

    let stdin = child.stdin.take().expect("failed to capture mock backend stdin");
    let stdout = child
        .stdout
        .take()
        .expect("failed to capture mock backend stdout");

    BackendState {
        process: std::sync::Mutex::new(BackendProcess {
            _child: child,
            stdin,
            stdout: BufReader::new(stdout),
        }),
    }
}

fn create_backend_state_from_shell_script(script: &str) -> BackendState {
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("failed to spawn scripted mock backend process");

    let stdin = child
        .stdin
        .take()
        .expect("failed to capture scripted mock backend stdin");
    let stdout = child
        .stdout
        .take()
        .expect("failed to capture scripted mock backend stdout");

    BackendState {
        process: std::sync::Mutex::new(BackendProcess {
            _child: child,
            stdin,
            stdout: BufReader::new(stdout),
        }),
    }
}

fn create_backend_state_from_process(mut child: std::process::Child) -> BackendState {
    let stdin = child
        .stdin
        .take()
        .expect("failed to capture process stdin");
    let stdout = child
        .stdout
        .take()
        .expect("failed to capture process stdout");

    BackendState {
        process: std::sync::Mutex::new(BackendProcess {
            _child: child,
            stdin,
            stdout: BufReader::new(stdout),
        }),
    }
}

fn spawn_python_backend_inline(script: &str) -> Option<BackendState> {
    let python = resolve_python_executable()?;
    let child = Command::new(python)
        .arg("-u")
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .ok()?;

    Some(create_backend_state_from_process(child))
}

fn find_repo_root() -> Option<PathBuf> {
    let mut current = std::env::current_dir().ok()?;
    loop {
        if current.join("app/backend/ipc_server.py").exists() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

fn is_ci() -> bool {
    std::env::var("CI")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true" || normalized == "yes"
        })
        .unwrap_or(false)
}

fn python_has_fastsurfer_runtime(python_bin: &str, repo_root: &Path) -> bool {
    Command::new(python_bin)
        .arg("-c")
        .arg("import torch, FastSurferCNN")
        .current_dir(repo_root)
        .env("PYTHONPATH", repo_root.to_string_lossy().to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[test]
fn validate_empty_path_should_return_path_is_empty_error() {
    let result = validate_result_path("   ");

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Path is empty");
}

#[test]
fn validate_missing_path_should_return_path_does_not_exist_error() {
    let missing_path = std::env::temp_dir().join(format!(
        "desktop_test_missing_{}.nii.gz",
        next_test_id()
    ));
    let result = validate_result_path(&missing_path.to_string_lossy());

    assert!(result.is_err());
    assert!(result.unwrap_err().starts_with("Path does not exist:"));
}

#[test]
fn validate_existing_file_path_should_return_canonical_file_path() {
    let temp_file = std::env::temp_dir().join(format!(
        "desktop_test_existing_file_{}.txt",
        next_test_id()
    ));
    fs::write(&temp_file, b"ok").expect("failed to create temp file for test");

    let result = validate_result_path(&temp_file.to_string_lossy());
    let canonical = temp_file
        .canonicalize()
        .expect("failed to canonicalize temp file path");

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), canonical);

    fs::remove_file(&temp_file).expect("failed to cleanup temp file");
}

#[test]
fn validate_existing_directory_path_should_return_canonical_directory_path() {
    let temp_dir = std::env::temp_dir().join(format!(
        "desktop_test_existing_dir_{}",
        next_test_id()
    ));
    fs::create_dir_all(&temp_dir).expect("failed to create temp dir for test");

    let result = validate_result_path(&temp_dir.to_string_lossy());
    let canonical = temp_dir
        .canonicalize()
        .expect("failed to canonicalize temp dir path");

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), canonical);

    fs::remove_dir_all(&temp_dir).expect("failed to cleanup temp dir");
}

#[test]
fn resolve_backend_binary_from_paths_should_return_first_existing_candidate() {
    let root = std::env::temp_dir().join(format!("desktop_test_root_{}", next_test_id()));
    let exe_dir = root.join("bin/app");
    let candidate_dir = root.join("backend");
    let candidate_path = candidate_dir.join("main");

    fs::create_dir_all(&exe_dir).expect("failed to create mock exe dir");
    fs::create_dir_all(&candidate_dir).expect("failed to create mock backend dir");
    fs::write(&candidate_path, b"mock").expect("failed to create mock backend binary");

    let result = resolve_backend_binary_path_from(&root, &exe_dir).expect("expected backend path resolution to succeed");

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
fn run_ipc_request_with_ok_response_should_return_result_value() {
    let backend = create_backend_state_with_mocked_responses(&[
        r#"{"ok":true,"result":{"ack_message":"Accepted"}}"#,
    ]);

    let result = backend
        .run_ipc_request("ping", json!({ "hello": "world" }))
        .expect("expected ipc request to succeed");

    assert_eq!(result.get("ack_message"), Some(&Value::String("Accepted".to_string())));
}

#[test]
fn run_ipc_request_with_non_json_prelude_should_wait_for_json_response() {
    let backend = create_backend_state_from_shell_script(
        "while IFS= read -r _line; do printf '%s\\n' 'backend log line'; printf '%s\\n' '{\"ok\":true,\"result\":{\"ack_message\":\"Accepted\"}}'; done",
    );

    let result = backend
        .run_ipc_request("ping", json!({ "hello": "world" }))
        .expect("expected ipc request to ignore prelude and parse json response");

    assert_eq!(
        result.get("ack_message"),
        Some(&Value::String("Accepted".to_string()))
    );
}

#[test]
fn run_ipc_request_with_real_python_process_should_pipe_request_and_receive_response() {
    let backend = spawn_python_backend_inline(
        "import json,sys\nfor line in sys.stdin:\n line=line.strip()\n if not line: continue\n req=json.loads(line)\n sys.stdout.write(json.dumps({'ok': True, 'result': {'echo_method': req.get('method')}}) + '\\n')\n sys.stdout.flush()",
    );

    let Some(backend) = backend else {
        if is_ci() {
            eprintln!("python executable unavailable in CI; skipping real python subprocess test");
            return;
        }
        panic!("python executable unavailable in local environment; set FASTSURFER_PYTHON_BIN or install python3");
    };

    let result = backend
        .run_ipc_request("health", json!({}))
        .expect("expected run_ipc_request to communicate with real python process");

    assert_eq!(
        result.get("echo_method"),
        Some(&Value::String("health".to_string()))
    );
}

#[test]
fn run_ipc_request_with_backend_error_should_return_formatted_error() {
    let backend = create_backend_state_with_mocked_responses(&[
        r#"{"ok":false,"error":{"message":"boom"}}"#,
    ]);

    let result = backend.run_ipc_request("predict_batch", json!({}));

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Backend predict_batch failed: boom");
}

#[test]
fn run_ipc_request_with_invalid_json_should_return_invalid_json_error() {
    let backend = create_backend_state_with_mocked_responses(&["{\"ok\":true"]);

    let result = backend.run_ipc_request("predict_batch", json!({}));

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .starts_with("Invalid IPC response JSON:"));
}

#[test]
fn start_predict_batch_with_valid_response_should_return_ack_and_paths() {
    let backend = create_backend_state_with_mocked_responses(&[
        r#"{"ok":true,"result":{"ack_message":"queued","requested_paths":["a.nii.gz","b.nii.gz"]}}"#,
    ]);

    let (ack, requested) = backend
        .start_predict_batch(&["a.nii.gz".to_string()], &["/data".to_string()])
        .expect("expected start_predict_batch to succeed");

    assert_eq!(ack, "queued");
    assert_eq!(requested, vec!["a.nii.gz".to_string(), "b.nii.gz".to_string()]);
}

#[test]
fn start_predict_batch_without_ack_message_should_return_missing_ack_error() {
    let backend = create_backend_state_with_mocked_responses(&[
        r#"{"ok":true,"result":{"requested_paths":["a.nii.gz"]}}"#,
    ]);

    let result = backend.start_predict_batch(&[], &[]);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Missing ack_message in IPC response");
}

#[test]
fn predict_batch_with_result_entries_should_return_ack_and_deduplicated_directories() {
    let backend = create_backend_state_with_mocked_responses(&[
        r#"{"ok":true,"result":{"ack_message":"done","requested_paths":["x"],"results":[{"input_path":"/in/a.nii.gz","output_path":"/tmp/out/a.mgz","output_filename":"a.mgz","run_result":"ok"},{"input_path":"/in/b.nii.gz","output_path":"/tmp/out/b.mgz","output_filename":"b.mgz","run_result":1}]}}"#,
    ]);

    let result = backend
        .predict_batch(&[], &[], "fallback", &["fallback_path".to_string()])
        .expect("expected predict_batch to succeed");

    assert_eq!(result.ack_message, "done Results directory: /tmp/out");
    assert_eq!(result.requested_paths, vec!["x".to_string()]);
    assert_eq!(result.result_directories, vec!["/tmp/out".to_string()]);
    assert_eq!(result.results.len(), 2);
}

#[test]
fn predict_batch_without_requested_paths_should_use_fallback_requested_paths() {
    let backend = create_backend_state_with_mocked_responses(&[
        r#"{"ok":true,"result":{"ack_message":"done","results":[{"input_path":"/in/a.nii.gz","output_path":"/tmp/out/a.mgz","output_filename":"a.mgz","run_result":"ok"}]}}"#,
    ]);

    let fallback_paths = vec!["fallback/path".to_string()];
    let result = backend
        .predict_batch(&[], &[], "fallback-ack", &fallback_paths)
        .expect("expected predict_batch to succeed");

    assert_eq!(result.requested_paths, fallback_paths);
}

#[test]
fn run_fastsurfer_inference_with_backend_should_merge_start_and_predict_results() {
    let backend = create_backend_state_with_mocked_responses(&[
        r#"{"ok":true,"result":{"ack_message":"queued","requested_paths":["/in/a.nii.gz"]}}"#,
        r#"{"ok":true,"result":{"ack_message":"done","requested_paths":["/in/a.nii.gz"],"results":[{"input_path":"/in/a.nii.gz","output_path":"/tmp/out/a.mgz","output_filename":"a.mgz","run_result":"ok"}]}}"#,
    ]);

    let result = run_fastsurfer_inference_with_backend(
        &backend,
        vec!["/in/a.nii.gz".to_string()],
        vec![],
    )
    .expect("expected run_fastsurfer_inference_with_backend to succeed");

    assert_eq!(result.requested_paths, vec!["/in/a.nii.gz".to_string()]);
    assert_eq!(result.results.len(), 1);
    assert!(result.ack_message.starts_with("done"));
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

#[test]
#[ignore = "Runs real FastSurfer inference via python IPC server and test data"]
fn run_fastsurfer_inference_with_test_data_should_produce_output_file() {
    let Some(repo_root) = find_repo_root() else {
        panic!("failed to locate repo root for e2e inference test");
    };

    let cwd = repo_root.join("app/gui/desktop/src-tauri");
    let exe_dir = cwd.join("target/debug");
    load_desktop_env(&cwd, &exe_dir);
    let launch = resolve_backend_launch_command_from(&cwd, &exe_dir)
        .expect("failed to resolve backend launch command");

    let mut launch_cmd = Command::new(&launch.program);
    launch_cmd
        .args(&launch.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());

    if !launch.args.is_empty() {
        let python_bin = resolve_python_executable().expect("python executable not found");
        if !python_has_fastsurfer_runtime(&python_bin, &repo_root) {
            if is_ci() {
                eprintln!(
                    "python runtime missing FastSurfer dependencies in CI; skipping e2e inference test"
                );
                return;
            }
            panic!(
                "python runtime missing FastSurfer dependencies in local environment (required: torch, FastSurferCNN)"
            );
        }

        launch_cmd.current_dir(&repo_root);
        launch_cmd.env("PYTHONPATH", repo_root.to_string_lossy().to_string());
    }

    let child = launch_cmd
        .spawn()
        .expect("failed to spawn backend process");

    let backend = create_backend_state_from_process(child);
    let input_path = repo_root.join("app/gui/desktop/src-tauri/testing/data/Subject140/140_orig.mgz");
    assert!(input_path.exists(), "test input file not found: {}", input_path.display());

    let result = run_fastsurfer_inference_with_backend(
        &backend,
        vec![input_path.to_string_lossy().to_string()],
        vec![],
    )
    .expect("expected e2e inference call to succeed");

    assert!(!result.results.is_empty());
    let output_path = PathBuf::from(&result.results[0].output_path);
    assert!(output_path.exists(), "inference output not found: {}", output_path.display());
}

#[test]
fn open_result_in_file_manager_with_empty_path_should_return_path_is_empty_error() {
    let result = open_result_in_file_manager(" ".to_string());

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Path is empty");
}

#[test]
fn open_result_in_file_manager_with_missing_path_should_return_path_does_not_exist_error() {
    let missing_path = std::env::temp_dir()
        .join(format!("desktop_test_nonexistent_open_{}", next_test_id()));

    let result = open_result_in_file_manager(missing_path.to_string_lossy().to_string());

    assert!(result.is_err());
    assert!(result.unwrap_err().starts_with("Path does not exist:"));
}

#[test]
fn validate_existing_file_parent_should_be_directory() {
    let temp_file = std::env::temp_dir().join(format!(
        "desktop_test_parent_dir_file_{}.txt",
        next_test_id()
    ));
    fs::write(&temp_file, b"ok").expect("failed to create temp file for test");

    let canonical = validate_result_path(&temp_file.to_string_lossy())
        .expect("expected validate_result_path to return canonical file path");

    let parent = canonical.parent().expect("file should have a parent directory");
    assert!(Path::new(parent).is_dir());

    fs::remove_file(&temp_file).expect("failed to cleanup temp file");
}
