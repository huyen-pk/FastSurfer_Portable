// This test suite integrates with test-containers for environment isolation.
use super::support::{
    compare_label_volumes_per_plane_single_slice_in_rust,
    compare_label_volumes_with_python, compare_preprocess_slice_with_python,
    configured_slice_indices, convert_mgz_to_nii_gz,
    create_backend_state_via_process, ensure_native_input_nifti,
    find_repo_root, is_ci, python_has_fastsurfer_runtime,
    python_has_nibabel_runtime, resolve_python_with_nibabel,
    run_native_inference_with_timeout,
};
use crate::inference::preprocess::{
    InferencePlane, load_input_volume, transformed_volume_shape,
};
use crate::prediction::run_fastsurfer_inference_with_backend;
use crate::process_mgmt::{
    load_desktop_env, resolve_backend_launch_command_from,
    resolve_python_executable,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn test_run_timestamp() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn create_timestamped_results_dir(
    repo_root: &Path,
    prefix: &str,
) -> Result<PathBuf, String> {
    let root = repo_root.join("app/gui/desktop/src-tauri/testing/rust/results");
    fs::create_dir_all(&root).map_err(|error| {
        format!(
            "failed to create rust test results root '{}': {error}",
            root.display()
        )
    })?;
    let run_dir = root.join(format!("{}_{}", prefix, test_run_timestamp()));
    fs::create_dir_all(&run_dir).map_err(|error| {
        format!(
            "failed to create test run dir '{}': {error}",
            run_dir.display()
        )
    })?;
    Ok(run_dir)
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
        let python_bin =
            resolve_python_executable().expect("python executable not found");
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

    let child = launch_cmd.spawn().expect("failed to spawn backend process");

    let backend = create_backend_state_via_process(child);
    let input_path = repo_root
        .join("app/gui/desktop/src-tauri/testing/data/Subject140/140_orig.mgz");
    assert!(
        input_path.exists(),
        "test input file not found: {}",
        input_path.display()
    );

    let file_paths = vec![input_path.to_string_lossy().to_string()];
    let folder_paths: Vec<String> = vec![];
    let result = run_fastsurfer_inference_with_backend(
        &backend,
        &file_paths,
        &folder_paths,
    )
    .expect("expected e2e inference call to succeed");

    assert!(!result.results.is_empty());
    let output_path = PathBuf::from(&result.results[0].output_path);
    assert!(
        output_path.exists(),
        "inference output not found: {}",
        output_path.display()
    );
}

#[test]
#[ignore = "Runs real FastSurfer inference via python IPC server and test data"]
fn predict_single_path_with_test_data_should_emit_progress_events() {
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
        let python_bin =
            resolve_python_executable().expect("python executable not found");
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

    let child = launch_cmd.spawn().expect("failed to spawn backend process");

    let backend = create_backend_state_via_process(child);
    let input_path = repo_root
        .join("app/gui/desktop/src-tauri/testing/data/Subject140/140_orig.mgz");
    assert!(
        input_path.exists(),
        "test input file not found: {}",
        input_path.display()
    );

    let mut captured_progress: Vec<(usize, String)> = Vec::new();
    let prediction = backend
        .predict_single_path(
            &input_path.to_string_lossy(),
            "task-e2e-progress",
            Some(&mut |progress, message| {
                captured_progress.push((progress, message));
            }),
        )
        .expect("expected e2e predict_single_path to succeed");

    assert!(
        !captured_progress.is_empty(),
        "expected at least one progress event from backend"
    );
    assert!(
        captured_progress.iter().any(|(progress, _)| *progress >= 2),
        "expected parsed tqdm progress events"
    );

    let output_path = PathBuf::from(&prediction.output_path);
    assert!(
        output_path.exists(),
        "inference output not found: {}",
        output_path.display()
    );
}

#[test]
#[ignore = "Runs real Python+Rust inference parity check on Subject140 test data"]
fn parity_native_rust_inference_with_test_data_should_generate_output_and_compare_to_python()
 {
    let Some(repo_root) = find_repo_root() else {
        panic!("failed to locate repo root for native e2e parity test");
    };

    let python_bin =
        resolve_python_executable().expect("python executable not found");
    if !python_has_fastsurfer_runtime(&python_bin, &repo_root)
        || !python_has_nibabel_runtime(&python_bin, &repo_root)
    {
        let _ = is_ci();
        eprintln!(
            "python runtime missing required deps (torch, FastSurferCNN, nibabel, numpy); skipping native e2e parity test"
        );
        return;
    }

    let input_mgz = repo_root
        .join("app/gui/desktop/src-tauri/testing/data/Subject140/140_orig.mgz");
    assert!(
        input_mgz.exists(),
        "test input file not found: {}",
        input_mgz.display()
    );

    let fixture_dir = repo_root
        .join("app/gui/desktop/src-tauri/testing/data/.tmp_e2e_output_py");
    fs::create_dir_all(&fixture_dir)
        .expect("failed to create temporary e2e fixture output directory");
    let input_nii = fixture_dir.join("140_orig.native_input.nii.gz");
    convert_mgz_to_nii_gz(&python_bin, &input_mgz, &input_nii)
        .expect("failed to convert test mgz input to temporary nii.gz");
    assert!(
        input_nii.exists(),
        "temporary converted input not found: {}",
        input_nii.display()
    );

    // Environment must be set externally for this test.
    if std::env::var("FASTSURFER_REPO_ROOT").is_err() {
        eprintln!(
            "Skipping native e2e parity test: FASTSURFER_REPO_ROOT not set"
        );
        return;
    }
    let native_result = run_native_inference_with_timeout(
        &[input_nii.to_string_lossy().to_string()],
        &[],
        Duration::from_secs(60),
    )
    .expect("expected native rust inference call to succeed");
    assert!(
        !native_result.results.is_empty(),
        "expected at least one native result"
    );

    let rust_pred = PathBuf::from(&native_result.results[0].output_path);
    assert!(
        rust_pred.exists(),
        "native rust output not found: {}",
        rust_pred.display()
    );

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
        .stderr(Stdio::inherit())
        .current_dir(&repo_root)
        .env("PYTHONPATH", repo_root.to_string_lossy().to_string());

    let child = launch_cmd
        .spawn()
        .expect("failed to spawn backend process for parity comparison");

    let backend = create_backend_state_via_process(child);
    let py_result = run_fastsurfer_inference_with_backend(
        &backend,
        &[input_mgz.to_string_lossy().to_string()],
        &[],
    )
    .expect("expected python backend inference call to succeed");

    assert!(
        !py_result.results.is_empty(),
        "expected at least one python result"
    );
    let py_pred = PathBuf::from(&py_result.results[0].output_path);
    assert!(
        py_pred.exists(),
        "python output not found: {}",
        py_pred.display()
    );

    compare_label_volumes_with_python(&python_bin, &rust_pred, &py_pred)
        .expect("failed comparing rust output volume to python output volume");
}

#[test]
#[ignore = "Runs Rust-only parity check against precomputed golden NIfTI files in testing/data/.tmp_e2e_output_py"]
fn parity_label_volume_ratio_rust_inference_with_golden_files_should_compare_without_python_runtime()
 {
    let Some(repo_root) = find_repo_root() else {
        panic!("failed to locate repo root for native golden parity test");
    };

    let fixture_dir = repo_root
        .join("app/gui/desktop/src-tauri/testing/data/.tmp_e2e_output_py");
    let input_nii = fixture_dir.join("140_orig.native_input.nii.gz");
    let golden_pred = fixture_dir.join("140_orig.python_pred.nii.gz");

    if !input_nii.exists() || !golden_pred.exists() {
        eprintln!(
            "missing golden fixture files; expected '{}' and '{}'. skipping rust-only golden parity test",
            input_nii.display(),
            golden_pred.display()
        );
        return;
    }

    // Environment must be set externally for this test.
    if std::env::var("FASTSURFER_REPO_ROOT").is_err() {
        eprintln!(
            "Skipping rust-only golden parity test: FASTSURFER_REPO_ROOT not set"
        );
        return;
    }

    let outer_timeout_secs =
        std::env::var("FASTSURFER_NATIVE_TEST_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(60);

    let run_dir = create_timestamped_results_dir(
        &repo_root,
        "parity_label_volume_ratio_one_slice",
    )
    .expect("failed to create timestamped one-slice parity results directory");

    // FASTSURFER_NATIVE_OUTPUT_ROOT must be handled externally or via default temp dir

    let native_result = run_native_inference_with_timeout(
        &[input_nii.to_string_lossy().to_string()],
        &[],
        Duration::from_secs(outer_timeout_secs),
    )
    .expect("expected native rust inference call to succeed");
    assert!(
        !native_result.results.is_empty(),
        "expected at least one native result"
    );

    let rust_pred = PathBuf::from(&native_result.results[0].output_path);
    assert!(
        rust_pred.exists(),
        "native rust output not found: {}",
        rust_pred.display()
    );

    let _ =
        fs::copy(&input_nii, run_dir.join("subject140_input_native.nii.gz"));
    let _ = fs::copy(&golden_pred, run_dir.join("python_pred_golden.nii.gz"));

    eprintln!("[parity][one-slice] artifacts: {}", run_dir.display());

    compare_label_volumes_per_plane_single_slice_in_rust(&rust_pred, &golden_pred, 0.25)
        .expect("failed comparing rust output to golden python output on 1 slice per plane");
}

#[test]
#[ignore = "Requires python3 + nibabel to compare Rust preprocessing slices with Python reference"]
fn parity_native_preprocess_should_match_python_for_planes_edge_and_center_slices()
 {
    let Some(repo_root) = find_repo_root() else {
        panic!("failed to locate repo root for preprocess parity test");
    };

    let Some(python_bin) = resolve_python_with_nibabel(&repo_root) else {
        eprintln!(
            "python runtime missing nibabel/numpy; skipping preprocess parity test"
        );
        return;
    };

    let input_nii = ensure_native_input_nifti(&repo_root, &python_bin)
        .expect("failed to ensure NIfTI input for preprocess parity test");

    let volume = load_input_volume(&input_nii.to_string_lossy())
        .expect("failed to load NIfTI input for preprocess parity test");

    let num_channels = 7usize;
    let base_res = 1.0f32;

    for plane in [
        InferencePlane::Coronal,
        InferencePlane::Axial,
        InferencePlane::Sagittal,
    ] {
        let [_, _, slices] = transformed_volume_shape(volume.shape_xyz, plane);
        assert!(
            slices > 0,
            "expected at least one slice for plane {}",
            plane.as_str()
        );

        let cases = configured_slice_indices(slices);
        assert!(
            !cases.is_empty(),
            "expected at least one configured slice for plane {}",
            plane.as_str()
        );
        for slice_index in cases {
            compare_preprocess_slice_with_python(
                &python_bin,
                &repo_root,
                &input_nii,
                plane,
                num_channels,
                base_res,
                slice_index,
            )
            .expect("rust/python preprocess parity mismatch");
        }
    }
}
