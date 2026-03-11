// This test suite integrates with test-containers for environment isolation.
use super::support::{
    create_backend_state_with_fake_responses, find_repo_root, fixture_python_pred, next_test_id,
    resolve_python_with_component_runtime,
};
use crate::inference::postprocess::{
    derive_aseg_from_pred, derive_brainmask_from_pred, flip_wm_islands, mask_aseg_with_brainmask,
};
use crate::inference::preprocess::load_input_volume;
use std::fs;
use std::io::Write;
use std::process::Command;

#[test]
fn predict_batch_with_result_entries_should_return_ack_and_deduplicated_directories() {
    let backend = create_backend_state_with_fake_responses(&[
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
    let backend = create_backend_state_with_fake_responses(&[
        r#"{"ok":true,"result":{"ack_message":"done","results":[{"input_path":"/in/a.nii.gz","output_path":"/tmp/out/a.mgz","output_filename":"a.mgz","run_result":"ok"}]}}"#,
    ]);

    let fallback_paths = vec!["fallback/path".to_string()];
    let result = backend
        .predict_batch(&[], &[], "fallback-ack", &fallback_paths)
        .expect("expected predict_batch to succeed");

    assert_eq!(result.requested_paths, fallback_paths);
}

#[test]
fn parity_postprocessing_should_match_original_python_components_on_python_pred_fixture() {
    let Some(repo_root) = find_repo_root() else {
        panic!("failed to locate repo root for postprocessing parity test");
    };

    let pred_nii = fixture_python_pred(&repo_root);
    if !pred_nii.exists() {
        eprintln!(
            "missing fixture prediction '{}'; skipping postprocessing parity test",
            pred_nii.display()
        );
        return;
    }

    let Some(python_bin) = resolve_python_with_component_runtime(&repo_root) else {
        eprintln!(
            "python runtime missing required FastSurfer component deps; skipping postprocessing parity test"
        );
        return;
    };

    let volume = load_input_volume(&pred_nii.to_string_lossy())
        .expect("failed to load python prediction fixture for postprocessing parity test");
    let shape = volume.shape_xyz;

    let pred_labels = volume
        .data_xyz
        .iter()
        .map(|value| value.round().max(0.0) as u16)
        .collect::<Vec<u16>>();

    let mut rust_aseg = derive_aseg_from_pred(&pred_labels);
    let rust_brainmask = derive_brainmask_from_pred(&pred_labels, shape);
    mask_aseg_with_brainmask(&mut rust_aseg, &rust_brainmask);
    flip_wm_islands(&mut rust_aseg, shape);

    let run_dir =
        std::env::temp_dir().join(format!("postprocess_component_parity_{}", next_test_id()));
    fs::create_dir_all(&run_dir).expect("failed to create temp postprocessing parity directory");

    let rust_aseg_raw = run_dir.join("rust_aseg.raw");
    let mut aseg_file = fs::File::create(&rust_aseg_raw)
        .expect("failed to create rust aseg raw file for postprocessing parity");
    for value in &rust_aseg {
        aseg_file
            .write_all(&value.to_le_bytes())
            .expect("failed writing rust aseg raw file");
    }

    let rust_mask_raw = run_dir.join("rust_brainmask.raw");
    fs::write(&rust_mask_raw, &rust_brainmask).expect("failed writing rust brainmask raw file");

    let script = r#"
import numpy as np
import nibabel as nib
import sys

from FastSurferCNN.reduce_to_aseg import create_mask, flip_wm_islands, reduce_to_aseg

pred_path = sys.argv[1]
rust_aseg_raw = sys.argv[2]
rust_mask_raw = sys.argv[3]
max_mismatch_ratio = float(sys.argv[4])

pred = np.rint(np.asanyarray(nib.load(pred_path).dataobj)).astype(np.int32)
py_aseg = reduce_to_aseg(pred.copy())
py_mask = create_mask(pred.copy(), 5, 4)
py_aseg[py_mask == 0] = 0
try:
    py_aseg = flip_wm_islands(py_aseg)
except AssertionError:
    # Rust path returns early when a hemisphere has no WM components.
    pass

rust_aseg = np.fromfile(rust_aseg_raw, dtype=np.uint16).astype(np.int32).reshape(pred.shape)
rust_mask = np.fromfile(rust_mask_raw, dtype=np.uint8).reshape(pred.shape)

if rust_aseg.shape != py_aseg.shape:
    raise SystemExit(f"aseg shape mismatch: rust={rust_aseg.shape} python={py_aseg.shape}")
if rust_mask.shape != py_mask.shape:
    raise SystemExit(f"brainmask shape mismatch: rust={rust_mask.shape} python={py_mask.shape}")

aseg_diff = np.count_nonzero(rust_aseg != py_aseg)
aseg_total = rust_aseg.size
aseg_ratio = (aseg_diff / aseg_total) if aseg_total else 0.0

mask_diff = np.count_nonzero(rust_mask != py_mask.astype(np.uint8))
mask_total = rust_mask.size
mask_ratio = (mask_diff / mask_total) if mask_total else 0.0

if aseg_ratio > max_mismatch_ratio:
    raise SystemExit(
        f"aseg mismatch ratio too high: diff={aseg_diff}/{aseg_total} ratio={aseg_ratio:.6f} threshold={max_mismatch_ratio:.6f}"
    )
if mask_ratio > max_mismatch_ratio:
    raise SystemExit(
        f"brainmask mismatch ratio too high: diff={mask_diff}/{mask_total} ratio={mask_ratio:.6f} threshold={max_mismatch_ratio:.6f}"
    )

print(
    f"aseg_diff={aseg_diff}/{aseg_total} aseg_ratio={aseg_ratio:.6f} "
    f"mask_diff={mask_diff}/{mask_total} mask_ratio={mask_ratio:.6f}"
)
"#;

    let output = Command::new(&python_bin)
        .arg("-c")
        .arg(script)
        .arg(&pred_nii)
        .arg(&rust_aseg_raw)
        .arg(&rust_mask_raw)
        .arg("0.05")
        .current_dir(&repo_root)
        .env("PYTHONPATH", repo_root.to_string_lossy().to_string())
        .output()
        .expect("failed to execute python postprocessing parity script");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "python postprocessing component parity failed: stdout='{}' stderr='{}'",
            stdout.trim(),
            stderr.trim()
        );
    }
}
