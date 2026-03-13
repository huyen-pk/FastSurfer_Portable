// This test suite integrates with test-containers for environment isolation.
use super::support::{
    create_backend_state_with_fake_responses, find_repo_root,
    fixture_native_input, next_test_id, resolve_python_with_component_runtime,
};
use crate::inference::pipeline::preprocess::{
    InferencePlane, PreparedPlaneInput, load_input_volume,
    prepare_plane_input_for_slice,
};
use std::fs;
use std::io::Write;
use std::process::Command;

fn write_prepared_tensor(
    run_dir: &std::path::Path,
    prepared: &PreparedPlaneInput,
) -> std::path::PathBuf {
    let rust_raw = run_dir.join("rust_tensor.raw");
    let mut raw_file = fs::File::create(&rust_raw)
        .expect("failed to create rust tensor raw file for parity check");
    for value in &prepared.tensor_data {
        raw_file
            .write_all(&value.to_le_bytes())
            .expect("failed writing rust tensor raw file");
    }
    rust_raw
}

struct PythonPreprocessComponentCheckArgs<'a> {
    python_bin: &'a str,
    repo_root: &'a std::path::Path,
    input_nii: &'a std::path::Path,
    plane: InferencePlane,
    num_channels: usize,
    base_res: f32,
    center: usize,
    prepared: &'a PreparedPlaneInput,
    rust_raw: &'a std::path::Path,
}

fn run_python_preprocess_component_check(
    args: &PythonPreprocessComponentCheckArgs<'_>,
) {
    let shape_csv = args
        .prepared
        .tensor_shape
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<String>>()
        .join(",");

    let script = r#"
import numpy as np
import nibabel as nib
import sys

from FastSurferCNN.data_loader.data_utils import get_thick_slices, transform_axial, transform_sagittal

input_nii = sys.argv[1]
plane = sys.argv[2]
num_channels = int(sys.argv[3])
base_res = float(sys.argv[4])
slice_index = int(sys.argv[5])
rust_scale0 = float(sys.argv[6])
rust_scale1 = float(sys.argv[7])
shape_csv = sys.argv[8]
rust_raw = sys.argv[9]

shape = tuple(int(x) for x in shape_csv.split(","))
img = nib.load(input_nii)
orig_data = np.asanyarray(img.dataobj)
orig_zoom = np.asarray(img.header.get_zooms()[:3], dtype=np.float32)

if plane == "sagittal":
    transformed = transform_sagittal(orig_data)
    zoom = orig_zoom[[2, 1]]
elif plane == "axial":
    transformed = transform_axial(orig_data)
    zoom = orig_zoom[[2, 0]]
else:
    transformed = orig_data
    zoom = orig_zoom[[0, 1]]

orig_thick = get_thick_slices(transformed, num_channels // 2)
orig_thick = np.transpose(orig_thick, (2, 0, 1, 3))
py_image = np.clip(orig_thick[slice_index].astype(np.float32) / 255.0, a_min=0.0, a_max=1.0)
py_image = py_image.transpose((2, 0, 1))
py_scale = base_res / zoom

rust = np.fromfile(rust_raw, dtype=np.float32).reshape(shape)
rust_image = rust[0]
rust_scale = np.asarray([rust_scale0, rust_scale1], dtype=np.float32)

if rust_image.shape != py_image.shape:
    raise SystemExit(f"shape mismatch: rust={rust_image.shape} python={py_image.shape}")
if not np.allclose(rust_scale, py_scale, rtol=0.0, atol=1e-6):
    raise SystemExit(f"scale mismatch: rust={rust_scale.tolist()} python={py_scale.tolist()}")
if not np.allclose(rust_image, py_image, rtol=0.0, atol=1e-6):
    diff = np.abs(rust_image - py_image)
    raise SystemExit(
        f"tensor mismatch: max_abs_diff={float(diff.max()):.8f} mean_abs_diff={float(diff.mean()):.8f}"
    )
"#;

    let output = Command::new(args.python_bin)
        .arg("-c")
        .arg(script)
        .arg(args.input_nii)
        .arg(args.plane.as_str())
        .arg(args.num_channels.to_string())
        .arg(format!("{}", args.base_res))
        .arg(args.center.to_string())
        .arg(format!("{:.8}", args.prepared.scale_factor[0]))
        .arg(format!("{:.8}", args.prepared.scale_factor[1]))
        .arg(shape_csv)
        .arg(args.rust_raw)
        .current_dir(args.repo_root)
        .env("PYTHONPATH", args.repo_root.to_string_lossy().to_string())
        .output()
        .expect("failed to execute python preprocessing parity script");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "python preprocessing component parity failed for plane={} slice={}: stdout='{}' stderr='{}'",
            args.plane.as_str(),
            args.center,
            stdout.trim(),
            stderr.trim()
        );
    }
}

#[test]
fn start_predict_batch_with_valid_response_should_return_ack_and_paths() {
    let backend = create_backend_state_with_fake_responses(&[
        r#"{"ok":true,"result":{"ack_message":"queued","requested_paths":["a.nii.gz","b.nii.gz"]}}"#,
    ]);

    let (ack, requested) = backend
        .start_predict_batch(&["a.nii.gz".to_string()], &["/data".to_string()])
        .expect("expected start_predict_batch to succeed");

    assert_eq!(ack, "queued");
    assert_eq!(
        requested,
        vec!["a.nii.gz".to_string(), "b.nii.gz".to_string()]
    );
}

#[test]
fn start_predict_batch_without_ack_message_should_return_missing_ack_error() {
    let backend = create_backend_state_with_fake_responses(&[
        r#"{"ok":true,"result":{"requested_paths":["a.nii.gz"]}}"#,
    ]);

    let result = backend.start_predict_batch(&[], &[]);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Missing ack_message in IPC response");
}

#[test]
fn parity_preprocessing_should_match_original_python_components_on_native_input_fixture()
 {
    let Some(repo_root) = find_repo_root() else {
        panic!("failed to locate repo root for preprocessing parity test");
    };

    let input_nii = fixture_native_input(&repo_root);
    if !input_nii.exists() {
        eprintln!(
            "missing fixture input '{}'; skipping preprocessing parity test",
            input_nii.display()
        );
        return;
    }

    let Some(python_bin) = resolve_python_with_component_runtime(&repo_root)
    else {
        eprintln!(
            "python runtime missing required FastSurfer component deps; skipping preprocessing parity test"
        );
        return;
    };

    let volume = load_input_volume(&input_nii.to_string_lossy())
        .expect("failed to load fixture input for preprocessing parity test");

    let num_channels = 7usize;
    let base_res = 1.0f32;

    for plane in [
        InferencePlane::Coronal,
        InferencePlane::Axial,
        InferencePlane::Sagittal,
    ] {
        let center = match plane {
            InferencePlane::Coronal => volume.shape_xyz[2] / 2,
            InferencePlane::Axial => volume.shape_xyz[1] / 2,
            InferencePlane::Sagittal => volume.shape_xyz[0] / 2,
        };

        let prepared = prepare_plane_input_for_slice(
            &volume,
            plane,
            num_channels,
            base_res,
            center,
        )
        .expect("failed to prepare Rust plane input");

        let run_dir = std::env::temp_dir().join(format!(
            "preprocess_component_parity_{}_{}",
            next_test_id(),
            plane.as_str()
        ));
        fs::create_dir_all(&run_dir)
            .expect("failed to create temp parity directory");

        let rust_raw = write_prepared_tensor(&run_dir, &prepared);
        let parity_args = PythonPreprocessComponentCheckArgs {
            python_bin: &python_bin,
            repo_root: &repo_root,
            input_nii: &input_nii,
            plane,
            num_channels,
            base_res,
            center,
            prepared: &prepared,
            rust_raw: &rust_raw,
        };
        run_python_preprocess_component_check(&parity_args);
    }
}
