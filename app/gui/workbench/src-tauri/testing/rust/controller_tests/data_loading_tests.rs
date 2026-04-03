use super::support::{
    find_repo_root, fixture_native_input, next_test_id,
    resolve_python_with_component_runtime,
};
use crate::inference::pipeline::preprocess::load_and_conform_input_volume;
use serde_json::Value;
use std::process::{Command, Stdio};

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn rounded_u8(value: f32) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

fn parse_json_usize_list(parsed: &Value, key: &str) -> Vec<usize> {
    parsed[key]
        .as_array()
        .unwrap_or_else(|| panic!("python output missing {key}"))
        .iter()
        .map(|value| {
            usize::try_from(value.as_u64().expect("invalid usize value"))
                .expect("python usize value should fit into usize")
        })
        .collect::<Vec<usize>>()
}

fn parse_json_f32_list(parsed: &Value, key: &str) -> Vec<f32> {
    parsed[key]
        .as_array()
        .unwrap_or_else(|| panic!("python output missing {key}"))
        .iter()
        .map(|value| {
            value
                .as_f64()
                .expect("invalid f64 value")
                .to_string()
                .parse::<f32>()
                .expect("python f64 value should fit into f32")
        })
        .collect::<Vec<f32>>()
}

fn parse_json_u64(parsed: &Value, key: &str) -> u64 {
    parsed[key]
        .as_u64()
        .unwrap_or_else(|| panic!("python output missing {key}"))
}

fn parse_json_u8(parsed: &Value, key: &str) -> u8 {
    u8::try_from(
        parsed[key]
            .as_u64()
            .unwrap_or_else(|| panic!("python output missing {key}")),
    )
    .expect("python u8 value should fit into u8")
}

// !!! IMAGE LOADING FUNCTIONS IN data_utils.py are irrelevant
// USE run_prediction.conform_and_save_orig instead
fn python_has_run_prediction_data_loading_runtime(
    python_bin: &str,
    repo_root: &std::path::Path,
) -> bool {
    let probe = r"
from FastSurferCNN.run_prediction import RunModelOnData
from FastSurferCNN.utils.common import SubjectDirectory
";

    Command::new(python_bin)
        .arg("-c")
        .arg(probe)
        .current_dir(repo_root)
        .env("PYTHONPATH", repo_root.to_string_lossy().to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn run_python_conformity_check(
    python_bin: &str,
    repo_root: &std::path::Path,
    input_nii: &std::path::Path,
    conformed_output: &std::path::Path,
    coords: &[Vec<usize>],
) -> Value {
    let script = r#"
import json
import sys

import numpy as np

from FastSurferCNN.run_prediction import RunModelOnData
from FastSurferCNN.utils.common import SubjectDirectory

in_path = sys.argv[1]
out_path = sys.argv[2]
coords = json.loads(sys.argv[3])

runner = RunModelOnData.__new__(RunModelOnData)
runner._threads = 1
runner._async_io = False
runner.orientation = "lia"
runner.image_size = "auto"
runner.vox_size = "min"
runner.conform_to_1mm_threshold = 0.95

subject = SubjectDirectory(
    id="subject140",
    orig_name=in_path,
    conf_name=out_path,
)
image, data = runner.conform_and_save_orig(subject)

shape = list(data.shape[:3])
zoom = [float(v) for v in image.header.get_zooms()[:3]]
samples = [int(data[x, y, z]) for x, y, z in coords]
sum_value = int(np.sum(data.astype(np.uint64), dtype=np.uint64))
sum_sq_value = int(np.sum(np.square(data.astype(np.uint64)), dtype=np.uint64))
min_value = int(np.min(data))
max_value = int(np.max(data))
print(json.dumps({
    "shape": shape,
    "zoom": zoom,
    "samples": samples,
    "sum": sum_value,
    "sum_sq": sum_sq_value,
    "min": min_value,
    "max": max_value,
}))
"#;

    let output = Command::new(python_bin)
        .arg("-c")
        .arg(script)
        .arg(input_nii)
        .arg(conformed_output)
        .arg(
            serde_json::to_string(coords)
                .expect("failed to serialize coordinate list"),
        )
        .current_dir(repo_root)
        .env("PYTHONPATH", repo_root.to_string_lossy().to_string())
        .output()
        .expect("failed to execute python data-loading parity script");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("python data-loading parity failed: {stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(stdout.trim())
        .expect("failed to parse python data-loading parity json output")
}

fn assert_zoom_matches(rust_zoom: [f32; 3], py_zoom: &[f32]) {
    for (idx, py) in py_zoom.iter().enumerate() {
        let rust = rust_zoom[idx];
        assert!(
            (rust - *py).abs() <= 1e-5,
            "zoom mismatch at axis {idx}: rust={rust} python={py}"
        );
    }
}

fn assert_sample_matches(
    shape: [usize; 3],
    rust_data: &[f32],
    coords: &[Vec<usize>],
    py_samples: &[usize],
) {
    for (index, coord) in coords.iter().enumerate() {
        let [x, y, z] = [coord[0], coord[1], coord[2]];
        let rust_offset = (x * shape[1] * shape[2]) + (y * shape[2]) + z;
        let rust_value = usize::from(rounded_u8(rust_data[rust_offset]));
        let py_value = py_samples[index];
        assert!(
            rust_value == py_value,
            "sample mismatch at {coord:?}: rust={rust_value} python={py_value}"
        );
    }
}

fn rust_volume_checksums(rust_data: &[f32]) -> (u64, u64, u8, u8) {
    let mut sum = 0u64;
    let mut sum_sq = 0u64;
    let mut min_value = u8::MAX;
    let mut max_value = u8::MIN;

    for value in rust_data.iter().copied() {
        let value = rounded_u8(value);
        let widened = u64::from(value);
        sum += widened;
        sum_sq += widened * widened;
        min_value = min_value.min(value);
        max_value = max_value.max(value);
    }

    (sum, sum_sq, min_value, max_value)
}

#[test]
fn parity_data_loading_should_match_python_run_prediction_conform_and_save_orig_component()
 {
    let Some(repo_root) = find_repo_root() else {
        panic!("failed to locate repo root for data loading parity test");
    };

    let input_nii = fixture_native_input(&repo_root);
    assert!(
        input_nii.exists(),
        "input fixture for data loading parity test not found: {}",
        input_nii.display()
    );

    let Some(python_bin) = resolve_python_with_component_runtime(&repo_root)
    else {
        panic!("python runtime missing required FastSurfer component deps");
    };

    assert!(
        python_has_run_prediction_data_loading_runtime(&python_bin, &repo_root),
        "python runtime missing required FastSurfer run_prediction data-loading deps"
    );

    let conformed_output = std::env::temp_dir().join(format!(
        "fastsurfer_data_loading_parity_{}_{}.nii.gz",
        std::process::id(),
        next_test_id()
    ));

    let rust_volume =
        load_and_conform_input_volume(&input_nii.to_string_lossy())
            .expect("failed to conform input fixture with Rust data loader");

    let py_conformed = run_python_conformity_check(
        &python_bin,
        &repo_root,
        &input_nii,
        &conformed_output,
        &[],
    );

    let py_shape = parse_json_usize_list(&py_conformed, "shape");
    assert_eq!(py_shape, rust_volume.shape_xyz.to_vec(), "shape mismatch");

    let py_zoom = parse_json_f32_list(&py_conformed, "zoom");
    assert_zoom_matches(rust_volume.zoom_xyz, &py_zoom);

    let (rust_sum, rust_sum_sq, rust_min, rust_max) =
        rust_volume_checksums(&rust_volume.data_xyz);
    assert_eq!(
        rust_sum,
        parse_json_u64(&py_conformed, "sum"),
        "sum mismatch"
    );
    assert_eq!(
        rust_sum_sq,
        parse_json_u64(&py_conformed, "sum_sq"),
        "sum of squares mismatch"
    );
    assert_eq!(
        rust_min,
        parse_json_u8(&py_conformed, "min"),
        "min mismatch"
    );
    assert_eq!(
        rust_max,
        parse_json_u8(&py_conformed, "max"),
        "max mismatch"
    );

    let shape = rust_volume.shape_xyz;
    let coords = vec![
        vec![0usize, 0usize, 0usize],
        vec![shape[0] / 2, shape[1] / 2, shape[2] / 2],
        vec![
            shape[0].saturating_sub(1), // ~ shape[0] - 1
            shape[1].saturating_sub(1), // => find the last index of the dimension
            shape[2].saturating_sub(1), // and handle underflow (<0)
        ],
        vec![shape[0] / 4, shape[1] / 3, shape[2] / 5],
        vec![shape[0] / 3, shape[1] / 5, shape[2] / 7],
    ];

    let py_conformed = run_python_conformity_check(
        &python_bin,
        &repo_root,
        &input_nii,
        &conformed_output,
        &coords,
    );

    let py_samples = parse_json_usize_list(&py_conformed, "samples");
    assert_sample_matches(shape, &rust_volume.data_xyz, &coords, &py_samples);

    let _ = std::fs::remove_file(&conformed_output);
}
