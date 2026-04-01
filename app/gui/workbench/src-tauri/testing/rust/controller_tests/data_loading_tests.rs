// This test suite integrates with test-containers for environment isolation.
use super::support::{
    find_repo_root, fixture_native_input, resolve_python_with_component_runtime,
};
use crate::inference::pipeline::preprocess::load_input_volume;
use serde_json::Value;
use std::process::Command;

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

fn run_python_data_loading_parity(
    python_bin: &str,
    repo_root: &std::path::Path,
    input_nii: &std::path::Path,
    coords: &[Vec<usize>],
) -> Value {
    let script = r#"
import json
import sys

from FastSurferCNN.data_loader.data_utils import load_and_conform_image

in_path = sys.argv[1]
coords = json.loads(sys.argv[2])
header, _affine, data = load_and_conform_image(in_path, vox_size=1.0)
shape = list(data.shape[:3])
zoom = [float(v) for v in header.get_zooms()[:3]]
samples = [float(data[x, y, z]) for x, y, z in coords]
print(json.dumps({"shape": shape, "zoom": zoom, "samples": samples}))
"#;

    let output = Command::new(python_bin)
        .arg("-c")
        .arg(script)
        .arg(input_nii)
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
    py_samples: &[f32],
) {
    for (index, coord) in coords.iter().enumerate() {
        let [x, y, z] = [coord[0], coord[1], coord[2]];
        let rust_offset = (x * shape[1] * shape[2]) + (y * shape[2]) + z;
        let rust_value = rust_data[rust_offset];
        let py_value = py_samples[index];
        assert!(
            (rust_value - py_value).abs() <= 1e-5,
            "sample mismatch at {coord:?}: rust={rust_value} python={py_value}"
        );
    }
}

#[test]
fn parity_data_loading_should_match_python_load_and_conform_image_component() {
    let Some(repo_root) = find_repo_root() else {
        panic!("failed to locate repo root for data loading parity test");
    };

    let input_nii = fixture_native_input(&repo_root);
    if !input_nii.exists() {
        eprintln!(
            "missing fixture input '{}'; skipping data loading parity test",
            input_nii.display()
        );
        return;
    }

    let Some(python_bin) = resolve_python_with_component_runtime(&repo_root)
    else {
        eprintln!(
            "python runtime missing required FastSurfer component deps; skipping data loading parity test"
        );
        return;
    };

    let rust_volume = load_input_volume(&input_nii.to_string_lossy())
        .expect("failed to load fixture input with Rust data loader");

    let shape = rust_volume.shape_xyz;
    let coords = vec![
        vec![0usize, 0usize, 0usize],
        vec![shape[0] / 2, shape[1] / 2, shape[2] / 2],
        vec![
            shape[0].saturating_sub(1),
            shape[1].saturating_sub(1),
            shape[2].saturating_sub(1),
        ],
        vec![shape[0] / 4, shape[1] / 3, shape[2] / 5],
        vec![shape[0] / 3, shape[1] / 5, shape[2] / 7],
    ];

    let parsed = run_python_data_loading_parity(
        &python_bin,
        &repo_root,
        &input_nii,
        &coords,
    );

    let py_shape = parse_json_usize_list(&parsed, "shape");
    assert_eq!(py_shape, rust_volume.shape_xyz.to_vec(), "shape mismatch");

    let py_zoom = parse_json_f32_list(&parsed, "zoom");
    assert_zoom_matches(rust_volume.zoom_xyz, &py_zoom);

    let py_samples = parse_json_f32_list(&parsed, "samples");
    assert_sample_matches(shape, &rust_volume.data_xyz, &coords, &py_samples);
}
