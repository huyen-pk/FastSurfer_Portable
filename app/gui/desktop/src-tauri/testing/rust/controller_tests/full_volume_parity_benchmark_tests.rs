// This test suite integrates with test-containers for environment isolation.
use super::support::{
    convert_mgz_to_nii_gz, create_backend_state_via_process, find_repo_root,
    fixture_native_input, fixture_python_pred, python_has_fastsurfer_runtime,
    python_has_nibabel_runtime, resolve_python_with_component_runtime,
    run_native_inference_with_timeout,
};
use crate::inference::postprocess::{
    derive_aseg_from_pred, derive_brainmask_from_pred, flip_wm_islands,
    mask_aseg_with_brainmask,
};
use crate::inference::preprocess::{
    InferencePlane, InputVolume, load_input_volume,
    prepare_plane_input_for_slice, transformed_volume_shape,
};
use crate::prediction::run_fastsurfer_inference_with_backend;
use crate::process_mgmt::{
    load_desktop_env, resolve_backend_launch_command_from,
};
use ndarray::Array3;
use nifti::writer::WriterOptions;
use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MIN_DICE_FOREGROUND: f64 = 0.90;
const MIN_DICE_MACRO: f64 = 0.85;
const MIN_ICC_2_1: f64 = 0.95;
const MAX_ASSD_FOREGROUND: f64 = 5.0;
const MAX_HD95_FOREGROUND: f64 = 15.0;
const MAX_POST_ASEG_MISMATCH_RATIO: f64 = 0.05;
const MAX_POST_BRAINMASK_MISMATCH_RATIO: f64 = 0.05;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StageTiming {
    rust_ms: f64,
    python_ms: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StageAccuracy {
    summary: Value,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StageReport {
    timing: StageTiming,
    accuracy: StageAccuracy,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FullVolumeParityReport {
    created_at_unix: u64,
    case_id: String,
    input_mgz: String,
    input_native_nii: String,
    rust_pred: String,
    python_pred: String,
    preprocess: StageReport,
    inference: StageReport,
    postprocess: StageReport,
    artifacts_dir: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PipelinePerfSample {
    elapsed_ms: f64,
    images_per_min: f64,
    speed_note: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InferencePerformanceReport {
    rust: PipelinePerfSample,
    python: PipelinePerfSample,
    rust_vs_python_speedup: f64,
    cpu_note: String,
    memory_note: String,
}

#[derive(Default)]
struct PreprocessStats {
    count: usize,
    sum: f64,
    sumsq: f64,
    min: f32,
    max: f32,
}

impl PreprocessStats {
    fn new() -> Self {
        Self {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            ..Self::default()
        }
    }

    fn update_slice(&mut self, data: &[f32]) {
        for value in data {
            self.count += 1;
            self.sum += f64::from(*value);
            self.sumsq += f64::from(*value) * f64::from(*value);
            if *value < self.min {
                self.min = *value;
            }
            if *value > self.max {
                self.max = *value;
            }
        }
    }

    fn as_json(&self) -> Value {
        if self.count == 0 {
            return serde_json::json!({
                "count": 0,
                "mean": null,
                "std": null,
                "min": null,
                "max": null,
            });
        }
        let n = self.count as f64;
        let mean = self.sum / n;
        let var = (self.sumsq / n) - (mean * mean);
        let std = if var.is_sign_negative() {
            0.0
        } else {
            var.sqrt()
        };
        serde_json::json!({
            "count": self.count,
            "mean": mean,
            "std": std,
            "min": self.min,
            "max": self.max,
        })
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn now_unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn run_python_json(
    python_bin: &str,
    repo_root: &Path,
    script: &str,
    args: &[String],
) -> Result<Value, String> {
    let mut command = Command::new(python_bin);
    command
        .arg("-c")
        .arg(script)
        .current_dir(repo_root)
        .env("PYTHONPATH", repo_root.to_string_lossy().to_string());
    for arg in args {
        command.arg(arg);
    }

    let output = command
        .output()
        .map_err(|error| format!("failed to run python helper: {error}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "python helper failed: stdout='{}' stderr='{}'",
            stdout.trim(),
            stderr.trim()
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(stdout.trim()).map_err(|error| {
        format!(
            "failed to parse python helper json: {error}; raw='{}'",
            stdout.trim()
        )
    })
}

fn write_f32_nifti(
    volume: &InputVolume,
    shape_xyz: [usize; 3],
    data: Vec<f32>,
    output_path: &Path,
) -> Result<(), String> {
    let expected = shape_xyz[0] * shape_xyz[1] * shape_xyz[2];
    if data.len() != expected {
        return Err(format!(
            "f32 nifti data length mismatch: got {}, expected {}",
            data.len(),
            expected
        ));
    }

    let array = Array3::from_shape_vec(
        (shape_xyz[0], shape_xyz[1], shape_xyz[2]),
        data,
    )
    .map_err(|error| {
        format!("failed to shape preprocess volume for nifti write: {error}")
    })?;

    WriterOptions::new(output_path)
        .reference_header(&volume.header)
        .write_nifti(&array)
        .map_err(|error| {
            format!(
                "failed to write preprocess nifti '{}': {error}",
                output_path.display()
            )
        })
}

fn preprocess_stage_rust_full_volume(
    input_nii: &Path,
    preprocess_dir: &Path,
) -> Result<(f64, Value), String> {
    let volume = load_input_volume(&input_nii.to_string_lossy())?;
    let mut stats = PreprocessStats::new();
    let t0 = Instant::now();

    for plane in [
        InferencePlane::Coronal,
        InferencePlane::Axial,
        InferencePlane::Sagittal,
    ] {
        let [height, width, slices_count] =
            transformed_volume_shape(volume.shape_xyz, plane);
        let mut center_channel_volume =
            vec![0f32; height * width * slices_count];

        for slice_index in 0..slices_count {
            let prepared = prepare_plane_input_for_slice(
                &volume,
                plane,
                7,
                1.0,
                slice_index,
            )?;
            stats.update_slice(&prepared.tensor_data);

            let channels = prepared.tensor_shape[1];
            let center_channel = channels / 2;
            let hw = height * width;
            let base = center_channel * hw;
            for row in 0..height {
                for col in 0..width {
                    let plane_offset =
                        ((row * width) + col) * slices_count + slice_index;
                    let slice_offset = base + (row * width) + col;
                    center_channel_volume[plane_offset] =
                        prepared.tensor_data[slice_offset];
                }
            }
        }

        let output_path = preprocess_dir
            .join(format!("rust_preprocess_{}.nii.gz", plane.as_str()));
        write_f32_nifti(
            &volume,
            [height, width, slices_count],
            center_channel_volume,
            &output_path,
        )?;
    }

    Ok((t0.elapsed().as_secs_f64() * 1000.0, stats.as_json()))
}

fn preprocess_stage_python_full_volume(
    python_bin: &str,
    repo_root: &Path,
    input_nii: &Path,
    preprocess_dir: &Path,
) -> Result<(f64, Value), String> {
    let script = r#"
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
from FastSurferCNN.data_loader.data_utils import get_thick_slices, transform_axial, transform_sagittal

input_nii = sys.argv[1]
preprocess_dir = sys.argv[2]
num_channels = 7
slice_thickness = num_channels // 2

def update_stats(stats, arr):
    data = arr.astype(np.float64, copy=False).ravel()
    stats["count"] += int(data.size)
    stats["sum"] += float(data.sum())
    stats["sumsq"] += float(np.square(data).sum())
    local_min = float(data.min())
    local_max = float(data.max())
    if stats["min"] is None or local_min < stats["min"]:
        stats["min"] = local_min
    if stats["max"] is None or local_max > stats["max"]:
        stats["max"] = local_max

img = nib.load(input_nii)
orig = np.asanyarray(img.dataobj)
affine = img.affine
header = img.header.copy()

stats = {"count": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None}
start = time.perf_counter()

for plane in ("coronal", "axial", "sagittal"):
    if plane == "axial":
        transformed = transform_axial(orig)
    elif plane == "sagittal":
        transformed = transform_sagittal(orig)
    else:
        transformed = orig

    thick = get_thick_slices(transformed, slice_thickness)
    thick = np.transpose(thick, (2, 0, 1, 3))
    center_volume = np.zeros((thick.shape[1], thick.shape[2], thick.shape[0]), dtype=np.float32)
    for idx in range(thick.shape[0]):
        image = np.clip(thick[idx].astype(np.float32) / 255.0, a_min=0.0, a_max=1.0)
        image = image.transpose((2, 0, 1))
        update_stats(stats, image)
        center_volume[:, :, idx] = image[slice_thickness, :, :]

    out_path = os.path.join(preprocess_dir, f"python_preprocess_{plane}.nii.gz")
    nib.save(nib.Nifti1Image(center_volume, affine, header=header), out_path)

elapsed_ms = (time.perf_counter() - start) * 1000.0
if stats["count"] > 0:
    n = float(stats["count"])
    mean = stats["sum"] / n
    var = max(0.0, (stats["sumsq"] / n) - (mean * mean))
    std = var ** 0.5
else:
    mean = None
    std = None

print(json.dumps({
    "elapsed_ms": elapsed_ms,
    "stats": {
        "count": stats["count"],
        "mean": mean,
        "std": std,
        "min": stats["min"],
        "max": stats["max"],
    }
}))
"#;

    let parsed = run_python_json(
        python_bin,
        repo_root,
        script,
        &[
            input_nii.to_string_lossy().to_string(),
            preprocess_dir.to_string_lossy().to_string(),
        ],
    )?;
    let elapsed = parsed["elapsed_ms"].as_f64().ok_or_else(|| {
        "python preprocess helper missing elapsed_ms".to_string()
    })?;
    Ok((elapsed, parsed["stats"].clone()))
}

fn inference_stage_python(
    repo_root: &Path,
    input_path: &Path,
) -> Result<(f64, PathBuf), String> {
    let cwd = repo_root.join("app/gui/desktop/src-tauri");
    let exe_dir = cwd.join("target/debug");
    load_desktop_env(&cwd, &exe_dir);
    let launch = resolve_backend_launch_command_from(&cwd, &exe_dir).map_err(
        |error| format!("failed to resolve backend launch command: {error}"),
    )?;

    let mut launch_cmd = Command::new(&launch.program);
    launch_cmd
        .args(&launch.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .current_dir(repo_root)
        .env("PYTHONPATH", repo_root.to_string_lossy().to_string());

    let child = launch_cmd
        .spawn()
        .map_err(|error| format!("failed to spawn backend process: {error}"))?;

    let backend = create_backend_state_via_process(child);

    let t0 = Instant::now();
    let py_result = run_fastsurfer_inference_with_backend(
        &backend,
        &[input_path.to_string_lossy().to_string()],
        &[],
    )
    .map_err(|error| format!("python backend inference failed: {error}"))?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if py_result.results.is_empty() {
        return Err("python backend inference returned no outputs".to_string());
    }

    let py_pred = PathBuf::from(&py_result.results[0].output_path);
    if !py_pred.exists() {
        return Err(format!(
            "python prediction output not found: {}",
            py_pred.display()
        ));
    }

    Ok((elapsed_ms, py_pred))
}

fn ensure_python_golden_fixture(
    repo_root: &Path,
    native_input_nii: &Path,
) -> Result<PathBuf, String> {
    let fs_reference_pred = repo_root.join(
        "app/gui/desktop/src-tauri/testing/data/fs_reference/Subject140/forward pass/python_forward_pred.nii.gz",
    );
    if fs_reference_pred.exists() {
        let fs_volume = load_input_volume(&fs_reference_pred.to_string_lossy())
            .map_err(|error| {
                format!(
                    "failed loading fs_reference python golden fixture: {error}"
                )
            })?;
        let native_volume =
            load_input_volume(&native_input_nii.to_string_lossy()).map_err(|error| {
                format!("failed loading native input for fs_reference validation: {error}")
            })?;

        if fs_volume.shape_xyz == native_volume.shape_xyz {
            return Ok(fs_reference_pred);
        }
    }

    let fixture_pred = fixture_python_pred(repo_root);

    if fixture_pred.exists() {
        let fixture_volume = load_input_volume(&fixture_pred.to_string_lossy())
            .map_err(|error| {
                format!(
                    "failed loading existing python golden fixture: {error}"
                )
            })?;
        let native_volume = load_input_volume(
            &native_input_nii.to_string_lossy(),
        )
        .map_err(|error| {
            format!(
                "failed loading native input for fixture validation: {error}"
            )
        })?;

        if fixture_volume.shape_xyz == native_volume.shape_xyz {
            return Ok(fixture_pred);
        }
    }

    if let Some(parent) = fixture_pred.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!("failed creating python golden fixture directory: {error}")
        })?;
    }

    let (_elapsed_ms, generated_pred) =
        inference_stage_python(repo_root, native_input_nii)?;
    fs::copy(&generated_pred, &fixture_pred).map_err(|error| {
        format!(
            "failed copying generated python prediction '{}' to golden fixture '{}': {error}",
            generated_pred.display(),
            fixture_pred.display()
        )
    })?;

    Ok(fixture_pred)
}

fn postprocess_stage_python_compare(
    python_bin: &str,
    repo_root: &Path,
    pred_nii: &Path,
    rust_aseg_raw: &Path,
    rust_mask_raw: &Path,
    shape_xyz: [usize; 3],
    python_aseg_nii: &Path,
    python_brainmask_nii: &Path,
    rust_aseg_nii: &Path,
    rust_brainmask_nii: &Path,
) -> Result<(f64, Value), String> {
    let script = r#"
import json
import sys
import time

import nibabel as nib
import numpy as np

from FastSurferCNN.reduce_to_aseg import create_mask, flip_wm_islands, reduce_to_aseg

pred_path = sys.argv[1]
rust_aseg_raw = sys.argv[2]
rust_mask_raw = sys.argv[3]
sx = int(sys.argv[4])
sy = int(sys.argv[5])
sz = int(sys.argv[6])
python_aseg_nii = sys.argv[7]
python_brainmask_nii = sys.argv[8]
rust_aseg_nii = sys.argv[9]
rust_brainmask_nii = sys.argv[10]

pred_img = nib.load(pred_path)
pred = np.rint(np.asanyarray(pred_img.dataobj)).astype(np.int32)
affine = pred_img.affine
header = pred_img.header.copy()
start = time.perf_counter()
py_aseg = reduce_to_aseg(pred.copy())
py_mask = create_mask(pred.copy(), 5, 4)
py_aseg[py_mask == 0] = 0
try:
    py_aseg = flip_wm_islands(py_aseg)
except AssertionError:
    pass
elapsed_ms = (time.perf_counter() - start) * 1000.0

rust_aseg = np.fromfile(rust_aseg_raw, dtype=np.uint16).astype(np.int32).reshape((sx, sy, sz))
rust_mask = np.fromfile(rust_mask_raw, dtype=np.uint8).reshape((sx, sy, sz))

nib.save(nib.Nifti1Image(py_aseg.astype(np.int16), affine, header=header), python_aseg_nii)
nib.save(nib.Nifti1Image(py_mask.astype(np.uint8), affine, header=header), python_brainmask_nii)
nib.save(nib.Nifti1Image(rust_aseg.astype(np.int16), affine, header=header), rust_aseg_nii)
nib.save(nib.Nifti1Image(rust_mask.astype(np.uint8), affine, header=header), rust_brainmask_nii)

aseg_diff = int(np.count_nonzero(rust_aseg != py_aseg))
mask_diff = int(np.count_nonzero(rust_mask != py_mask.astype(np.uint8)))

aseg_total = int(rust_aseg.size)
mask_total = int(rust_mask.size)

aseg_ratio = (aseg_diff / aseg_total) if aseg_total else 0.0
mask_ratio = (mask_diff / mask_total) if mask_total else 0.0

print(json.dumps({
    "elapsed_ms": elapsed_ms,
    "accuracy": {
        "aseg_diff": aseg_diff,
        "aseg_total": aseg_total,
        "aseg_mismatch_ratio": aseg_ratio,
        "brainmask_diff": mask_diff,
        "brainmask_total": mask_total,
        "brainmask_mismatch_ratio": mask_ratio,
    }
}))
"#;

    let parsed = run_python_json(
        python_bin,
        repo_root,
        script,
        &[
            pred_nii.to_string_lossy().to_string(),
            rust_aseg_raw.to_string_lossy().to_string(),
            rust_mask_raw.to_string_lossy().to_string(),
            shape_xyz[0].to_string(),
            shape_xyz[1].to_string(),
            shape_xyz[2].to_string(),
            python_aseg_nii.to_string_lossy().to_string(),
            python_brainmask_nii.to_string_lossy().to_string(),
            rust_aseg_nii.to_string_lossy().to_string(),
            rust_brainmask_nii.to_string_lossy().to_string(),
        ],
    )?;

    let elapsed = parsed["elapsed_ms"].as_f64().ok_or_else(|| {
        "python postprocess helper missing elapsed_ms".to_string()
    })?;
    Ok((elapsed, parsed["accuracy"].clone()))
}

fn copy_if_exists(src: &Path, dst: &Path) -> Result<(), String> {
    if src.exists() {
        fs::copy(src, dst).map_err(|error| {
            format!(
                "failed copying '{}' -> '{}': {error}",
                src.display(),
                dst.display()
            )
        })?;
    }
    Ok(())
}

fn read_nested_f64(value: &Value, path: &[&str]) -> Option<f64> {
    let mut cursor = value;
    for key in path {
        cursor = cursor.get(*key)?;
    }
    cursor.as_f64()
}

fn prefixed_file_name(src: &Path, prefix: &str, fallback: &str) -> String {
    match src.file_name().and_then(|name| name.to_str()) {
        Some(name) => format!("{}{}", prefix, name),
        None => fallback.to_string(),
    }
}

fn parity_slices_per_plane() -> usize {
    std::env::var("FASTSURFER_PARITY_SLICES_PER_PLANE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(10)
}

fn parity_run_python_benchmark() -> bool {
    std::env::var("FASTSURFER_PARITY_RUN_PY_BENCH")
        .ok()
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !(normalized == "0"
                || normalized == "false"
                || normalized == "no"
                || normalized == "off")
        })
        .unwrap_or(true)
}

#[test]
#[ignore = "Heavy full-volume parity benchmark for pipeline stages and metrics"]
fn parity_dice_assd_hd95_hdmax_icc_full_volume_pipeline_should_generate_stage_benchmark_report()
 {
    let Some(repo_root) = find_repo_root() else {
        panic!("failed to locate repo root for full-volume parity benchmark");
    };

    let Some(python_bin) = resolve_python_with_component_runtime(&repo_root)
    else {
        eprintln!(
            "python runtime missing component deps; skipping full-volume parity benchmark"
        );
        return;
    };

    if !python_has_fastsurfer_runtime(&python_bin, &repo_root)
        || !python_has_nibabel_runtime(&python_bin, &repo_root)
    {
        eprintln!(
            "python runtime missing FastSurfer/nibabel deps; skipping full-volume parity benchmark"
        );
        return;
    }

    let input_mgz = repo_root
        .join("app/gui/desktop/src-tauri/testing/data/Subject140/140_orig.mgz");
    if !input_mgz.exists() {
        eprintln!(
            "missing Subject140 input at '{}'; skipping",
            input_mgz.display()
        );
        return;
    }

    let fixture_native = fixture_native_input(&repo_root);
    if !fixture_native.exists() {
        if let Some(parent) = fixture_native.parent() {
            fs::create_dir_all(parent)
                .expect("failed to create native fixture parent directory");
        }
        convert_mgz_to_nii_gz(&python_bin, &input_mgz, &fixture_native)
            .expect("failed to convert mgz to native input nii for benchmark");
    }

    let reports_root =
        repo_root.join("app/gui/desktop/src-tauri/testing/rust/results");
    fs::create_dir_all(&reports_root)
        .expect("failed to create parity reports root directory");

    let run_dir = reports_root.join(format!(
        "parity_dice_assd_hd95_hdmax_icc_full_volume_subject140_{}",
        now_unix_millis()
    ));
    fs::create_dir_all(&run_dir)
        .expect("failed to create parity run directory");

    let preprocess_dir = run_dir.join("preprocess");
    let forward_dir = run_dir.join("forward_pass");
    let postprocess_dir = run_dir.join("postprocess");
    fs::create_dir_all(&preprocess_dir)
        .expect("failed to create preprocess artifacts directory");
    fs::create_dir_all(&forward_dir)
        .expect("failed to create forward artifacts directory");
    fs::create_dir_all(&postprocess_dir)
        .expect("failed to create postprocess artifacts directory");

    let input_mgz_copy = run_dir.join("input_subject140.mgz");
    let input_native_copy =
        run_dir.join("input_subject140.native_input.nii.gz");
    copy_if_exists(&input_mgz, &input_mgz_copy)
        .expect("failed copying input mgz artifact");
    copy_if_exists(&fixture_native, &input_native_copy)
        .expect("failed copying input native nii artifact");

    let (rust_pre_ms, rust_pre_stats) =
        preprocess_stage_rust_full_volume(&fixture_native, &preprocess_dir)
            .expect("rust preprocess stage failed");
    let (python_pre_ms, python_pre_stats) =
        preprocess_stage_python_full_volume(
            &python_bin,
            &repo_root,
            &fixture_native,
            &preprocess_dir,
        )
        .expect("python preprocess stage failed");

    let _slices_per_plane = parity_slices_per_plane();

    // Environment must be set externally for this test.
    if std::env::var("FASTSURFER_REPO_ROOT").is_err() {
        eprintln!(
            "Skipping full-volume parity benchmark: FASTSURFER_REPO_ROOT not set"
        );
        return;
    }

    let rust_inf_t0 = Instant::now();
    let rust_result = run_native_inference_with_timeout(
        &[fixture_native.to_string_lossy().to_string()],
        &[],
        Duration::from_secs(900),
    )
    .expect("rust full-volume inference failed");
    let rust_inf_ms = rust_inf_t0.elapsed().as_secs_f64() * 1000.0;

    assert!(
        !rust_result.results.is_empty(),
        "rust full-volume returned no results"
    );
    let rust_pred = PathBuf::from(&rust_result.results[0].output_path);
    assert!(
        rust_pred.exists(),
        "rust pred output not found: {}",
        rust_pred.display()
    );

    let python_golden_pred =
        ensure_python_golden_fixture(&repo_root, &fixture_native)
            .expect("failed to ensure python golden prediction fixture");

    let run_python_benchmark = parity_run_python_benchmark();
    let (python_inf_ms, python_pred_benchmark) = if run_python_benchmark {
        inference_stage_python(&repo_root, &fixture_native)
            .expect("python inference benchmark stage failed")
    } else {
        (0.0, python_golden_pred.clone())
    };

    let metrics_script = repo_root
        .join("app/gui/desktop/src-tauri/testing/python/parity_metrics.py");
    let inference_metrics_json = forward_dir.join("inference_metrics.json");
    let status = Command::new(&python_bin)
        .arg(&metrics_script)
        .arg(&rust_pred)
        .arg(&python_golden_pred)
        .arg(&inference_metrics_json)
        .current_dir(&repo_root)
        .env("PYTHONPATH", repo_root.to_string_lossy().to_string())
        .status()
        .expect("failed to run parity_metrics.py for inference stage");
    assert!(
        status.success(),
        "parity_metrics.py failed for inference stage"
    );

    let inference_metrics = serde_json::from_str::<Value>(
        &fs::read_to_string(&inference_metrics_json)
            .expect("failed reading inference metrics json file"),
    )
    .expect("failed parsing inference metrics json");

    let dice_fg =
        read_nested_f64(&inference_metrics, &["aggregate", "dice_foreground"])
            .unwrap_or(0.0);
    let dice_macro =
        read_nested_f64(&inference_metrics, &["aggregate", "dice_macro"])
            .unwrap_or(0.0);
    let icc_2_1 =
        read_nested_f64(&inference_metrics, &["aggregate", "icc_2_1_volumes"])
            .unwrap_or(0.0);
    let assd_fg =
        read_nested_f64(&inference_metrics, &["aggregate", "assd_foreground"])
            .unwrap_or(f64::INFINITY);
    let hd95_fg =
        read_nested_f64(&inference_metrics, &["aggregate", "hd95_foreground"])
            .unwrap_or(f64::INFINITY);

    assert!(
        dice_fg >= MIN_DICE_FOREGROUND,
        "inference gate failed: dice_foreground {} < {}",
        dice_fg,
        MIN_DICE_FOREGROUND
    );
    assert!(
        dice_macro >= MIN_DICE_MACRO,
        "inference gate failed: dice_macro {} < {}",
        dice_macro,
        MIN_DICE_MACRO
    );
    assert!(
        icc_2_1 >= MIN_ICC_2_1,
        "inference gate failed: icc_2_1_volumes {} < {}",
        icc_2_1,
        MIN_ICC_2_1
    );
    assert!(
        assd_fg <= MAX_ASSD_FOREGROUND,
        "inference gate failed: assd_foreground {} > {}",
        assd_fg,
        MAX_ASSD_FOREGROUND
    );
    assert!(
        hd95_fg <= MAX_HD95_FOREGROUND,
        "inference gate failed: hd95_foreground {} > {}",
        hd95_fg,
        MAX_HD95_FOREGROUND
    );

    let post_input = python_golden_pred.clone();

    let post_volume = load_input_volume(&post_input.to_string_lossy())
        .expect("failed loading postprocess input volume");
    let post_shape = post_volume.shape_xyz;
    let post_labels = post_volume
        .data_xyz
        .iter()
        .map(|v| v.round().max(0.0) as u16)
        .collect::<Vec<u16>>();

    let post_t0 = Instant::now();
    let mut rust_aseg = derive_aseg_from_pred(&post_labels);
    let rust_brainmask = derive_brainmask_from_pred(&post_labels, post_shape);
    mask_aseg_with_brainmask(&mut rust_aseg, &rust_brainmask);
    flip_wm_islands(&mut rust_aseg, post_shape);
    let rust_post_ms = post_t0.elapsed().as_secs_f64() * 1000.0;

    let rust_aseg_raw = postprocess_dir.join("rust_post_aseg.raw");
    let mut aseg_file = fs::File::create(&rust_aseg_raw)
        .expect("failed to create rust_post_aseg.raw");
    for value in &rust_aseg {
        aseg_file
            .write_all(&value.to_le_bytes())
            .expect("failed writing rust_post_aseg.raw");
    }
    let rust_brainmask_raw = postprocess_dir.join("rust_post_brainmask.raw");
    fs::write(&rust_brainmask_raw, &rust_brainmask)
        .expect("failed writing rust_post_brainmask.raw");

    let python_aseg_nii = postprocess_dir.join("python_post_aseg.nii.gz");
    let python_brainmask_nii =
        postprocess_dir.join("python_post_brainmask.nii.gz");
    let rust_aseg_nii = postprocess_dir.join("rust_post_aseg.nii.gz");
    let rust_brainmask_nii = postprocess_dir.join("rust_post_brainmask.nii.gz");

    let (python_post_ms, post_accuracy) = postprocess_stage_python_compare(
        &python_bin,
        &repo_root,
        &post_input,
        &rust_aseg_raw,
        &rust_brainmask_raw,
        post_shape,
        &python_aseg_nii,
        &python_brainmask_nii,
        &rust_aseg_nii,
        &rust_brainmask_nii,
    )
    .expect("python postprocess compare failed");

    let post_aseg_ratio = post_accuracy["aseg_mismatch_ratio"]
        .as_f64()
        .expect("post accuracy missing aseg_mismatch_ratio");
    let post_brainmask_ratio = post_accuracy["brainmask_mismatch_ratio"]
        .as_f64()
        .expect("post accuracy missing brainmask_mismatch_ratio");

    assert!(
        post_aseg_ratio <= MAX_POST_ASEG_MISMATCH_RATIO,
        "postprocess gate failed: aseg_mismatch_ratio {} > {}",
        post_aseg_ratio,
        MAX_POST_ASEG_MISMATCH_RATIO
    );
    assert!(
        post_brainmask_ratio <= MAX_POST_BRAINMASK_MISMATCH_RATIO,
        "postprocess gate failed: brainmask_mismatch_ratio {} > {}",
        post_brainmask_ratio,
        MAX_POST_BRAINMASK_MISMATCH_RATIO
    );

    let rust_pred_copy = forward_dir.join(prefixed_file_name(
        &rust_pred,
        "rust_forward_",
        "rust_forward_pred.nii.gz",
    ));
    let python_golden_pred_copy = forward_dir.join(prefixed_file_name(
        &python_golden_pred,
        "python_forward_golden_",
        "python_forward_golden_pred.nii.gz",
    ));
    let python_benchmark_pred_copy = forward_dir.join(prefixed_file_name(
        &python_pred_benchmark,
        "python_forward_benchmark_",
        "python_forward_benchmark_pred.nii.gz",
    ));
    copy_if_exists(&rust_pred, &rust_pred_copy)
        .expect("failed copying rust pred artifact");
    copy_if_exists(&python_golden_pred, &python_golden_pred_copy)
        .expect("failed copying python golden pred artifact");
    copy_if_exists(&python_pred_benchmark, &python_benchmark_pred_copy)
        .expect("failed copying python benchmark pred artifact");

    let rust_images_per_min = if rust_inf_ms > 0.0 {
        60_000.0 / rust_inf_ms
    } else {
        0.0
    };
    let python_images_per_min = if python_inf_ms > 0.0 {
        60_000.0 / python_inf_ms
    } else {
        0.0
    };
    let rust_vs_python_speedup = if python_inf_ms > 0.0 {
        python_inf_ms / rust_inf_ms.max(1e-9)
    } else {
        0.0
    };

    let inference_performance = InferencePerformanceReport {
        rust: PipelinePerfSample {
            elapsed_ms: rust_inf_ms,
            images_per_min: rust_images_per_min,
            speed_note: "native sparse ONNX pipeline".to_string(),
        },
        python: PipelinePerfSample {
            elapsed_ms: python_inf_ms,
            images_per_min: python_images_per_min,
            speed_note: if run_python_benchmark {
                "python backend runtime pipeline".to_string()
            } else {
                "python benchmark skipped (FASTSURFER_PARITY_RUN_PY_BENCH=0)".to_string()
            },
        },
        rust_vs_python_speedup,
        cpu_note: "cpu utilization and per-process cpu breakdown are not captured by this test yet"
            .to_string(),
        memory_note: "peak RSS is not captured by this test yet".to_string(),
    };

    let report = FullVolumeParityReport {
        created_at_unix: now_unix(),
        case_id: "Subject140".to_string(),
        input_mgz: input_mgz.to_string_lossy().to_string(),
        input_native_nii: fixture_native.to_string_lossy().to_string(),
        rust_pred: rust_pred.to_string_lossy().to_string(),
        python_pred: python_golden_pred.to_string_lossy().to_string(),
        preprocess: StageReport {
            timing: StageTiming {
                rust_ms: rust_pre_ms,
                python_ms: python_pre_ms,
            },
            accuracy: StageAccuracy {
                summary: serde_json::json!({
                    "rust": rust_pre_stats,
                    "python": python_pre_stats,
                }),
            },
        },
        inference: StageReport {
            timing: StageTiming {
                rust_ms: rust_inf_ms,
                python_ms: python_inf_ms,
            },
            accuracy: StageAccuracy {
                summary: serde_json::json!({
                    "comparison_mode": "rust_prediction_vs_python_golden_fixture",
                    "python_golden_pred": python_golden_pred,
                    "python_benchmark_pred": python_pred_benchmark,
                    "metrics": inference_metrics,
                    "performance": inference_performance,
                    "gates": {
                        "dice_foreground_min": MIN_DICE_FOREGROUND,
                        "dice_macro_min": MIN_DICE_MACRO,
                        "icc_2_1_min": MIN_ICC_2_1,
                        "assd_foreground_max": MAX_ASSD_FOREGROUND,
                        "hd95_foreground_max": MAX_HD95_FOREGROUND,
                        "observed": {
                            "dice_foreground": dice_fg,
                            "dice_macro": dice_macro,
                            "icc_2_1_volumes": icc_2_1,
                            "assd_foreground": assd_fg,
                            "hd95_foreground": hd95_fg,
                        },
                    }
                }),
            },
        },
        postprocess: StageReport {
            timing: StageTiming {
                rust_ms: rust_post_ms,
                python_ms: python_post_ms,
            },
            accuracy: StageAccuracy {
                summary: serde_json::json!({
                    "metrics": post_accuracy,
                    "gates": {
                        "aseg_mismatch_ratio_max": MAX_POST_ASEG_MISMATCH_RATIO,
                        "brainmask_mismatch_ratio_max": MAX_POST_BRAINMASK_MISMATCH_RATIO,
                        "observed": {
                            "aseg_mismatch_ratio": post_aseg_ratio,
                            "brainmask_mismatch_ratio": post_brainmask_ratio,
                        },
                    }
                }),
            },
        },
        artifacts_dir: run_dir.to_string_lossy().to_string(),
    };

    let report_path = run_dir.join("full_volume_parity_report.json");
    fs::write(
        &report_path,
        serde_json::to_string_pretty(&report)
            .expect("failed to serialize full volume parity report"),
    )
    .expect("failed writing full volume parity report json");

    eprintln!("[parity][full-volume] report: {}", report_path.display());
}
