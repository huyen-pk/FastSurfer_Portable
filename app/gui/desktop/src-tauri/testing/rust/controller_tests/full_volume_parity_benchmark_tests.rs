// This test suite integrates with test-containers for environment isolation.
use super::support::{
    convert_mgz_to_nii_gz, create_backend_state_via_process, find_repo_root,
    fixture_native_input, fixture_python_pred, python_has_fastsurfer_runtime,
    python_has_nibabel_runtime, resolve_python_with_component_runtime,
    run_native_inference_with_timeout,
};
use crate::inference::pipeline::postprocess::{
    derive_aseg_from_pred, derive_brainmask_from_pred, flip_wm_islands,
    mask_aseg_with_brainmask,
};
use crate::inference::pipeline::preprocess::{
    InferencePlane, InputVolume, load_input_volume,
    prepare_plane_input_for_slice, transformed_volume_shape,
};
use crate::prediction::legacy_run_fastsurfer_inference_with_backend;
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

struct FullVolumeRunContext {
    repo_root: PathBuf,
    python_bin: String,
    input_mgz: PathBuf,
    fixture_native: PathBuf,
    run_dir: PathBuf,
    preprocess_dir: PathBuf,
    forward_dir: PathBuf,
    postprocess_dir: PathBuf,
}

struct InferenceStageOutput {
    rust_inf_ms: f64,
    python_inf_ms: f64,
    rust_pred: PathBuf,
    python_golden_pred: PathBuf,
    python_pred_benchmark: PathBuf,
    run_python_benchmark: bool,
    inference_metrics: Value,
    dice_fg: f64,
    dice_macro: f64,
    icc_2_1: f64,
    assd_fg: f64,
    hd95_fg: f64,
}

struct PostprocessStageOutput {
    rust_post_ms: f64,
    python_post_ms: f64,
    post_accuracy: Value,
    post_aseg_ratio: f64,
    post_brainmask_ratio: f64,
}

struct InferenceGateMetrics {
    dice_fg: f64,
    dice_macro: f64,
    icc_2_1: f64,
    assd_fg: f64,
    hd95_fg: f64,
}

struct PostprocessCompareArgs<'a> {
    pred_nii: &'a Path,
    rust_aseg_raw: &'a Path,
    rust_mask_raw: &'a Path,
    shape_xyz: [usize; 3],
    python_aseg_nii: &'a Path,
    python_brainmask_nii: &'a Path,
    rust_aseg_nii: &'a Path,
    rust_brainmask_nii: &'a Path,
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
        let n = f64::from(
            u32::try_from(self.count).expect("count should fit into u32"),
        );
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
    let py_result = legacy_run_fastsurfer_inference_with_backend(
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
    args: &PostprocessCompareArgs<'_>,
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
            args.pred_nii.to_string_lossy().to_string(),
            args.rust_aseg_raw.to_string_lossy().to_string(),
            args.rust_mask_raw.to_string_lossy().to_string(),
            args.shape_xyz[0].to_string(),
            args.shape_xyz[1].to_string(),
            args.shape_xyz[2].to_string(),
            args.python_aseg_nii.to_string_lossy().to_string(),
            args.python_brainmask_nii.to_string_lossy().to_string(),
            args.rust_aseg_nii.to_string_lossy().to_string(),
            args.rust_brainmask_nii.to_string_lossy().to_string(),
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
        Some(name) => format!("{prefix}{name}"),
        None => fallback.to_string(),
    }
}

fn round_label_to_u16(value: f32) -> u16 {
    value
        .round()
        .max(0.0)
        .to_string()
        .parse::<u16>()
        .expect("rounded label value should fit into u16")
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
        .is_none_or(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !(normalized == "0"
                || normalized == "false"
                || normalized == "no"
                || normalized == "off")
        })
}

fn load_inference_metrics(
    context: &FullVolumeRunContext,
    rust_pred: &Path,
    python_golden_pred: &Path,
) -> Result<Value, String> {
    let metrics_script = context
        .repo_root
        .join("app/gui/desktop/src-tauri/testing/python/parity_metrics.py");
    let inference_metrics_json =
        context.forward_dir.join("inference_metrics.json");
    let status = Command::new(&context.python_bin)
        .arg(&metrics_script)
        .arg(rust_pred)
        .arg(python_golden_pred)
        .arg(&inference_metrics_json)
        .current_dir(&context.repo_root)
        .env(
            "PYTHONPATH",
            context.repo_root.to_string_lossy().to_string(),
        )
        .status()
        .map_err(|error| {
            format!(
                "failed to run parity_metrics.py for inference stage: {error}"
            )
        })?;
    if !status.success() {
        return Err("parity_metrics.py failed for inference stage".to_string());
    }

    serde_json::from_str::<Value>(
        &fs::read_to_string(&inference_metrics_json).map_err(|error| {
            format!("failed reading inference metrics json file: {error}")
        })?,
    )
    .map_err(|error| format!("failed parsing inference metrics json: {error}"))
}

fn extract_inference_gate_metrics(
    inference_metrics: &Value,
) -> InferenceGateMetrics {
    InferenceGateMetrics {
        dice_fg: read_nested_f64(
            inference_metrics,
            &["aggregate", "dice_foreground"],
        )
        .unwrap_or(0.0),
        dice_macro: read_nested_f64(
            inference_metrics,
            &["aggregate", "dice_macro"],
        )
        .unwrap_or(0.0),
        icc_2_1: read_nested_f64(
            inference_metrics,
            &["aggregate", "icc_2_1_volumes"],
        )
        .unwrap_or(0.0),
        assd_fg: read_nested_f64(
            inference_metrics,
            &["aggregate", "assd_foreground"],
        )
        .unwrap_or(f64::INFINITY),
        hd95_fg: read_nested_f64(
            inference_metrics,
            &["aggregate", "hd95_foreground"],
        )
        .unwrap_or(f64::INFINITY),
    }
}

fn assert_inference_gate_metrics(metrics: &InferenceGateMetrics) {
    assert!(
        metrics.dice_fg >= MIN_DICE_FOREGROUND,
        "inference gate failed: dice_foreground {} < {}",
        metrics.dice_fg,
        MIN_DICE_FOREGROUND
    );
    assert!(
        metrics.dice_macro >= MIN_DICE_MACRO,
        "inference gate failed: dice_macro {} < {}",
        metrics.dice_macro,
        MIN_DICE_MACRO
    );
    assert!(
        metrics.icc_2_1 >= MIN_ICC_2_1,
        "inference gate failed: icc_2_1_volumes {} < {}",
        metrics.icc_2_1,
        MIN_ICC_2_1
    );
    assert!(
        metrics.assd_fg <= MAX_ASSD_FOREGROUND,
        "inference gate failed: assd_foreground {} > {}",
        metrics.assd_fg,
        MAX_ASSD_FOREGROUND
    );
    assert!(
        metrics.hd95_fg <= MAX_HD95_FOREGROUND,
        "inference gate failed: hd95_foreground {} > {}",
        metrics.hd95_fg,
        MAX_HD95_FOREGROUND
    );
}

fn copy_forward_artifacts(
    context: &FullVolumeRunContext,
    inference: &InferenceStageOutput,
) -> Result<(), String> {
    let rust_pred_copy = context.forward_dir.join(prefixed_file_name(
        &inference.rust_pred,
        "rust_forward_",
        "rust_forward_pred.nii.gz",
    ));
    let python_golden_pred_copy = context.forward_dir.join(prefixed_file_name(
        &inference.python_golden_pred,
        "python_forward_golden_",
        "python_forward_golden_pred.nii.gz",
    ));
    let python_benchmark_pred_copy =
        context.forward_dir.join(prefixed_file_name(
            &inference.python_pred_benchmark,
            "python_forward_benchmark_",
            "python_forward_benchmark_pred.nii.gz",
        ));
    copy_if_exists(&inference.rust_pred, &rust_pred_copy)?;
    copy_if_exists(&inference.python_golden_pred, &python_golden_pred_copy)?;
    copy_if_exists(
        &inference.python_pred_benchmark,
        &python_benchmark_pred_copy,
    )?;
    Ok(())
}

fn build_inference_performance(
    inference: &InferenceStageOutput,
) -> InferencePerformanceReport {
    let rust_images_per_min = if inference.rust_inf_ms > 0.0 {
        60_000.0 / inference.rust_inf_ms
    } else {
        0.0
    };
    let python_images_per_min = if inference.python_inf_ms > 0.0 {
        60_000.0 / inference.python_inf_ms
    } else {
        0.0
    };
    let rust_vs_python_speedup = if inference.python_inf_ms > 0.0 {
        inference.python_inf_ms / inference.rust_inf_ms.max(1e-9)
    } else {
        0.0
    };

    InferencePerformanceReport {
        rust: PipelinePerfSample {
            elapsed_ms: inference.rust_inf_ms,
            images_per_min: rust_images_per_min,
            speed_note: "native sparse ONNX pipeline".to_string(),
        },
        python: PipelinePerfSample {
            elapsed_ms: inference.python_inf_ms,
            images_per_min: python_images_per_min,
            speed_note: if inference.run_python_benchmark {
                "python backend runtime pipeline".to_string()
            } else {
                "python benchmark skipped (FASTSURFER_PARITY_RUN_PY_BENCH=0)".to_string()
            },
        },
        rust_vs_python_speedup,
        cpu_note: "cpu utilization and per-process cpu breakdown are not captured by this test yet"
            .to_string(),
        memory_note: "peak RSS is not captured by this test yet".to_string(),
    }
}

fn setup_full_volume_run_context(
    repo_root: PathBuf,
    python_bin: String,
) -> Result<FullVolumeRunContext, String> {
    let input_mgz = repo_root
        .join("app/gui/desktop/src-tauri/testing/data/Subject140/140_orig.mgz");
    if !input_mgz.exists() {
        return Err(format!(
            "missing Subject140 input at '{}'",
            input_mgz.display()
        ));
    }

    let fixture_native = fixture_native_input(&repo_root);
    if !fixture_native.exists() {
        if let Some(parent) = fixture_native.parent() {
            fs::create_dir_all(parent).map_err(|error| {
                format!(
                    "failed to create native fixture parent directory: {error}"
                )
            })?;
        }
        convert_mgz_to_nii_gz(&python_bin, &input_mgz, &fixture_native)?;
    }

    let reports_root =
        repo_root.join("app/gui/desktop/src-tauri/testing/rust/results");
    fs::create_dir_all(&reports_root).map_err(|error| {
        format!("failed to create parity reports root directory: {error}")
    })?;

    let run_dir = reports_root.join(format!(
        "parity_dice_assd_hd95_hdmax_icc_full_volume_subject140_{}",
        now_unix_millis()
    ));
    let preprocess_dir = run_dir.join("preprocess");
    let forward_dir = run_dir.join("forward_pass");
    let postprocess_dir = run_dir.join("postprocess");

    fs::create_dir_all(&preprocess_dir).map_err(|error| {
        format!("failed to create preprocess artifacts directory: {error}")
    })?;
    fs::create_dir_all(&forward_dir).map_err(|error| {
        format!("failed to create forward artifacts directory: {error}")
    })?;
    fs::create_dir_all(&postprocess_dir).map_err(|error| {
        format!("failed to create postprocess artifacts directory: {error}")
    })?;

    copy_if_exists(&input_mgz, &run_dir.join("input_subject140.mgz"))?;
    copy_if_exists(
        &fixture_native,
        &run_dir.join("input_subject140.native_input.nii.gz"),
    )?;

    Ok(FullVolumeRunContext {
        repo_root,
        python_bin,
        input_mgz,
        fixture_native,
        run_dir,
        preprocess_dir,
        forward_dir,
        postprocess_dir,
    })
}

fn run_inference_stage(
    context: &FullVolumeRunContext,
) -> Result<InferenceStageOutput, String> {
    if std::env::var("FASTSURFER_REPO_ROOT").is_err() {
        return Err("FASTSURFER_REPO_ROOT not set".to_string());
    }

    let rust_inf_t0 = Instant::now();
    let rust_result = run_native_inference_with_timeout(
        &[context.fixture_native.to_string_lossy().to_string()],
        &[],
        Duration::from_secs(900),
    )?;
    let rust_inf_ms = rust_inf_t0.elapsed().as_secs_f64() * 1000.0;

    let rust_pred = rust_result
        .results
        .first()
        .map(|result| PathBuf::from(&result.output_path))
        .ok_or_else(|| "rust full-volume returned no results".to_string())?;
    if !rust_pred.exists() {
        return Err(format!(
            "rust pred output not found: {}",
            rust_pred.display()
        ));
    }

    let python_golden_pred = ensure_python_golden_fixture(
        &context.repo_root,
        &context.fixture_native,
    )?;
    let run_python_benchmark = parity_run_python_benchmark();
    let (python_inf_ms, python_pred_benchmark) = if run_python_benchmark {
        inference_stage_python(&context.repo_root, &context.fixture_native)?
    } else {
        (0.0, python_golden_pred.clone())
    };

    let inference_metrics =
        load_inference_metrics(context, &rust_pred, &python_golden_pred)?;
    let metrics = extract_inference_gate_metrics(&inference_metrics);
    assert_inference_gate_metrics(&metrics);

    Ok(InferenceStageOutput {
        rust_inf_ms,
        python_inf_ms,
        rust_pred,
        python_golden_pred,
        python_pred_benchmark,
        run_python_benchmark,
        inference_metrics,
        dice_fg: metrics.dice_fg,
        dice_macro: metrics.dice_macro,
        icc_2_1: metrics.icc_2_1,
        assd_fg: metrics.assd_fg,
        hd95_fg: metrics.hd95_fg,
    })
}

fn run_postprocess_stage(
    context: &FullVolumeRunContext,
    post_input: &Path,
) -> Result<PostprocessStageOutput, String> {
    let post_volume = load_input_volume(&post_input.to_string_lossy())?;
    let post_shape = post_volume.shape_xyz;
    let post_labels = post_volume
        .data_xyz
        .iter()
        .map(|value| round_label_to_u16(*value))
        .collect::<Vec<u16>>();

    let post_t0 = Instant::now();
    let mut rust_aseg = derive_aseg_from_pred(&post_labels);
    let rust_brainmask = derive_brainmask_from_pred(&post_labels, post_shape);
    mask_aseg_with_brainmask(&mut rust_aseg, &rust_brainmask);
    flip_wm_islands(&mut rust_aseg, post_shape);
    let rust_post_ms = post_t0.elapsed().as_secs_f64() * 1000.0;

    let rust_aseg_raw = context.postprocess_dir.join("rust_post_aseg.raw");
    let mut aseg_file = fs::File::create(&rust_aseg_raw).map_err(|error| {
        format!("failed to create rust_post_aseg.raw: {error}")
    })?;
    for value in &rust_aseg {
        aseg_file.write_all(&value.to_le_bytes()).map_err(|error| {
            format!("failed writing rust_post_aseg.raw: {error}")
        })?;
    }
    let rust_brainmask_raw =
        context.postprocess_dir.join("rust_post_brainmask.raw");
    fs::write(&rust_brainmask_raw, &rust_brainmask).map_err(|error| {
        format!("failed writing rust_post_brainmask.raw: {error}")
    })?;

    let compare_args = PostprocessCompareArgs {
        pred_nii: post_input,
        rust_aseg_raw: &rust_aseg_raw,
        rust_mask_raw: &rust_brainmask_raw,
        shape_xyz: post_shape,
        python_aseg_nii: &context
            .postprocess_dir
            .join("python_post_aseg.nii.gz"),
        python_brainmask_nii: &context
            .postprocess_dir
            .join("python_post_brainmask.nii.gz"),
        rust_aseg_nii: &context.postprocess_dir.join("rust_post_aseg.nii.gz"),
        rust_brainmask_nii: &context
            .postprocess_dir
            .join("rust_post_brainmask.nii.gz"),
    };
    let (python_post_ms, post_accuracy) = postprocess_stage_python_compare(
        &context.python_bin,
        &context.repo_root,
        &compare_args,
    )?;

    let post_aseg_ratio = post_accuracy["aseg_mismatch_ratio"]
        .as_f64()
        .ok_or_else(|| {
            "post accuracy missing aseg_mismatch_ratio".to_string()
        })?;
    let post_brainmask_ratio = post_accuracy["brainmask_mismatch_ratio"]
        .as_f64()
        .ok_or_else(|| {
            "post accuracy missing brainmask_mismatch_ratio".to_string()
        })?;

    assert!(
        post_aseg_ratio <= MAX_POST_ASEG_MISMATCH_RATIO,
        "postprocess gate failed: aseg_mismatch_ratio {post_aseg_ratio} > {MAX_POST_ASEG_MISMATCH_RATIO}"
    );
    assert!(
        post_brainmask_ratio <= MAX_POST_BRAINMASK_MISMATCH_RATIO,
        "postprocess gate failed: brainmask_mismatch_ratio {post_brainmask_ratio} > {MAX_POST_BRAINMASK_MISMATCH_RATIO}"
    );

    Ok(PostprocessStageOutput {
        rust_post_ms,
        python_post_ms,
        post_accuracy,
        post_aseg_ratio,
        post_brainmask_ratio,
    })
}

fn write_full_volume_report(
    context: &FullVolumeRunContext,
    rust_pre_ms: f64,
    rust_pre_stats: &Value,
    python_pre_ms: f64,
    python_pre_stats: &Value,
    inference: &InferenceStageOutput,
    postprocess: &PostprocessStageOutput,
) -> Result<PathBuf, String> {
    copy_forward_artifacts(context, inference)?;
    let inference_performance = build_inference_performance(inference);

    let report = FullVolumeParityReport {
        created_at_unix: now_unix(),
        case_id: "Subject140".to_string(),
        input_mgz: context.input_mgz.to_string_lossy().to_string(),
        input_native_nii: context.fixture_native.to_string_lossy().to_string(),
        rust_pred: inference.rust_pred.to_string_lossy().to_string(),
        python_pred: inference.python_golden_pred.to_string_lossy().to_string(),
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
                rust_ms: inference.rust_inf_ms,
                python_ms: inference.python_inf_ms,
            },
            accuracy: StageAccuracy {
                summary: serde_json::json!({
                    "comparison_mode": "rust_prediction_vs_python_golden_fixture",
                    "python_golden_pred": inference.python_golden_pred,
                    "python_benchmark_pred": inference.python_pred_benchmark,
                    "metrics": inference.inference_metrics,
                    "performance": inference_performance,
                    "gates": {
                        "dice_foreground_min": MIN_DICE_FOREGROUND,
                        "dice_macro_min": MIN_DICE_MACRO,
                        "icc_2_1_min": MIN_ICC_2_1,
                        "assd_foreground_max": MAX_ASSD_FOREGROUND,
                        "hd95_foreground_max": MAX_HD95_FOREGROUND,
                        "observed": {
                            "dice_foreground": inference.dice_fg,
                            "dice_macro": inference.dice_macro,
                            "icc_2_1_volumes": inference.icc_2_1,
                            "assd_foreground": inference.assd_fg,
                            "hd95_foreground": inference.hd95_fg,
                        },
                    }
                }),
            },
        },
        postprocess: StageReport {
            timing: StageTiming {
                rust_ms: postprocess.rust_post_ms,
                python_ms: postprocess.python_post_ms,
            },
            accuracy: StageAccuracy {
                summary: serde_json::json!({
                    "metrics": postprocess.post_accuracy,
                    "gates": {
                        "aseg_mismatch_ratio_max": MAX_POST_ASEG_MISMATCH_RATIO,
                        "brainmask_mismatch_ratio_max": MAX_POST_BRAINMASK_MISMATCH_RATIO,
                        "observed": {
                            "aseg_mismatch_ratio": postprocess.post_aseg_ratio,
                            "brainmask_mismatch_ratio": postprocess.post_brainmask_ratio,
                        },
                    }
                }),
            },
        },
        artifacts_dir: context.run_dir.to_string_lossy().to_string(),
    };

    let report_path = context.run_dir.join("full_volume_parity_report.json");
    fs::write(
        &report_path,
        serde_json::to_string_pretty(&report).map_err(|error| {
            format!("failed to serialize full volume parity report: {error}")
        })?,
    )
    .map_err(|error| {
        format!("failed writing full volume parity report json: {error}")
    })?;

    Ok(report_path)
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

    let context = match setup_full_volume_run_context(repo_root, python_bin) {
        Ok(context) => context,
        Err(error) => {
            eprintln!("{error}; skipping");
            return;
        }
    };

    let (rust_pre_ms, rust_pre_stats) = preprocess_stage_rust_full_volume(
        &context.fixture_native,
        &context.preprocess_dir,
    )
    .expect("rust preprocess stage failed");
    let (python_pre_ms, python_pre_stats) =
        preprocess_stage_python_full_volume(
            &context.python_bin,
            &context.repo_root,
            &context.fixture_native,
            &context.preprocess_dir,
        )
        .expect("python preprocess stage failed");

    let _slices_per_plane = parity_slices_per_plane();
    let inference = match run_inference_stage(&context) {
        Ok(inference) => inference,
        Err(error) if error == "FASTSURFER_REPO_ROOT not set" => {
            eprintln!("Skipping full-volume parity benchmark: {error}");
            return;
        }
        Err(error) => panic!("{error}"),
    };

    let postprocess =
        run_postprocess_stage(&context, &inference.python_golden_pred)
            .expect("python postprocess compare failed");

    let report_path = write_full_volume_report(
        &context,
        rust_pre_ms,
        &rust_pre_stats,
        python_pre_ms,
        &python_pre_stats,
        &inference,
        &postprocess,
    )
    .expect("failed to write full volume parity report");

    eprintln!("[parity][full-volume] report: {}", report_path.display());
}
