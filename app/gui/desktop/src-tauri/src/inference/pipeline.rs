use crate::inference::onnx_loader::NativeOnnxSessions;
use crate::inference::postprocess::{
    derive_aseg_from_pred, derive_brainmask_from_pred, flip_wm_islands, mask_aseg_with_brainmask,
    split_cortex_labels,
};
use crate::inference::preprocess::{
    InferencePlane, InputVolume, build_legacy_preprocessing_trace, load_input_volume, oriented_to_xyz,
    prepare_plane_input_for_slice, transformed_volume_shape,
};
use crate::inference::qc::evaluate_qc;
use crate::models::{
    InferenceArtifacts, InferenceOutput, InferenceProgressEvent, InferenceQc, ProcessingRunResult,
};
use ndarray::Array3;
use nifti::writer::WriterOptions;
use std::collections::{BTreeSet, HashMap};
use std::io::{BufRead, BufReader};
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter};

#[derive(Clone)]
struct PlaneForwardResult {
    summary: String,
    logits: Vec<f32>,
    shape_chw: [usize; 3],
    slice_index: usize,
}

struct NativeSingleResult {
    prediction: InferenceOutput,
    result_directory: String,
    qc_summary: String,
}

fn discovery_summary(models: &NativeOnnxSessions) -> String {
    format!(
        "axial='{}', coronal='{}', sagittal='{}'",
        models.registry.axial_model_path,
        models.registry.coronal_model_path,
        models.registry.sagittal_model_path
    )
}

fn session_channels_or_default(input_shape: &Option<Vec<usize>>) -> usize {
    input_shape
        .as_ref()
        .and_then(|dims| dims.get(1).copied())
        .filter(|channels| *channels > 0)
        .unwrap_or(7)
}

fn native_trace_timing_enabled() -> bool {
    std::env::var("FASTSURFER_NATIVE_TRACE_TIMING")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on"
        })
        .unwrap_or(false)
}

fn parse_native_slices_per_plane() -> Option<usize> {
    std::env::var("FASTSURFER_NATIVE_SLICES_PER_PLANE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn native_sparse_parallel_enabled() -> bool {
    std::env::var("FASTSURFER_NATIVE_SPARSE_PARALLEL")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !(normalized == "0" || normalized == "false" || normalized == "no" || normalized == "off")
        })
        .unwrap_or(true)
}

fn native_slice_indices_for_plane(total_slices: usize) -> Vec<usize> {
    if total_slices == 0 {
        return Vec::new();
    }

    let Some(requested) = parse_native_slices_per_plane() else {
        return (0..total_slices).collect::<Vec<usize>>();
    };

    let requested = requested.min(total_slices);
    if requested >= total_slices {
        return (0..total_slices).collect::<Vec<usize>>();
    }

    if requested == 1 {
        return vec![total_slices / 2];
    }

    let max_idx = total_slices - 1;
    let mut indices = Vec::<usize>::with_capacity(requested);
    for i in 0..requested {
        let idx = (i * max_idx) / (requested - 1);
        if indices.last().copied() != Some(idx) {
            indices.push(idx);
        }
    }

    if indices.is_empty() {
        vec![total_slices / 2]
    } else {
        indices
    }
}

fn legacy_preprocessing_summary(models: &NativeOnnxSessions) -> String {
    let base_res = 1.0f32;
    let synthetic_zoom = [1.0f32, 1.0f32, 1.0f32];

    let axial_channels = session_channels_or_default(&models.axial.input_shape);
    let coronal_channels = session_channels_or_default(&models.coronal.input_shape);
    let sagittal_channels = session_channels_or_default(&models.sagittal.input_shape);

    let axial = build_legacy_preprocessing_trace(
        InferencePlane::Axial,
        axial_channels,
        base_res,
        synthetic_zoom,
    );
    let coronal = build_legacy_preprocessing_trace(
        InferencePlane::Coronal,
        coronal_channels,
        base_res,
        synthetic_zoom,
    );
    let sagittal = build_legacy_preprocessing_trace(
        InferencePlane::Sagittal,
        sagittal_channels,
        base_res,
        synthetic_zoom,
    );

    format!("{axial} || {coronal} || {sagittal}")
}

fn run_single_plane_forward_at_slice(
    sessions: &NativeOnnxSessions,
    volume: &InputVolume,
    plane: InferencePlane,
    slice_index: usize,
) -> Result<PlaneForwardResult, String> {
    let trace_timing = native_trace_timing_enabled();
    let t0 = Instant::now();
    let session = match plane {
        InferencePlane::Coronal => &sessions.coronal,
        InferencePlane::Axial => &sessions.axial,
        InferencePlane::Sagittal => &sessions.sagittal,
    };

    let channel_count = session_channels_or_default(&session.input_shape);
    let t_pre = Instant::now();
    let prepared = prepare_plane_input_for_slice(volume, plane, channel_count, 1.0, slice_index)?;
    let pre_elapsed = t_pre.elapsed();

    let t_forward = Instant::now();
    let run = session.run(
        prepared.tensor_shape.as_slice(),
        prepared.tensor_data.as_slice(),
        prepared.scale_factor,
        plane.as_str(),
    )?;
    let forward_elapsed = t_forward.elapsed();

    let output_shapes = run
        .output_shapes
        .iter()
        .map(|shape| format!("{:?}", shape))
        .collect::<Vec<String>>()
        .join("; ");

    let summary = format!(
        "plane='{}' slice={} scale_factor=[{:.6}, {:.6}] outputs={} output_shapes={}",
        prepared.plane.as_str(),
        prepared.slice_index,
        prepared.scale_factor[0],
        prepared.scale_factor[1],
        run.output_shapes.len(),
        output_shapes
    );

    if trace_timing {
        eprintln!(
            "[trace][native-timing] plane={} slice={} prep_ms={} fwd_ms={} total_ms={}",
            plane.as_str(),
            slice_index,
            pre_elapsed.as_millis(),
            forward_elapsed.as_millis(),
            t0.elapsed().as_millis()
        );
    }

    Ok(PlaneForwardResult {
        summary,
        logits: run.logits,
        shape_chw: run.shape_chw,
        slice_index,
    })
}

fn sagittal_index_map_for_classes(num_classes: usize) -> Option<Vec<usize>> {
    let ranges_to_vec = |ranges: Vec<Vec<usize>>| ranges.into_iter().flatten().collect::<Vec<usize>>();

    match num_classes {
        96 => Some(ranges_to_vec(vec![
            vec![0],
            (5..14).collect(),
            (1..4).collect(),
            vec![14, 15, 4],
            (16..19).collect(),
            (5..51).collect(),
            (20..51).collect(),
        ])),
        51 => Some(ranges_to_vec(vec![
            vec![0],
            (5..14).collect(),
            (1..4).collect(),
            vec![14, 15, 4],
            (16..19).collect(),
            (5..51).collect(),
            vec![20, 22, 27],
            (29..32).collect(),
            vec![33, 34],
            (38..43).collect(),
            vec![45],
        ])),
        21 => Some(ranges_to_vec(vec![
            vec![0],
            (5..15).collect(),
            (1..4).collect(),
            vec![15, 16, 4],
            (17..20).collect(),
            (5..21).collect(),
        ])),
        _ => None,
    }
}

fn remap_sagittal_logits_to_full_space(
    logits: &[f32],
    shape_chw: [usize; 3],
) -> Result<(Vec<f32>, [usize; 3]), String> {
    let [classes, h, w] = shape_chw;
    let idx_map = sagittal_index_map_for_classes(classes)
        .ok_or_else(|| format!("Unsupported sagittal class count for remap: {classes}"))?;

    let expected_len = classes * h * w;
    if logits.len() != expected_len {
        return Err(format!(
            "Sagittal logits length mismatch before remap: got {}, expected {}",
            logits.len(),
            expected_len
        ));
    }

    let full_classes = idx_map.len();
    let mut remapped = vec![0f32; full_classes * h * w];
    let hw = h * w;

    for (full_class_idx, source_class_idx) in idx_map.iter().enumerate() {
        if *source_class_idx >= classes {
            return Err(format!(
                "Sagittal remap index out of range: source class {} >= {}",
                source_class_idx,
                classes
            ));
        }
        let src_offset = source_class_idx * hw;
        let dst_offset = full_class_idx * hw;
        remapped[dst_offset..(dst_offset + hw)]
            .copy_from_slice(&logits[src_offset..(src_offset + hw)]);
    }

    Ok((remapped, [full_classes, h, w]))
}

fn merge_plane_logits_into_volume(
    merged_logits: &mut [f32],
    plane_result: &PlaneForwardResult,
    plane: InferencePlane,
    volume_shape_xyz: [usize; 3],
    plane_weight: f32,
    expected_classes: usize,
) -> Result<(), String> {
    let [classes, h, w] = plane_result.shape_chw;
    let [expected_h, expected_w, expected_slices] = transformed_volume_shape(volume_shape_xyz, plane);

    if h != expected_h || w != expected_w {
        return Err(format!(
            "{} plane logits shape mismatch. expected hw=({expected_h},{expected_w}), got ({h},{w})",
            plane.as_str()
        ));
    }

    if plane_result.slice_index >= expected_slices {
        return Err(format!(
            "{} plane slice index {} out of range (slices={expected_slices})",
            plane.as_str(),
            plane_result.slice_index
        ));
    }

    let per_slice_len = classes * h * w;
    if plane_result.logits.len() != per_slice_len {
        return Err(format!(
            "{} plane logits length mismatch: got {}, expected {}",
            plane.as_str(),
            plane_result.logits.len(),
            per_slice_len
        ));
    }

    let [sx, sy, sz] = volume_shape_xyz;
    let voxels = sx * sy * sz;

    if classes != expected_classes {
        return Err(format!(
            "{} plane class mismatch after remap: expected {}, got {}",
            plane.as_str(),
            expected_classes,
            classes
        ));
    }

    for class_index in 0..classes {
        let class_offset = class_index * h * w;
        for ih in 0..h {
            for iw in 0..w {
                let slice_offset = class_offset + (ih * w) + iw;
                let (x, y, z) = oriented_to_xyz(plane, ih, iw, plane_result.slice_index);
                if x >= sx || y >= sy || z >= sz {
                    continue;
                }
                let voxel_offset = (x * sy * sz) + (y * sz) + z;
                let merged_offset = (class_index * voxels) + voxel_offset;
                merged_logits[merged_offset] += plane_weight * plane_result.logits[slice_offset];
            }
        }
    }

    Ok(())
}

fn merge_plane_logits_into_sparse_volume(
    merged_logits: &mut [f32],
    plane_result: &PlaneForwardResult,
    plane: InferencePlane,
    volume_shape_xyz: [usize; 3],
    plane_weight: f32,
    expected_classes: usize,
    sampled_offsets: &HashMap<usize, usize>,
) -> Result<(), String> {
    let [classes, h, w] = plane_result.shape_chw;
    let [expected_h, expected_w, expected_slices] = transformed_volume_shape(volume_shape_xyz, plane);

    if h != expected_h || w != expected_w {
        return Err(format!(
            "{} plane logits shape mismatch. expected hw=({expected_h},{expected_w}), got ({h},{w})",
            plane.as_str()
        ));
    }

    if plane_result.slice_index >= expected_slices {
        return Err(format!(
            "{} plane slice index {} out of range (slices={expected_slices})",
            plane.as_str(),
            plane_result.slice_index
        ));
    }

    let per_slice_len = classes * h * w;
    if plane_result.logits.len() != per_slice_len {
        return Err(format!(
            "{} plane logits length mismatch: got {}, expected {}",
            plane.as_str(),
            plane_result.logits.len(),
            per_slice_len
        ));
    }

    if classes != expected_classes {
        return Err(format!(
            "{} plane class mismatch after remap: expected {}, got {}",
            plane.as_str(),
            expected_classes,
            classes
        ));
    }

    let [sx, sy, sz] = volume_shape_xyz;
    let sampled_voxels = sampled_offsets.len();

    for class_index in 0..classes {
        let class_offset = class_index * h * w;
        for ih in 0..h {
            for iw in 0..w {
                let slice_offset = class_offset + (ih * w) + iw;
                let (x, y, z) = oriented_to_xyz(plane, ih, iw, plane_result.slice_index);
                if x >= sx || y >= sy || z >= sz {
                    continue;
                }
                let voxel_offset = (x * sy * sz) + (y * sz) + z;
                let Some(sampled_ix) = sampled_offsets.get(&voxel_offset) else {
                    continue;
                };
                let merged_offset = (class_index * sampled_voxels) + sampled_ix;
                merged_logits[merged_offset] += plane_weight * plane_result.logits[slice_offset];
            }
        }
    }

    Ok(())
}

fn argmax_labels_from_logits(
    merged_logits: &[f32],
    classes: usize,
    voxels: usize,
) -> Result<(Vec<u16>, Vec<usize>), String> {
    if merged_logits.len() != classes * voxels {
        return Err(format!(
            "Merged logits size mismatch: got {}, expected {}",
            merged_logits.len(),
            classes * voxels
        ));
    }

    let mut labels = vec![0u16; voxels];
    let mut class_hist = vec![0usize; classes];

    for voxel in 0..voxels {
        let mut best_class = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for class_index in 0..classes {
            let value = merged_logits[(class_index * voxels) + voxel];
            if value > best_value {
                best_value = value;
                best_class = class_index;
            }
        }
        labels[voxel] = best_class as u16;
        class_hist[best_class] += 1;
    }

    Ok((labels, class_hist))
}

fn create_native_output_dir() -> Result<PathBuf, String> {
    let epoch_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("System clock error while creating output directory: {error}"))?
        .as_nanos();

    let output_dir = std::env::temp_dir().join(format!("fastsurfer_ipc_out_{epoch_ns}"));
    fs::create_dir_all(&output_dir).map_err(|error| {
        format!(
            "Failed to create output directory '{}': {error}",
            output_dir.display()
        )
    })?;

    Ok(output_dir)
}

fn is_supported_nifti_path(path: &Path) -> bool {
    let lower = path.to_string_lossy().to_ascii_lowercase();
    lower.ends_with(".nii") || lower.ends_with(".nii.gz")
}

fn collect_nifti_files_recursive(dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_nifti_files_recursive(&path, out);
        } else if path.is_file() && is_supported_nifti_path(&path) {
            out.push(path.to_string_lossy().to_string());
        }
    }
}

fn discover_lut_path() -> Option<PathBuf> {
    if let Ok(explicit_path) = std::env::var("FASTSURFER_LUT_PATH") {
        let path = PathBuf::from(explicit_path);
        if path.exists() && path.is_file() {
            return Some(path);
        }
    }

    let mut candidates = Vec::<PathBuf>::new();
    if let Ok(repo_root) = std::env::var("FASTSURFER_REPO_ROOT") {
        candidates.push(PathBuf::from(repo_root).join("FastSurferCNN/config/FreeSurferColorLUT.txt"));
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("FastSurferCNN/config/FreeSurferColorLUT.txt"));
        candidates.push(cwd.join("..").join("..").join("..").join("..").join("FastSurferCNN/config/FreeSurferColorLUT.txt"));
    }

    candidates
        .into_iter()
        .find(|candidate| candidate.exists() && candidate.is_file())
}

fn load_lut_ids() -> Result<Vec<u16>, String> {
    let lut_path = discover_lut_path().ok_or_else(|| {
        "Could not discover FreeSurferColorLUT.txt. Set FASTSURFER_LUT_PATH or FASTSURFER_REPO_ROOT."
            .to_string()
    })?;

    let file = fs::File::open(&lut_path)
        .map_err(|error| format!("Failed to open LUT '{}': {error}", lut_path.display()))?;
    let reader = BufReader::new(file);

    let mut ids = Vec::<u16>::new();
    for line in reader.lines() {
        let line = line
            .map_err(|error| format!("Failed reading LUT '{}': {error}", lut_path.display()))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let Some(first) = parts.next() else {
            continue;
        };
        if let Ok(id) = first.parse::<u16>() {
            ids.push(id);
        }
    }

    if ids.is_empty() {
        return Err(format!(
            "No class IDs parsed from LUT '{}'.",
            lut_path.display()
        ));
    }

    Ok(ids)
}

fn map_label_indices_to_lut(label_indices: &[u16], lut_ids: &[u16]) -> Result<Vec<u16>, String> {
    if lut_ids.is_empty() {
        return Err("LUT IDs are empty; cannot map class indices to label space.".to_string());
    }

    let mut mapped = Vec::<u16>::with_capacity(label_indices.len());
    for index in label_indices {
        let idx = *index as usize;
        let label = lut_ids.get(idx).ok_or_else(|| {
            format!(
                "Predicted class index {} is out of LUT bounds (len={}).",
                idx,
                lut_ids.len()
            )
        })?;
        mapped.push(*label);
    }
    Ok(mapped)
}

fn write_pred_nifti(
    volume: &InputVolume,
    labels_xyz: Vec<u16>,
    output_path: &Path,
) -> Result<(), String> {
    let [sx, sy, sz] = volume.shape_xyz;
    let expected_len = sx * sy * sz;
    if labels_xyz.len() != expected_len {
        return Err(format!(
            "Label volume length mismatch: got {}, expected {}",
            labels_xyz.len(),
            expected_len
        ));
    }

    let array = Array3::from_shape_vec((sx, sy, sz), labels_xyz)
        .map_err(|error| format!("Failed to shape label array for NIfTI write: {error}"))?;

    WriterOptions::new(output_path)
        .reference_header(&volume.header)
        .write_nifti(&array)
        .map_err(|error| {
            format!(
                "Failed to write prediction NIfTI '{}': {error}",
                output_path.display()
            )
        })
}

fn write_u8_nifti(
    volume: &InputVolume,
    labels_xyz: Vec<u8>,
    output_path: &Path,
) -> Result<(), String> {
    let [sx, sy, sz] = volume.shape_xyz;
    let expected_len = sx * sy * sz;
    if labels_xyz.len() != expected_len {
        return Err(format!(
            "Label volume length mismatch: got {}, expected {}",
            labels_xyz.len(),
            expected_len
        ));
    }

    let array = Array3::from_shape_vec((sx, sy, sz), labels_xyz)
        .map_err(|error| format!("Failed to shape label array for NIfTI write: {error}"))?;

    WriterOptions::new(output_path)
        .reference_header(&volume.header)
        .write_nifti(&array)
        .map_err(|error| {
            format!(
                "Failed to write NIfTI '{}': {error}",
                output_path.display()
            )
        })
}

fn run_native_single_path<F>(
    sessions: &NativeOnnxSessions,
    lut_ids: &[u16],
    input_path: &str,
    mut on_progress: F,
) -> Result<NativeSingleResult, String>
where
    F: FnMut(u8, String),
{
    on_progress(2, "Loading input volume...".to_string());
    let volume = load_input_volume(input_path)?;

    let [sx, sy, sz] = volume.shape_xyz;
    let voxels = sx * sy * sz;

    let [_, _, coronal_slices] = transformed_volume_shape(volume.shape_xyz, InferencePlane::Coronal);
    let [_, _, axial_slices] = transformed_volume_shape(volume.shape_xyz, InferencePlane::Axial);
    let [_, _, sagittal_slices] = transformed_volume_shape(volume.shape_xyz, InferencePlane::Sagittal);
    let coronal_indices = native_slice_indices_for_plane(coronal_slices);
    let axial_indices = native_slice_indices_for_plane(axial_slices);
    let sagittal_indices = native_slice_indices_for_plane(sagittal_slices);
    let total_slices = coronal_indices.len() + axial_indices.len() + sagittal_indices.len();

    if total_slices == 0 {
        return Err("Input volume has zero slices after orientation transforms".to_string());
    }

    let coronal_channels = session_channels_or_default(&sessions.coronal.input_shape);
    let axial_channels = session_channels_or_default(&sessions.axial.input_shape);
    let sagittal_channels = session_channels_or_default(&sessions.sagittal.input_shape);

    let class_count = sessions
        .coronal
        .output_shapes
        .first()
        .and_then(|shape| shape.as_ref())
        .and_then(|shape| shape.get(1).copied())
        .unwrap_or(79usize);

    let sparse_mode = parse_native_slices_per_plane().is_some();

    let mut sampled_order = Vec::<usize>::new();
    let mut sampled_lookup = HashMap::<usize, usize>::new();
    if sparse_mode {
        for (plane, indices) in [
            (InferencePlane::Coronal, &coronal_indices),
            (InferencePlane::Sagittal, &sagittal_indices),
            (InferencePlane::Axial, &axial_indices),
        ] {
            let [h, w, _] = transformed_volume_shape(volume.shape_xyz, plane);
            for &slice_index in indices {
                for ih in 0..h {
                    for iw in 0..w {
                        let (x, y, z) = oriented_to_xyz(plane, ih, iw, slice_index);
                        let voxel_offset = (x * sy * sz) + (y * sz) + z;
                        if let std::collections::hash_map::Entry::Vacant(entry) = sampled_lookup.entry(voxel_offset) {
                            let ix = sampled_order.len();
                            sampled_order.push(voxel_offset);
                            entry.insert(ix);
                        }
                    }
                }
            }
        }
    }

    let sampled_voxels = if sparse_mode {
        sampled_lookup.len().max(1)
    } else {
        voxels
    };
    let mut merged_logits = vec![0f32; class_count * sampled_voxels];
    let mut processed_slices = 0usize;

    let mut plane_first_summaries: Vec<String> = Vec::new();

    let plane_specs = [
        (InferencePlane::Coronal, coronal_indices, coronal_channels, 0.4f32),
        (InferencePlane::Sagittal, sagittal_indices, sagittal_channels, 0.2f32),
        (InferencePlane::Axial, axial_indices, axial_channels, 0.4f32),
    ];

    let use_sparse_parallel = sparse_mode && native_sparse_parallel_enabled();

    if use_sparse_parallel {
        on_progress(5, "Running sparse per-plane ONNX forward passes...".to_string());

        let mut plane_results = Vec::<(InferencePlane, f32, Vec<PlaneForwardResult>)>::new();
        let sessions_ref = sessions;
        let volume_ref = &volume;
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for (plane, slice_indices, _channels, weight) in &plane_specs {
                if slice_indices.is_empty() {
                    continue;
                }
                let slice_indices = slice_indices.clone();
                let plane_value = *plane;
                let weight_value = *weight;
                handles.push(scope.spawn(move || {
                    let mut collected = Vec::<PlaneForwardResult>::with_capacity(slice_indices.len());
                    for slice_index in slice_indices {
                        let plane_result = run_single_plane_forward_at_slice(
                            sessions_ref,
                            volume_ref,
                            plane_value,
                            slice_index,
                        )?;

                        let plane_result = if plane_value == InferencePlane::Sagittal {
                            let (logits, shape_chw) = remap_sagittal_logits_to_full_space(
                                &plane_result.logits,
                                plane_result.shape_chw,
                            )?;
                            PlaneForwardResult {
                                summary: plane_result.summary,
                                logits,
                                shape_chw,
                                slice_index: plane_result.slice_index,
                            }
                        } else {
                            plane_result
                        };
                        collected.push(plane_result);
                    }

                    Ok::<(InferencePlane, f32, Vec<PlaneForwardResult>), String>((
                        plane_value,
                        weight_value,
                        collected,
                    ))
                }));
            }

            for handle in handles {
                plane_results.push(handle.join().map_err(|_| {
                    "native sparse plane worker panicked before returning result".to_string()
                })??);
            }

            Ok::<(), String>(())
        })?;

        for (plane, weight, results_for_plane) in plane_results {
            if let Some(first) = results_for_plane.first() {
                plane_first_summaries.push(first.summary.clone());
            }

            let slice_count = results_for_plane.len();
            for (position, plane_result) in results_for_plane.iter().enumerate() {
                merge_plane_logits_into_sparse_volume(
                    &mut merged_logits,
                    plane_result,
                    plane,
                    volume.shape_xyz,
                    weight,
                    class_count,
                    &sampled_lookup,
                )?;

                processed_slices += 1;
                if position == 0 || position + 1 == slice_count || (position + 1) % 16 == 0 {
                    let file_progress = 5 + ((processed_slices * 85) / total_slices).min(85);
                    on_progress(
                        file_progress as u8,
                        format!(
                            "Aggregating {} plane slices: {}/{}",
                            plane.as_str(),
                            position + 1,
                            slice_count
                        ),
                    );
                }
            }
        }
    } else {
        for (plane, slice_indices, _channels, weight) in plane_specs {
            if slice_indices.is_empty() {
                continue;
            }

            on_progress(
                5,
                format!("Running {} plane ONNX forward passes...", plane.as_str()),
            );

            let slice_count = slice_indices.len();
            for (position, slice_index) in slice_indices.into_iter().enumerate() {
                let plane_result = run_single_plane_forward_at_slice(sessions, &volume, plane, slice_index)?;

                let plane_result = if plane == InferencePlane::Sagittal {
                    let (logits, shape_chw) = remap_sagittal_logits_to_full_space(
                        &plane_result.logits,
                        plane_result.shape_chw,
                    )?;
                    PlaneForwardResult {
                        summary: plane_result.summary,
                        logits,
                        shape_chw,
                        slice_index: plane_result.slice_index,
                    }
                } else {
                    plane_result
                };

                if position == 0 {
                    plane_first_summaries.push(plane_result.summary.clone());
                }

                if sparse_mode {
                    merge_plane_logits_into_sparse_volume(
                        &mut merged_logits,
                        &plane_result,
                        plane,
                        volume.shape_xyz,
                        weight,
                        class_count,
                        &sampled_lookup,
                    )?;
                } else {
                    merge_plane_logits_into_volume(
                        &mut merged_logits,
                        &plane_result,
                        plane,
                        volume.shape_xyz,
                        weight,
                        class_count,
                    )?;
                }

                processed_slices += 1;
                if position == 0 || position + 1 == slice_count || (position + 1) % 16 == 0 {
                    let file_progress = 5 + ((processed_slices * 85) / total_slices).min(85);
                    on_progress(
                        file_progress as u8,
                        format!(
                            "Aggregating {} plane slices: {}/{}",
                            plane.as_str(),
                            position + 1,
                            slice_count
                        ),
                    );
                }
            }
        }
    }

    on_progress(92, "Converting logits to label volume...".to_string());
    let (label_indices_xyz, class_hist) = if sparse_mode {
        let (sampled_labels, mut sampled_hist) =
            argmax_labels_from_logits(&merged_logits, class_count, sampled_voxels)?;
        let mut full_labels = vec![0u16; voxels];
        for (sample_ix, voxel_offset) in sampled_order.iter().copied().enumerate() {
            full_labels[voxel_offset] = sampled_labels[sample_ix];
        }
        if sampled_hist.len() < class_count {
            sampled_hist.resize(class_count, 0);
        }
        (full_labels, sampled_hist)
    } else {
        argmax_labels_from_logits(&merged_logits, class_count, voxels)?
    };
    let mut pred_labels_xyz = map_label_indices_to_lut(&label_indices_xyz, lut_ids)?;
    split_cortex_labels(&mut pred_labels_xyz, volume.shape_xyz);

    let mut ranked = class_hist
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(class_id, count)| (class_id, *count))
        .collect::<Vec<(usize, usize)>>();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));

    let top_classes = ranked
        .iter()
        .take(5)
        .map(|(class_id, count)| format!("{}:{}", class_id, count))
        .collect::<Vec<String>>()
        .join(", ");

    on_progress(95, "Writing native prediction outputs...".to_string());

    let output_dir = create_native_output_dir()?;
    let pred_path = output_dir.join("pred.nii.gz");
    write_pred_nifti(&volume, pred_labels_xyz.clone(), &pred_path)?;

    let mut aseg_labels = derive_aseg_from_pred(&pred_labels_xyz);
    let brainmask = derive_brainmask_from_pred(&pred_labels_xyz, volume.shape_xyz);
    mask_aseg_with_brainmask(&mut aseg_labels, &brainmask);
    flip_wm_islands(&mut aseg_labels, volume.shape_xyz);

    let aseg_path = output_dir.join("aseg.nii.gz");
    write_pred_nifti(&volume, aseg_labels.clone(), &aseg_path)?;

    let brainmask_path = output_dir.join("brainmask.nii.gz");
    write_u8_nifti(&volume, brainmask, &brainmask_path)?;

    let voxvol_mm3 = f64::from(volume.zoom_xyz[0])
        * f64::from(volume.zoom_xyz[1])
        * f64::from(volume.zoom_xyz[2]);
    let qc = evaluate_qc(&pred_labels_xyz, volume.shape_xyz, voxvol_mm3)?;

    on_progress(100, "Rust ONNX inference completed.".to_string());

    if native_trace_timing_enabled() {
        let probe = sessions.run_dummy_probe()?;
        eprintln!(
            "[trace][native-inference] loaded ONNX sessions from {} ({}) | probe={} | preproc={} | input='{}' | {} | classes={} voxels={} top_classes=[{}] output='{}'",
            sessions.registry.source_dir,
            discovery_summary(sessions),
            probe.join(" | "),
            legacy_preprocessing_summary(sessions),
            input_path,
            plane_first_summaries.join(" | "),
            class_count,
            voxels,
            top_classes,
            pred_path.display()
        );
    }

    let output_path_str = pred_path.to_string_lossy().to_string();
    let output_filename = pred_path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "pred.nii.gz".to_string());

    let qc_message = qc
        .message
        .clone()
        .unwrap_or_else(|| "qc=unavailable".to_string());

    Ok(NativeSingleResult {
        prediction: InferenceOutput {
            input_path: input_path.to_string(),
            output_path: output_path_str,
            output_filename,
            run_result: "rust-onnx-native".to_string(),
            artifacts: Some(InferenceArtifacts {
                brainmask_path: Some(brainmask_path.to_string_lossy().to_string()),
                aseg_path: Some(aseg_path.to_string_lossy().to_string()),
            }),
            qc: Some(InferenceQc {
                passed: qc.passed,
                message: qc.message,
            }),
        },
        result_directory: output_dir.to_string_lossy().to_string(),
        qc_summary: format!(
            "Native logits fusion completed (weights coronal=0.4 axial=0.4 sagittal=0.2, classes={class_count}, top_classes=[{top_classes}]) | {qc_message}"
        ),
    })
}

fn validate_inputs(file_paths: &[String], folder_paths: &[String]) -> Result<Vec<String>, String> {
    let mut resolved = Vec::<String>::new();

    for file in file_paths {
        let path = PathBuf::from(file);
        if path.exists() && path.is_file() && is_supported_nifti_path(&path) {
            resolved.push(path.to_string_lossy().to_string());
        }
    }

    for folder in folder_paths {
        let path = PathBuf::from(folder);
        if path.exists() && path.is_dir() {
            collect_nifti_files_recursive(&path, &mut resolved);
        }
    }

    if resolved.is_empty() {
        return Err(
            "Rust native inference requires at least one valid NIfTI path (.nii/.nii.gz) from files or folders."
                .to_string(),
        );
    }

    resolved.sort();
    resolved.dedup();

    Ok(resolved)
}

pub(crate) fn run_native_inference(
    file_paths: &[String],
    folder_paths: &[String],
) -> Result<ProcessingRunResult, String> {
    let requested_paths = validate_inputs(file_paths, folder_paths)?;
    let sessions = NativeOnnxSessions::load_default()?;
    let lut_ids = load_lut_ids()?;

    let mut results = Vec::with_capacity(requested_paths.len());
    let mut result_directories = BTreeSet::new();
    let mut qc_summaries = Vec::new();

    for input_path in &requested_paths {
        let single = run_native_single_path(&sessions, &lut_ids, input_path, |_progress, _message| {})?;
        result_directories.insert(single.result_directory);
        qc_summaries.push(single.qc_summary);
        results.push(single.prediction);
    }

    let result_directories = result_directories.into_iter().collect::<Vec<String>>();
    let ack_message = if result_directories.is_empty() {
        format!("Processing started for {} path(s).", requested_paths.len())
    } else {
        format!(
            "Processing started for {} path(s). Results directory: {}",
            requested_paths.len(),
            result_directories.join(", ")
        )
    };

    Ok(ProcessingRunResult {
        ack_message,
        requested_paths,
        result_directories,
        qc_summary: Some(qc_summaries.join(" | ")),
        results,
    })
}

pub(crate) fn run_native_inference_with_progress(
    app_handle: &AppHandle,
    task_id: &str,
    file_paths: &[String],
    folder_paths: &[String],
) -> Result<ProcessingRunResult, String> {
    let requested_paths = match validate_inputs(file_paths, folder_paths) {
        Ok(paths) => paths,
        Err(error) => {
            let _ = app_handle.emit(
                "fastsurfer://inference-progress",
                InferenceProgressEvent {
                    task_id: task_id.to_string(),
                    status: "failed".to_string(),
                    message: error.clone(),
                    total: 0,
                    completed: 0,
                    progress: 0,
                    current_path: None,
                    output_path: None,
                },
            );
            return Err(error);
        }
    };

    let total = requested_paths.len();

    let _ = app_handle.emit(
        "fastsurfer://inference-progress",
        InferenceProgressEvent {
            task_id: task_id.to_string(),
            status: "started".to_string(),
            message: "Rust ONNX mode selected. Initializing ONNX sessions...".to_string(),
            total,
            completed: 0,
            progress: 0,
            current_path: None,
            output_path: None,
        },
    );

    let sessions = match NativeOnnxSessions::load_default() {
        Ok(sessions) => sessions,
        Err(error) => {
            let _ = app_handle.emit(
                "fastsurfer://inference-progress",
                InferenceProgressEvent {
                    task_id: task_id.to_string(),
                    status: "failed".to_string(),
                    message: error.clone(),
                    total,
                    completed: 0,
                    progress: 0,
                    current_path: None,
                    output_path: None,
                },
            );
            return Err(error);
        }
    };

    let lut_ids = match load_lut_ids() {
        Ok(ids) => ids,
        Err(error) => {
            let _ = app_handle.emit(
                "fastsurfer://inference-progress",
                InferenceProgressEvent {
                    task_id: task_id.to_string(),
                    status: "failed".to_string(),
                    message: error.clone(),
                    total,
                    completed: 0,
                    progress: 0,
                    current_path: None,
                    output_path: None,
                },
            );
            return Err(error);
        }
    };

    let mut results = Vec::with_capacity(total);
    let mut result_directories = BTreeSet::new();
    let mut qc_summaries = Vec::new();

    for (index, input_path) in requested_paths.iter().enumerate() {
        let single = run_native_single_path(&sessions, &lut_ids, input_path, |file_progress, file_message| {
            let overall_progress = if total == 0 {
                file_progress
            } else {
                ((((index * 100) + file_progress as usize) / total).min(99)) as u8
            };

            let _ = app_handle.emit(
                "fastsurfer://inference-progress",
                InferenceProgressEvent {
                    task_id: task_id.to_string(),
                    status: "item_progress".to_string(),
                    message: file_message,
                    total,
                    completed: index,
                    progress: overall_progress,
                    current_path: Some(input_path.clone()),
                    output_path: None,
                },
            );
        });

        match single {
            Ok(single) => {
                let completed = index + 1;
                let progress = if total == 0 {
                    100
                } else {
                    (((completed * 100) / total).min(100)) as u8
                };

                let output_path = single.prediction.output_path.clone();
                result_directories.insert(single.result_directory);
                qc_summaries.push(single.qc_summary);
                results.push(single.prediction);

                let _ = app_handle.emit(
                    "fastsurfer://inference-progress",
                    InferenceProgressEvent {
                        task_id: task_id.to_string(),
                        status: "item_completed".to_string(),
                        message: format!("Processed {completed}/{total}"),
                        total,
                        completed,
                        progress,
                        current_path: Some(input_path.clone()),
                        output_path: Some(output_path),
                    },
                );
            }
            Err(error) => {
                let progress = if total == 0 {
                    0
                } else {
                    (((index * 100) / total).min(100)) as u8
                };

                let _ = app_handle.emit(
                    "fastsurfer://inference-progress",
                    InferenceProgressEvent {
                        task_id: task_id.to_string(),
                        status: "failed".to_string(),
                        message: error.clone(),
                        total,
                        completed: index,
                        progress,
                        current_path: Some(input_path.clone()),
                        output_path: None,
                    },
                );

                return Err(error);
            }
        }
    }

    let result_directories = result_directories.into_iter().collect::<Vec<String>>();
    let ack_message = if result_directories.is_empty() {
        format!("Processing started for {} path(s).", requested_paths.len())
    } else {
        format!(
            "Processing started for {} path(s). Results directory: {}",
            requested_paths.len(),
            result_directories.join(", ")
        )
    };

    let _ = app_handle.emit(
        "fastsurfer://inference-progress",
        InferenceProgressEvent {
            task_id: task_id.to_string(),
            status: "completed".to_string(),
            message: ack_message.clone(),
            total,
            completed: total,
            progress: 100,
            current_path: None,
            output_path: None,
        },
    );

    Ok(ProcessingRunResult {
        ack_message,
        requested_paths,
        result_directories,
        qc_summary: Some(qc_summaries.join(" | ")),
        results,
    })
}
