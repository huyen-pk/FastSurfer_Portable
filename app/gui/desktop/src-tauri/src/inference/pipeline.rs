use crate::inference::onnx_loader_candle as candle_loader;
use crate::inference::onnx_loader_ort as ort_loader;
use crate::inference::postprocess;
use crate::inference::preprocess;
use crate::inference::qc::evaluate_qc;
use crate::models::{
    InferenceArtifacts, InferenceOutput, InferencePlane,
    InferenceProgressEvent, InferenceQc, InputVolume, ProcessingRunResult,
};
use ndarray::Array3;
use nifti::writer::WriterOptions;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tauri::Emitter;
use tokio::sync::mpsc;

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

const NATIVE_CANCELLED_MESSAGE: &str = "Task cancelled by user.";

fn cancelled_error() -> String {
    NATIVE_CANCELLED_MESSAGE.to_string()
}

fn is_task_cancelled(
    cancelled_tasks: &Arc<Mutex<BTreeSet<String>>>,
    task_id: &str,
) -> Result<bool, String> {
    cancelled_tasks
        .lock()
        .map_err(|_| "Cancelled tasks mutex was poisoned".to_string())
        .map(|set| set.contains(task_id))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeOnnxRuntime {
    Candle,
    Ort,
}

#[derive(Clone)]
enum NativeOnnxSessions {
    Candle(candle_loader::NativeOnnxSessions),
    Ort(ort_loader::NativeOnnxSessions),
}

enum PlaneSessionRef<'a> {
    Candle(&'a candle_loader::PlaneSession),
    Ort(&'a ort_loader::PlaneSession),
}

struct PlaneRun {
    logits: Vec<f32>,
    shape_chw: [usize; 3],
    output_shapes: Vec<Vec<usize>>,
}

fn resolve_native_onnx_runtime() -> NativeOnnxRuntime {
    let raw = std::env::var("FASTSURFER_NATIVE_RUNTIME")
        .unwrap_or_else(|_| "candle".to_string())
        .trim()
        .to_ascii_lowercase();

    match raw.as_str() {
        "ort" | "onnxruntime" | "onnx-runtime" => NativeOnnxRuntime::Ort,
        _ => NativeOnnxRuntime::Candle,
    }
}

impl NativeOnnxRuntime {
    fn as_str(self) -> &'static str {
        match self {
            Self::Candle => "candle",
            Self::Ort => "ort",
        }
    }
}

impl NativeOnnxSessions {
    fn runtime(&self) -> NativeOnnxRuntime {
        match self {
            Self::Candle(_) => NativeOnnxRuntime::Candle,
            Self::Ort(_) => NativeOnnxRuntime::Ort,
        }
    }

    fn load_default() -> Result<Self, String> {
        match resolve_native_onnx_runtime() {
            NativeOnnxRuntime::Candle => {
                candle_loader::NativeOnnxSessions::load_default()
                    .map(Self::Candle)
            }
            NativeOnnxRuntime::Ort => {
                ort_loader::NativeOnnxSessions::load_default().map(Self::Ort)
            }
        }
    }

    fn source_dir(&self) -> &str {
        match self {
            Self::Candle(models) => &models.registry.source_dir,
            Self::Ort(models) => &models.registry.source_dir,
        }
    }

    fn model_paths_summary(&self) -> String {
        match self {
            Self::Candle(models) => format!(
                "axial='{}', coronal='{}', sagittal='{}'",
                models.registry.axial_model_path,
                models.registry.coronal_model_path,
                models.registry.sagittal_model_path
            ),
            Self::Ort(models) => format!(
                "axial='{}', coronal='{}', sagittal='{}'",
                models.registry.axial_model_path,
                models.registry.coronal_model_path,
                models.registry.sagittal_model_path
            ),
        }
    }

    fn run_dummy_probe(&self) -> Result<Vec<String>, String> {
        match self {
            Self::Candle(models) => models.run_dummy_probe(),
            Self::Ort(models) => models.run_dummy_probe(),
        }
    }

    fn plane_session(&self, plane: InferencePlane) -> PlaneSessionRef<'_> {
        match (self, plane) {
            (Self::Candle(models), InferencePlane::Coronal) => {
                PlaneSessionRef::Candle(&models.coronal)
            }
            (Self::Candle(models), InferencePlane::Axial) => {
                PlaneSessionRef::Candle(&models.axial)
            }
            (Self::Candle(models), InferencePlane::Sagittal) => {
                PlaneSessionRef::Candle(&models.sagittal)
            }
            (Self::Ort(models), InferencePlane::Coronal) => {
                PlaneSessionRef::Ort(&models.coronal)
            }
            (Self::Ort(models), InferencePlane::Axial) => {
                PlaneSessionRef::Ort(&models.axial)
            }
            (Self::Ort(models), InferencePlane::Sagittal) => {
                PlaneSessionRef::Ort(&models.sagittal)
            }
        }
    }

    fn class_count_or_default(&self) -> usize {
        let output_shapes = match self {
            Self::Candle(models) => &models.coronal.output_shapes,
            Self::Ort(models) => &models.coronal.output_shapes,
        };

        output_shapes
            .first()
            .and_then(std::option::Option::as_deref)
            .and_then(|dims| dims.get(1).copied())
            .unwrap_or(79usize)
    }
}

impl PlaneSessionRef<'_> {
    fn input_shape(&self) -> Option<&[usize]> {
        match self {
            Self::Candle(session) => session.input_shape.as_deref(),
            Self::Ort(session) => session.input_shape.as_deref(),
        }
    }

    fn run(
        &self,
        input_shape: &[usize],
        input_data: &[f32],
        scale_factor: [f32; 2],
        plane: &str,
    ) -> Result<PlaneRun, String> {
        match self {
            Self::Candle(session) => {
                let run = session.run(
                    input_shape,
                    input_data,
                    scale_factor,
                    plane,
                )?;
                Ok(PlaneRun {
                    logits: run.logits,
                    shape_chw: run.shape_chw,
                    output_shapes: run.output_shapes,
                })
            }
            Self::Ort(session) => {
                let run = session.run(
                    input_shape,
                    input_data,
                    scale_factor,
                    plane,
                )?;
                Ok(PlaneRun {
                    logits: run.logits,
                    shape_chw: run.shape_chw,
                    output_shapes: run.output_shapes,
                })
            }
        }
    }
}

fn discovery_summary(models: &NativeOnnxSessions) -> String {
    models.model_paths_summary()
}

fn session_channels_or_default(input_shape: Option<&[usize]>) -> usize {
    input_shape
        .and_then(|dims| dims.get(1).copied())
        .filter(|channels| *channels > 0)
        .unwrap_or(7)
}

fn native_trace_timing_enabled() -> bool {
    std::env::var("FASTSURFER_NATIVE_TRACE_TIMING")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            normalized == "1"
                || normalized == "true"
                || normalized == "yes"
                || normalized == "on"
        })
        .unwrap_or(false)
}

fn parse_native_slices_per_plane() -> Option<usize> {
    std::env::var("FASTSURFER_NATIVE_SLICES_PER_PLANE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn native_parallel_enabled() -> bool {
    std::env::var("FASTSURFER_NATIVE_PARALLEL")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !(normalized == "0"
                || normalized == "false"
                || normalized == "no"
                || normalized == "off")
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

fn build_sampled_lookup(
    volume: &InputVolume,
    coronal_indices: &[usize],
    sagittal_indices: &[usize],
    axial_indices: &[usize],
) -> (Vec<usize>, HashMap<usize, usize>) {
    let mut sampled_order = Vec::<usize>::new();
    let mut sampled_lookup = HashMap::<usize, usize>::new();

    let [_, sy, sz] = volume.shape_xyz;

    for (plane, indices) in [
        (InferencePlane::Coronal, coronal_indices),
        (InferencePlane::Sagittal, sagittal_indices),
        (InferencePlane::Axial, axial_indices),
    ] {
        let [height, width, _slice_count] =
            preprocess::transformed_volume_shape(volume.shape_xyz, plane);
        for &slice_index in indices {
            for ih in 0..height {
                for iw in 0..width {
                    let (x, y, z) =
                        preprocess::oriented_to_xyz(plane, ih, iw, slice_index);
                    let voxel_offset = (x * sy * sz) + (y * sz) + z;
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        sampled_lookup.entry(voxel_offset)
                    {
                        let ix = sampled_order.len();
                        sampled_order.push(voxel_offset);
                        entry.insert(ix);
                    }
                }
            }
        }
    }

    (sampled_order, sampled_lookup)
}

struct FinalizeNativeResultInput<'a> {
    volume: &'a InputVolume,
    label_indices_xyz: &'a [u16],
    class_hist: &'a [usize],
    plane_first_summaries: &'a [String],
    sessions: &'a NativeOnnxSessions,
    lut_ids: &'a [u16],
    input_path: &'a str,
}

fn finalize_native_result(
    input: &FinalizeNativeResultInput<'_>,
    on_progress: &mut dyn FnMut(u8, String),
) -> Result<NativeSingleResult, String> {
    let FinalizeNativeResultInput {
        volume,
        label_indices_xyz,
        class_hist,
        plane_first_summaries,
        sessions,
        lut_ids,
        input_path,
    } = *input;

    on_progress(92, "Converting logits to label volume...".to_string());

    let mut pred_labels_xyz =
        map_label_indices_to_lut(label_indices_xyz, lut_ids)?;
    postprocess::split_cortex_labels(&mut pred_labels_xyz, volume.shape_xyz);

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
        .map(|(class_id, count)| format!("{class_id}:{count}"))
        .collect::<Vec<String>>()
        .join(", ");

    on_progress(95, "Writing native prediction outputs...".to_string());

    let (output_dir, pred_path, aseg_path, brainmask_path) =
        write_prediction_artifacts(volume, &pred_labels_xyz)?;

    let [sx, sy, sz] = volume.shape_xyz;
    let voxvol_mm3 = f64::from(volume.zoom_xyz[0])
        * f64::from(volume.zoom_xyz[1])
        * f64::from(volume.zoom_xyz[2]);
    let qc = evaluate_qc(&pred_labels_xyz, volume.shape_xyz, voxvol_mm3)?;

    on_progress(100, "Rust ONNX inference completed.".to_string());

    if native_trace_timing_enabled() {
        let probe = sessions.run_dummy_probe()?;
        eprintln!(
            "[trace][native-inference] runtime={} loaded ONNX sessions from {} ({}) | probe={} | preproc={} | input='{}' | {} | classes={} voxels={} top_classes=[{}] output='{}'",
            sessions.runtime().as_str(),
            sessions.source_dir(),
            discovery_summary(sessions),
            probe.join(" | "),
            legacy_preprocessing_summary(sessions),
            input_path,
            plane_first_summaries.join(" | "),
            class_hist.len().saturating_sub(0),
            sx * sy * sz,
            top_classes,
            pred_path.display()
        );
    }

    let output_path_str = pred_path.to_string_lossy().to_string();
    let output_filename = pred_path.file_name().map_or_else(
        || "pred.nii.gz".to_string(),
        |name| name.to_string_lossy().to_string(),
    );

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
                brainmask_path: Some(
                    brainmask_path.to_string_lossy().to_string(),
                ),
                aseg_path: Some(aseg_path.to_string_lossy().to_string()),
            }),
            qc: Some(InferenceQc {
                passed: qc.passed,
                message: qc.message,
            }),
        },
        result_directory: output_dir.to_string_lossy().to_string(),
        qc_summary: format!(
            "Native logits fusion completed (weights coronal=0.4 axial=0.4 sagittal=0.2, classes={} , top_classes=[{}]) | {}",
            class_hist.len().saturating_sub(0),
            top_classes,
            qc_message
        ),
    })
}

fn legacy_preprocessing_summary(models: &NativeOnnxSessions) -> String {
    let base_res = 1.0f32;
    let synthetic_zoom = [1.0f32, 1.0f32, 1.0f32];

    let axial_channels = session_channels_or_default(
        models.plane_session(InferencePlane::Axial).input_shape(),
    );
    let coronal_channels = session_channels_or_default(
        models.plane_session(InferencePlane::Coronal).input_shape(),
    );
    let sagittal_channels = session_channels_or_default(
        models.plane_session(InferencePlane::Sagittal).input_shape(),
    );

    let axial = preprocess::build_legacy_preprocessing_trace(
        InferencePlane::Axial,
        axial_channels,
        base_res,
        synthetic_zoom,
    );
    let coronal = preprocess::build_legacy_preprocessing_trace(
        InferencePlane::Coronal,
        coronal_channels,
        base_res,
        synthetic_zoom,
    );
    let sagittal = preprocess::build_legacy_preprocessing_trace(
        InferencePlane::Sagittal,
        sagittal_channels,
        base_res,
        synthetic_zoom,
    );

    format!("{axial} || {coronal} || {sagittal}")
}

struct ForwardPassRequest<'a> {
    sessions: &'a NativeOnnxSessions,
    volume: &'a InputVolume,
    plane_specs: &'a [(InferencePlane, &'a [usize], usize, f32)],
}

fn execute_forward_pass(
    rt: &tokio::runtime::Handle,
    request: &ForwardPassRequest<'_>,
    ctx: &mut ForwardPassContext<'_>,
) -> Result<(), String> {
    let ForwardPassRequest {
        sessions,
        volume,
        plane_specs,
    } = *request;

    // Decide parallel vs sequential based on runtime flag.
    // Note: `sparse_mode` (sampling slices) is orthogonal and may be
    // executed in either parallel or sequential mode depending on the
    // `FASTSURFER_NATIVE_PARALLEL` environment setting.
    let parallel_enabled = native_parallel_enabled();
    if parallel_enabled {
        execute_forward_pass_parallel(rt, sessions, volume, plane_specs, ctx)
    } else {
        execute_forward_pass_sequential(rt, sessions, volume, plane_specs, ctx)
    }
}

struct ForwardPassContext<'a> {
    sparse_mode: bool,
    sampled_lookup: &'a HashMap<usize, usize>,
    merged_logits: &'a mut [f32],
    should_cancel: &'a dyn Fn() -> bool,
    on_progress: &'a mut dyn FnMut(u8, String),
    plane_first_summaries: &'a mut Vec<String>,
}

fn execute_forward_pass_parallel(
    rt: &tokio::runtime::Handle,
    sessions: &NativeOnnxSessions,
    volume: &InputVolume,
    plane_specs: &[(InferencePlane, &[usize], usize, f32)],
    ctx: &mut ForwardPassContext,
) -> Result<(), String> {
    if (ctx.should_cancel)() {
        return Err(cancelled_error());
    }

    (ctx.on_progress)(5, "Initializing parallel ONNX inference...".to_string());

    let (tx, mut rx) = mpsc::unbounded_channel::<
        Result<(InferencePlane, f32, PlaneForwardResult), String>,
    >();

    let sessions_arc = Arc::new(sessions.clone());
    let volume_arc = Arc::new(volume.clone());

    let mut handles = Vec::new();

    // Materialize an owned job list so worker closures do not borrow
    // `plane_specs` (which would prevent spawning 'static tasks).
    let jobs: Vec<(InferencePlane, Vec<usize>, usize, f32)> = plane_specs
        .iter()
        .copied()
        .map(|(p, s, ch, w)| (p, s.to_owned(), ch, w))
        .collect();

    for (plane_value, slice_indices_owned, _channels_value, weight_value) in
        jobs
    {
        if slice_indices_owned.is_empty() {
            continue;
        }

        if (ctx.should_cancel)() {
            return Err(cancelled_error());
        }

        let tx_clone = tx.clone();
        let sessions_worker = sessions_arc.clone();
        let volume_worker = volume_arc.clone();

        handles.push(rt.spawn_blocking(move || {
            for &slice_index in &slice_indices_owned {
                let mut plane_result = run_single_plane_forward_at_slice(
                    &sessions_worker,
                    &volume_worker,
                    plane_value,
                    slice_index,
                )?;
                plane_result = remap_if_sagittal(plane_value, plane_result)?;

                if tx_clone
                    .send(Ok((plane_value, weight_value, plane_result)))
                    .is_err()
                {
                    break;
                }
            }
            Ok::<(), String>(())
        }));
    }

    drop(tx);

    while let Some(msg) = rt.block_on(rx.recv()) {
        if (ctx.should_cancel)() {
            return Err(cancelled_error());
        }
        let (plane, weight, plane_result) = msg?;
        let mut pr_ctx = PlaneResultContext {
            sampled_lookup: ctx.sampled_lookup,
            volume_shape_xyz: volume.shape_xyz,
            class_count: sessions.class_count_or_default(),
            plane_first_summaries: ctx.plane_first_summaries,
            on_progress: ctx.on_progress,
        };
        handle_plane_result(
            ctx.merged_logits,
            &plane_result,
            plane,
            weight,
            ctx.sparse_mode,
            &mut pr_ctx,
        )?;
    }

    for handle in handles {
        rt.block_on(handle).map_err(|_| "worker task panicked")??;
    }

    Ok(())
}

fn execute_forward_pass_sequential(
    _rt: &tokio::runtime::Handle,
    sessions: &NativeOnnxSessions,
    volume: &InputVolume,
    plane_specs: &[(InferencePlane, &[usize], usize, f32)],
    ctx: &mut ForwardPassContext,
) -> Result<(), String> {
    for &(plane, slice_indices, _channels, weight) in plane_specs {
        (ctx.on_progress)(
            5,
            format!("Running {} plane ONNX forward passes...", plane.as_str()),
        );

        for (position, slice_index) in slice_indices.iter().copied().enumerate()
        {
            if (ctx.should_cancel)() {
                return Err(cancelled_error());
            }

            let mut plane_result = run_single_plane_forward_at_slice(
                sessions,
                volume,
                plane,
                slice_index,
            )?;

            plane_result = remap_if_sagittal(plane, plane_result)?;

            if position == 0 {
                ctx.plane_first_summaries.push(plane_result.summary.clone());
            }

            let mut pr_ctx = PlaneResultContext {
                sampled_lookup: ctx.sampled_lookup,
                volume_shape_xyz: volume.shape_xyz,
                class_count: sessions.class_count_or_default(),
                plane_first_summaries: ctx.plane_first_summaries,
                on_progress: ctx.on_progress,
            };
            handle_plane_result(
                ctx.merged_logits,
                &plane_result,
                plane,
                weight,
                ctx.sparse_mode,
                &mut pr_ctx,
            )?;
        }
    }

    Ok(())
}

struct PlaneResultContext<'a> {
    sampled_lookup: &'a HashMap<usize, usize>,
    volume_shape_xyz: [usize; 3],
    class_count: usize,
    plane_first_summaries: &'a mut Vec<String>,
    on_progress: &'a mut dyn FnMut(u8, String),
}

fn handle_plane_result(
    merged_logits: &mut [f32],
    plane_result: &PlaneForwardResult,
    plane: InferencePlane,
    weight: f32,
    sparse_mode: bool,
    ctx: &mut PlaneResultContext,
) -> Result<(), String> {
    if plane_result.slice_index == 0
        || ctx
            .plane_first_summaries
            .iter()
            .all(|s| !s.contains(plane.as_str()))
    {
        ctx.plane_first_summaries.push(plane_result.summary.clone());
    }

    if sparse_mode {
        merge_plane_logits_into_sparse_volume(
            merged_logits,
            plane_result,
            plane,
            ctx.volume_shape_xyz,
            weight,
            ctx.class_count,
            ctx.sampled_lookup,
        )?;
    } else {
        merge_plane_logits_into_volume(
            merged_logits,
            plane_result,
            plane,
            ctx.volume_shape_xyz,
            weight,
            ctx.class_count,
        )?;
    }

    (ctx.on_progress)(
        5,
        format!("Aggregating progress... slice {}", plane_result.slice_index),
    );
    Ok(())
}

fn run_single_plane_forward_at_slice(
    sessions: &NativeOnnxSessions,
    volume: &InputVolume,
    plane: InferencePlane,
    slice_index: usize,
) -> Result<PlaneForwardResult, String> {
    let trace_timing = native_trace_timing_enabled();
    let t0 = Instant::now();
    let session = sessions.plane_session(plane);

    let channel_count = session_channels_or_default(session.input_shape());
    let t_pre = Instant::now();
    let prepared = preprocess::prepare_plane_input_for_slice(
        volume,
        plane,
        channel_count,
        1.0,
        slice_index,
    )?;
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
        .map(|shape| format!("{shape:?}"))
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
    let ranges_to_vec = |ranges: Vec<Vec<usize>>| {
        ranges.into_iter().flatten().collect::<Vec<usize>>()
    };

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
    let idx_map = sagittal_index_map_for_classes(classes).ok_or_else(|| {
        format!("Unsupported sagittal class count for remap: {classes}")
    })?;

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
                "Sagittal remap index out of range: source class {source_class_idx} >= {classes}"
            ));
        }
        let src_offset = source_class_idx * hw;
        let dst_offset = full_class_idx * hw;
        remapped[dst_offset..(dst_offset + hw)]
            .copy_from_slice(&logits[src_offset..(src_offset + hw)]);
    }

    Ok((remapped, [full_classes, h, w]))
}

fn remap_if_sagittal(
    plane: InferencePlane,
    plane_result: PlaneForwardResult,
) -> Result<PlaneForwardResult, String> {
    if plane == InferencePlane::Sagittal {
        let (logits, shape_chw) = remap_sagittal_logits_to_full_space(
            &plane_result.logits,
            plane_result.shape_chw,
        )?;
        Ok(PlaneForwardResult {
            summary: plane_result.summary,
            logits,
            shape_chw,
            slice_index: plane_result.slice_index,
        })
    } else {
        Ok(plane_result)
    }
}

fn merge_plane_logits_into_volume(
    merged_logits: &mut [f32],
    plane_result: &PlaneForwardResult,
    plane: InferencePlane,
    volume_shape_xyz: [usize; 3],
    plane_weight: f32,
    expected_classes: usize,
) -> Result<(), String> {
    let [classes, height, width] = plane_result.shape_chw;
    let [expected_height, expected_width, expected_slices] =
        preprocess::transformed_volume_shape(volume_shape_xyz, plane);

    if height != expected_height || width != expected_width {
        return Err(format!(
            "{} plane logits shape mismatch. expected hw=({expected_height},{expected_width}), got ({height},{width})",
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

    let per_slice_len = classes * height * width;
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
        let class_offset = class_index * height * width;
        for row in 0..height {
            for col in 0..width {
                let slice_offset = class_offset + (row * width) + col;
                let (vx, vy, vz) = preprocess::oriented_to_xyz(
                    plane,
                    row,
                    col,
                    plane_result.slice_index,
                );
                if vx >= sx || vy >= sy || vz >= sz {
                    continue;
                }
                let voxel_offset = (vx * sy * sz) + (vy * sz) + vz;
                let merged_offset = (class_index * voxels) + voxel_offset;
                merged_logits[merged_offset] +=
                    plane_weight * plane_result.logits[slice_offset];
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
    let [classes, height, width] = plane_result.shape_chw;
    let [expected_height, expected_width, expected_slices] =
        preprocess::transformed_volume_shape(volume_shape_xyz, plane);

    if height != expected_height || width != expected_width {
        return Err(format!(
            "{} plane logits shape mismatch. expected hw=({expected_height},{expected_width}), got ({height},{width})",
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

    let per_slice_len = classes * height * width;
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
        let class_offset = class_index * height * width;
        for row in 0..height {
            for col in 0..width {
                let slice_offset = class_offset + (row * width) + col;
                let (vx, vy, vz) = preprocess::oriented_to_xyz(
                    plane,
                    row,
                    col,
                    plane_result.slice_index,
                );
                if vx >= sx || vy >= sy || vz >= sz {
                    continue;
                }
                let voxel_offset = (vx * sy * sz) + (vy * sz) + vz;
                let Some(sampled_ix) = sampled_offsets.get(&voxel_offset)
                else {
                    continue;
                };
                let merged_offset = (class_index * sampled_voxels) + sampled_ix;
                merged_logits[merged_offset] +=
                    plane_weight * plane_result.logits[slice_offset];
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
        labels[voxel] = u16::try_from(best_class).map_err(|_| {
            format!("Predicted class index {best_class} cannot be cast to u16")
        })?;
        class_hist[best_class] += 1;
    }

    Ok((labels, class_hist))
}

fn create_native_output_dir() -> Result<PathBuf, String> {
    let epoch_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            format!(
                "System clock error while creating output directory: {error}"
            )
        })?
        .as_nanos();

    let base_output_root = std::env::var("FASTSURFER_NATIVE_OUTPUT_ROOT")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map_or_else(std::env::temp_dir, PathBuf::from);

    fs::create_dir_all(&base_output_root).map_err(|error| {
        format!(
            "Failed to create output root directory '{}': {error}",
            base_output_root.display()
        )
    })?;

    let output_dir =
        base_output_root.join(format!("fastsurfer_ipc_out_{epoch_ns}"));
    fs::create_dir_all(&output_dir).map_err(|error| {
        format!(
            "Failed to create output directory '{}': {error}",
            output_dir.display()
        )
    })?;

    Ok(output_dir)
}

fn is_supported_nifti_path(path: &Path) -> bool {
    path.extension().is_some_and(|ext| {
        ext.eq_ignore_ascii_case("nii")
            || ext.eq_ignore_ascii_case("mgz")
            || ext.eq_ignore_ascii_case("mgh")
    }) || path
        .to_string_lossy()
        .to_ascii_lowercase()
        .ends_with(".nii.gz")
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
        candidates.push(
            PathBuf::from(repo_root)
                .join("FastSurferCNN/config/FreeSurferColorLUT.txt"),
        );
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates
            .push(cwd.join("FastSurferCNN/config/FreeSurferColorLUT.txt"));
        candidates.push(
            cwd.join("..")
                .join("..")
                .join("..")
                .join("..")
                .join("FastSurferCNN/config/FreeSurferColorLUT.txt"),
        );
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

    let file = fs::File::open(&lut_path).map_err(|error| {
        format!("Failed to open LUT '{}': {error}", lut_path.display())
    })?;
    let reader = BufReader::new(file);

    let mut ids = Vec::<u16>::new();
    for line in reader.lines() {
        let line = line.map_err(|error| {
            format!("Failed reading LUT '{}': {error}", lut_path.display())
        })?;
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

fn map_label_indices_to_lut(
    label_indices: &[u16],
    lut_ids: &[u16],
) -> Result<Vec<u16>, String> {
    if lut_ids.is_empty() {
        return Err(
            "LUT IDs are empty; cannot map class indices to label space."
                .to_string(),
        );
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

    let array =
        Array3::from_shape_vec((sx, sy, sz), labels_xyz).map_err(|error| {
            format!("Failed to shape label array for NIfTI write: {error}")
        })?;

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

fn write_prediction_artifacts(
    volume: &InputVolume,
    pred_labels_xyz: &[u16],
) -> Result<(PathBuf, PathBuf, PathBuf, PathBuf), String> {
    let output_dir = create_native_output_dir()?;
    let pred_path = output_dir.join("pred.nii.gz");
    write_pred_nifti(volume, pred_labels_xyz.to_owned(), &pred_path)?;

    let mut aseg_labels = postprocess::derive_aseg_from_pred(pred_labels_xyz);
    let brainmask = postprocess::derive_brainmask_from_pred(
        pred_labels_xyz,
        volume.shape_xyz,
    );
    postprocess::mask_aseg_with_brainmask(&mut aseg_labels, &brainmask);
    postprocess::flip_wm_islands(&mut aseg_labels, volume.shape_xyz);

    let aseg_path = output_dir.join("aseg.nii.gz");
    write_pred_nifti(volume, aseg_labels.clone(), &aseg_path)?;

    let brainmask_path = output_dir.join("brainmask.nii.gz");
    write_u8_nifti(volume, brainmask, &brainmask_path)?;

    Ok((output_dir, pred_path, aseg_path, brainmask_path))
}

fn finalize_logits_to_labels(
    merged_logits: &[f32],
    class_count: usize,
    voxels: usize,
    sparse_mode: bool,
    sampled_order: &[usize],
    sampled_voxels: usize,
) -> Result<(Vec<u16>, Vec<usize>), String> {
    if sparse_mode {
        let (sampled_labels, mut sampled_hist) = argmax_labels_from_logits(
            merged_logits,
            class_count,
            sampled_voxels,
        )?;
        let mut full_labels = vec![0u16; voxels];
        for (sample_ix, voxel_offset) in
            sampled_order.iter().copied().enumerate()
        {
            full_labels[voxel_offset] = sampled_labels[sample_ix];
        }
        if sampled_hist.len() < class_count {
            sampled_hist.resize(class_count, 0);
        }
        Ok((full_labels, sampled_hist))
    } else {
        argmax_labels_from_logits(merged_logits, class_count, voxels)
    }
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

    let array =
        Array3::from_shape_vec((sx, sy, sz), labels_xyz).map_err(|error| {
            format!("Failed to shape label array for NIfTI write: {error}")
        })?;

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
    should_cancel: &dyn Fn() -> bool,
    mut on_progress: F,
) -> Result<NativeSingleResult, String>
where
    F: FnMut(u8, String),
{
    let rt = tokio::runtime::Handle::try_current().map_err(|_| {
        "Failed to get current tokio runtime handle".to_string()
    })?;
    if should_cancel() {
        return Err(cancelled_error());
    }

    on_progress(2, "Loading input volume...".to_string());
    let volume = preprocess::load_input_volume(input_path)?;

    let [sx, sy, sz] = volume.shape_xyz;
    let voxels = sx * sy * sz;

    let [_, _, coronal_slices] = preprocess::transformed_volume_shape(
        volume.shape_xyz,
        InferencePlane::Coronal,
    );
    let [_, _, axial_slices] = preprocess::transformed_volume_shape(
        volume.shape_xyz,
        InferencePlane::Axial,
    );
    let [_, _, sagittal_slices] = preprocess::transformed_volume_shape(
        volume.shape_xyz,
        InferencePlane::Sagittal,
    );
    let coronal_indices = native_slice_indices_for_plane(coronal_slices);
    let axial_indices = native_slice_indices_for_plane(axial_slices);
    let sagittal_indices = native_slice_indices_for_plane(sagittal_slices);
    let total_slices =
        coronal_indices.len() + axial_indices.len() + sagittal_indices.len();

    if total_slices == 0 {
        return Err(
            "Input volume has zero slices after orientation transforms"
                .to_string(),
        );
    }

    let coronal_channels = session_channels_or_default(
        sessions
            .plane_session(InferencePlane::Coronal)
            .input_shape(),
    );
    let axial_channels = session_channels_or_default(
        sessions.plane_session(InferencePlane::Axial).input_shape(),
    );
    let sagittal_channels = session_channels_or_default(
        sessions
            .plane_session(InferencePlane::Sagittal)
            .input_shape(),
    );

    let class_count = sessions.class_count_or_default();

    let sparse_mode = parse_native_slices_per_plane().is_some();

    let (
        merged_logits,
        sampled_order,
        _sampled_lookup,
        sampled_voxels,
        plane_first_summaries,
    ) = run_forward_and_merge(
        &rt,
        sessions,
        &volume,
        &PlaneSampling {
            coronal_indices: &coronal_indices,
            sagittal_indices: &sagittal_indices,
            axial_indices: &axial_indices,
            coronal_channels,
            sagittal_channels,
            axial_channels,
        },
        class_count,
        sparse_mode,
        &mut ForwardCallbacks {
            should_cancel,
            on_progress: &mut on_progress as &mut dyn FnMut(u8, String),
        },
    )?;

    if should_cancel() {
        return Err(cancelled_error());
    }

    on_progress(92, "Converting logits to label volume...".to_string());
    let (label_indices_xyz, class_hist) = finalize_logits_to_labels(
        &merged_logits,
        class_count,
        voxels,
        sparse_mode,
        &sampled_order,
        sampled_voxels,
    )?;

    finalize_native_result(
        &FinalizeNativeResultInput {
            volume: &volume,
            label_indices_xyz: &label_indices_xyz,
            class_hist: &class_hist,
            plane_first_summaries: &plane_first_summaries,
            sessions,
            lut_ids,
            input_path,
        },
        &mut on_progress,
    )
}

type ForwardMergeResult = (
    Vec<f32>,
    Vec<usize>,
    HashMap<usize, usize>,
    usize,
    Vec<String>,
);

struct PlaneSampling<'a> {
    coronal_indices: &'a [usize],
    sagittal_indices: &'a [usize],
    axial_indices: &'a [usize],
    coronal_channels: usize,
    sagittal_channels: usize,
    axial_channels: usize,
}

struct ForwardCallbacks<'a> {
    should_cancel: &'a dyn Fn() -> bool,
    on_progress: &'a mut dyn FnMut(u8, String),
}

fn run_forward_and_merge(
    rt: &tokio::runtime::Handle,
    sessions: &NativeOnnxSessions,
    volume: &InputVolume,
    plane_sampling: &PlaneSampling<'_>,
    class_count: usize,
    sparse_mode: bool,
    callbacks: &mut ForwardCallbacks<'_>,
) -> Result<ForwardMergeResult, String> {
    let PlaneSampling {
        coronal_indices,
        sagittal_indices,
        axial_indices,
        coronal_channels,
        sagittal_channels,
        axial_channels,
    } = *plane_sampling;
    let ForwardCallbacks {
        should_cancel,
        on_progress,
    } = callbacks;

    let (sampled_order, sampled_lookup) = if sparse_mode {
        build_sampled_lookup(
            volume,
            coronal_indices,
            sagittal_indices,
            axial_indices,
        )
    } else {
        (Vec::<usize>::new(), HashMap::<usize, usize>::new())
    };
    let [sx, sy, sz] = volume.shape_xyz;
    let voxels = sx * sy * sz;
    let sampled_voxels = if sparse_mode {
        sampled_lookup.len().max(1)
    } else {
        voxels
    };
    let mut merged_logits = vec![0f32; class_count * sampled_voxels];
    let mut plane_first_summaries: Vec<String> = Vec::new();

    let plane_specs_vec = vec![
        (
            InferencePlane::Coronal,
            coronal_indices,
            coronal_channels,
            0.4f32,
        ),
        (
            InferencePlane::Sagittal,
            sagittal_indices,
            sagittal_channels,
            0.2f32,
        ),
        (InferencePlane::Axial, axial_indices, axial_channels, 0.4f32),
    ];

    let mut forward_ctx = ForwardPassContext {
        sparse_mode,
        sampled_lookup: &sampled_lookup,
        merged_logits: &mut merged_logits,
        should_cancel,
        on_progress: *on_progress,
        plane_first_summaries: &mut plane_first_summaries,
    };

    execute_forward_pass(
        rt,
        &ForwardPassRequest {
            sessions,
            volume,
            plane_specs: &plane_specs_vec,
        },
        &mut forward_ctx,
    )?;

    Ok((
        merged_logits,
        sampled_order,
        sampled_lookup,
        sampled_voxels,
        plane_first_summaries,
    ))
}

fn validate_inputs(
    file_paths: &[String],
    folder_paths: &[String],
) -> Result<Vec<String>, String> {
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
            "Rust native inference requires at least one valid input path (.nii/.nii.gz/.mgz/.mgh) from files or folders."
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
        let single = run_native_single_path(
            &sessions,
            &lut_ids,
            input_path,
            &|| false,
            |_progress, _message| {},
        )?;
        result_directories.insert(single.result_directory);
        qc_summaries.push(single.qc_summary);
        results.push(single.prediction);
    }

    let result_directories =
        result_directories.into_iter().collect::<Vec<String>>();
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

fn load_runtime_dependencies<R: tauri::Runtime>(
    app_handle: &tauri::AppHandle<R>,
    task_id: &str,
    file_paths: &[String],
    folder_paths: &[String],
) -> Result<(Vec<String>, NativeOnnxSessions, Vec<u16>), String> {
    let requested_paths = match validate_inputs(file_paths, folder_paths) {
        Ok(paths) => paths,
        Err(error) => {
            emit_failed_progress(app_handle, task_id, 0, 0, error.clone());
            return Err(error);
        }
    };

    let total = requested_paths.len();
    emit_inference_progress(
        app_handle,
        task_id,
        ProgressUpdate {
            status: "started",
            message: format!(
                "Rust ONNX mode selected (runtime={}). Initializing ONNX sessions...",
                resolve_native_onnx_runtime().as_str()
            ),
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
            emit_failed_progress(app_handle, task_id, total, 0, error.clone());
            return Err(error);
        }
    };

    let lut_ids = match load_lut_ids() {
        Ok(ids) => ids,
        Err(error) => {
            emit_failed_progress(app_handle, task_id, total, 0, error.clone());
            return Err(error);
        }
    };

    Ok((requested_paths, sessions, lut_ids))
}

pub(crate) fn run_native_inference_with_progress<R: tauri::Runtime>(
    app_handle: &tauri::AppHandle<R>,
    cancelled_tasks: &Arc<Mutex<BTreeSet<String>>>,
    task_id: &str,
    file_paths: &[String],
    folder_paths: &[String],
) -> Result<ProcessingRunResult, String> {
    let (requested_paths, sessions, lut_ids) = load_runtime_dependencies(
        app_handle,
        task_id,
        file_paths,
        folder_paths,
    )?;

    let total = requested_paths.len();

    let mut results = Vec::with_capacity(total);
    let mut result_directories = BTreeSet::new();
    let mut qc_summaries = Vec::new();
    for (index, input_path) in requested_paths.iter().enumerate() {
        let single = process_single_input(
            &SingleInputContext {
                app_handle,
                cancelled_tasks,
                task_id,
                sessions: &sessions,
                lut_ids: &lut_ids,
                index,
                total,
            },
            input_path,
        )?;

        let completed = index + 1;
        let progress = if total == 0 {
            100u8
        } else {
            let v = ((completed * 100) / total).min(100);
            u8::try_from(v).unwrap_or(100u8)
        };
        let output_path = single.prediction.output_path.clone();
        result_directories.insert(single.result_directory);
        qc_summaries.push(single.qc_summary);
        results.push(single.prediction);

        emit_inference_progress(
            app_handle,
            task_id,
            ProgressUpdate {
                status: "item_completed",
                message: format!("Processed {completed}/{total}"),
                total,
                completed,
                progress,
                current_path: Some(input_path.clone()),
                output_path: Some(output_path),
            },
        );
    }

    if let Ok(mut cancelled) = cancelled_tasks.lock() {
        cancelled.remove(task_id);
    }

    let result_directories =
        result_directories.into_iter().collect::<Vec<String>>();
    let ack_message = if result_directories.is_empty() {
        format!("Processing started for {} path(s).", requested_paths.len())
    } else {
        format!(
            "Processing started for {} path(s). Results directory: {}",
            requested_paths.len(),
            result_directories.join(", ")
        )
    };

    emit_inference_progress(
        app_handle,
        task_id,
        ProgressUpdate {
            status: "completed",
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

struct ProgressUpdate<'a> {
    status: &'a str,
    message: String,
    total: usize,
    completed: usize,
    progress: u8,
    current_path: Option<String>,
    output_path: Option<String>,
}

fn emit_failed_progress<R: tauri::Runtime>(
    app_handle: &tauri::AppHandle<R>,
    task_id: &str,
    total: usize,
    completed: usize,
    message: String,
) {
    emit_inference_progress(
        app_handle,
        task_id,
        ProgressUpdate {
            status: "failed",
            message,
            total,
            completed,
            progress: progress_for_completed(total, completed, 100),
            current_path: None,
            output_path: None,
        },
    );
}

fn progress_for_completed(total: usize, completed: usize, fallback: u8) -> u8 {
    if total == 0 {
        fallback
    } else {
        let value = ((completed * 100) / total).min(100);
        u8::try_from(value).unwrap_or(100)
    }
}

fn remove_cancelled_task(
    cancelled_tasks: &Arc<Mutex<BTreeSet<String>>>,
    task_id: &str,
) {
    if let Ok(mut cancelled) = cancelled_tasks.lock() {
        cancelled.remove(task_id);
    }
}

fn emit_single_input_status<R: tauri::Runtime>(
    ctx: &SingleInputContext<'_, R>,
    input_path: &str,
    status: &'static str,
    message: String,
    progress: u8,
) {
    emit_inference_progress(
        ctx.app_handle,
        ctx.task_id,
        ProgressUpdate {
            status,
            message,
            total: ctx.total,
            completed: ctx.index,
            progress,
            current_path: Some(input_path.to_string()),
            output_path: None,
        },
    );
}

fn emit_cancelled_single_input<R: tauri::Runtime>(
    ctx: &SingleInputContext<'_, R>,
    input_path: &str,
) {
    emit_single_input_status(
        ctx,
        input_path,
        "cancelled",
        NATIVE_CANCELLED_MESSAGE.to_string(),
        progress_for_completed(ctx.total, ctx.index, 0),
    );
}

fn emit_failed_single_input<R: tauri::Runtime>(
    ctx: &SingleInputContext<'_, R>,
    input_path: &str,
    message: String,
) {
    emit_single_input_status(
        ctx,
        input_path,
        "failed",
        message,
        progress_for_completed(ctx.total, ctx.index, 0),
    );
}

fn emit_inference_progress<R: tauri::Runtime>(
    app_handle: &tauri::AppHandle<R>,
    task_id: &str,
    update: ProgressUpdate<'_>,
) {
    let ProgressUpdate {
        status,
        message,
        total,
        completed,
        progress,
        current_path,
        output_path,
    } = update;

    let _ = app_handle.emit(
        "fastsurfer://inference-progress",
        InferenceProgressEvent {
            task_id: task_id.to_string(),
            status: status.to_string(),
            message,
            total,
            completed,
            progress,
            current_path,
            output_path,
        },
    );
}

struct SingleInputContext<'a, R: tauri::Runtime> {
    app_handle: &'a tauri::AppHandle<R>,
    cancelled_tasks: &'a Arc<Mutex<BTreeSet<String>>>,
    task_id: &'a str,
    sessions: &'a NativeOnnxSessions,
    lut_ids: &'a [u16],
    index: usize,
    total: usize,
}

fn process_single_input<R: tauri::Runtime>(
    ctx: &SingleInputContext<'_, R>,
    input_path: &str,
) -> Result<NativeSingleResult, String> {
    if is_task_cancelled(ctx.cancelled_tasks, ctx.task_id)? {
        emit_cancelled_single_input(ctx, input_path);
        remove_cancelled_task(ctx.cancelled_tasks, ctx.task_id);
        return Err(cancelled_error());
    }

    let should_cancel = || {
        ctx.cancelled_tasks
            .lock()
            .map(|set| set.contains(ctx.task_id))
            .unwrap_or(false)
    };

    let single = run_native_single_path(
        ctx.sessions,
        ctx.lut_ids,
        input_path,
        &should_cancel,
        |file_progress, file_message| {
            let overall_progress = if ctx.total == 0 {
                file_progress
            } else {
                let v = (((ctx.index * 100) + file_progress as usize)
                    / ctx.total)
                    .min(99);
                u8::try_from(v).unwrap_or(99u8)
            };

            emit_inference_progress(
                ctx.app_handle,
                ctx.task_id,
                ProgressUpdate {
                    status: "item_progress",
                    message: file_message,
                    total: ctx.total,
                    completed: ctx.index,
                    progress: overall_progress,
                    current_path: Some(input_path.to_string()),
                    output_path: None,
                },
            );
        },
    );

    match single {
        Ok(single) => Ok(single),
        Err(error) => {
            let was_cancelled = error == NATIVE_CANCELLED_MESSAGE
                || is_task_cancelled(ctx.cancelled_tasks, ctx.task_id)?;

            if was_cancelled {
                emit_cancelled_single_input(ctx, input_path);
                remove_cancelled_task(ctx.cancelled_tasks, ctx.task_id);

                return Err(cancelled_error());
            }

            emit_failed_single_input(ctx, input_path, error.clone());

            Err(error)
        }
    }
}
