use crate::inference::engine::onnx_loader_candle as candle_loader;
use crate::inference::engine::onnx_loader_ort as ort_loader;
use crate::inference::entities::InferencePlane;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeOnnxRuntime {
    Candle,
    Ort,
}

#[derive(Clone)]
pub(crate) enum NativeOnnxSessions {
    Candle(candle_loader::NativeOnnxSessions),
    Ort(ort_loader::NativeOnnxSessions),
}

pub(crate) struct PlaneRun {
    pub logits: Vec<f32>,
    pub shape_chw: [usize; 3],
    pub output_shapes: Vec<Vec<usize>>,
}

pub(crate) enum PlaneSessionRef<'a> {
    Candle(&'a candle_loader::PlaneSession),
    Ort(&'a ort_loader::PlaneSession),
}

pub(crate) fn resolve_native_onnx_runtime() -> NativeOnnxRuntime {
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
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Candle => "candle",
            Self::Ort => "ort",
        }
    }
}

impl NativeOnnxSessions {
    pub(crate) fn runtime(&self) -> NativeOnnxRuntime {
        match self {
            Self::Candle(_) => NativeOnnxRuntime::Candle,
            Self::Ort(_) => NativeOnnxRuntime::Ort,
        }
    }

    pub(crate) fn load_default() -> Result<Self, String> {
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

    pub(crate) fn source_dir(&self) -> &str {
        match self {
            Self::Candle(models) => &models.registry.source_dir,
            Self::Ort(models) => &models.registry.source_dir,
        }
    }

    pub(crate) fn model_paths_summary(&self) -> String {
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

    pub(crate) fn run_dummy_probe(&self) -> Result<Vec<String>, String> {
        match self {
            Self::Candle(models) => models.run_dummy_probe(),
            Self::Ort(models) => models.run_dummy_probe(),
        }
    }

    pub(crate) fn plane_session(
        &self,
        plane: InferencePlane,
    ) -> PlaneSessionRef<'_> {
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

    pub(crate) fn class_count_or_default(&self) -> usize {
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
    pub(crate) fn input_shape(&self) -> Option<&[usize]> {
        match self {
            Self::Candle(session) => session.input_shape.as_deref(),
            Self::Ort(session) => session.input_shape.as_deref(),
        }
    }

    pub(crate) fn run(
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

pub(crate) fn discovery_summary(models: &NativeOnnxSessions) -> String {
    models.model_paths_summary()
}

pub(crate) fn session_channels_or_default(
    input_shape: Option<&[usize]>,
) -> usize {
    input_shape
        .and_then(|dims| dims.get(1).copied())
        .filter(|channels| *channels > 0)
        .unwrap_or(7)
}

pub(crate) fn native_trace_timing_enabled() -> bool {
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

pub(crate) fn parse_native_slices_per_plane() -> Option<usize> {
    std::env::var("FASTSURFER_NATIVE_SLICES_PER_PLANE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
}

pub(crate) fn native_parallel_enabled() -> bool {
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

pub(crate) fn native_slice_indices_for_plane(
    total_slices: usize,
) -> Vec<usize> {
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

pub(crate) fn load_runtime_dependencies(
    file_paths: &[String],
    folder_paths: &[String],
) -> Result<(Vec<String>, NativeOnnxSessions, Vec<u16>), String> {
    let requested_paths =
        crate::inference::file_io::validate_inputs(file_paths, folder_paths)?;

    let sessions = NativeOnnxSessions::load_default()?;

    let lut_ids = crate::inference::file_io::load_lut_ids()?;

    Ok((requested_paths, sessions, lut_ids))
}
