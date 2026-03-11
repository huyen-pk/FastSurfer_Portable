use ort::session::{Session, SessionInputValue};
use ort::value::{Tensor, ValueType};
use std::collections::BTreeSet;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

#[derive(Clone)]
pub(crate) struct OnnxModelRegistry {
    pub axial_model_path: String,
    pub coronal_model_path: String,
    pub sagittal_model_path: String,
    pub source_dir: String,
}

#[derive(Clone)]
pub(crate) struct NativeOnnxSessions {
    pub axial: PlaneSession,
    pub coronal: PlaneSession,
    pub sagittal: PlaneSession,
    pub registry: OnnxModelRegistry,
}

#[derive(Clone)]
pub(crate) struct PlaneSession {
    pub session: Arc<Mutex<Session>>,
    pub model_path: String,
    pub input_name: String,
    pub required_input_names: Vec<String>,
    pub required_input_shapes: HashMap<String, Option<Vec<usize>>>,
    pub output_names: Vec<String>,
    pub input_shape: Option<Vec<usize>>,
    pub output_shapes: Vec<Option<Vec<usize>>>,
}

pub(crate) struct PlaneRunResult {
    pub logits: Vec<f32>,
    pub shape_chw: [usize; 3],
    pub output_shapes: Vec<Vec<usize>>,
}

static ORT_INIT: OnceLock<()> = OnceLock::new();

fn ort_trace_timing_enabled() -> bool {
    let value = std::env::var("FASTSURFER_ORT_TRACE_TIMING")
        .or_else(|_| std::env::var("FASTSURFER_NATIVE_TRACE_TIMING"));

    value
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on"
        })
        .unwrap_or(false)
}

fn ort_cpu_threads_override() -> Option<usize> {
    std::env::var("FASTSURFER_ORT_CPU_THREADS")
        .or_else(|_| std::env::var("FASTSURFER_NATIVE_CPU_THREADS"))
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn ensure_ort_initialized() -> Result<(), String> {
    if ORT_INIT.get().is_some() {
        return Ok(());
    }

    let init = if std::env::var_os("ORT_DYLIB_PATH").is_none()
        && let Some(path) = resolve_ort_dylib_path()
    {
        ort::init_from(path)
    } else {
        Ok(ort::init())
    }
    .map_err(|e| e.to_string())?;

    let _ = init.with_name("fastsurfer-desktop").commit();

    let _ = ORT_INIT.set(());
    Ok(())
}

fn resolve_ort_dylib_path() -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os("FASTSURFER_ORT_DYLIB_PATH") {
        let path = PathBuf::from(explicit);
        if path.exists() && path.is_file() {
            return Some(path);
        }
    }

    let mut candidates = Vec::<PathBuf>::new();
    let names = ort_dylib_file_names();

    if let Ok(cwd) = std::env::current_dir() {
        for name in names {
            candidates.push(cwd.join(name));
            candidates.push(
                cwd.join("resources")
                    .join("ort")
                    .join(platform_folder())
                    .join(name),
            );
        }
    }

    if let Ok(exe) = std::env::current_exe()
        && let Some(exe_dir) = exe.parent()
    {
        for name in names {
            candidates.push(exe_dir.join(name));
            candidates.push(
                exe_dir
                    .join("resources")
                    .join("ort")
                    .join(platform_folder())
                    .join(name),
            );
            candidates.push(exe_dir.join("_internal").join(name));
            candidates.push(exe_dir.join("..").join("Resources").join(name));
            candidates.push(
                exe_dir
                    .join("..")
                    .join("Resources")
                    .join("ort")
                    .join(platform_folder())
                    .join(name),
            );
        }

        candidates.extend(find_ort_dylibs_near(exe_dir, 3));
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for name in names {
        candidates.push(
            manifest_dir
                .join("resources")
                .join("ort")
                .join(platform_folder())
                .join(name),
        );
    }

    candidates
        .into_iter()
        .find(|path| path.exists() && path.is_file())
}

fn find_ort_dylibs_near(root: &Path, max_depth: usize) -> Vec<PathBuf> {
    if max_depth == 0 {
        return Vec::new();
    }

    let mut found = Vec::<PathBuf>::new();
    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(name) = path.file_name().and_then(|name| name.to_str())
                    && is_ort_dylib_filename(name)
                {
                    found.push(path);
                }
            } else if path.is_dir() {
                found.extend(find_ort_dylibs_near(&path, max_depth.saturating_sub(1)));
            }
        }
    }

    found
}

fn is_ort_dylib_filename(name: &str) -> bool {
    #[cfg(target_os = "windows")]
    {
        name.eq_ignore_ascii_case("onnxruntime.dll")
    }

    #[cfg(target_os = "linux")]
    {
        name.starts_with("libonnxruntime") && name.contains(".so")
    }

    #[cfg(target_os = "macos")]
    {
        name.starts_with("libonnxruntime") && name.ends_with(".dylib")
    }
}

fn platform_folder() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "windows"
    }

    #[cfg(target_os = "linux")]
    {
        "linux"
    }

    #[cfg(target_os = "macos")]
    {
        "macos"
    }
}

fn ort_dylib_file_names() -> &'static [&'static str] {
    #[cfg(target_os = "windows")]
    {
        &["onnxruntime.dll"]
    }

    #[cfg(target_os = "linux")]
    {
        &["libonnxruntime.so", "libonnxruntime.so.1"]
    }

    #[cfg(target_os = "macos")]
    {
        &["libonnxruntime.dylib"]
    }
}

impl OnnxModelRegistry {
    pub(crate) fn discover_default() -> Result<Self, String> {
        let candidate_dirs = candidate_onnx_dirs();

        for dir in &candidate_dirs {
            let axial = find_model(
                dir,
                &["FastSurferVINN_Axial.onnx", "FastSurferVINN_axial.onnx"],
            );
            let coronal = find_model(
                dir,
                &["FastSurferVINN_Coronal.onnx", "FastSurferVINN_coronal.onnx"],
            );
            let sagittal = find_model(
                dir,
                &[
                    "FastSurferVINN_Sagittal.onnx",
                    "FastSurferVINN_sagittal.onnx",
                ],
            );

            if let (Some(axial_path), Some(coronal_path), Some(sagittal_path)) =
                (axial, coronal, sagittal)
            {
                return Ok(Self {
                    axial_model_path: axial_path.to_string_lossy().to_string(),
                    coronal_model_path: coronal_path.to_string_lossy().to_string(),
                    sagittal_model_path: sagittal_path.to_string_lossy().to_string(),
                    source_dir: dir.to_string_lossy().to_string(),
                });
            }
        }

        let searched = candidate_dirs
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect::<Vec<String>>()
            .join(", ");

        Err(format!(
            "Could not discover ONNX model trio (axial/coronal/sagittal). Searched directories: {searched}"
        ))
    }
}

impl NativeOnnxSessions {
    pub(crate) fn load_default() -> Result<Self, String> {
        ensure_ort_initialized()?;
        let trace_timing = ort_trace_timing_enabled();
        let t0 = Instant::now();
        let registry = OnnxModelRegistry::discover_default()?;

        let t_ax = Instant::now();
        let axial = load_runnable_model(&registry.axial_model_path, "axial")?;
        if trace_timing {
            eprintln!(
                "[trace][ort-onnx] loaded axial model in {} ms",
                t_ax.elapsed().as_millis()
            );
        }

        let t_cor = Instant::now();
        let coronal = load_runnable_model(&registry.coronal_model_path, "coronal")?;
        if trace_timing {
            eprintln!(
                "[trace][ort-onnx] loaded coronal model in {} ms",
                t_cor.elapsed().as_millis()
            );
        }

        let t_sag = Instant::now();
        let sagittal = load_runnable_model(&registry.sagittal_model_path, "sagittal")?;
        if trace_timing {
            eprintln!(
                "[trace][ort-onnx] loaded sagittal model in {} ms",
                t_sag.elapsed().as_millis()
            );
            eprintln!(
                "[trace][ort-onnx] total session load {} ms",
                t0.elapsed().as_millis()
            );
        }

        Ok(Self {
            axial,
            coronal,
            sagittal,
            registry,
        })
    }

    pub(crate) fn run_dummy_probe(&self) -> Result<Vec<String>, String> {
        Ok(vec![
            self.axial.run_dummy_probe("axial")?,
            self.coronal.run_dummy_probe("coronal")?,
            self.sagittal.run_dummy_probe("sagittal")?,
        ])
    }
}

impl PlaneSession {
    fn run_dummy_probe(&self, plane: &str) -> Result<String, String> {
        let Some(shape) = self.input_shape.as_ref() else {
            return Ok(format!(
                "{plane}: skipped dummy run (symbolic input shape), outputs={}",
                self.output_shapes
                    .iter()
                    .map(|shape| match shape {
                        Some(dims) => format!("{dims:?}"),
                        None => "symbolic".to_string(),
                    })
                    .collect::<Vec<String>>()
                    .join("; ")
            ));
        };

        let element_count = shape
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| format!("{plane}: input shape overflow for {shape:?}"))?;

        let max_probe_elements = 16_000_000usize;
        if element_count > max_probe_elements {
            return Ok(format!(
                "{plane}: skipped dummy run (shape too large for probe) input_shape={shape:?} elements={element_count}"
            ));
        }

        let data = vec![0f32; element_count];
        let outputs = self.run(shape, &data, [1.0, 1.0], plane)?;

        Ok(format!(
            "{plane}: dummy run ok input_shape={shape:?} outputs={} model_path='{}'",
            outputs.output_shapes.len(),
            self.model_path
        ))
    }

    pub(crate) fn run(
        &self,
        input_shape: &[usize],
        input_data: &[f32],
        scale_factor: [f32; 2],
        plane: &str,
    ) -> Result<PlaneRunResult, String> {
        let trace_timing = ort_trace_timing_enabled();
        let run_started = Instant::now();

        let expected_len = input_shape.iter().product::<usize>();
        if expected_len != input_data.len() {
            return Err(format!(
                "{plane}: input tensor length mismatch, got {}, expected {} for shape {:?}",
                input_data.len(),
                expected_len,
                input_shape
            ));
        }

        let input_tensor = Tensor::from_array((input_shape.to_vec(), input_data.to_vec()))
            .map_err(|error| {
                format!(
                    "{plane}: failed to create ORT input tensor for shape {:?}: {error}",
                    input_shape
                )
            })?;

        let mut inputs: Vec<(String, SessionInputValue<'_>)> = Vec::new();
        inputs.push((self.input_name.clone(), input_tensor.into()));

        for required_name in &self.required_input_names {
            if required_name == &self.input_name {
                continue;
            }

            let lower = required_name.to_ascii_lowercase();
            if lower.contains("scale") {
                let aux_shape = self
                    .required_input_shapes
                    .get(required_name)
                    .and_then(|shape| shape.clone())
                    .unwrap_or_else(|| vec![2usize]);

                let expected_aux_len = aux_shape.iter().product::<usize>();
                let fallback_aux_shape = vec![2usize];
                let actual_shape = if expected_aux_len == 2 {
                    aux_shape
                } else {
                    fallback_aux_shape
                };

                let aux_tensor = Tensor::from_array((actual_shape, vec![scale_factor[0], scale_factor[1]]))
                    .map_err(|error| {
                        format!(
                            "{plane}: failed to create ORT auxiliary input '{required_name}' tensor: {error}"
                        )
                    })?;

                inputs.push((required_name.clone(), aux_tensor.into()));
                continue;
            }

            return Err(format!(
                "{plane}: unsupported required auxiliary input '{required_name}' for model '{}'",
                self.model_path
            ));
        }

        let mut session = self
            .session
            .lock()
            .map_err(|_| format!("{plane}: ORT session mutex was poisoned"))?;

        let eval_started = Instant::now();
        let outputs = session
            .run(inputs)
            .map_err(|error| format!("{plane}: ONNX Runtime forward pass failed: {error}"))?;

        if trace_timing {
            eprintln!(
                "[trace][ort-onnx] run end plane={} eval_ms={} total_ms={}",
                plane,
                eval_started.elapsed().as_millis(),
                run_started.elapsed().as_millis()
            );
        }

        let primary_name = self
            .output_names
            .first()
            .ok_or_else(|| format!("{plane}: model has no declared graph outputs"))?;

        let primary = outputs.get(primary_name).ok_or_else(|| {
            format!("{plane}: expected output '{primary_name}' not found in ORT outputs")
        })?;

        let (shape, tensor_data) = primary
            .try_extract_tensor::<f32>()
            .map_err(|error| format!("{plane}: failed to extract f32 output tensor: {error}"))?;

        let shape = shape
            .iter()
            .map(|dim| {
                if *dim < 0 {
                    Err(format!(
                        "{plane}: dynamic output shape is not supported: {shape:?}"
                    ))
                } else {
                    Ok(*dim as usize)
                }
            })
            .collect::<Result<Vec<usize>, String>>()?;

        if shape.len() != 4 {
            return Err(format!(
                "{plane}: output shape must be rank-4 [N,C,H,W], got {:?}",
                shape
            ));
        }

        if shape[0] != 1 {
            return Err(format!(
                "{plane}: output batch dimension must be 1, got {}",
                shape[0]
            ));
        }

        let mut output_shapes = Vec::<Vec<usize>>::new();
        for name in &self.output_names {
            if let Some(value) = outputs.get(name)
                && let Ok((dims, _)) = value.try_extract_tensor::<f32>()
            {
                let parsed = dims
                    .iter()
                    .map(|dim| if *dim < 0 { 0usize } else { *dim as usize })
                    .collect::<Vec<usize>>();
                output_shapes.push(parsed);
            }
        }

        Ok(PlaneRunResult {
            logits: tensor_data.to_vec(),
            shape_chw: [shape[1], shape[2], shape[3]],
            output_shapes,
        })
    }
}

fn load_runnable_model(path: &str, plane: &str) -> Result<PlaneSession, String> {
    ensure_ort_initialized()?;

    let mut builder = Session::builder()
        .map_err(|error| format!("Failed to create ORT session builder for {plane}: {error}"))?;

    if let Some(thread_count) = ort_cpu_threads_override() {
        builder = builder.with_intra_threads(thread_count).map_err(|error| {
            format!(
                "Failed to configure ORT intra-op threads ({thread_count}) for {plane}: {error}"
            )
        })?;
    }

    let session = builder.commit_from_file(path).map_err(|error| {
        format!("Failed to load {plane} ONNX model in ORT at '{path}': {error}")
    })?;

    let initializers = session
        .overridable_initializers()
        .iter()
        .map(|initializer| initializer.name().to_string())
        .collect::<HashSet<String>>();

    let required_input_names = session
        .inputs()
        .iter()
        .filter(|input| !initializers.contains(input.name()))
        .map(|input| input.name().to_string())
        .collect::<Vec<String>>();

    let required_input_shapes = session
        .inputs()
        .iter()
        .filter(|input| !initializers.contains(input.name()))
        .map(|input| {
            (
                input.name().to_string(),
                dims_from_value_type(input.dtype()),
            )
        })
        .collect::<HashMap<String, Option<Vec<usize>>>>();

    let input_name = required_input_names
        .iter()
        .find(|name| {
            required_input_shapes
                .get(*name)
                .and_then(|shape| shape.as_ref())
                .map(|shape| shape.len() == 4)
                .unwrap_or(false)
        })
        .cloned()
        .or_else(|| required_input_names.first().cloned())
        .ok_or_else(|| format!("Failed to discover {plane} ORT input name at '{path}'"))?;

    let output_names = session
        .outputs()
        .iter()
        .map(|output| output.name().to_string())
        .collect::<Vec<String>>();

    if output_names.is_empty() {
        return Err(format!(
            "Failed to discover {plane} ORT output names at '{path}'"
        ));
    }

    let input_shape = session
        .inputs()
        .iter()
        .find(|input| input.name() == input_name)
        .and_then(|input| dims_from_value_type(input.dtype()));

    let output_shapes = output_names
        .iter()
        .map(|name| {
            session
                .outputs()
                .iter()
                .find(|output| output.name() == name)
                .and_then(|output| dims_from_value_type(output.dtype()))
        })
        .collect::<Vec<Option<Vec<usize>>>>();

    Ok(PlaneSession {
        session: Arc::new(Mutex::new(session)),
        model_path: path.to_string(),
        input_name,
        required_input_names,
        required_input_shapes,
        output_names,
        input_shape,
        output_shapes,
    })
}

fn dims_from_value_type(dtype: &ValueType) -> Option<Vec<usize>> {
    match dtype {
        ValueType::Tensor { shape, .. } => shape
            .iter()
            .map(|dim| if *dim < 0 { None } else { Some(*dim as usize) })
            .collect::<Option<Vec<usize>>>(),
        _ => None,
    }
}

fn find_model(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    names
        .iter()
        .map(|name| dir.join(name))
        .find(|candidate| candidate.exists() && candidate.is_file())
}

fn candidate_onnx_dirs() -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(explicit_dir) = std::env::var("FASTSURFER_ONNX_DIR") {
        let explicit_path = PathBuf::from(explicit_dir);
        candidates.push(explicit_path.clone());
        candidates.push(explicit_path.join("onnx"));
    }

    if let Ok(explicit_repo_root) = std::env::var("FASTSURFER_REPO_ROOT") {
        let repo_path = PathBuf::from(explicit_repo_root);
        candidates.push(repo_path.join("onnx"));
    }

    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("onnx"));
        candidates.push(cwd.join("..").join("..").join("..").join("..").join("onnx"));
    }

    if let Ok(exe) = std::env::current_exe()
        && let Some(exe_dir) = exe.parent()
    {
        candidates.push(exe_dir.join("onnx"));
        candidates.push(exe_dir.join("_internal").join("onnx"));
        candidates.push(exe_dir.join("..").join("Resources").join("onnx"));
        candidates.push(exe_dir.join("..").join("..").join("onnx"));
    }

    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut unique_existing: Vec<PathBuf> = Vec::new();
    for candidate in candidates {
        let key = candidate.to_string_lossy().to_string();
        if seen.contains(&key) {
            continue;
        }
        seen.insert(key);

        if candidate.exists() && candidate.is_dir() {
            unique_existing.push(candidate);
        }
    }

    unique_existing
}
