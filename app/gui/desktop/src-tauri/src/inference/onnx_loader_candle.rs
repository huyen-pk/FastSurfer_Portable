use candle::{DType, Device, Tensor};
use candle_onnx::{onnx, read_file, simple_eval};
use std::collections::BTreeSet;
use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;
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
    pub model: Arc<onnx::ModelProto>,
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

fn native_trace_timing_enabled() -> bool {
    std::env::var("FASTSURFER_NATIVE_TRACE_TIMING")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on"
        })
        .unwrap_or(false)
}

fn native_plane_timeout_secs() -> Option<u64> {
    std::env::var("FASTSURFER_NATIVE_PLANE_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
}

fn apply_native_cpu_thread_override() {
    if let Ok(value) = std::env::var("FASTSURFER_NATIVE_CPU_THREADS")
        && let Ok(parsed) = value.trim().parse::<usize>()
        && parsed > 0
    {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(parsed)
            .build_global();
    }
}

fn resolve_native_device_mode() -> &'static str {
    let requested = std::env::var("FASTSURFER_NATIVE_DEVICE")
        .unwrap_or_else(|_| "cpu".to_string())
        .to_ascii_lowercase();

    match requested.as_str() {
        "gpu" | "cuda" => {
            if candle::utils::cuda_is_available() {
                "cuda"
            } else {
                "cpu"
            }
        }
        _ => "cpu",
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
        apply_native_cpu_thread_override();
        let trace_timing = native_trace_timing_enabled();
        let t0 = Instant::now();
        if trace_timing {
            let requested =
                std::env::var("FASTSURFER_NATIVE_DEVICE").unwrap_or_else(|_| "cpu".to_string());
            let resolved = resolve_native_device_mode();
            eprintln!(
                "[trace][native-onnx] requested_device={} resolved_device={} cuda_available={} metal_available={} rayon_threads={}",
                requested,
                resolved,
                candle::utils::cuda_is_available(),
                candle::utils::metal_is_available(),
                candle::utils::get_num_threads()
            );
            if (requested.eq_ignore_ascii_case("gpu") || requested.eq_ignore_ascii_case("cuda"))
                && resolved == "cpu"
            {
                eprintln!(
                    "[trace][native-onnx] GPU requested but Candle ONNX path is running on CPU (cuda feature unavailable in this build)."
                );
            }
        }
        let registry = OnnxModelRegistry::discover_default()?;

        let t_ax = Instant::now();
        let axial = load_runnable_model(&registry.axial_model_path, "axial")?;
        if trace_timing {
            eprintln!(
                "[trace][native-onnx] loaded axial model in {} ms",
                t_ax.elapsed().as_millis()
            );
        }

        let t_cor = Instant::now();
        let coronal = load_runnable_model(&registry.coronal_model_path, "coronal")?;
        if trace_timing {
            eprintln!(
                "[trace][native-onnx] loaded coronal model in {} ms",
                t_cor.elapsed().as_millis()
            );
        }

        let t_sag = Instant::now();
        let sagittal = load_runnable_model(&registry.sagittal_model_path, "sagittal")?;
        if trace_timing {
            eprintln!(
                "[trace][native-onnx] loaded sagittal model in {} ms",
                t_sag.elapsed().as_millis()
            );
            eprintln!(
                "[trace][native-onnx] total session load {} ms",
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
        let trace_timing = native_trace_timing_enabled();
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

        let input =
            Tensor::from_vec(input_data.to_vec(), input_shape, &Device::Cpu).map_err(|error| {
                format!(
                    "{plane}: failed to create input tensor for shape {:?}: {error}",
                    input_shape
                )
            })?;

        let mut inputs = HashMap::new();
        inputs.insert(self.input_name.clone(), input);

        for required_name in &self.required_input_names {
            if required_name == &self.input_name {
                continue;
            }

            let lower = required_name.to_ascii_lowercase();
            if lower.contains("scale") {
                let aux_shape = self
                    .required_input_shapes
                    .get(required_name)
                    .and_then(|shape| shape.clone());

                let scale_values = vec![scale_factor[0], scale_factor[1]];
                let tensor = match aux_shape {
                    Some(shape) if shape.iter().product::<usize>() == scale_values.len() => {
                        Tensor::from_vec(scale_values, shape, &Device::Cpu).map_err(|error| {
                            format!(
                                "{plane}: failed to create auxiliary input '{required_name}' tensor: {error}"
                            )
                        })?
                    }
                    _ => Tensor::from_vec(scale_values, &[2], &Device::Cpu).map_err(|error| {
                        format!(
                            "{plane}: failed to create fallback auxiliary input '{required_name}' tensor: {error}"
                        )
                    })?,
                };
                inputs.insert(required_name.clone(), tensor);
                continue;
            }

            return Err(format!(
                "{plane}: unsupported required auxiliary input '{required_name}' for model '{}'",
                self.model_path
            ));
        }

        if trace_timing {
            eprintln!(
                "[trace][native-onnx] simple_eval start plane={} input_shape={:?}",
                plane, input_shape
            );
        }
        let eval_started = Instant::now();
        let outputs = if let Some(timeout_secs) = native_plane_timeout_secs() {
            let (tx, rx) = mpsc::channel();
            let model = Arc::clone(&self.model);
            let plane_name = plane.to_string();
            thread::spawn(move || {
                let result = simple_eval(&model, inputs)
                    .map_err(|error| format!("{plane_name}: ONNX forward pass failed: {error}"));
                let _ = tx.send(result);
            });

            match rx.recv_timeout(Duration::from_secs(timeout_secs)) {
                Ok(result) => result?,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    return Err(format!(
                        "{plane}: ONNX forward pass timed out after {}s",
                        timeout_secs
                    ));
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(format!(
                        "{plane}: ONNX forward pass worker disconnected before returning result"
                    ));
                }
            }
        } else {
            simple_eval(&self.model, inputs)
                .map_err(|error| format!("{plane}: ONNX forward pass failed: {error}"))?
        };
        if trace_timing {
            eprintln!(
                "[trace][native-onnx] simple_eval end plane={} eval_ms={} total_ms={}",
                plane,
                eval_started.elapsed().as_millis(),
                run_started.elapsed().as_millis()
            );
        }

        let available = outputs
            .keys()
            .map(ToString::to_string)
            .collect::<Vec<String>>()
            .join(", ");

        let primary_name = self
            .output_names
            .first()
            .ok_or_else(|| format!("{plane}: model has no declared graph outputs"))?;

        let primary = outputs.get(primary_name).ok_or_else(|| {
            format!(
                "{plane}: expected output '{primary_name}' not found. Available keys: [{available}]"
            )
        })?;

        let shape = primary.dims().to_vec();
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

        let output_shapes = self
            .output_names
            .iter()
            .filter_map(|name| outputs.get(name))
            .map(|tensor| tensor.dims().to_vec())
            .collect::<Vec<Vec<usize>>>();

        let primary_f32 = if primary.dtype() == DType::F32 {
            primary.clone()
        } else {
            primary
                .to_dtype(DType::F32)
                .map_err(|error| format!("{plane}: failed to cast output tensor to f32: {error}"))?
        };

        let logits = primary_f32
            .flatten_all()
            .map_err(|error| format!("{plane}: failed to flatten output tensor: {error}"))?
            .to_vec1::<f32>()
            .map_err(|error| format!("{plane}: failed to read output tensor as f32: {error}"))?;

        Ok(PlaneRunResult {
            logits,
            shape_chw: [shape[1], shape[2], shape[3]],
            output_shapes,
        })
    }
}

fn load_runnable_model(path: &str, plane: &str) -> Result<PlaneSession, String> {
    let mut model = read_file(path)
        .map_err(|error| format!("Failed to parse {plane} ONNX model at '{path}': {error}"))?;

    hydrate_external_initializers(path, &mut model, plane)?;
    rewrite_scalar_prelu_nodes(&mut model);
    rewrite_max_nodes(&mut model);
    rewrite_maxpool_extra_outputs(&mut model);

    let graph = model.graph.as_ref().ok_or_else(|| {
        format!("Failed to load {plane} ONNX model at '{path}': graph is missing")
    })?;

    let initializers = graph
        .initializer
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<HashSet<String>>();

    let required_input_names = graph
        .input
        .iter()
        .filter(|input| !initializers.contains(&input.name))
        .map(|input| input.name.clone())
        .collect::<Vec<String>>();

    let required_input_shapes = graph
        .input
        .iter()
        .filter(|input| !initializers.contains(&input.name))
        .map(|input| (input.name.clone(), dims_from_value_info(input)))
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
        .ok_or_else(|| format!("Failed to discover {plane} ONNX input name at '{path}'"))?;

    let output_names = graph
        .output
        .iter()
        .map(|output| output.name.clone())
        .collect::<Vec<String>>();

    if output_names.is_empty() {
        return Err(format!(
            "Failed to discover {plane} ONNX output names at '{path}'"
        ));
    }

    let input_shape = graph
        .input
        .iter()
        .find(|input| input.name == input_name)
        .and_then(dims_from_value_info);

    let output_shapes = output_names
        .iter()
        .map(|name| {
            graph
                .output
                .iter()
                .find(|output| &output.name == name)
                .and_then(dims_from_value_info)
        })
        .collect::<Vec<Option<Vec<usize>>>>();

    Ok(PlaneSession {
        model: Arc::new(model),
        model_path: path.to_string(),
        input_name,
        required_input_names,
        required_input_shapes,
        output_names,
        input_shape,
        output_shapes,
    })
}

fn hydrate_external_initializers(
    model_path: &str,
    model: &mut onnx::ModelProto,
    plane: &str,
) -> Result<(), String> {
    let Some(graph) = model.graph.as_mut() else {
        return Ok(());
    };

    let model_dir = Path::new(model_path)
        .parent()
        .ok_or_else(|| format!("{plane}: invalid model path '{model_path}'"))?;

    let external_tag = onnx::tensor_proto::DataLocation::External as i32;

    for tensor in &mut graph.initializer {
        if tensor.data_location != external_tag {
            continue;
        }
        if !tensor.raw_data.is_empty() {
            continue;
        }

        let mut location: Option<String> = None;
        let mut offset: u64 = 0;
        let mut length: Option<u64> = None;

        for entry in &tensor.external_data {
            match entry.key.as_str() {
                "location" => location = Some(entry.value.clone()),
                "offset" => {
                    offset = entry.value.parse::<u64>().map_err(|error| {
                        format!(
                            "{plane}: invalid external_data offset '{}' for tensor '{}': {error}",
                            entry.value, tensor.name
                        )
                    })?
                }
                "length" => {
                    length = Some(entry.value.parse::<u64>().map_err(|error| {
                        format!(
                            "{plane}: invalid external_data length '{}' for tensor '{}': {error}",
                            entry.value, tensor.name
                        )
                    })?)
                }
                _ => {}
            }
        }

        let location = location.ok_or_else(|| {
            format!(
                "{plane}: missing external_data location for tensor '{}' in '{}'",
                tensor.name, model_path
            )
        })?;

        let external_path = model_dir.join(location);
        let mut file = std::fs::File::open(&external_path).map_err(|error| {
            format!(
                "{plane}: failed opening external data file '{}' for tensor '{}': {error}",
                external_path.display(),
                tensor.name
            )
        })?;

        file.seek(SeekFrom::Start(offset)).map_err(|error| {
            format!(
                "{plane}: failed seeking external data file '{}' at offset {} for tensor '{}': {error}",
                external_path.display(),
                offset,
                tensor.name
            )
        })?;

        let mut bytes = Vec::<u8>::new();
        match length {
            Some(len) => {
                bytes.resize(len as usize, 0u8);
                file.read_exact(&mut bytes).map_err(|error| {
                    format!(
                        "{plane}: failed reading {} bytes from '{}' for tensor '{}': {error}",
                        len,
                        external_path.display(),
                        tensor.name
                    )
                })?;
            }
            None => {
                file.read_to_end(&mut bytes).map_err(|error| {
                    format!(
                        "{plane}: failed reading external data '{}' for tensor '{}': {error}",
                        external_path.display(),
                        tensor.name
                    )
                })?;
            }
        }

        tensor.raw_data = bytes;
    }

    Ok(())
}

fn rewrite_scalar_prelu_nodes(model: &mut onnx::ModelProto) {
    let Some(graph) = model.graph.as_mut() else {
        return;
    };

    let mut scalar_slopes = HashMap::<String, f32>::new();
    for tensor in &graph.initializer {
        if let Some(value) = scalar_f32_tensor_value(tensor) {
            scalar_slopes.insert(tensor.name.clone(), value);
        }
    }

    for node in &mut graph.node {
        if node.op_type != "PRelu" || node.input.len() < 2 {
            continue;
        }

        let slope_name = node.input[1].clone();
        let Some(alpha) = scalar_slopes.get(&slope_name).copied() else {
            continue;
        };

        let x_input = node.input[0].clone();
        node.op_type = "LeakyRelu".to_string();
        node.input = vec![x_input];
        node.attribute = vec![onnx::AttributeProto {
            name: "alpha".to_string(),
            ref_attr_name: String::new(),
            doc_string: String::new(),
            r#type: onnx::attribute_proto::AttributeType::Float as i32,
            f: alpha,
            i: 0,
            s: Vec::new(),
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: Vec::new(),
            ints: Vec::new(),
            strings: Vec::new(),
            tensors: Vec::new(),
            graphs: Vec::new(),
            sparse_tensors: Vec::new(),
            type_protos: Vec::new(),
        }];
    }
}

fn scalar_f32_tensor_value(tensor: &onnx::TensorProto) -> Option<f32> {
    let element_count = tensor
        .dims
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul((*dim).max(0) as usize))?;
    if element_count != 1 {
        return None;
    }

    if let Some(value) = tensor.float_data.first() {
        return Some(*value);
    }

    if tensor.raw_data.len() >= 4 {
        let bytes: [u8; 4] = tensor.raw_data[0..4].try_into().ok()?;
        return Some(f32::from_le_bytes(bytes));
    }

    if let Some(value) = tensor.double_data.first() {
        return Some(*value as f32);
    }

    if let Some(value) = tensor.int32_data.first() {
        return Some(*value as f32);
    }

    if let Some(value) = tensor.int64_data.first() {
        return Some(*value as f32);
    }

    None
}

fn rewrite_max_nodes(model: &mut onnx::ModelProto) {
    let Some(graph) = model.graph.as_mut() else {
        return;
    };

    let mut rewritten_nodes = Vec::<onnx::NodeProto>::with_capacity(graph.node.len());
    let mut rewrite_index = 0usize;

    for node in graph.node.iter() {
        if node.op_type != "Max" || node.input.len() < 2 {
            rewritten_nodes.push(node.clone());
            continue;
        }

        let node_name = if node.name.is_empty() {
            format!("max_{rewrite_index}")
        } else {
            node.name.clone()
        };
        rewrite_index += 1;

        let final_value = format!("{node_name}_value");

        let mut current = node.input[0].clone();
        for (pair_idx, next_input) in node.input.iter().skip(1).enumerate() {
            let is_last = pair_idx + 1 == node.input.len() - 1;
            let cond_output = format!("{node_name}_cond_{pair_idx}");
            let where_output = if is_last {
                final_value.clone()
            } else {
                format!("{node_name}_tmp_{pair_idx}")
            };

            rewritten_nodes.push(onnx::NodeProto {
                input: vec![current.clone(), next_input.clone()],
                output: vec![cond_output.clone()],
                name: format!("{node_name}_greater_{pair_idx}"),
                op_type: "Greater".to_string(),
                domain: String::new(),
                attribute: Vec::new(),
                doc_string: String::new(),
            });

            rewritten_nodes.push(onnx::NodeProto {
                input: vec![cond_output, current.clone(), next_input.clone()],
                output: vec![where_output.clone()],
                name: format!("{node_name}_where_{pair_idx}"),
                op_type: "Where".to_string(),
                domain: String::new(),
                attribute: Vec::new(),
                doc_string: String::new(),
            });

            current = where_output;
        }

        if node.output.is_empty() {
            rewritten_nodes.push(onnx::NodeProto {
                input: vec![final_value.clone()],
                output: vec![format!("{node_name}_out")],
                name: format!("{node_name}_identity_out"),
                op_type: "Identity".to_string(),
                domain: String::new(),
                attribute: Vec::new(),
                doc_string: String::new(),
            });
        } else {
            for (output_idx, output_name) in node.output.iter().enumerate() {
                rewritten_nodes.push(onnx::NodeProto {
                    input: vec![final_value.clone()],
                    output: vec![output_name.clone()],
                    name: format!("{node_name}_identity_{output_idx}"),
                    op_type: "Identity".to_string(),
                    domain: String::new(),
                    attribute: Vec::new(),
                    doc_string: String::new(),
                });
            }
        }
    }

    graph.node = rewritten_nodes;
}

fn rewrite_maxpool_extra_outputs(model: &mut onnx::ModelProto) {
    let Some(graph) = model.graph.as_mut() else {
        return;
    };

    let mut rewritten_nodes = Vec::<onnx::NodeProto>::with_capacity(graph.node.len());

    for node in graph.node.iter() {
        if node.op_type != "MaxPool" || node.output.len() <= 1 {
            rewritten_nodes.push(node.clone());
            continue;
        }

        let primary_output = node.output[0].clone();
        let mut primary_node = node.clone();
        primary_node.output = vec![primary_output.clone()];
        rewritten_nodes.push(primary_node);

        let node_name = if node.name.is_empty() {
            "maxpool".to_string()
        } else {
            node.name.clone()
        };

        for (output_idx, extra_output) in node.output.iter().skip(1).enumerate() {
            let zeros_like_output = format!("{node_name}_extra_zeros_like_{output_idx}");
            let zeros_i64_output = format!("{node_name}_extra_zeros_i64_{output_idx}");
            rewritten_nodes.push(onnx::NodeProto {
                input: vec![primary_output.clone(), primary_output.clone()],
                output: vec![zeros_like_output.clone()],
                name: format!("{node_name}_extra_zero_like_op_{output_idx}"),
                op_type: "Sub".to_string(),
                domain: String::new(),
                attribute: Vec::new(),
                doc_string: String::new(),
            });

            rewritten_nodes.push(onnx::NodeProto {
                input: vec![zeros_like_output],
                output: vec![zeros_i64_output.clone()],
                name: format!("{node_name}_extra_cast_i64_op_{output_idx}"),
                op_type: "Cast".to_string(),
                domain: String::new(),
                attribute: vec![onnx::AttributeProto {
                    name: "to".to_string(),
                    ref_attr_name: String::new(),
                    doc_string: String::new(),
                    r#type: onnx::attribute_proto::AttributeType::Int as i32,
                    f: 0.0,
                    i: onnx::tensor_proto::DataType::Int64 as i64,
                    s: Vec::new(),
                    t: None,
                    g: None,
                    sparse_tensor: None,
                    tp: None,
                    floats: Vec::new(),
                    ints: Vec::new(),
                    strings: Vec::new(),
                    tensors: Vec::new(),
                    graphs: Vec::new(),
                    sparse_tensors: Vec::new(),
                    type_protos: Vec::new(),
                }],
                doc_string: String::new(),
            });

            rewritten_nodes.push(onnx::NodeProto {
                input: vec![zeros_i64_output],
                output: vec![extra_output.clone()],
                name: format!("{node_name}_extra_identity_{output_idx}"),
                op_type: "Identity".to_string(),
                domain: String::new(),
                attribute: Vec::new(),
                doc_string: String::new(),
            });
        }
    }

    graph.node = rewritten_nodes;
}

fn dims_from_value_info(value: &onnx::ValueInfoProto) -> Option<Vec<usize>> {
    let type_proto = value.r#type.as_ref()?;
    let tensor_type = match type_proto.value.as_ref()? {
        onnx::type_proto::Value::TensorType(tt) => tt,
        _ => return None,
    };
    let shape = tensor_type.shape.as_ref()?;

    shape
        .dim
        .iter()
        .map(|dim| match dim.value.as_ref()? {
            onnx::tensor_shape_proto::dimension::Value::DimValue(v) => Some(*v as usize),
            onnx::tensor_shape_proto::dimension::Value::DimParam(_) => None,
        })
        .collect::<Option<Vec<usize>>>()
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
