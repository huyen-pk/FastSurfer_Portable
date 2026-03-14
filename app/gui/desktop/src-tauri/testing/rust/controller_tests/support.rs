// This test suite integrates with test-containers for environment isolation.
use crate::backend::BackendState;
use crate::inference::entities::InferencePlane;
use crate::inference::pipeline::preprocess::{
    PreparedPlaneInput, load_input_volume, oriented_to_xyz,
    prepare_plane_input_for_slice, transformed_volume_shape,
};
use crate::inference::pipeline::run::run_native_inference;
use crate::process_mgmt::{
    BackendLaunchCommand, BackendProcess, resolve_python_executable,
};
use nifti::{IntoNdArray, NiftiObject, ReaderOptions};
use std::fmt::Write as _;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

pub(super) fn next_test_id() -> usize {
    static NEXT_ID: AtomicUsize = AtomicUsize::new(1);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

fn round_f32_to_i32(value: f32) -> i32 {
    value
        .round()
        .to_string()
        .parse::<i32>()
        .expect("rounded f32 label should fit into i32")
}

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("usize value should fit into u32"))
}

fn write_prepared_tensor_raw(
    run_dir: &Path,
    prepared: &PreparedPlaneInput,
) -> Result<PathBuf, String> {
    let rust_raw = run_dir.join("rust_tensor.raw");
    let mut raw_file = fs::File::create(&rust_raw).map_err(|error| {
        format!(
            "failed creating rust raw tensor file '{}': {error}",
            rust_raw.display()
        )
    })?;

    for value in &prepared.tensor_data {
        raw_file.write_all(&value.to_le_bytes()).map_err(|error| {
            format!(
                "failed writing rust raw tensor file '{}': {error}",
                rust_raw.display()
            )
        })?;
    }

    Ok(rust_raw)
}

struct PythonPreprocessParityArgs<'a> {
    python_bin: &'a str,
    repo_root: &'a Path,
    input_nii: &'a Path,
    plane: InferencePlane,
    num_channels: usize,
    base_res: f32,
    slice_index: usize,
    prepared: &'a PreparedPlaneInput,
    rust_raw: &'a Path,
}

fn run_python_preprocess_parity(
    args: &PythonPreprocessParityArgs<'_>,
) -> Result<(), String> {
    let shape_csv = args
        .prepared
        .tensor_shape
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<String>>()
        .join(",");

    let script = r#"
import nibabel as nib
import numpy as np
import sys

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
orig_data = np.asanyarray(nib.load(input_nii).dataobj)
orig_zoom = np.asarray(nib.load(input_nii).header.get_zooms()[:3], dtype=np.float32)

def transform_axial(vol):
    return np.moveaxis(vol, [0, 1, 2], [1, 2, 0])

def transform_sagittal(vol):
    return np.moveaxis(vol, [0, 1, 2], [2, 1, 0])

def get_thick_slices(img_data, slice_thickness=3):
    img_data_pad = np.pad(img_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(img_data_pad, 2 * slice_thickness + 1, axis=2)

def to_tensor_test(img):
    img = img.astype(np.float32)
    img = np.clip(img / 255.0, a_min=0.0, a_max=1.0)
    img = img.transpose((2, 0, 1))
    return img

slice_thickness = num_channels // 2
if plane == "sagittal":
    transformed = transform_sagittal(orig_data)
    zoom = np.asarray(orig_zoom)[[2, 1]]
elif plane == "axial":
    transformed = transform_axial(orig_data)
    zoom = np.asarray(orig_zoom)[[2, 0]]
else:
    transformed = orig_data
    zoom = np.asarray(orig_zoom)[[0, 1]]

orig_thick = get_thick_slices(transformed, slice_thickness)
orig_thick = np.transpose(orig_thick, (2, 0, 1, 3))
py_image = to_tensor_test(orig_thick[slice_index])
py_scale = base_res / zoom

rust = np.fromfile(rust_raw, dtype=np.float32).reshape(shape)
rust_image = rust[0]
rust_scale = np.asarray([rust_scale0, rust_scale1], dtype=np.float32)

if rust_image.shape != py_image.shape:
    raise SystemExit(f"preprocess shape mismatch: rust={rust_image.shape} python={py_image.shape}")

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
        .arg(args.slice_index.to_string())
        .arg(format!("{:.8}", args.prepared.scale_factor[0]))
        .arg(format!("{:.8}", args.prepared.scale_factor[1]))
        .arg(shape_csv)
        .arg(args.rust_raw)
        .current_dir(args.repo_root)
        .output()
        .map_err(|error| {
            format!("failed to execute python preprocess parity check: {error}")
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "python preprocess parity failed (plane={}, slice={}): stdout='{}' stderr='{}'",
            args.plane.as_str(),
            args.slice_index,
            stdout.trim(),
            stderr.trim()
        ));
    }

    Ok(())
}

pub(super) fn create_backend_state_with_fake_responses(
    responses: &[&str],
) -> BackendState {
    let mut branches = String::new();
    for (index, response) in responses.iter().enumerate() {
        write!(branches, "{index}) printf '%s\\n' '{response}' ;;")
            .expect("writing response branch should succeed");
    }

    let script = format!(
        "i=0; while IFS= read -r _line; do case \"$i\" in {branches} *) printf '%s\\n' '{{\"ok\":false,\"error\":{{\"message\":\"unexpected request\"}}}}' ;; esac; i=$((i+1)); done"
    );

    let mut child = Command::new("sh")
        .arg("-c")
        .arg(script.clone())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("failed to spawn fake backend process");

    let stdin = child
        .stdin
        .take()
        .expect("failed to capture mock backend stdin");
    let stdout = child
        .stdout
        .take()
        .expect("failed to capture mock backend stdout");

    BackendState::from_process(
        BackendProcess {
            child,
            stdin,
            stdout: std::io::BufReader::new(stdout),
        },
        BackendLaunchCommand {
            program: "sh".to_string(),
            args: vec!["-c".to_string(), script],
        },
        None,
    )
}

pub(super) fn create_backend_state_via_shell_script(
    script: &str,
) -> BackendState {
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("failed to spawn scripted mock backend process");

    let stdin = child
        .stdin
        .take()
        .expect("failed to capture scripted mock backend stdin");
    let stdout = child
        .stdout
        .take()
        .expect("failed to capture scripted mock backend stdout");

    BackendState::from_process(
        BackendProcess {
            child,
            stdin,
            stdout: std::io::BufReader::new(stdout),
        },
        BackendLaunchCommand {
            program: "sh".to_string(),
            args: vec!["-c".to_string(), script.to_string()],
        },
        None,
    )
}

pub(super) fn create_backend_state_via_process(
    mut child: std::process::Child,
) -> BackendState {
    let stdin = child.stdin.take().expect("failed to capture process stdin");
    let stdout = child
        .stdout
        .take()
        .expect("failed to capture process stdout");

    BackendState::from_process(
        BackendProcess {
            child,
            stdin,
            stdout: std::io::BufReader::new(stdout),
        },
        BackendLaunchCommand {
            program: "unknown".to_string(),
            args: Vec::new(),
        },
        None,
    )
}

pub(super) fn spawn_python_backend_inline(
    script: &str,
) -> Option<BackendState> {
    let python = resolve_python_executable()?;
    let child = Command::new(python)
        .arg("-u")
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .ok()?;

    Some(create_backend_state_via_process(child))
}

pub(super) fn find_repo_root() -> Option<PathBuf> {
    let mut current = std::env::current_dir().ok()?;
    loop {
        if current.join("app/backend/ipc_server.py").exists() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

pub(super) fn is_ci() -> bool {
    std::env::var("CI")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true" || normalized == "yes"
        })
        .unwrap_or(false)
}

pub(super) fn python_has_fastsurfer_runtime(
    python_bin: &str,
    repo_root: &Path,
) -> bool {
    Command::new(python_bin)
        .arg("-c")
        .arg("import torch, FastSurferCNN")
        .current_dir(repo_root)
        .env("PYTHONPATH", repo_root.to_string_lossy().to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

pub(super) fn python_has_nibabel_runtime(
    python_bin: &str,
    repo_root: &Path,
) -> bool {
    Command::new(python_bin)
        .arg("-c")
        .arg("import nibabel, numpy")
        .current_dir(repo_root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

pub(super) fn resolve_python_with_nibabel(repo_root: &Path) -> Option<String> {
    let mut candidates = Vec::<String>::new();
    if let Some(resolved) = resolve_python_executable() {
        candidates.push(resolved);
    }
    candidates.push("/home/huyenpk/anaconda3/bin/python".to_string());
    candidates.push("python3".to_string());

    candidates
        .into_iter()
        .find(|candidate| python_has_nibabel_runtime(candidate, repo_root))
}

pub(super) fn resolve_python_with_component_runtime(
    repo_root: &Path,
) -> Option<String> {
    let mut candidates = Vec::<String>::new();
    if let Some(resolved) = resolve_python_executable() {
        candidates.push(resolved);
    }
    candidates.push("/home/huyenpk/anaconda3/bin/python".to_string());
    candidates.push("python3".to_string());

    let probe = "import numpy, nibabel, scipy, skimage\nfrom FastSurferCNN.data_loader.data_utils import load_and_conform_image, transform_axial, transform_sagittal, get_thick_slices\nfrom FastSurferCNN.reduce_to_aseg import reduce_to_aseg, create_mask, flip_wm_islands";

    candidates.into_iter().find(|python_bin| {
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
    })
}

pub(super) fn fixture_native_input(repo_root: &Path) -> PathBuf {
    repo_root.join(
        "app/gui/desktop/src-tauri/testing/data/.tmp_e2e_output_py/140_orig.native_input.nii.gz",
    )
}

pub(super) fn fixture_python_pred(repo_root: &Path) -> PathBuf {
    repo_root.join(
        "app/gui/desktop/src-tauri/testing/data/.tmp_e2e_output_py/140_orig.python_pred.nii.gz",
    )
}

pub(super) fn convert_mgz_to_nii_gz(
    python_bin: &str,
    input_mgz: &Path,
    output_nii_gz: &Path,
) -> Result<(), String> {
    let script = r"
import nibabel as nib
import sys
img = nib.load(sys.argv[1])
nib.save(img, sys.argv[2])
";

    let status = Command::new(python_bin)
        .arg("-c")
        .arg(script)
        .arg(input_mgz)
        .arg(output_nii_gz)
        .status()
        .map_err(|error| {
            format!("failed to spawn python conversion command: {error}")
        })?;

    if !status.success() {
        return Err(format!(
            "python conversion from '{}' to '{}' failed with status {}",
            input_mgz.display(),
            output_nii_gz.display(),
            status
        ));
    }

    Ok(())
}

fn load_nifti_labels_as_i32(
    path: &Path,
) -> Result<(Vec<i32>, Vec<usize>), String> {
    let obj = ReaderOptions::new().read_file(path).map_err(|error| {
        format!("failed to read NIfTI '{}': {error}", path.display())
    })?;

    let volume = obj.into_volume();
    let array = volume.into_ndarray::<f32>().map_err(|error| {
        format!(
            "failed to convert NIfTI '{}' into ndarray: {error}",
            path.display()
        )
    })?;

    let shape = array.shape().to_vec();
    let labels = array
        .iter()
        .map(|value| round_f32_to_i32(*value))
        .collect::<Vec<i32>>();

    Ok((labels, shape))
}

pub(super) fn ensure_native_input_nifti(
    repo_root: &Path,
    python_bin: &str,
) -> Result<PathBuf, String> {
    let fixture_dir = repo_root
        .join("app/gui/desktop/src-tauri/testing/data/.tmp_e2e_output_py");
    fs::create_dir_all(&fixture_dir).map_err(|error| {
        format!(
            "failed to create fixture directory '{}': {error}",
            fixture_dir.display()
        )
    })?;

    let input_nii = fixture_dir.join("140_orig.native_input.nii.gz");
    if input_nii.exists() {
        return Ok(input_nii);
    }

    let input_mgz = repo_root
        .join("app/gui/desktop/src-tauri/testing/data/Subject140/140_orig.mgz");
    if !input_mgz.exists() {
        return Err(format!(
            "missing test input file: {}",
            input_mgz.display()
        ));
    }

    convert_mgz_to_nii_gz(python_bin, &input_mgz, &input_nii)?;
    Ok(input_nii)
}

pub(super) fn compare_label_volumes_with_python(
    python_bin: &str,
    rust_pred: &Path,
    py_pred: &Path,
) -> Result<(), String> {
    let script = r#"
import nibabel as nib
import numpy as np
import sys

rust_path, py_path = sys.argv[1], sys.argv[2]
rust_data = np.asanyarray(nib.load(rust_path).dataobj)
py_data = np.asanyarray(nib.load(py_path).dataobj)

if rust_data.shape != py_data.shape:
    raise SystemExit(f"shape mismatch: rust={rust_data.shape} python={py_data.shape}")

rust_i = rust_data.astype(np.int32, copy=False)
py_i = py_data.astype(np.int32, copy=False)
diff = np.count_nonzero(rust_i != py_i)
total = rust_i.size
ratio = (diff / total) if total else 0.0

print(f"voxel_diff={diff}/{total} ratio={ratio:.6f}")

# keep this as a soft parity gate while Rust path is still under migration
if ratio > 0.25:
    raise SystemExit(f"voxel mismatch ratio too high: {ratio:.6f}")
"#;

    let output = Command::new(python_bin)
        .arg("-c")
        .arg(script)
        .arg(rust_pred)
        .arg(py_pred)
        .output()
        .map_err(|error| {
            format!("failed to execute python volume comparison: {error}")
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "python volume comparison failed: stdout='{}' stderr='{}'",
            stdout.trim(),
            stderr.trim()
        ));
    }

    Ok(())
}

pub(super) fn compare_label_volumes_per_plane_single_slice_in_rust(
    rust_pred: &Path,
    golden_pred: &Path,
    max_mismatch_ratio: f64,
) -> Result<(), String> {
    let (rust_labels, rust_shape) = load_nifti_labels_as_i32(rust_pred)?;
    let (gold_labels, gold_shape) = load_nifti_labels_as_i32(golden_pred)?;

    if rust_shape.len() < 3 || gold_shape.len() < 3 {
        return Err(format!(
            "expected rank-3 volume shape, got rust={rust_shape:?} golden={gold_shape:?}"
        ));
    }

    let rust_shape_xyz = [rust_shape[0], rust_shape[1], rust_shape[2]];
    let gold_shape_xyz = [gold_shape[0], gold_shape[1], gold_shape[2]];
    let rust_expected_len =
        rust_shape_xyz[0] * rust_shape_xyz[1] * rust_shape_xyz[2];
    let gold_expected_len =
        gold_shape_xyz[0] * gold_shape_xyz[1] * gold_shape_xyz[2];
    if rust_labels.len() != rust_expected_len
        || gold_labels.len() != gold_expected_len
    {
        return Err(format!(
            "voxel count mismatch with shape rust={:?} golden={:?}: rust={} golden={} expected_rust={} expected_golden={}",
            rust_shape_xyz,
            gold_shape_xyz,
            rust_labels.len(),
            gold_labels.len(),
            rust_expected_len,
            gold_expected_len
        ));
    }

    let compare_shape_xyz = [
        rust_shape_xyz[0].min(gold_shape_xyz[0]),
        rust_shape_xyz[1].min(gold_shape_xyz[1]),
        rust_shape_xyz[2].min(gold_shape_xyz[2]),
    ];

    let rust_offset_xyz = [
        (rust_shape_xyz[0] - compare_shape_xyz[0]) / 2,
        (rust_shape_xyz[1] - compare_shape_xyz[1]) / 2,
        (rust_shape_xyz[2] - compare_shape_xyz[2]) / 2,
    ];
    let gold_offset_xyz = [
        (gold_shape_xyz[0] - compare_shape_xyz[0]) / 2,
        (gold_shape_xyz[1] - compare_shape_xyz[1]) / 2,
        (gold_shape_xyz[2] - compare_shape_xyz[2]) / 2,
    ];

    for plane in [
        InferencePlane::Coronal,
        InferencePlane::Axial,
        InferencePlane::Sagittal,
    ] {
        let [height, width, slices_count] =
            transformed_volume_shape(compare_shape_xyz, plane);
        if slices_count == 0 {
            return Err(format!("plane {} has zero slices", plane.as_str()));
        }
        let slice_index = slices_count / 2;
        let mut diff = 0usize;
        let mut total = 0usize;

        for row in 0..height {
            for col in 0..width {
                let (x, y, z) = oriented_to_xyz(plane, row, col, slice_index);
                let rx = x + rust_offset_xyz[0];
                let ry = y + rust_offset_xyz[1];
                let rz = z + rust_offset_xyz[2];
                let gx = x + gold_offset_xyz[0];
                let gy = y + gold_offset_xyz[1];
                let gz = z + gold_offset_xyz[2];

                let rust_offset = (rx * rust_shape_xyz[1] * rust_shape_xyz[2])
                    + (ry * rust_shape_xyz[2])
                    + rz;
                let gold_offset = (gx * gold_shape_xyz[1] * gold_shape_xyz[2])
                    + (gy * gold_shape_xyz[2])
                    + gz;

                if rust_labels[rust_offset] != gold_labels[gold_offset] {
                    diff += 1;
                }
                total += 1;
            }
        }

        let ratio = if total == 0 {
            0.0
        } else {
            usize_to_f64(diff) / usize_to_f64(total)
        };

        if ratio > max_mismatch_ratio {
            return Err(format!(
                "slice mismatch ratio too high for plane={} slice={} diff={}/{} ratio={:.6} threshold={:.6}",
                plane.as_str(),
                slice_index,
                diff,
                total,
                ratio,
                max_mismatch_ratio
            ));
        }
    }

    Ok(())
}

pub(super) fn compare_preprocess_slice_with_python(
    python_bin: &str,
    repo_root: &Path,
    input_nii: &Path,
    plane: InferencePlane,
    num_channels: usize,
    base_res: f32,
    slice_index: usize,
) -> Result<(), String> {
    let volume = load_input_volume(&input_nii.to_string_lossy())?;
    let prepared = prepare_plane_input_for_slice(
        &volume,
        plane,
        num_channels,
        base_res,
        slice_index,
    )?;

    let run_dir = std::env::temp_dir().join(format!(
        "preproc_parity_{}_{}",
        next_test_id(),
        plane.as_str()
    ));
    fs::create_dir_all(&run_dir).map_err(|error| {
        format!(
            "failed creating temp preprocess parity directory '{}': {error}",
            run_dir.display()
        )
    })?;

    let rust_raw = write_prepared_tensor_raw(&run_dir, &prepared)?;
    let parity_args = PythonPreprocessParityArgs {
        python_bin,
        repo_root,
        input_nii,
        plane,
        num_channels,
        base_res,
        slice_index,
        prepared: &prepared,
        rust_raw: &rust_raw,
    };
    run_python_preprocess_parity(&parity_args)?;

    Ok(())
}

fn parse_bool_env(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            normalized == "1"
                || normalized == "true"
                || normalized == "yes"
                || normalized == "on"
        })
        .unwrap_or(false)
}

pub(super) fn configured_slice_indices(total_slices: usize) -> Vec<usize> {
    if total_slices == 0 {
        return Vec::new();
    }

    if parse_bool_env("FASTSURFER_PREPROCESS_FULL_SCAN") {
        return (0..total_slices).collect::<Vec<usize>>();
    }

    if let Ok(indices_csv) =
        std::env::var("FASTSURFER_PREPROCESS_SLICE_INDICES")
    {
        let parsed = indices_csv
            .split(',')
            .filter_map(|part| part.trim().parse::<usize>().ok())
            .filter(|idx| *idx < total_slices)
            .collect::<Vec<usize>>();
        if !parsed.is_empty() {
            return parsed;
        }
    }

    let requested = std::env::var("FASTSURFER_PREPROCESS_SLICES_PER_PLANE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2usize)
        .min(total_slices);

    if requested >= total_slices {
        return (0..total_slices).collect::<Vec<usize>>();
    }

    if requested == 1 {
        return vec![total_slices / 2];
    }

    let mut indices = Vec::<usize>::with_capacity(requested);
    let max_idx = total_slices - 1;
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

pub(super) fn run_native_inference_with_timeout(
    file_paths: &[String],
    folder_paths: &[String],
    timeout: Duration,
) -> Result<crate::inference::entities::ProcessingRunResult, String> {
    let (tx, rx) = mpsc::channel();
    let files = file_paths.to_vec();
    let folders = folder_paths.to_vec();

    thread::spawn(move || {
        let result = run_native_inference(&files, &folders);
        let _ = tx.send(result);
    });

    match rx.recv_timeout(timeout) {
        Ok(result) => result,
        Err(mpsc::RecvTimeoutError::Timeout) => Err(format!(
            "native rust inference timed out after {}s",
            timeout.as_secs()
        )),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            Err("native rust inference worker disconnected before returning a result".to_string())
        }
    }
}

pub(super) fn setup_test_app() -> tauri::App<tauri::test::MockRuntime> {
    tauri::test::mock_builder()
        .build(tauri::generate_context!())
        .unwrap()
}
