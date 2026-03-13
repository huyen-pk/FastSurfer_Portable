use crate::inference::entities::InputVolume;
use crate::inference::pipeline::postprocess;
use ndarray::Array3;
use nifti::writer::WriterOptions;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

pub(crate) fn create_native_output_dir() -> Result<PathBuf, String> {
    let epoch_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
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

pub(crate) fn is_supported_nifti_path(path: &Path) -> bool {
    path.extension().is_some_and(|ext| {
        ext.eq_ignore_ascii_case("nii")
            || ext.eq_ignore_ascii_case("mgz")
            || ext.eq_ignore_ascii_case("mgh")
    }) || path
        .to_string_lossy()
        .to_ascii_lowercase()
        .ends_with(".nii.gz")
}

pub(crate) fn collect_nifti_files_recursive(dir: &Path, out: &mut Vec<String>) {
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

pub(crate) fn discover_lut_path() -> Option<PathBuf> {
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

pub(crate) fn load_lut_ids() -> Result<Vec<u16>, String> {
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

pub(crate) fn map_label_indices_to_lut(
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

pub(crate) fn write_pred_nifti(
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

pub(crate) fn write_u8_nifti(
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

pub(crate) fn write_prediction_artifacts(
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

pub(crate) fn validate_inputs(
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
            "Rust native inference requires at least one valid input path (.nii/.nii.gz/.mgz/.mgh) from files or folders.".to_string(),
        );
    }

    resolved.sort();
    resolved.dedup();

    Ok(resolved)
}
