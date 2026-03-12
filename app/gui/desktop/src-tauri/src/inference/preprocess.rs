use nifti::{IntoNdArray, NiftiObject, ReaderOptions};
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub use crate::models::{InferencePlane, InputVolume, PreparedPlaneInput};

fn is_supported_native_input_path(path: &str) -> bool {
    let p = std::path::Path::new(path);
    if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
        if ext.eq_ignore_ascii_case("nii")
            || ext.eq_ignore_ascii_case("mgz")
            || ext.eq_ignore_ascii_case("mgh")
        {
            return true;
        }
    }
    // fallback for .nii.gz style names
    if let Some(fname) = p.file_name().and_then(|s| s.to_str()) {
        return fname.to_ascii_lowercase().ends_with(".nii.gz");
    }
    false
}

fn is_mgz_or_mgh_path(path: &str) -> bool {
    let p = std::path::Path::new(path);
    if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
        return ext.eq_ignore_ascii_case("mgz")
            || ext.eq_ignore_ascii_case("mgh");
    }
    // fallback for .mgz/.mgh in filenames
    if let Some(fname) = p.file_name().and_then(|s| s.to_str()) {
        return fname.to_ascii_lowercase().ends_with(".mgz")
            || fname.to_ascii_lowercase().ends_with(".mgh");
    }
    false
}

fn to_temp_nifti_path() -> Result<PathBuf, String> {
    let epoch_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            format!(
                "System clock error while creating temp NIfTI path: {error}"
            )
        })?
        .as_nanos();

    Ok(std::env::temp_dir()
        .join(format!("fastsurfer_native_input_{epoch_ns}.nii.gz")))
}

fn run_mri_convert(input_path: &str, output_path: &str) -> Result<(), String> {
    let program = std::env::var("FASTSURFER_MRI_CONVERT_BIN")
        .unwrap_or_else(|_| "mri_convert".to_string());
    let status = Command::new(&program)
        .arg(input_path)
        .arg(output_path)
        .status()
        .map_err(|error| {
            format!("Failed to execute '{program}' for MGZ conversion: {error}")
        })?;

    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "'{program}' returned non-zero exit status during MGZ conversion"
        ))
    }
}

fn run_python_nibabel_convert(
    input_path: &str,
    output_path: &str,
) -> Result<(), String> {
    let python_bin = std::env::var("FASTSURFER_PYTHON_BIN")
        .unwrap_or_else(|_| "python3".to_string());
    let script = [
        "import nibabel as nib",
        "import sys",
        "img = nib.load(sys.argv[1])",
        "nib.save(img, sys.argv[2])",
    ]
    .join("; ");

    let status = Command::new(&python_bin)
        .arg("-c")
        .arg(script)
        .arg(input_path)
        .arg(output_path)
        .status()
        .map_err(|error| {
            format!(
                "Failed to execute '{python_bin}' for MGZ conversion: {error}"
            )
        })?;

    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "'{python_bin}' returned non-zero exit status during nibabel MGZ conversion"
        ))
    }
}

fn convert_mgz_to_nifti(input_path: &str) -> Result<PathBuf, String> {
    let output_path = to_temp_nifti_path()?;
    let output_str = output_path.to_string_lossy().to_string();

    match run_mri_convert(input_path, &output_str) {
        Ok(()) => return Ok(output_path),
        Err(_error) => {}
    }

    run_python_nibabel_convert(input_path, &output_str)
        .map(|()| output_path)
        .map_err(|error| {
            format!(
                "Failed to convert MGZ/MGH input '{input_path}' to NIfTI. Tried mri_convert and python nibabel fallback. Last error: {error}"
            )
        })
}

pub(crate) fn load_input_volume(path: &str) -> Result<InputVolume, String> {
    if !is_supported_native_input_path(path) {
        return Err(format!(
            "Native Rust inference supports .nii/.nii.gz and .mgz/.mgh input, got: {path}"
        ));
    }

    let mut converted_temp_file: Option<PathBuf> = None;
    let load_path = if is_mgz_or_mgh_path(path) {
        let converted = convert_mgz_to_nifti(path)?;
        let converted_str = converted.to_string_lossy().to_string();
        converted_temp_file = Some(converted);
        converted_str
    } else {
        path.to_string()
    };

    let obj = ReaderOptions::new()
        .read_file(&load_path)
        .map_err(|error| {
            format!("Failed to read NIfTI file '{load_path}': {error}")
        })?;

    let header = obj.header().clone();
    let volume = obj.into_volume();
    let array = volume
        .into_ndarray::<f32>()
        .map_err(|error| format!("Failed to materialize NIfTI volume '{load_path}' as ndarray: {error}"))?;

    let shape = array.shape();
    if shape.len() < 3 {
        return Err(format!(
            "Input volume must be 3D (or greater with singleton extras), got shape={shape:?}"
        ));
    }

    let sx = shape[0];
    let sy = shape[1];
    let sz = shape[2];
    let shape_xyz = [sx, sy, sz];
    let data_xyz = array
        .iter()
        .copied()
        .take(sx * sy * sz)
        .collect::<Vec<f32>>();

    let zoom_xyz = [header.pixdim[1], header.pixdim[2], header.pixdim[3]];

    if let Some(temp_file) = converted_temp_file {
        let _ = std::fs::remove_file(temp_file);
    }

    Ok(InputVolume {
        data_xyz,
        shape_xyz,
        zoom_xyz,
        header,
    })
}

#[allow(dead_code)]
pub(crate) fn prepare_single_plane_input(
    volume: &InputVolume,
    plane: InferencePlane,
    num_channels: usize,
    base_res: f32,
) -> Result<PreparedPlaneInput, String> {
    let [_, _, count] = transformed_volume_shape(volume.shape_xyz, plane);
    let center = count / 2;
    prepare_plane_input_for_slice(volume, plane, num_channels, base_res, center)
}

pub(crate) fn prepare_plane_input_for_slice(
    volume: &InputVolume,
    plane: InferencePlane,
    num_channels: usize,
    base_res: f32,
    slice_index: usize,
) -> Result<PreparedPlaneInput, String> {
    let slice_thickness = slice_thickness_from_num_channels(num_channels);
    let channels = thick_slice_channel_count(slice_thickness);
    let [h, w, count] = transformed_volume_shape(volume.shape_xyz, plane);

    if count == 0 || h == 0 || w == 0 {
        return Err(
            "Invalid transformed volume shape encountered in preprocessing"
                .to_string(),
        );
    }

    if slice_index >= count {
        return Err(format!(
            "Requested slice index {slice_index} out of range for plane {} with {count} slices",
            plane.as_str()
        ));
    }

    let mut tensor_data = vec![0f32; channels * h * w];

    for ch in 0..channels {
        let relative = isize::try_from(ch).unwrap_or(0)
            - isize::try_from(slice_thickness).unwrap_or(0);
        let slice = clamp_index(
            isize::try_from(slice_index).unwrap_or(0) + relative,
            count,
        );
        for ih in 0..h {
            for iw in 0..w {
                let voxel = oriented_voxel(volume, plane, ih, iw, slice);
                let normalized = (voxel / 255.0).clamp(0.0, 1.0);
                let offset = (ch * h * w) + (ih * w) + iw;
                tensor_data[offset] = normalized;
            }
        }
    }

    let scale =
        scale_factor(base_res, transformed_zoom(volume.zoom_xyz, plane));

    Ok(PreparedPlaneInput {
        tensor_data,
        tensor_shape: [1, channels, h, w],
        scale_factor: scale,
        plane,
        slice_index,
    })
}

fn oriented_voxel(
    volume: &InputVolume,
    plane: InferencePlane,
    ih: usize,
    iw: usize,
    islice: usize,
) -> f32 {
    let [sx, sy, sz] = volume.shape_xyz;

    let (x, y, z) = oriented_to_xyz(plane, ih, iw, islice);

    if x >= sx || y >= sy || z >= sz {
        return 0.0;
    }

    let index = (x * sy * sz) + (y * sz) + z;
    volume.data_xyz.get(index).copied().unwrap_or(0.0)
}

pub(crate) fn oriented_to_xyz(
    plane: InferencePlane,
    ih: usize,
    iw: usize,
    islice: usize,
) -> (usize, usize, usize) {
    match plane {
        InferencePlane::Coronal => (ih, iw, islice),
        InferencePlane::Axial => (iw, islice, ih),
        InferencePlane::Sagittal => (islice, iw, ih),
    }
}

fn clamp_index(value: isize, upper_exclusive: usize) -> usize {
    if upper_exclusive == 0 {
        return 0;
    }
    if value <= 0 {
        return 0;
    }
    let max_index = upper_exclusive - 1;
    let as_usize = usize::try_from(value).unwrap_or(0);
    if as_usize > max_index {
        max_index
    } else {
        as_usize
    }
}

pub(crate) fn slice_thickness_from_num_channels(num_channels: usize) -> usize {
    num_channels / 2
}

pub(crate) fn thick_slice_channel_count(slice_thickness: usize) -> usize {
    (2 * slice_thickness) + 1
}

pub(crate) fn transformed_volume_shape(
    shape_coronal: [usize; 3],
    plane: InferencePlane,
) -> [usize; 3] {
    let [x, y, z] = shape_coronal;
    match plane {
        InferencePlane::Coronal => [x, y, z],
        InferencePlane::Axial => [z, x, y],
        InferencePlane::Sagittal => [z, y, x],
    }
}

pub(crate) fn transformed_zoom(
    orig_zoom_xyz: [f32; 3],
    plane: InferencePlane,
) -> [f32; 2] {
    let [zx, zy, zz] = orig_zoom_xyz;
    match plane {
        InferencePlane::Coronal => [zx, zy],
        InferencePlane::Axial => [zz, zx],
        InferencePlane::Sagittal => [zz, zy],
    }
}

pub(crate) fn scale_factor(
    base_res: f32,
    transformed_zoom: [f32; 2],
) -> [f32; 2] {
    [
        base_res / transformed_zoom[0],
        base_res / transformed_zoom[1],
    ]
}

pub(crate) fn batched_tensor_shape(
    oriented_shape: [usize; 3],
    slice_thickness: usize,
) -> [usize; 4] {
    let [h, w, count] = oriented_shape;
    let channels = thick_slice_channel_count(slice_thickness);
    [count, channels, h, w]
}

pub(crate) fn build_legacy_preprocessing_trace(
    plane: InferencePlane,
    num_channels: usize,
    base_res: f32,
    orig_zoom_xyz: [f32; 3],
) -> String {
    let slice_thickness = slice_thickness_from_num_channels(num_channels);
    let channels = thick_slice_channel_count(slice_thickness);
    let canonical_coronal_shape = [256usize, 256usize, 256usize];
    let oriented_shape =
        transformed_volume_shape(canonical_coronal_shape, plane);
    let batched_shape = batched_tensor_shape(oriented_shape, slice_thickness);
    let zoom = transformed_zoom(orig_zoom_xyz, plane);
    let scale = scale_factor(base_res, zoom);

    format!(
        concat!(
            "plane={} ",
            "legacy_preproc=",
            "transform_(coronal->plane)",
            " -> get_thick_slices(axis=2, edge-pad, channels={})",
            " -> transpose((2,0,1,3))",
            " -> ToTensorTest(float32, clip(img/255,0,1), transpose((2,0,1)))",
            " | synthetic_oriented_shape={:?}",
            " | synthetic_batched_shape={:?}",
            " | scale_factor=base_res/zoom={}"
        ),
        plane.as_str(),
        channels,
        oriented_shape,
        batched_shape,
        format!("[{:.6}, {:.6}]", scale[0], scale[1])
    )
}
