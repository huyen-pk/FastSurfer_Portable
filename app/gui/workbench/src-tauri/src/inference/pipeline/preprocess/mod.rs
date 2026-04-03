#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::similar_names
)]

mod conform;

pub use crate::inference::entities::{
    InferencePlane, InputVolume, PreparedPlaneInput,
};
pub use conform::{
    FastSurferConformOptions, FastSurferImageSize, FastSurferOrientation,
    FastSurferVoxSize, load_and_conform_input_volume,
    load_and_conform_input_volume_with_options, load_input_volume,
};

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

/// Prepare a single slice tensor for one inference plane.
///
/// # Errors
/// Returns an error if the transformed volume shape is invalid or the slice
/// index is out of bounds.
pub fn prepare_plane_input_for_slice(
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

#[must_use]
pub fn transformed_volume_shape(
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
