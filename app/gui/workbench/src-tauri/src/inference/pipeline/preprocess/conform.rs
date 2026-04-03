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

use nifti::{IntoNdArray, NiftiHeader, NiftiObject, ReaderOptions};
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::inference::entities::InputVolume;

const NIFTI_UINT8_DATATYPE: i16 = 2;
const FASTSURFER_CONFORM_VOX_EPS: f64 = 1e-4;
const FASTSURFER_CONFORM_ROT_EPS: f64 = 1e-6;

type Matrix3 = [[f64; 3]; 3];
type Matrix4 = [[f64; 4]; 4];

#[derive(Clone, Copy, Debug)]
pub enum FastSurferVoxSize {
    Min,
    Value(f32),
    Any,
}

#[derive(Clone, Copy, Debug)]
pub enum FastSurferImageSize {
    Auto,
    Fov,
    Value(usize),
    Any,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastSurferOrientation {
    Native,
    Lia,
}

#[derive(Clone, Copy, Debug)]
pub struct FastSurferConformOptions {
    pub vox_size: FastSurferVoxSize,
    pub image_size: FastSurferImageSize,
    pub orientation: FastSurferOrientation,
    pub conform_to_1mm_threshold: Option<f32>,
    pub rescale_max: Option<f32>,
}

impl Default for FastSurferConformOptions {
    fn default() -> Self {
        Self {
            vox_size: FastSurferVoxSize::Min,
            image_size: FastSurferImageSize::Auto,
            orientation: FastSurferOrientation::Lia,
            conform_to_1mm_threshold: Some(0.95),
            rescale_max: Some(255.0),
        }
    }
}

#[derive(Clone)]
struct LoadedInputImage {
    data_xyz: Vec<f32>,
    shape_xyz: [usize; 3],
    zoom_xyz: [f32; 3],
    header: NiftiHeader,
    affine: Matrix4,
}

#[derive(Clone, Copy)]
struct TargetGeometry {
    vox_size: Option<[f32; 3]>,
    shape_xyz: Option<[usize; 3]>,
}

#[derive(Clone, Copy)]
struct RescaleParameters {
    src_min: f64,
    scale: f64,
    dst_min: f64,
    dst_max: f64,
}

/// Load an input volume and conform it with the default FastSurferCNN
/// preprocessing settings used by
/// `RunModelOnData::conform_and_save_orig`.
pub fn load_and_conform_input_volume(
    path: &str,
) -> Result<InputVolume, String> {
    load_and_conform_input_volume_with_options(
        path,
        FastSurferConformOptions::default(),
    )
}

/// Load an input volume and conform it with explicit FastSurfer-compatible
/// preprocessing options.
pub fn load_and_conform_input_volume_with_options(
    path: &str,
    options: FastSurferConformOptions,
) -> Result<InputVolume, String> {
    let loaded = load_native_image_with_affine(path)?;
    if input_is_conformed(&loaded, options)? {
        return Ok(InputVolume {
            data_xyz: loaded.data_xyz,
            shape_xyz: loaded.shape_xyz,
            zoom_xyz: loaded.zoom_xyz,
            header: loaded.header,
        });
    }

    conform_loaded_image(&loaded, options)
}

fn load_native_image_with_affine(
    path: &str,
) -> Result<LoadedInputImage, String> {
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

    let result = (|| {
        let obj =
            ReaderOptions::new()
                .read_file(&load_path)
                .map_err(|error| {
                    format!("Failed to read NIfTI file '{load_path}': {error}")
                })?;

        let header = obj.header().clone();
        let affine = choose_affine_from_header(&header)?;
        let volume = obj.into_volume();
        let array = volume.into_ndarray::<f32>().map_err(|error| {
            format!(
                "Failed to materialize NIfTI volume '{load_path}' as ndarray: {error}"
            )
        })?;

        let full_shape = array.shape().to_vec();
        if full_shape.len() < 3 {
            return Err(format!(
                "Input volume must be 3D (or greater with singleton extras), got shape={full_shape:?}"
            ));
        }
        if full_shape.len() > 3 && full_shape[3..].iter().any(|dim| *dim != 1) {
            return Err(format!(
                "Multiple input frames are not supported, got shape={full_shape:?}"
            ));
        }

        let shape_xyz = [full_shape[0], full_shape[1], full_shape[2]];
        let voxel_count = shape_xyz[0] * shape_xyz[1] * shape_xyz[2];
        let data_xyz = array
            .iter()
            .copied()
            .take(voxel_count)
            .collect::<Vec<f32>>();
        let zoom_xyz = [header.pixdim[1], header.pixdim[2], header.pixdim[3]];

        Ok(LoadedInputImage {
            data_xyz,
            shape_xyz,
            zoom_xyz,
            header,
            affine,
        })
    })();

    if let Some(temp_file) = converted_temp_file {
        let _ = std::fs::remove_file(temp_file);
    }

    result
}

fn choose_affine_from_header(header: &NiftiHeader) -> Result<Matrix4, String> {
    let sform = sform_affine(header);
    let qform = qform_affine(header)?;
    let has_sform = header.sform_code != 0;
    let has_qform = header.qform_code != 0;

    if has_qform && (!has_sform || !matrix4_close(&sform, &qform, 1e-3)) {
        return Ok(qform);
    }

    let affine = if has_sform {
        sform
    } else if has_qform {
        qform
    } else {
        fallback_affine(header)
    };

    let affine_zoom = affine_column_norms(&affine);
    let header_zoom = [
        header.pixdim[1] as f64,
        header.pixdim[2] as f64,
        header.pixdim[3] as f64,
    ];
    if !vector3_close(&affine_zoom, &header_zoom, 1e-3) {
        return Err(format!(
            "Invalid NIfTI header: affine voxel sizes {affine_zoom:?} differ from header {header_zoom:?}"
        ));
    }

    Ok(affine)
}

fn sform_affine(header: &NiftiHeader) -> Matrix4 {
    [
        [
            header.srow_x[0] as f64,
            header.srow_x[1] as f64,
            header.srow_x[2] as f64,
            header.srow_x[3] as f64,
        ],
        [
            header.srow_y[0] as f64,
            header.srow_y[1] as f64,
            header.srow_y[2] as f64,
            header.srow_y[3] as f64,
        ],
        [
            header.srow_z[0] as f64,
            header.srow_z[1] as f64,
            header.srow_z[2] as f64,
            header.srow_z[3] as f64,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn qform_affine(header: &NiftiHeader) -> Result<Matrix4, String> {
    let b = header.quatern_b as f64;
    let c = header.quatern_c as f64;
    let d = header.quatern_d as f64;
    let mut a_sq = 1.0 - (b * b) - (c * c) - (d * d);
    if a_sq < 0.0 && a_sq > -1e-7 {
        a_sq = 0.0;
    }
    if a_sq < 0.0 {
        return Err("Invalid qform quaternion in NIfTI header".to_string());
    }
    let a = a_sq.sqrt();

    let dx = header.pixdim[1] as f64;
    let dy = header.pixdim[2] as f64;
    let mut dz = header.pixdim[3] as f64;
    if header.pixdim[0] < 0.0 {
        dz = -dz;
    }

    let rot = [
        [
            (a * a) + (b * b) - (c * c) - (d * d),
            (2.0 * b * c) - (2.0 * a * d),
            (2.0 * b * d) + (2.0 * a * c),
        ],
        [
            (2.0 * b * c) + (2.0 * a * d),
            (a * a) + (c * c) - (b * b) - (d * d),
            (2.0 * c * d) - (2.0 * a * b),
        ],
        [
            (2.0 * b * d) - (2.0 * a * c),
            (2.0 * c * d) + (2.0 * a * b),
            (a * a) + (d * d) - (b * b) - (c * c),
        ],
    ];

    Ok([
        [
            rot[0][0] * dx,
            rot[0][1] * dy,
            rot[0][2] * dz,
            header.quatern_x as f64,
        ],
        [
            rot[1][0] * dx,
            rot[1][1] * dy,
            rot[1][2] * dz,
            header.quatern_y as f64,
        ],
        [
            rot[2][0] * dx,
            rot[2][1] * dy,
            rot[2][2] * dz,
            header.quatern_z as f64,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn fallback_affine(header: &NiftiHeader) -> Matrix4 {
    [
        [header.pixdim[1] as f64, 0.0, 0.0, 0.0],
        [0.0, header.pixdim[2] as f64, 0.0, 0.0],
        [0.0, 0.0, header.pixdim[3] as f64, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn input_is_conformed(
    loaded: &LoadedInputImage,
    options: FastSurferConformOptions,
) -> Result<bool, String> {
    let target =
        determine_target_geometry(loaded.shape_xyz, loaded.zoom_xyz, options)?;

    let vox_ok =
        target.vox_size.is_none_or(|vox_size| {
            loaded.zoom_xyz.iter().zip(vox_size.iter()).all(
                |(actual, expected)| {
                    (f64::from(*actual) - f64::from(*expected)).abs()
                        <= FASTSURFER_CONFORM_VOX_EPS
                },
            )
        });
    let shape_ok = target
        .shape_xyz
        .is_none_or(|shape_xyz| shape_xyz == loaded.shape_xyz);
    let dtype_ok = loaded.header.datatype == NIFTI_UINT8_DATATYPE;
    let orientation_ok = match options.orientation {
        FastSurferOrientation::Native => true,
        FastSurferOrientation::Lia => {
            affine_to_axcodes(&loaded.affine) == ['L', 'I', 'A']
        }
    };

    Ok(vox_ok && shape_ok && dtype_ok && orientation_ok)
}

fn determine_target_geometry(
    shape_xyz: [usize; 3],
    zoom_xyz: [f32; 3],
    options: FastSurferConformOptions,
) -> Result<TargetGeometry, String> {
    let target_vox_size = determine_target_vox_size(zoom_xyz, options)?;
    let mut image_size = options.image_size;
    if matches!(image_size, FastSurferImageSize::Any)
        && target_vox_size.is_some()
    {
        image_size = FastSurferImageSize::Fov;
    }

    let target_shape = match image_size {
        FastSurferImageSize::Any => None,
        FastSurferImageSize::Value(value) => Some([value, value, value]),
        FastSurferImageSize::Fov | FastSurferImageSize::Auto => {
            let target = if let Some(vox_size) = target_vox_size {
                let mut adjusted = shape_xyz;
                for axis in 0..3 {
                    let fov =
                        f64::from(zoom_xyz[axis]) * shape_xyz[axis] as f64;
                    adjusted[axis] =
                        stable_positive_ceil(fov / f64::from(vox_size[axis]));
                }
                adjusted
            } else {
                shape_xyz
            };

            if matches!(image_size, FastSurferImageSize::Auto)
                && target_vox_size.is_some_and(|vox_size| {
                    let threshold = options
                        .conform_to_1mm_threshold
                        .map_or(0.0, |value| (1.0 - value).abs());
                    vox_size.iter().all(|value| {
                        (f64::from(*value) - 1.0).abs() <= f64::from(threshold)
                    })
                })
            {
                Some([256, 256, 256])
            } else if matches!(image_size, FastSurferImageSize::Auto) {
                let max_dim = target.into_iter().max().unwrap_or(256).max(256);
                Some([max_dim, max_dim, max_dim])
            } else {
                Some(target)
            }
        }
    };

    Ok(TargetGeometry {
        vox_size: target_vox_size,
        shape_xyz: target_shape,
    })
}

fn determine_target_vox_size(
    zoom_xyz: [f32; 3],
    options: FastSurferConformOptions,
) -> Result<Option<[f32; 3]>, String> {
    match options.vox_size {
        FastSurferVoxSize::Any => Ok(None),
        FastSurferVoxSize::Value(value) if value > 0.0 && value <= 1.0 => {
            Ok(Some([value, value, value]))
        }
        FastSurferVoxSize::Value(value) => Err(format!(
            "Invalid FastSurfer target voxel size {value}; expected 0 < value <= 1"
        )),
        FastSurferVoxSize::Min => {
            let min_zoom = zoom_xyz
                .into_iter()
                .map(f64::from)
                .fold(f64::INFINITY, f64::min);
            let rounded = round_to_decimals(min_zoom, 4).min(1.0);
            let threshold =
                options.conform_to_1mm_threshold.map_or(0.0, f64::from);
            let target = if threshold > 0.0 && rounded > threshold {
                1.0
            } else {
                rounded
            } as f32;
            Ok(Some([target, target, target]))
        }
    }
}

fn conform_loaded_image(
    loaded: &LoadedInputImage,
    options: FastSurferConformOptions,
) -> Result<InputVolume, String> {
    let target =
        determine_target_geometry(loaded.shape_xyz, loaded.zoom_xyz, options)?;
    let target_zoom = target.vox_size.unwrap_or(loaded.zoom_xyz);
    let target_shape = target.shape_xyz.unwrap_or(loaded.shape_xyz);
    let target_affine = build_target_affine(
        loaded,
        target_zoom,
        target_shape,
        options.orientation,
    )?;
    let mapped = resample_linear_volume(
        &loaded.data_xyz,
        loaded.shape_xyz,
        &loaded.affine,
        &target_affine,
        target_shape,
    )?;

    let conformed_u8 = if let Some(rescale_max) = options.rescale_max {
        let params = compute_rescale_parameters(
            &loaded.data_xyz,
            f64::from(rescale_max),
        )?;
        apply_rescale_and_round(&mapped, params)
    } else {
        mapped
            .iter()
            .map(|value| value.round().clamp(0.0, 255.0) as u8)
            .collect::<Vec<u8>>()
    };

    let mut header = loaded.header.clone();
    update_header_for_conformed_volume(
        &mut header,
        target_shape,
        target_zoom,
        &target_affine,
    );

    Ok(InputVolume {
        data_xyz: conformed_u8
            .into_iter()
            .map(f32::from)
            .collect::<Vec<f32>>(),
        shape_xyz: target_shape,
        zoom_xyz: target_zoom,
        header,
    })
}

fn build_target_affine(
    loaded: &LoadedInputImage,
    target_zoom: [f32; 3],
    target_shape: [usize; 3],
    orientation: FastSurferOrientation,
) -> Result<Matrix4, String> {
    let basis = match orientation {
        FastSurferOrientation::Native => {
            normalized_affine_basis(&loaded.affine)?
        }
        FastSurferOrientation::Lia => orientation_basis(['L', 'I', 'A']),
    };
    let linear = scale_basis_matrix(&basis, target_zoom);
    let source_center = apply_affine(
        &loaded.affine,
        [
            loaded.shape_xyz[0] as f64 / 2.0,
            loaded.shape_xyz[1] as f64 / 2.0,
            loaded.shape_xyz[2] as f64 / 2.0,
        ],
    );
    let target_center_offset = mat3_vec_mul(
        &linear,
        [
            target_shape[0] as f64 / 2.0,
            target_shape[1] as f64 / 2.0,
            target_shape[2] as f64 / 2.0,
        ],
    );

    Ok([
        [
            linear[0][0],
            linear[0][1],
            linear[0][2],
            source_center[0] - target_center_offset[0],
        ],
        [
            linear[1][0],
            linear[1][1],
            linear[1][2],
            source_center[1] - target_center_offset[1],
        ],
        [
            linear[2][0],
            linear[2][1],
            linear[2][2],
            source_center[2] - target_center_offset[2],
        ],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn resample_linear_volume(
    data_xyz: &[f32],
    source_shape: [usize; 3],
    source_affine: &Matrix4,
    target_affine: &Matrix4,
    target_shape: [usize; 3],
) -> Result<Vec<f64>, String> {
    let source_inv = invert_affine(source_affine)?;
    let target_to_source = mat4_mul(&source_inv, target_affine);
    let voxel_count = target_shape[0] * target_shape[1] * target_shape[2];
    let mut out = vec![0.0; voxel_count];

    for x in 0..target_shape[0] {
        for y in 0..target_shape[1] {
            for z in 0..target_shape[2] {
                let source = apply_affine(
                    &target_to_source,
                    [x as f64, y as f64, z as f64],
                );
                let value = trilinear_sample(data_xyz, source_shape, source);
                let offset = (x * target_shape[1] * target_shape[2])
                    + (y * target_shape[2])
                    + z;
                out[offset] = value;
            }
        }
    }

    Ok(out)
}

fn trilinear_sample(
    data_xyz: &[f32],
    shape_xyz: [usize; 3],
    coord: [f64; 3],
) -> f64 {
    let [x, y, z] = coord;
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let z0 = z.floor() as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let z1 = z0 + 1;

    let wx1 = x - x0 as f64;
    let wy1 = y - y0 as f64;
    let wz1 = z - z0 as f64;
    let wx0 = 1.0 - wx1;
    let wy0 = 1.0 - wy1;
    let wz0 = 1.0 - wz1;

    let neighbors = [
        (x0, y0, z0, wx0 * wy0 * wz0),
        (x1, y0, z0, wx1 * wy0 * wz0),
        (x0, y1, z0, wx0 * wy1 * wz0),
        (x1, y1, z0, wx1 * wy1 * wz0),
        (x0, y0, z1, wx0 * wy0 * wz1),
        (x1, y0, z1, wx1 * wy0 * wz1),
        (x0, y1, z1, wx0 * wy1 * wz1),
        (x1, y1, z1, wx1 * wy1 * wz1),
    ];

    neighbors
        .into_iter()
        .fold(0.0, |acc, (ix, iy, iz, weight)| {
            acc + (weight * sampled_voxel(data_xyz, shape_xyz, ix, iy, iz))
        })
}

fn sampled_voxel(
    data_xyz: &[f32],
    shape_xyz: [usize; 3],
    x: isize,
    y: isize,
    z: isize,
) -> f64 {
    if x < 0 || y < 0 || z < 0 {
        return 0.0;
    }

    let x = x as usize;
    let y = y as usize;
    let z = z as usize;
    if x >= shape_xyz[0] || y >= shape_xyz[1] || z >= shape_xyz[2] {
        return 0.0;
    }

    let offset = (x * shape_xyz[1] * shape_xyz[2]) + (y * shape_xyz[2]) + z;
    f64::from(data_xyz[offset])
}

fn compute_rescale_parameters(
    data_xyz: &[f32],
    dst_max: f64,
) -> Result<RescaleParameters, String> {
    if data_xyz.is_empty() {
        return Err(
            "Cannot compute rescale parameters for an empty image".to_string()
        );
    }

    let (mut data_min, mut data_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut non_zero_voxels = 0usize;
    for value in data_xyz.iter().copied().map(f64::from) {
        data_min = data_min.min(value);
        data_max = data_max.max(value);
        if value.abs() >= 1e-15 {
            non_zero_voxels += 1;
        }
    }

    if (data_max - data_min).abs() <= f64::EPSILON {
        return Ok(RescaleParameters {
            src_min: data_min,
            scale: 1.0,
            dst_min: 0.0,
            dst_max,
        });
    }

    let bins = 1000usize;
    let width = (data_max - data_min) / bins as f64;
    let mut hist = vec![0usize; bins];
    for value in data_xyz.iter().copied().map(f64::from) {
        let mut index = ((value - data_min) / width).floor() as isize;
        if index < 0 {
            index = 0;
        }
        if index as usize >= bins {
            index = (bins - 1) as isize;
        }
        hist[index as usize] += 1;
    }

    let mut cum_hist = Vec::with_capacity(bins + 1);
    cum_hist.push(0usize);
    let mut running = 0usize;
    for count in &hist {
        running += *count;
        cum_hist.push(running);
    }

    let lower_cutoff = 0usize;
    let lower_binedge_index = cum_hist
        .iter()
        .position(|value| *value >= lower_cutoff)
        .unwrap_or(0usize);
    let src_min =
        histogram_edge(data_min, width, lower_binedge_index.min(bins));

    let total_voxels = data_xyz.len();
    let upper_cutoff =
        total_voxels.saturating_sub((0.001 * non_zero_voxels as f64) as usize);
    let first_ge = cum_hist.iter().position(|value| *value >= upper_cutoff);
    let upper_binedge_index = if let Some(index) = first_ge {
        index as isize - 2
    } else if non_zero_voxels < 10 {
        -1
    } else {
        return Err("rescale upper bound not found".to_string());
    };
    let src_max = if upper_binedge_index < 0 {
        data_max
    } else {
        histogram_edge(data_min, width, upper_binedge_index as usize)
    };

    let scale = if (src_max - src_min).abs() <= f64::EPSILON {
        1.0
    } else {
        dst_max / (src_max - src_min)
    };

    Ok(RescaleParameters {
        src_min,
        scale,
        dst_min: 0.0,
        dst_max,
    })
}

fn apply_rescale_and_round(
    mapped: &[f64],
    params: RescaleParameters,
) -> Vec<u8> {
    mapped
        .iter()
        .map(|value| {
            let was_zero = value.abs() <= 1e-8;
            let mut scaled =
                params.dst_min + (params.scale * (value - params.src_min));
            scaled = scaled.clamp(params.dst_min, params.dst_max);
            if was_zero {
                scaled = 0.0;
            }
            scaled.round().clamp(params.dst_min, params.dst_max) as u8
        })
        .collect::<Vec<u8>>()
}

fn update_header_for_conformed_volume(
    header: &mut NiftiHeader,
    shape_xyz: [usize; 3],
    zoom_xyz: [f32; 3],
    affine: &Matrix4,
) {
    header.dim[0] = 3;
    header.dim[1] = shape_xyz[0] as u16;
    header.dim[2] = shape_xyz[1] as u16;
    header.dim[3] = shape_xyz[2] as u16;
    header.pixdim[1] = zoom_xyz[0];
    header.pixdim[2] = zoom_xyz[1];
    header.pixdim[3] = zoom_xyz[2];
    header.datatype = NIFTI_UINT8_DATATYPE;
    header.bitpix = 8;
    header.sform_code = 1;
    header.qform_code = 0;
    header.srow_x = [
        affine[0][0] as f32,
        affine[0][1] as f32,
        affine[0][2] as f32,
        affine[0][3] as f32,
    ];
    header.srow_y = [
        affine[1][0] as f32,
        affine[1][1] as f32,
        affine[1][2] as f32,
        affine[1][3] as f32,
    ];
    header.srow_z = [
        affine[2][0] as f32,
        affine[2][1] as f32,
        affine[2][2] as f32,
        affine[2][3] as f32,
    ];
}

fn affine_to_axcodes(affine: &Matrix4) -> [char; 3] {
    let mut used_axes = [false; 3];
    let mut codes = ['R', 'A', 'S'];

    for (axis, code) in codes.iter_mut().enumerate() {
        let mut best_dim = 0usize;
        let mut best_value = f64::NEG_INFINITY;
        for dim in 0..3 {
            if used_axes[dim] {
                continue;
            }
            let value = affine[dim][axis].abs();
            if value > best_value {
                best_value = value;
                best_dim = dim;
            }
        }
        used_axes[best_dim] = true;
        let positive = affine[best_dim][axis] > 0.0;
        *code = match best_dim {
            0 => {
                if positive {
                    'R'
                } else {
                    'L'
                }
            }
            1 => {
                if positive {
                    'A'
                } else {
                    'P'
                }
            }
            _ => {
                if positive {
                    'S'
                } else {
                    'I'
                }
            }
        };
    }

    codes
}

fn orientation_basis(codes: [char; 3]) -> Matrix3 {
    let mut basis = [[0.0; 3]; 3];
    for (axis, code) in codes.into_iter().enumerate() {
        let vector = match code {
            'L' => [-1.0, 0.0, 0.0],
            'R' => [1.0, 0.0, 0.0],
            'P' => [0.0, -1.0, 0.0],
            'A' => [0.0, 1.0, 0.0],
            'I' => [0.0, 0.0, -1.0],
            'S' => [0.0, 0.0, 1.0],
            _ => [0.0, 0.0, 0.0],
        };
        for dim in 0..3 {
            basis[dim][axis] = vector[dim];
        }
    }
    basis
}

fn normalized_affine_basis(affine: &Matrix4) -> Result<Matrix3, String> {
    let mut basis = [[0.0; 3]; 3];
    for axis in 0..3 {
        let norm = (affine[0][axis].powi(2)
            + affine[1][axis].powi(2)
            + affine[2][axis].powi(2))
        .sqrt();
        if norm <= FASTSURFER_CONFORM_ROT_EPS {
            return Err(
                "Cannot normalize affine with a zero-length basis vector"
                    .to_string(),
            );
        }
        for dim in 0..3 {
            basis[dim][axis] = affine[dim][axis] / norm;
        }
    }
    Ok(basis)
}

fn scale_basis_matrix(basis: &Matrix3, zoom_xyz: [f32; 3]) -> Matrix3 {
    let mut scaled = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            scaled[row][column] =
                basis[row][column] * f64::from(zoom_xyz[column]);
        }
    }
    scaled
}

fn apply_affine(affine: &Matrix4, point: [f64; 3]) -> [f64; 3] {
    [
        (affine[0][0] * point[0])
            + (affine[0][1] * point[1])
            + (affine[0][2] * point[2])
            + affine[0][3],
        (affine[1][0] * point[0])
            + (affine[1][1] * point[1])
            + (affine[1][2] * point[2])
            + affine[1][3],
        (affine[2][0] * point[0])
            + (affine[2][1] * point[1])
            + (affine[2][2] * point[2])
            + affine[2][3],
    ]
}

fn invert_affine(matrix: &Matrix4) -> Result<Matrix4, String> {
    let linear = [
        [matrix[0][0], matrix[0][1], matrix[0][2]],
        [matrix[1][0], matrix[1][1], matrix[1][2]],
        [matrix[2][0], matrix[2][1], matrix[2][2]],
    ];
    let linear_inv = invert_3x3(&linear)?;
    let translation = [matrix[0][3], matrix[1][3], matrix[2][3]];
    let inv_translation = mat3_vec_mul(
        &linear_inv,
        [-translation[0], -translation[1], -translation[2]],
    );

    Ok([
        [
            linear_inv[0][0],
            linear_inv[0][1],
            linear_inv[0][2],
            inv_translation[0],
        ],
        [
            linear_inv[1][0],
            linear_inv[1][1],
            linear_inv[1][2],
            inv_translation[1],
        ],
        [
            linear_inv[2][0],
            linear_inv[2][1],
            linear_inv[2][2],
            inv_translation[2],
        ],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn invert_3x3(matrix: &Matrix3) -> Result<Matrix3, String> {
    let determinant = (matrix[0][0]
        * ((matrix[1][1] * matrix[2][2]) - (matrix[1][2] * matrix[2][1])))
        - (matrix[0][1]
            * ((matrix[1][0] * matrix[2][2]) - (matrix[1][2] * matrix[2][0])))
        + (matrix[0][2]
            * ((matrix[1][0] * matrix[2][1]) - (matrix[1][1] * matrix[2][0])));
    if determinant.abs() <= FASTSURFER_CONFORM_ROT_EPS {
        return Err("Cannot invert singular affine matrix".to_string());
    }

    let inv_det = 1.0 / determinant;
    Ok([
        [
            ((matrix[1][1] * matrix[2][2]) - (matrix[1][2] * matrix[2][1]))
                * inv_det,
            ((matrix[0][2] * matrix[2][1]) - (matrix[0][1] * matrix[2][2]))
                * inv_det,
            ((matrix[0][1] * matrix[1][2]) - (matrix[0][2] * matrix[1][1]))
                * inv_det,
        ],
        [
            ((matrix[1][2] * matrix[2][0]) - (matrix[1][0] * matrix[2][2]))
                * inv_det,
            ((matrix[0][0] * matrix[2][2]) - (matrix[0][2] * matrix[2][0]))
                * inv_det,
            ((matrix[0][2] * matrix[1][0]) - (matrix[0][0] * matrix[1][2]))
                * inv_det,
        ],
        [
            ((matrix[1][0] * matrix[2][1]) - (matrix[1][1] * matrix[2][0]))
                * inv_det,
            ((matrix[0][1] * matrix[2][0]) - (matrix[0][0] * matrix[2][1]))
                * inv_det,
            ((matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0]))
                * inv_det,
        ],
    ])
}

fn mat3_vec_mul(matrix: &Matrix3, vector: [f64; 3]) -> [f64; 3] {
    [
        (matrix[0][0] * vector[0])
            + (matrix[0][1] * vector[1])
            + (matrix[0][2] * vector[2]),
        (matrix[1][0] * vector[0])
            + (matrix[1][1] * vector[1])
            + (matrix[1][2] * vector[2]),
        (matrix[2][0] * vector[0])
            + (matrix[2][1] * vector[1])
            + (matrix[2][2] * vector[2]),
    ]
}

fn mat4_mul(lhs: &Matrix4, rhs: &Matrix4) -> Matrix4 {
    let mut out = [[0.0; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            out[row][column] = (0..4)
                .map(|index| lhs[row][index] * rhs[index][column])
                .sum::<f64>();
        }
    }
    out
}

fn matrix4_close(lhs: &Matrix4, rhs: &Matrix4, tolerance: f64) -> bool {
    lhs.iter().zip(rhs.iter()).all(|(lhs_row, rhs_row)| {
        lhs_row
            .iter()
            .zip(rhs_row.iter())
            .all(|(lhs_value, rhs_value)| {
                (lhs_value - rhs_value).abs() <= tolerance
            })
    })
}

fn vector3_close(lhs: &[f64; 3], rhs: &[f64; 3], tolerance: f64) -> bool {
    lhs.iter().zip(rhs.iter()).all(|(lhs_value, rhs_value)| {
        (lhs_value - rhs_value).abs() <= tolerance
    })
}

fn affine_column_norms(affine: &Matrix4) -> [f64; 3] {
    [
        (affine[0][0].powi(2) + affine[1][0].powi(2) + affine[2][0].powi(2))
            .sqrt(),
        (affine[0][1].powi(2) + affine[1][1].powi(2) + affine[2][1].powi(2))
            .sqrt(),
        (affine[0][2].powi(2) + affine[1][2].powi(2) + affine[2][2].powi(2))
            .sqrt(),
    ]
}

fn round_to_decimals(value: f64, decimals: u32) -> f64 {
    let factor = 10f64.powi(i32::try_from(decimals).unwrap_or_default());
    (value * factor).round() / factor
}

fn stable_positive_ceil(value: f64) -> usize {
    ((value * 10_000.0).floor() / 10_000.0).ceil() as usize
}

fn histogram_edge(data_min: f64, width: f64, index: usize) -> f64 {
    data_min + (width * index as f64)
}

fn is_supported_native_input_path(path: &str) -> bool {
    let p = std::path::Path::new(path);
    if let Some(ext) = p.extension().and_then(|s| s.to_str())
        && (ext.eq_ignore_ascii_case("nii")
            || ext.eq_ignore_ascii_case("mgz")
            || ext.eq_ignore_ascii_case("mgh"))
    {
        return true;
    }
    if let Some(fname) = p.file_name().and_then(|s| s.to_str()) {
        return fname.to_ascii_lowercase().ends_with(".nii.gz");
    }
    false
}

fn is_mgz_or_mgh_path(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|ext| {
            ext.eq_ignore_ascii_case("mgz") || ext.eq_ignore_ascii_case("mgh")
        })
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

/// Load a supported native inference input volume from disk.
///
/// # Errors
/// Returns an error if the path is unsupported, conversion from `MGZ`/`MGH` to
/// `NIfTI` fails, or the `NIfTI` volume cannot be read/materialized.
pub fn load_input_volume(path: &str) -> Result<InputVolume, String> {
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
