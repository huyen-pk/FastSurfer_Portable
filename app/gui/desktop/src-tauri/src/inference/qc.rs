#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::bool_to_int_with_if,
    clippy::case_sensitive_file_extension_comparisons,
    clippy::ignored_unit_patterns,
    clippy::uninlined_format_args
)]

pub(crate) struct QcResult {
    pub passed: Option<bool>,
    pub message: Option<String>,
}

const QC_TOTAL_VOLUME_MIN_LITERS: f64 = 0.70;
const VENT_LABELS: [u16; 4] = [4, 43, 31, 63];

fn is_vent_label(label: u16) -> bool {
    VENT_LABELS.contains(&label)
}

pub(crate) fn evaluate_qc(
    labels_xyz: &[u16],
    shape_xyz: [usize; 3],
    voxvol_mm3: f64,
) -> Result<QcResult, String> {
    let [sx, sy, sz] = shape_xyz;
    let expected = sx * sy * sz;
    if labels_xyz.len() != expected {
        return Err(format!(
            "QC label length mismatch: got {}, expected {}",
            labels_xyz.len(),
            expected,
        ));
    }

    let non_bg_voxels = labels_xyz.iter().filter(|v| **v > 0).count() as u64;
    let total_volume_liters = (non_bg_voxels as f64) * voxvol_mm3 / 1_000_000.0;

    let vent_bg_touching = count_vent_bg_touching(labels_xyz, sx, sy, sz);

    let vent_bg_intersection_mm3 = (vent_bg_touching as f64) * voxvol_mm3;
    let passed = total_volume_liters >= QC_TOTAL_VOLUME_MIN_LITERS;
    let message = format!(
        "total_volume_liters={:.3} threshold_liters={:.2} vent_bg_intersection_mm3={:.2}",
        total_volume_liters,
        QC_TOTAL_VOLUME_MIN_LITERS,
        vent_bg_intersection_mm3,
    );

    Ok(QcResult {
        passed: Some(passed),
        message: Some(message),
    })
}

fn count_vent_bg_touching(
    labels_xyz: &[u16],
    sx: usize,
    sy: usize,
    sz: usize,
) -> usize {
    let mut vent_bg_touching = 0usize;

    for x in 0..sx {
        for y in 0..sy {
            for z in 0..sz {
                let idx = (x * sy * sz) + (y * sz) + z;
                let label = labels_xyz[idx];
                if !is_vent_label(label) {
                    continue;
                }

                if voxel_touches_background(labels_xyz, sx, sy, sz, x, y, z) {
                    vent_bg_touching += 1;
                }
            }
        }
    }

    vent_bg_touching
}

fn voxel_touches_background(
    labels_xyz: &[u16],
    sx: usize,
    sy: usize,
    sz: usize,
    x: usize,
    y: usize,
    z: usize,
) -> bool {
    for dx in -1isize..=1 {
        for dy in -1isize..=1 {
            for dz in -1isize..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }

                let nx = isize::try_from(x).unwrap_or(0) + dx;
                let ny = isize::try_from(y).unwrap_or(0) + dy;
                let nz = isize::try_from(z).unwrap_or(0) + dz;
                if nx < 0 || ny < 0 || nz < 0 {
                    continue;
                }
                if nx >= isize::try_from(sx).unwrap_or(0)
                    || ny >= isize::try_from(sy).unwrap_or(0)
                    || nz >= isize::try_from(sz).unwrap_or(0)
                {
                    continue;
                }

                let nidx = (usize::try_from(nx).unwrap_or(0) * sy * sz)
                    + (usize::try_from(ny).unwrap_or(0) * sz)
                    + usize::try_from(nz).unwrap_or(0);
                if labels_xyz[nidx] == 0 {
                    return true;
                }
            }
        }
    }

    false
}
