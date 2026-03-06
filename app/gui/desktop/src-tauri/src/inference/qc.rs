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
            expected
        ));
    }

    let non_bg_voxels = labels_xyz.iter().filter(|value| **value > 0).count();
    let total_volume_liters = (non_bg_voxels as f64) * voxvol_mm3 / 1_000_000.0;

    let mut vent_bg_touching = 0usize;
    for x in 0..sx {
        for y in 0..sy {
            for z in 0..sz {
                let idx = (x * sy * sz) + (y * sz) + z;
                let label = labels_xyz[idx];
                if !is_vent_label(label) {
                    continue;
                }

                let mut touches_bg = false;
                for dx in -1isize..=1 {
                    for dy in -1isize..=1 {
                        for dz in -1isize..=1 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }

                            let nx = x as isize + dx;
                            let ny = y as isize + dy;
                            let nz = z as isize + dz;
                            if nx < 0
                                || ny < 0
                                || nz < 0
                                || nx >= sx as isize
                                || ny >= sy as isize
                                || nz >= sz as isize
                            {
                                continue;
                            }

                            let nidx = ((nx as usize) * sy * sz) + ((ny as usize) * sz) + (nz as usize);
                            if labels_xyz[nidx] == 0 {
                                touches_bg = true;
                                break;
                            }
                        }
                        if touches_bg {
                            break;
                        }
                    }
                    if touches_bg {
                        break;
                    }
                }

                if touches_bg {
                    vent_bg_touching += 1;
                }
            }
        }
    }

    let vent_bg_intersection_mm3 = (vent_bg_touching as f64) * voxvol_mm3;
    let passed = total_volume_liters >= QC_TOTAL_VOLUME_MIN_LITERS;
    let message = format!(
        "total_volume_liters={:.3} threshold_liters={:.2} vent_bg_intersection_mm3={:.2}",
        total_volume_liters, QC_TOTAL_VOLUME_MIN_LITERS, vent_bg_intersection_mm3
    );

    Ok(QcResult {
        passed: Some(passed),
        message: Some(message),
    })
}
