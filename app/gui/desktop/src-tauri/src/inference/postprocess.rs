use std::collections::VecDeque;

fn usz_to_isz(v: usize) -> isize {
    isize::try_from(v).unwrap_or(0)
}

fn isz_to_usz(v: isize) -> usize {
    usize::try_from(v).unwrap_or(0)
}

const FRONTAL_SPECIAL_LABELS: [u16; 4] = [1012, 2012, 1019, 2019];
const CORTEX_SPLIT_LABELS: [u16; 19] = [
    1003, 1006, 1007, 1008, 1009, 1011, 1015, 1018, 1019, 1020, 1025, 1026,
    1027, 1028, 1029, 1030, 1031, 1034, 1035,
];
const PROBLEMATIC_SPLIT_LABELS: [u16; 4] = [1011, 1019, 1026, 1029];

fn is_frontal_special(label: u16) -> bool {
    FRONTAL_SPECIAL_LABELS.contains(&label)
}

fn index_to_xyz(index: usize, shape: [usize; 3]) -> [f64; 3] {
    let [_sx, sy, sz] = shape;
    let x = index / (sy * sz);
    let rem = index % (sy * sz);
    let y = rem / sz;
    let z = rem % sz;
    [
        f64::from(u32::try_from(x).unwrap_or(0)),
        f64::from(u32::try_from(y).unwrap_or(0)),
        f64::from(u32::try_from(z).unwrap_or(0)),
    ]
}

fn squared_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx) + (dy * dy) + (dz * dz)
}

fn connected_components_for_mask(
    mask: &[u8],
    shape: [usize; 3],
) -> Vec<Vec<usize>> {
    let [sx, sy, sz] = shape;
    let mut visited = vec![false; mask.len()];
    let mut components = Vec::<Vec<usize>>::new();

    for x in 0..sx {
        for y in 0..sy {
            for z in 0..sz {
                let start = (x * sy * sz) + (y * sz) + z;
                if visited[start] || mask[start] == 0 {
                    continue;
                }
                let mut queue = VecDeque::<usize>::new();
                let mut component = Vec::<usize>::new();
                visited[start] = true;
                queue.push_back(start);

                while let Some(idx) = queue.pop_front() {
                    component.push(idx);
                    let x0 = idx / (sy * sz);
                    let rem = idx % (sy * sz);
                    let y0 = rem / sz;
                    let z0 = rem % sz;

                    for dx in -1isize..=1 {
                        for dy in -1isize..=1 {
                            for dz in -1isize..=1 {
                                if dx == 0 && dy == 0 && dz == 0 {
                                    continue;
                                }
                                let nx = usz_to_isz(x0) + dx;
                                let ny = usz_to_isz(y0) + dy;
                                let nz = usz_to_isz(z0) + dz;
                                if nx < 0
                                    || ny < 0
                                    || nz < 0
                                    || nx >= usz_to_isz(sx)
                                    || ny >= usz_to_isz(sy)
                                    || nz >= usz_to_isz(sz)
                                {
                                    continue;
                                }
                                let nidx = (isz_to_usz(nx) * sy * sz)
                                    + (isz_to_usz(ny) * sz)
                                    + isz_to_usz(nz);
                                if !visited[nidx] && mask[nidx] != 0 {
                                    visited[nidx] = true;
                                    queue.push_back(nidx);
                                }
                            }
                        }
                    }
                }

                components.push(component);
            }
        }
    }

    components
}

fn largest_component_indices(mask: &[u8], shape: [usize; 3]) -> Vec<usize> {
    connected_components_for_mask(mask, shape)
        .into_iter()
        .max_by_key(Vec::len)
        .unwrap_or_default()
}

fn centroid_of_indices(
    indices: &[usize],
    shape: [usize; 3],
) -> Option<[f64; 3]> {
    if indices.is_empty() {
        return None;
    }
    let mut sum = [0.0f64; 3];
    for idx in indices {
        let xyz = index_to_xyz(*idx, shape);
        sum[0] += xyz[0];
        sum[1] += xyz[1];
        sum[2] += xyz[2];
    }
    let n = f64::from(u32::try_from(indices.len()).unwrap_or(0));
    Some([sum[0] / n, sum[1] / n, sum[2] / n])
}

pub(crate) fn split_cortex_labels(
    pred_labels: &mut [u16],
    shape_xyz: [usize; 3],
) {
    let lh_wm_mask = pred_labels
        .iter()
        .map(|value| u8::from(*value == 2))
        .collect::<Vec<u8>>();
    let rh_wm_mask = pred_labels
        .iter()
        .map(|value| u8::from(*value == 41))
        .collect::<Vec<u8>>();

    let lh_centroid = centroid_of_indices(
        &largest_component_indices(&lh_wm_mask, shape_xyz),
        shape_xyz,
    );
    let rh_centroid = centroid_of_indices(
        &largest_component_indices(&rh_wm_mask, shape_xyz),
        shape_xyz,
    );
    let (Some(lh_centroid), Some(rh_centroid)) = (lh_centroid, rh_centroid)
    else {
        return;
    };

    for label_current in CORTEX_SPLIT_LABELS {
        let mask = pred_labels
            .iter()
            .map(|value| u8::from(*value == label_current))
            .collect::<Vec<u8>>();
        let components = connected_components_for_mask(&mask, shape_xyz);
        for component in components {
            if let Some(centroid) = centroid_of_indices(&component, shape_xyz) {
                let dist_left = squared_distance(centroid, lh_centroid);
                let dist_right = squared_distance(centroid, rh_centroid);
                if dist_right < dist_left {
                    for idx in component {
                        pred_labels[idx] = label_current + 1000;
                    }
                }
            }
        }
    }

    for prob_class_lh in PROBLEMATIC_SPLIT_LABELS {
        let prob_class_rh = prob_class_lh + 1000;
        for (idx, value) in pred_labels.iter_mut().enumerate() {
            if *value == prob_class_lh || *value == prob_class_rh {
                let xyz = index_to_xyz(idx, shape_xyz);
                let dist_left = squared_distance(xyz, lh_centroid);
                let dist_right = squared_distance(xyz, rh_centroid);
                *value = if dist_right < dist_left {
                    prob_class_rh
                } else {
                    prob_class_lh
                };
            }
        }
    }
}

pub(crate) fn flip_wm_islands(aseg_labels: &mut [u16], shape_xyz: [usize; 3]) {
    let [
        left_white_matter,
        left_gray_matter,
        right_white_matter,
        right_gray_matter,
    ] = [2u16, 3u16, 41u16, 42u16];

    let lh_wm_mask = aseg_labels
        .iter()
        .map(|value| u8::from(*value == left_white_matter))
        .collect::<Vec<u8>>();
    let rh_wm_mask = aseg_labels
        .iter()
        .map(|value| u8::from(*value == right_white_matter))
        .collect::<Vec<u8>>();

    let lh_components = connected_components_for_mask(&lh_wm_mask, shape_xyz);
    let rh_components = connected_components_for_mask(&rh_wm_mask, shape_xyz);

    if lh_components.is_empty() || rh_components.is_empty() {
        return;
    }

    let lh_main = lh_components
        .iter()
        .max_by_key(|component| component.len())
        .cloned()
        .unwrap_or_default();
    let rh_main = rh_components
        .iter()
        .max_by_key(|component| component.len())
        .cloned()
        .unwrap_or_default();

    let lh_context = aseg_labels
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| {
            if *value == left_white_matter || *value == left_gray_matter {
                Some(idx)
            } else {
                None
            }
        })
        .collect::<Vec<usize>>();
    let rh_context = aseg_labels
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| {
            if *value == right_white_matter || *value == right_gray_matter {
                Some(idx)
            } else {
                None
            }
        })
        .collect::<Vec<usize>>();

    let lh_centroid = centroid_of_indices(&lh_context, shape_xyz);
    let rh_centroid = centroid_of_indices(&rh_context, shape_xyz);
    let (Some(lh_centroid), Some(rh_centroid)) = (lh_centroid, rh_centroid)
    else {
        return;
    };

    for component in rh_components {
        if component == rh_main {
            continue;
        }
        for idx in component {
            let xyz = index_to_xyz(idx, shape_xyz);
            if squared_distance(xyz, lh_centroid)
                < squared_distance(xyz, rh_centroid)
            {
                aseg_labels[idx] = left_white_matter;
            }
        }
    }

    for component in lh_components {
        if component == lh_main {
            continue;
        }
        for idx in component {
            let xyz = index_to_xyz(idx, shape_xyz);
            if squared_distance(xyz, rh_centroid)
                < squared_distance(xyz, lh_centroid)
            {
                aseg_labels[idx] = right_white_matter;
            }
        }
    }
}

pub(crate) fn derive_aseg_from_pred(pred_labels: &[u16]) -> Vec<u16> {
    let mut aseg = pred_labels.to_vec();
    for value in &mut aseg {
        if *value >= 2000 {
            *value = 42;
        } else if *value >= 1000 {
            *value = 3;
        }
    }
    aseg
}

fn binary_dilate(mask: &[u8], shape: [usize; 3]) -> Vec<u8> {
    let [sx, sy, sz] = shape;
    let mut out = vec![0u8; mask.len()];
    for x in 0..sx {
        for y in 0..sy {
            for z in 0..sz {
                let mut any = false;
                for dx in -1isize..=1 {
                    for dy in -1isize..=1 {
                        for dz in -1isize..=1 {
                            let nx = usz_to_isz(x) + dx;
                            let ny = usz_to_isz(y) + dy;
                            let nz = usz_to_isz(z) + dz;
                            if nx < 0
                                || ny < 0
                                || nz < 0
                                || nx >= usz_to_isz(sx)
                                || ny >= usz_to_isz(sy)
                                || nz >= usz_to_isz(sz)
                            {
                                continue;
                            }
                            let nidx = (isz_to_usz(nx) * sy * sz)
                                + (isz_to_usz(ny) * sz)
                                + isz_to_usz(nz);
                            if mask[nidx] != 0 {
                                any = true;
                                break;
                            }
                        }
                        if any {
                            break;
                        }
                    }
                    if any {
                        break;
                    }
                }
                let idx = (x * sy * sz) + (y * sz) + z;
                out[idx] = u8::from(any);
            }
        }
    }
    out
}

fn binary_erode(mask: &[u8], shape: [usize; 3]) -> Vec<u8> {
    let [sx, sy, sz] = shape;
    let mut out = vec![0u8; mask.len()];
    for x in 0..sx {
        for y in 0..sy {
            for z in 0..sz {
                let mut all = true;
                for dx in -1isize..=1 {
                    for dy in -1isize..=1 {
                        for dz in -1isize..=1 {
                            let nx = usz_to_isz(x) + dx;
                            let ny = usz_to_isz(y) + dy;
                            let nz = usz_to_isz(z) + dz;
                            if nx < 0
                                || ny < 0
                                || nz < 0
                                || nx >= usz_to_isz(sx)
                                || ny >= usz_to_isz(sy)
                                || nz >= usz_to_isz(sz)
                            {
                                all = false;
                                break;
                            }
                            let nidx = (isz_to_usz(nx) * sy * sz)
                                + (isz_to_usz(ny) * sz)
                                + isz_to_usz(nz);
                            if mask[nidx] == 0 {
                                all = false;
                                break;
                            }
                        }
                        if !all {
                            break;
                        }
                    }
                    if !all {
                        break;
                    }
                }
                let idx = (x * sy * sz) + (y * sz) + z;
                out[idx] = u8::from(all);
            }
        }
    }
    out
}

fn keep_largest_connected_component(mask: &[u8], shape: [usize; 3]) -> Vec<u8> {
    let [sx, sy, sz] = shape;
    let mut visited = vec![false; mask.len()];
    let mut best_component = Vec::<usize>::new();

    for x in 0..sx {
        for y in 0..sy {
            for z in 0..sz {
                let start = (x * sy * sz) + (y * sz) + z;
                if visited[start] || mask[start] == 0 {
                    continue;
                }

                let mut queue = VecDeque::<usize>::new();
                let mut component = Vec::<usize>::new();
                visited[start] = true;
                queue.push_back(start);

                while let Some(idx) = queue.pop_front() {
                    component.push(idx);

                    let x0 = idx / (sy * sz);
                    let rem = idx % (sy * sz);
                    let y0 = rem / sz;
                    let z0 = rem % sz;

                    for dx in -1isize..=1 {
                        for dy in -1isize..=1 {
                            for dz in -1isize..=1 {
                                if dx == 0 && dy == 0 && dz == 0 {
                                    continue;
                                }
                                let nx = usz_to_isz(x0) + dx;
                                let ny = usz_to_isz(y0) + dy;
                                let nz = usz_to_isz(z0) + dz;
                                if nx < 0
                                    || ny < 0
                                    || nz < 0
                                    || nx >= usz_to_isz(sx)
                                    || ny >= usz_to_isz(sy)
                                    || nz >= usz_to_isz(sz)
                                {
                                    continue;
                                }
                                let nidx = (isz_to_usz(nx) * sy * sz)
                                    + (isz_to_usz(ny) * sz)
                                    + isz_to_usz(nz);
                                if !visited[nidx] && mask[nidx] != 0 {
                                    visited[nidx] = true;
                                    queue.push_back(nidx);
                                }
                            }
                        }
                    }
                }

                if component.len() > best_component.len() {
                    best_component = component;
                }
            }
        }
    }

    let mut out = vec![0u8; mask.len()];
    for idx in best_component {
        out[idx] = 1;
    }
    out
}

pub(crate) fn derive_brainmask_from_pred(
    pred_labels: &[u16],
    shape_xyz: [usize; 3],
) -> Vec<u8> {
    let mut initial = pred_labels
        .iter()
        .map(|value| u8::from(*value > 0 && !is_frontal_special(*value)))
        .collect::<Vec<u8>>();

    for _ in 0..5 {
        initial = binary_dilate(&initial, shape_xyz);
    }
    for _ in 0..4 {
        initial = binary_erode(&initial, shape_xyz);
    }

    let mut largest = keep_largest_connected_component(&initial, shape_xyz);
    for (index, value) in pred_labels.iter().enumerate() {
        if is_frontal_special(*value) {
            largest[index] = 1;
        }
    }
    largest
}

pub(crate) fn mask_aseg_with_brainmask(
    aseg_labels: &mut [u16],
    brainmask: &[u8],
) {
    for (label, mask) in aseg_labels.iter_mut().zip(brainmask.iter()) {
        if *mask == 0 {
            *label = 0;
        }
    }
}
