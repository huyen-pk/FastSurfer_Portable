from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)

from FastSurferCNN.utils.metrics import hd


def _dice_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.count_nonzero(a & b)
    denom = np.count_nonzero(a) + np.count_nonzero(b)
    if denom == 0:
        return 1.0
    return float((2.0 * inter) / denom)


def _surface_distances(result: np.ndarray, reference: np.ndarray, voxelspacing: tuple[float, float, float]) -> np.ndarray:
    footprint = generate_binary_structure(result.ndim, 1)
    result = result.astype(bool)
    reference = reference.astype(bool)

    if np.count_nonzero(result) == 0 or np.count_nonzero(reference) == 0:
        return np.array([], dtype=np.float64)

    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    return dt[result_border]


def _assd(a: np.ndarray, b: np.ndarray, voxelspacing: tuple[float, float, float]) -> float:
    d1 = _surface_distances(a, b, voxelspacing)
    d2 = _surface_distances(b, a, voxelspacing)
    if d1.size == 0 or d2.size == 0:
        return float("nan")
    return float((d1.mean() + d2.mean()) / 2.0)


def _icc_2_1(x: np.ndarray, y: np.ndarray) -> float:
    data = np.stack([x, y], axis=1).astype(np.float64)
    n, k = data.shape
    if n < 2:
        return float("nan")

    mean_row = data.mean(axis=1, keepdims=True)
    mean_col = data.mean(axis=0, keepdims=True)
    mean_all = data.mean()

    ss_rows = k * np.sum((mean_row - mean_all) ** 2)
    ss_cols = n * np.sum((mean_col - mean_all) ** 2)
    ss_err = np.sum((data - mean_row - mean_col + mean_all) ** 2)

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_err + (k * (ms_cols - ms_err) / n)
    if np.isclose(denom, 0.0):
        return float("nan")
    return float((ms_rows - ms_err) / denom)


def _icc_3_1(x: np.ndarray, y: np.ndarray) -> float:
    data = np.stack([x, y], axis=1).astype(np.float64)
    n, k = data.shape
    if n < 2:
        return float("nan")

    mean_row = data.mean(axis=1, keepdims=True)
    mean_col = data.mean(axis=0, keepdims=True)
    mean_all = data.mean()

    ss_rows = k * np.sum((mean_row - mean_all) ** 2)
    ss_err = np.sum((data - mean_row - mean_col + mean_all) ** 2)

    ms_rows = ss_rows / (n - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_err
    if np.isclose(denom, 0.0):
        return float("nan")
    return float((ms_rows - ms_err) / denom)


def _label_set(a: np.ndarray, b: np.ndarray) -> list[int]:
    labels = sorted(set(np.unique(a).tolist()) | set(np.unique(b).tolist()))
    return [int(v) for v in labels if int(v) != 0]


def _safe_float(value: float) -> float | None:
    if value is None:
        return None
    if np.isnan(value) or np.isinf(value):
        return None
    return float(value)


def compute_metrics(rust_seg: np.ndarray, py_seg: np.ndarray, voxelspacing: tuple[float, float, float]) -> dict:
    labels = _label_set(rust_seg, py_seg)

    per_label = []
    dice_values = []
    assd_values = []
    hd95_values = []
    hdmax_values = []

    rust_vols = []
    py_vols = []

    for label in labels:
        r = rust_seg == label
        p = py_seg == label
        dice = _dice_similarity(r, p)
        dice_values.append(dice)

        rust_vol = int(np.count_nonzero(r))
        py_vol = int(np.count_nonzero(p))
        rust_vols.append(rust_vol)
        py_vols.append(py_vol)

        assd = _safe_float(_assd(r, p, voxelspacing))

        hd_max = None
        hd_95 = None
        if rust_vol > 0 and py_vol > 0:
            try:
                hd_max_v, _hd50_v, hd95_v = hd(r, p, voxelspacing=voxelspacing, connectivity=1)
                hd_max = _safe_float(float(hd_max_v))
                hd_95 = _safe_float(float(hd95_v))
            except Exception:
                hd_max = None
                hd_95 = None

        if assd is not None:
            assd_values.append(assd)
        if hd_95 is not None:
            hd95_values.append(hd_95)
        if hd_max is not None:
            hdmax_values.append(hd_max)

        per_label.append(
            {
                "label": label,
                "dice": _safe_float(dice),
                "assd": assd,
                "hd95": hd_95,
                "hdmax": hd_max,
                "rust_voxels": rust_vol,
                "python_voxels": py_vol,
            }
        )

    rust_vols_np = np.asarray(rust_vols, dtype=np.float64)
    py_vols_np = np.asarray(py_vols, dtype=np.float64)

    if rust_vols_np.size == 0:
        icc21 = None
        icc31 = None
    else:
        icc21 = _safe_float(_icc_2_1(rust_vols_np, py_vols_np))
        icc31 = _safe_float(_icc_3_1(rust_vols_np, py_vols_np))

    fg_r = rust_seg > 0
    fg_p = py_seg > 0
    fg_dice = _safe_float(_dice_similarity(fg_r, fg_p))

    fg_assd = _safe_float(_assd(fg_r, fg_p, voxelspacing))
    fg_hd95 = None
    fg_hdmax = None
    if np.count_nonzero(fg_r) > 0 and np.count_nonzero(fg_p) > 0:
        try:
            hd_max_v, _hd50_v, hd95_v = hd(fg_r, fg_p, voxelspacing=voxelspacing, connectivity=1)
            fg_hdmax = _safe_float(float(hd_max_v))
            fg_hd95 = _safe_float(float(hd95_v))
        except Exception:
            pass

    return {
        "labels_evaluated": len(labels),
        "aggregate": {
            "dice_macro": _safe_float(float(np.mean(dice_values))) if dice_values else None,
            "assd_macro": _safe_float(float(np.mean(assd_values))) if assd_values else None,
            "hd95_macro": _safe_float(float(np.mean(hd95_values))) if hd95_values else None,
            "hdmax_macro": _safe_float(float(np.mean(hdmax_values))) if hdmax_values else None,
            "dice_foreground": fg_dice,
            "assd_foreground": fg_assd,
            "hd95_foreground": fg_hd95,
            "hdmax_foreground": fg_hdmax,
            "icc_2_1_volumes": icc21,
            "icc_3_1_volumes": icc31,
            "icc_python_variant": "not_available_in_metrics_py",
        },
        "per_label": per_label,
    }


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: python parity_metrics.py <rust_seg.nii.gz> <python_seg.nii.gz> <output.json>"
        )

    rust_path = Path(sys.argv[1])
    python_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    rust_img = nib.load(str(rust_path))
    py_img = nib.load(str(python_path))

    rust_seg = np.rint(np.asanyarray(rust_img.dataobj)).astype(np.int32)
    py_seg = np.rint(np.asanyarray(py_img.dataobj)).astype(np.int32)

    if rust_seg.shape != py_seg.shape:
        raise SystemExit(f"shape mismatch rust={rust_seg.shape} python={py_seg.shape}")

    voxelspacing = tuple(float(v) for v in rust_img.header.get_zooms()[:3])
    metrics = compute_metrics(rust_seg, py_seg, voxelspacing)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
