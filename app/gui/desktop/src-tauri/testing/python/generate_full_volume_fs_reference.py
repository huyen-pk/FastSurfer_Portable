#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

import FastSurferCNN.reduce_to_aseg as rta
from FastSurferCNN.data_loader import data_utils as du
from FastSurferCNN.data_loader.augmentation import ToTensorTest
from FastSurferCNN.data_loader.conform import to_target_orientation
from FastSurferCNN.data_loader.data_utils import (
    get_thick_slices,
    map_prediction_sagittal2full,
    transform_axial,
    transform_sagittal,
)
from FastSurferCNN.run_prediction import RunModelOnData
from FastSurferCNN.utils.checkpoint import get_config_file, load_checkpoint_config_defaults


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "FastSurferCNN").exists() and (candidate / "app").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {start}")


def save_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    image = nib.Nifti1Image(data, affine)
    image.set_data_dtype(data.dtype)
    nib.save(image, str(path))


def resolve_input_paths(repo_root: Path) -> tuple[Path, Path]:
    input_mgz = repo_root / "app/gui/desktop/src-tauri/testing/data/Subject140/140_orig.mgz"
    if not input_mgz.exists():
        raise FileNotFoundError(f"Input MGZ not found: {input_mgz}")

    native_input_nii = repo_root / "app/gui/desktop/src-tauri/testing/data/.tmp_e2e_output_py/140_orig.native_input.nii.gz"
    if not native_input_nii.exists():
        native_input_nii.parent.mkdir(parents=True, exist_ok=True)
        source_img = nib.load(str(input_mgz))
        nib.save(source_img, str(native_input_nii))

    return input_mgz, native_input_nii


def build_evaluator(repo_root: Path) -> RunModelOnData:
    config_file = get_config_file("FastSurferCNN")
    ckpts = load_checkpoint_config_defaults("checkpoint", config_file)
    cfgs = load_checkpoint_config_defaults("config", config_file)

    lut = repo_root / "FastSurferCNN/config/FreeSurferColorLUT.txt"
    if not lut.exists():
        raise FileNotFoundError(f"LUT not found: {lut}")

    return RunModelOnData(
        lut=lut,
        ckpt_ax=Path(str(ckpts["axial"])),
        ckpt_sag=Path(str(ckpts["sagittal"])),
        ckpt_cor=Path(str(ckpts["coronal"])),
        cfg_ax=Path(str(cfgs["axial"])),
        cfg_sag=Path(str(cfgs["sagittal"])),
        cfg_cor=Path(str(cfgs["coronal"])),
        device="cpu",
        viewagg_device="cpu",
        threads=1,
        batch_size=1,
        vox_size="min",
        orientation="lia",
        image_size=True,
        async_io=False,
        conform_to_1mm_threshold=0.95,
    )


def plane_zoom(orig_zoom: np.ndarray, plane: str) -> np.ndarray:
    if plane == "axial":
        return np.asarray(orig_zoom)[[2, 0]]
    if plane == "sagittal":
        return np.asarray(orig_zoom)[[2, 1]]
    return np.asarray(orig_zoom)[[0, 1]]


def generate_preprocess_artifacts(
    orig_data: np.ndarray,
    affine: np.ndarray,
    preprocess_dir: Path,
) -> None:
    orig_in_lia, _ = to_target_orientation(orig_data, affine, target_orientation="LIA")
    slice_thickness = 3

    for plane in ("coronal", "axial", "sagittal"):
        if plane == "axial":
            transformed = transform_axial(orig_in_lia)
        elif plane == "sagittal":
            transformed = transform_sagittal(orig_in_lia)
        else:
            transformed = orig_in_lia

        thick = get_thick_slices(transformed, slice_thickness)
        thick = np.transpose(thick, (2, 0, 1, 3))

        center_volume = np.zeros((thick.shape[1], thick.shape[2], thick.shape[0]), dtype=np.float32)
        for idx in range(thick.shape[0]):
            image = np.clip(thick[idx].astype(np.float32) / 255.0, a_min=0.0, a_max=1.0)
            image = image.transpose((2, 0, 1))
            center_volume[:, :, idx] = image[slice_thickness, :, :]

        save_nifti(
            preprocess_dir / f"python_preprocess_{plane}.nii.gz",
            center_volume,
            affine,
        )


def run_subject140_one_slice_geometry_generation(repo_root: Path) -> Path:
    input_mgz, native_input_nii = resolve_input_paths(repo_root)

    fs_reference_root = repo_root / "app/gui/desktop/src-tauri/testing/data/fs_reference/Subject140"
    run_dir = fs_reference_root / "one_slice_geometry" / f"iter_{int(time.time())}"
    preprocess_dir = run_dir / "preprocess"
    forward_dir = run_dir / "forward pass"
    postprocess_dir = run_dir / "postprocess"

    preprocess_dir.mkdir(parents=True, exist_ok=True)
    forward_dir.mkdir(parents=True, exist_ok=True)
    postprocess_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.perf_counter()

    native_img = nib.load(str(native_input_nii))
    affine = native_img.affine
    orig_data = np.asanyarray(native_img.dataobj)
    orig_zoom = np.asarray(native_img.header.get_zooms()[:3], dtype=np.float32)
    orig_in_lia, _ = to_target_orientation(orig_data, affine, target_orientation="LIA")

    evaluator = build_evaluator(repo_root)
    to_tensor = ToTensorTest()
    slice_thickness = 3

    slice_indices: dict[str, int] = {}
    timings = {"preprocess": 0.0, "forward": 0.0, "postprocess": 0.0}

    for plane in ("coronal", "axial", "sagittal"):
        t0 = time.perf_counter()
        if plane == "axial":
            transformed = transform_axial(orig_in_lia)
        elif plane == "sagittal":
            transformed = transform_sagittal(orig_in_lia)
        else:
            transformed = orig_in_lia

        thick = get_thick_slices(transformed, slice_thickness)
        thick = np.transpose(thick, (2, 0, 1, 3))
        slice_idx = thick.shape[0] // 2
        slice_indices[plane] = int(slice_idx)
        preprocessed = to_tensor(thick[slice_idx])
        center_slice = preprocessed[slice_thickness, :, :]
        save_nifti(
            preprocess_dir / f"python_preprocess_{plane}_slice_{slice_idx}.nii.gz",
            center_slice[:, :, None].astype(np.float32),
            affine,
        )
        timings["preprocess"] += (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        model = evaluator.models[plane]
        base_res = float(evaluator.view_ops[plane]["cfg"].MODEL.BASE_RES)
        scale_factor = (base_res / plane_zoom(orig_zoom, plane)).astype(np.float32)

        image_tensor = torch.from_numpy(preprocessed[None, ...]).to(model.device)
        scale_tensor = torch.from_numpy(scale_factor[None, ...]).to(model.device)
        with torch.no_grad():
            logits = model.model(image_tensor, scale_tensor, None)

        logits_cpu = logits.detach().cpu()
        if plane == "sagittal":
            logits_np = map_prediction_sagittal2full(
                logits_cpu.numpy(),
                num_classes=model.get_num_classes(),
                lut=evaluator.lut,
            )
            logits_cpu = torch.from_numpy(logits_np)

        pred_labels = torch.argmax(logits_cpu, dim=1)
        mapped = du.map_label2aparc_aseg(pred_labels, evaluator.labels)
        pred_classes = du.split_cortex_labels(mapped.cpu().numpy()).astype(np.int16)
        pred_slice_volume = pred_classes[0, :, :][:, :, None]

        save_nifti(
            forward_dir / f"python_forward_pred_{plane}_slice_{slice_idx}.nii.gz",
            pred_slice_volume,
            affine,
        )
        timings["forward"] += (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        post_aseg = rta.reduce_to_aseg(pred_slice_volume.copy())
        try:
            post_mask = rta.create_mask(pred_slice_volume.copy(), 5, 4).astype(np.uint8)
        except AssertionError:
            post_mask = (pred_slice_volume > 0).astype(np.uint8)
        post_aseg[post_mask == 0] = 0
        try:
            post_aseg = rta.flip_wm_islands(post_aseg)
        except AssertionError:
            pass

        save_nifti(
            postprocess_dir / f"python_post_aseg_{plane}_slice_{slice_idx}.nii.gz",
            post_aseg.astype(np.int16),
            affine,
        )
        save_nifti(
            postprocess_dir / f"python_post_brainmask_{plane}_slice_{slice_idx}.nii.gz",
            post_mask.astype(np.uint8),
            affine,
        )
        timings["postprocess"] += (time.perf_counter() - t2) * 1000.0

    metadata = {
        "mode": "one_slice_geometry",
        "created_at_unix": int(time.time()),
        "input_mgz": str(input_mgz),
        "native_input_nii": str(native_input_nii),
        "slice_indices": slice_indices,
        "stage_dirs": {
            "preprocess": str(preprocess_dir),
            "forward pass": str(forward_dir),
            "postprocess": str(postprocess_dir),
        },
        "stage_timings_ms": {
            "preprocess": timings["preprocess"],
            "forward": timings["forward"],
            "postprocess": timings["postprocess"],
            "total": (time.perf_counter() - start_total) * 1000.0,
        },
    }

    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    latest_ref = fs_reference_root / "one_slice_geometry" / "latest_run.txt"
    latest_ref.parent.mkdir(parents=True, exist_ok=True)
    latest_ref.write_text(str(run_dir), encoding="utf-8")

    return run_dir


def run_subject140_full_volume_generation(repo_root: Path) -> Path:
    input_mgz, native_input_nii = resolve_input_paths(repo_root)

    fs_reference_root = repo_root / "app/gui/desktop/src-tauri/testing/data/fs_reference/Subject140"
    preprocess_dir = fs_reference_root / "preprocess"
    forward_dir = fs_reference_root / "forward pass"
    postprocess_dir = fs_reference_root / "postprocess"

    preprocess_dir.mkdir(parents=True, exist_ok=True)
    forward_dir.mkdir(parents=True, exist_ok=True)
    postprocess_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.perf_counter()

    native_img = nib.load(str(native_input_nii))
    header = native_img.header
    affine = native_img.affine
    orig_data = np.asanyarray(native_img.dataobj)

    conformed_input_path = preprocess_dir / "subject140_conformed_input.nii.gz"
    save_nifti(conformed_input_path, orig_data.astype(np.uint8), affine)

    t0 = time.perf_counter()
    generate_preprocess_artifacts(orig_data, affine, preprocess_dir)
    preprocess_ms = (time.perf_counter() - t0) * 1000.0

    evaluator = build_evaluator(repo_root)

    t1 = time.perf_counter()
    pred_data = evaluator.get_prediction(
        str(native_input_nii),
        orig_data,
        header.get_zooms(),
        affine,
    )
    forward_ms = (time.perf_counter() - t1) * 1000.0

    forward_pred_path = forward_dir / "python_forward_pred.nii.gz"
    save_nifti(forward_pred_path, pred_data.astype(np.int16), affine)

    t2 = time.perf_counter()
    brainmask = rta.create_mask(pred_data.copy(), 5, 4).astype(np.uint8)
    aseg = rta.reduce_to_aseg(pred_data.copy())
    aseg[brainmask == 0] = 0
    try:
        aseg = rta.flip_wm_islands(aseg)
    except AssertionError:
        pass
    postprocess_ms = (time.perf_counter() - t2) * 1000.0

    save_nifti(postprocess_dir / "python_post_brainmask.nii.gz", brainmask, affine)
    save_nifti(postprocess_dir / "python_post_aseg.nii.gz", aseg.astype(np.int16), affine)

    metadata = {
        "created_at_unix": int(time.time()),
        "input_mgz": str(input_mgz),
        "native_input_nii": str(native_input_nii),
        "stage_dirs": {
            "preprocess": str(preprocess_dir),
            "forward pass": str(forward_dir),
            "postprocess": str(postprocess_dir),
        },
        "stage_timings_ms": {
            "preprocess": preprocess_ms,
            "forward": forward_ms,
            "postprocess": postprocess_ms,
            "total": (time.perf_counter() - start_total) * 1000.0,
        },
        "artifacts": {
            "conformed_input": str(conformed_input_path),
            "forward_pred": str(forward_pred_path),
            "post_aseg": str(postprocess_dir / "python_post_aseg.nii.gz"),
            "post_brainmask": str(postprocess_dir / "python_post_brainmask.nii.gz"),
        },
    }

    metadata_path = fs_reference_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return fs_reference_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate full-volume legacy FastSurfer staged reference fixture")
    parser.add_argument("--repo-root", type=Path, default=None, help="FastSurfer repo root (auto-detected if omitted)")
    parser.add_argument(
        "--full-volume",
        action="store_true",
        help="Generate full-volume fixture (default generates one-slice-per-plane geometry fixture)",
    )
    args = parser.parse_args()

    repo_root = find_repo_root(args.repo_root or Path(__file__).resolve())
    if args.full_volume:
        fixture_root = run_subject140_full_volume_generation(repo_root)
        print(f"Generated full-volume fs reference fixture at: {fixture_root}")
    else:
        fixture_root = run_subject140_one_slice_geometry_generation(repo_root)
        print(f"Generated one-slice geometry fixture at: {fixture_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
