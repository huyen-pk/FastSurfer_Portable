#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def _center_slice(data: np.ndarray) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {data.shape}")
    return data[:, :, data.shape[2] // 2]


def _load_slice(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    return _center_slice(data)


def _auto_pick_latest_run(results_root: Path) -> Path:
    runs = sorted(
        [p for p in results_root.glob("parity_dice_assd_hd95_hdmax_icc_full_volume_subject140_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if not runs:
        raise FileNotFoundError(f"No run folders found under {results_root}")
    return runs[-1]


def _resolve_stage_files(run_dir: Path, stage: str) -> tuple[Path, Path]:
    if stage == "preprocess":
        rust = run_dir / "preprocess" / "rust_preprocess_axial.nii.gz"
        py = run_dir / "preprocess" / "python_preprocess_axial.nii.gz"
    elif stage == "forward":
        rust = run_dir / "forward_pass" / "rust_pred.nii.gz"
        py = run_dir / "forward_pass" / "python_pred.nii.gz"
        if not py.exists():
            py = run_dir / "forward_pass" / "python_golden_pred.nii.gz"
    elif stage == "postprocess_aseg":
        rust = run_dir / "postprocess" / "rust_post_aseg.nii.gz"
        py = run_dir / "postprocess" / "python_post_aseg.nii.gz"
    elif stage == "postprocess_brainmask":
        rust = run_dir / "postprocess" / "rust_post_brainmask.nii.gz"
        py = run_dir / "postprocess" / "python_post_brainmask.nii.gz"
    else:
        raise ValueError(f"Unknown stage: {stage}")

    if not rust.exists():
        raise FileNotFoundError(f"Rust artifact missing: {rust}")
    if not py.exists():
        raise FileNotFoundError(f"Python artifact missing: {py}")

    return rust, py


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect parity stage artifacts (Rust vs Python)")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to parity run directory (defaults to latest under testing/rust/results)",
    )
    parser.add_argument(
        "--stage",
        choices=["preprocess", "forward", "postprocess_aseg", "postprocess_brainmask"],
        default="forward",
        help="Stage to visualize",
    )
    parser.add_argument("--cmap", default="gray", help="Matplotlib colormap for main views")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save figure")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[5]
    results_root = repo_root / "app/gui/desktop/src-tauri/testing/rust/results"
    run_dir = args.run_dir if args.run_dir is not None else _auto_pick_latest_run(results_root)

    rust_path, py_path = _resolve_stage_files(run_dir, args.stage)
    rust_slice = _load_slice(rust_path).astype(np.float32)
    py_slice = _load_slice(py_path).astype(np.float32)

    if rust_slice.shape != py_slice.shape:
        raise ValueError(f"Shape mismatch rust={rust_slice.shape} python={py_slice.shape}")

    diff = rust_slice - py_slice
    abs_diff = np.abs(diff)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(f"Stage: {args.stage}\nRun: {run_dir.name}", fontsize=12)

    im0 = axes[0, 0].imshow(rust_slice, cmap=args.cmap)
    axes[0, 0].set_title("Rust center slice")
    axes[0, 0].axis("off")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(py_slice, cmap=args.cmap)
    axes[0, 1].set_title("Python center slice")
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(diff, cmap="coolwarm")
    axes[1, 0].set_title("Signed diff (Rust - Python)")
    axes[1, 0].axis("off")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(abs_diff, cmap="magma")
    axes[1, 1].set_title("Absolute diff")
    axes[1, 1].axis("off")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    print(f"Run dir: {run_dir}")
    print(f"Rust: {rust_path}")
    print(f"Python: {py_path}")
    print(f"Slice shape: {rust_slice.shape}")
    print(f"max|diff|={float(abs_diff.max()):.6f} mean|diff|={float(abs_diff.mean()):.6f}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to: {args.save}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
