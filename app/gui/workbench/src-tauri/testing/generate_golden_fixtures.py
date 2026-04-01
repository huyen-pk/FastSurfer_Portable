from __future__ import annotations

import tempfile
from pathlib import Path

import nibabel as nib


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]

    input_mgz = repo_root / "app/gui/workbench/src-tauri/testing/data/Subject140/140_orig.mgz"
    if not input_mgz.exists():
        raise FileNotFoundError(f"missing fixture input: {input_mgz}")

    out_dir = repo_root / "app/gui/workbench/src-tauri/testing/data/.tmp_e2e_output_py"
    out_dir.mkdir(parents=True, exist_ok=True)

    native_input = out_dir / "140_orig.native_input.nii.gz"
    python_pred = out_dir / "140_orig.python_pred.nii.gz"

    nib.save(nib.load(str(input_mgz)), str(native_input))

    import sys

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from app.backend.inference_service import FastSurferInferenceService

    service = FastSurferInferenceService()
    with tempfile.TemporaryDirectory(prefix="fastsurfer_py_fixture_") as tmp_out:
        prediction = service.predict_from_path(input_path=str(input_mgz), output_dir=tmp_out)
        pred_path = Path(str(prediction["output_path"]))
        if pred_path.is_dir() or not pred_path.exists():
            candidates = sorted(
                list(Path(tmp_out).rglob("pred.mgz"))
                + list(Path(tmp_out).rglob("pred.mgh"))
                + list(Path(tmp_out).rglob("pred.nii"))
                + list(Path(tmp_out).rglob("pred.nii.gz"))
            )
            if not candidates:
                all_outputs = sorted(
                    list(Path(tmp_out).rglob("*.mgz"))
                    + list(Path(tmp_out).rglob("*.mgh"))
                    + list(Path(tmp_out).rglob("*.nii"))
                    + list(Path(tmp_out).rglob("*.nii.gz"))
                )
                raise RuntimeError(
                    f"Could not resolve segmentation output file. prediction={prediction}, found={all_outputs}"
                )
            pred_path = candidates[0]
        nib.save(nib.load(str(pred_path)), str(python_pred))

    print(f"generated: {native_input}")
    print(f"generated: {python_pred}")


if __name__ == "__main__":
    main()
