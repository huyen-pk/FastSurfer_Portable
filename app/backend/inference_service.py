from __future__ import annotations

import base64
import importlib
import shutil
import tempfile
from pathlib import Path
from typing import Any


def _get_default_ckpts_and_cfgs() -> tuple[dict[str, Any], dict[str, Any]]:
    from FastSurferCNN.utils.checkpoint import get_config_file, load_checkpoint_config_defaults

    config_file = get_config_file("FastSurferCNN")
    ckpts = load_checkpoint_config_defaults("checkpoint", config_file)
    cfgs = load_checkpoint_config_defaults("config", config_file)
    return ckpts, cfgs


class FastSurferInferenceService:
    def __init__(self):
        self.ckpts, self.cfgs = _get_default_ckpts_and_cfgs()
        self.lut_path = str(Path("FastSurferCNN") / "config" / "FreeSurferColorLUT.txt")

    def run_prediction(self, input_path: Path, output_dir: Path) -> tuple[int | str, Path]:
        run_prediction_module = importlib.import_module("FastSurferCNN.run_prediction")
        run_main = getattr(run_prediction_module, "main", None)

        if not callable(run_main) and run_main is not None:
            nested_main = getattr(run_main, "main", None)
            if callable(nested_main):
                run_main = nested_main

        if not callable(run_main):
            raise RuntimeError(
                f"FastSurferCNN.run_prediction.main is not callable (resolved type: {type(run_main).__name__})"
            )

        kwargs = {
            "orig_name": str(input_path),
            "out_dir": str(output_dir),
            "pred_name": "pred.mgz",
            "ckpt_ax": str(self.ckpts.get("axial")),
            "ckpt_sag": str(self.ckpts.get("sagittal")),
            "ckpt_cor": str(self.ckpts.get("coronal")),
            "cfg_ax": str(self.cfgs.get("axial")),
            "cfg_sag": str(self.cfgs.get("sagittal")),
            "cfg_cor": str(self.cfgs.get("coronal")),
            "async_io": False,
            "batch_size": 1,
            "device": "cpu",
            "viewagg_device": "cpu",
            "lut": self.lut_path,
        }

        try:
            run_result = run_main(**kwargs)
        except SystemExit as exc:
            run_result = exc.code

        seg_files = (
            list(output_dir.rglob("*.mgz"))
            + list(output_dir.rglob("*.mgh"))
            + list(output_dir.rglob("*.nii"))
            + list(output_dir.rglob("*.nii.gz"))
        )
        if not seg_files:
            raise RuntimeError(f"No segmentation file produced. run_prediction result={run_result}")

        return run_result, seg_files[0]

    @staticmethod
    def is_supported_image(path: Path) -> bool:
        lower = str(path).lower()
        return (
            lower.endswith(".nii")
            or lower.endswith(".nii.gz")
            or lower.endswith(".mgz")
            or lower.endswith(".mgh")
        )

    def collect_images_recursive(self, directory: Path) -> list[Path]:
        resolved_directory = directory.expanduser().resolve()
        if not resolved_directory.exists() or not resolved_directory.is_dir():
            return []

        collected: list[Path] = []
        for entry in resolved_directory.rglob("*"):
            if entry.is_file() and self.is_supported_image(entry):
                collected.append(entry)
        return collected

    def resolve_input_paths(self, *, file_paths: list[str], folder_paths: list[str]) -> list[Path]:
        resolved: list[Path] = []

        for file_path in file_paths:
            path = Path(file_path).expanduser().resolve()
            if path.exists() and path.is_file() and self.is_supported_image(path):
                resolved.append(path)

        for folder_path in folder_paths:
            folder = Path(folder_path).expanduser().resolve()
            if folder.exists() and folder.is_dir():
                resolved.extend(self.collect_images_recursive(folder))

        unique_sorted = sorted({str(path): path for path in resolved}.values(), key=lambda p: str(p))
        return unique_sorted

    def predict_from_path(
        self,
        input_path: str | Path,
        *,
        output_dir: str | Path,
        return_base64: bool = False,
    ) -> dict[str, Any]:
        resolved_input = Path(str(input_path)).expanduser().resolve()
        if not resolved_input.exists():
            raise FileNotFoundError(f"input_path does not exist: {resolved_input}")

        resolved_output = Path(str(output_dir)).expanduser().resolve()
        resolved_output.mkdir(parents=True, exist_ok=True)

        run_result, seg_path = self.run_prediction(input_path=resolved_input, output_dir=resolved_output)

        result: dict[str, Any] = {
            "run_result": run_result,
            "output_path": str(seg_path),
            "output_filename": seg_path.name,
        }
        if return_base64:
            result["output_base64"] = base64.b64encode(seg_path.read_bytes()).decode("utf-8")
        return result

    def predict_batch_from_paths(self, *, file_paths: list[str], folder_paths: list[str]) -> dict[str, Any]:
        requested_paths = self.resolve_input_paths(file_paths=file_paths, folder_paths=folder_paths)
        if not requested_paths:
            raise ValueError("No valid input image files selected (.nii, .nii.gz, .mgz, .mgh).")

        results: list[dict[str, Any]] = []
        result_directories: set[str] = set()
        for input_path in requested_paths:
            output_dir = Path(tempfile.mkdtemp(prefix="fastsurfer_ipc_out_"))
            prediction = self.predict_from_path(input_path=input_path, output_dir=output_dir)
            output_path = str(prediction["output_path"])
            parent_dir = str(Path(output_path).parent)
            if parent_dir:
                result_directories.add(parent_dir)
            results.append(
                {
                    "input_path": str(input_path),
                    "output_path": output_path,
                    "output_filename": prediction["output_filename"],
                    "run_result": prediction["run_result"],
                }
            )

        requested_as_str = [str(path) for path in requested_paths]
        sorted_result_directories = sorted(result_directories)
        if sorted_result_directories:
            ack_message = (
                f"Processing started for {len(requested_as_str)} path(s). "
                f"Results directory: {', '.join(sorted_result_directories)}"
            )
        else:
            ack_message = f"Processing started for {len(requested_as_str)} path(s)."

        return {
            "ack_message": ack_message,
            "requested_paths": requested_as_str,
            "results": results,
            "result_directories": sorted_result_directories,
        }

    def predict_from_bytes(
        self,
        file_base64: str,
        *,
        file_name: str = "input.nii.gz",
    ) -> dict[str, Any]:
        workdir = Path(tempfile.mkdtemp(prefix="fastsurfer_ipc_in_"))
        try:
            input_path = workdir / file_name
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_bytes(base64.b64decode(file_base64))

            output_dir = workdir / "out"
            output_dir.mkdir(parents=True, exist_ok=True)

            run_result, seg_path = self.run_prediction(input_path=input_path, output_dir=output_dir)
            return {
                "run_result": run_result,
                "output_filename": seg_path.name,
                "output_base64": base64.b64encode(seg_path.read_bytes()).decode("utf-8"),
            }
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
