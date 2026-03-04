from __future__ import annotations

import base64
import contextlib
import importlib
import io
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


_PERCENT_PATTERN = re.compile(r"(?<!\d)(\d{1,3})\s*%(?!\d)")
_FRACTION_PATTERN = re.compile(r"(?<!\d)(\d{1,5})\s*/\s*(\d{1,5})(?!\d)")


def _extract_progress_values_from_line(line: str) -> list[int]:
    progress_values: list[int] = []

    for match in _PERCENT_PATTERN.findall(line):
        try:
            progress_values.append(int(match))
        except ValueError:
            continue

    for done_raw, total_raw in _FRACTION_PATTERN.findall(line):
        try:
            done = int(done_raw)
            total = int(total_raw)
        except ValueError:
            continue

        if total <= 0 or done < 0:
            continue

        progress_values.append(int((done / total) * 100))

    deduped = sorted(set(progress_values))
    return deduped


def _get_default_ckpts_and_cfgs() -> tuple[dict[str, Any], dict[str, Any]]:
    from FastSurferCNN.utils.checkpoint import get_config_file, load_checkpoint_config_defaults

    config_file = get_config_file("FastSurferCNN")
    ckpts = load_checkpoint_config_defaults("checkpoint", config_file)
    cfgs = load_checkpoint_config_defaults("config", config_file)
    return ckpts, cfgs


def _normalize_path_map(paths: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in paths.items():
        if value is None:
            normalized[key] = value
            continue

        value_path = Path(str(value))
        if value_path.is_absolute() and value_path.exists():
            normalized[key] = str(value_path)
            continue

        resolved = _resolve_resource_path(str(value))
        normalized[key] = str(resolved)

    return normalized


def _ensure_onnx_model_aliases() -> None:
    onnx_dir = Path("onnx")
    if not onnx_dir.exists():
        return

    alias_pairs = [
        ("FastSurferVINN_Axial.onnx", "FastSurferVINN_axial.onnx"),
        ("FastSurferVINN_Coronal.onnx", "FastSurferVINN_coronal.onnx"),
        ("FastSurferVINN_Sagittal.onnx", "FastSurferVINN_sagittal.onnx"),
    ]

    for expected_name, existing_name in alias_pairs:
        expected_path = onnx_dir / expected_name
        existing_path = onnx_dir / existing_name
        if expected_path.exists() or not existing_path.exists():
            continue

        try:
            expected_path.symlink_to(existing_path.name)
        except OSError:
            shutil.copy2(existing_path, expected_path)


def _resource_base_dirs() -> list[Path]:
    bases: list[Path] = []

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        bases.append(Path(str(meipass)))

    if getattr(sys, "frozen", False):
        bases.append(Path(sys.executable).resolve().parent)

    bases.append(Path(__file__).resolve().parents[2])
    bases.append(Path.cwd())

    unique: list[Path] = []
    seen: set[str] = set()
    for base in bases:
        key = str(base)
        if key in seen:
            continue
        seen.add(key)
        unique.append(base)

    return unique


def _resolve_resource_path(relative_path: str) -> Path:
    for base in _resource_base_dirs():
        candidate = base / relative_path
        if candidate.exists():
            return candidate
    return Path(relative_path)


class FastSurferInferenceService:
    def __init__(self):
        ckpts, cfgs = _get_default_ckpts_and_cfgs()
        self.ckpts = _normalize_path_map(ckpts)
        self.cfgs = _normalize_path_map(cfgs)
        self.lut_path = str(
            _resolve_resource_path("FastSurferCNN/config/FreeSurferColorLUT.txt")
        )

    def run_prediction(
        self,
        input_path: Path,
        output_dir: Path,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> tuple[int | str, Path]:
        class _ProgressCapture(io.TextIOBase):
            def __init__(self, callback: Callable[[int, str], None] | None):
                self.callback = callback
                self._last_progress = -1
                self._residual = ""
                self._last_hint = ""

            def _emit(self, value: int, message: str) -> None:
                if not self.callback:
                    return

                bounded = max(1, min(95, int(value)))
                if bounded <= self._last_progress:
                    return

                self._last_progress = bounded
                self.callback(bounded, message)

            def write(self, s: str) -> int:
                if not s:
                    return 0

                text = self._residual + s
                lines = re.split(r"[\r\n]+", text)
                if text and text[-1] not in "\r\n":
                    self._residual = lines.pop() if lines else text
                else:
                    self._residual = ""

                for raw_line in lines:
                    line = raw_line.strip()
                    if not line:
                        continue

                    extracted_values = _extract_progress_values_from_line(line)
                    if extracted_values:
                        for extracted in extracted_values:
                            self._emit(extracted, f"Processing MRI: {max(1, min(95, extracted))}%")
                        continue

                    if self.callback and self._last_progress < 95 and line != self._last_hint:
                        self._last_hint = line
                        self._emit(self._last_progress + 1, "Processing MRI...")

                return len(s)

            def flush(self) -> None:
                if not self._residual:
                    return

                line = self._residual.strip()
                self._residual = ""
                if not line:
                    return

                extracted_values = _extract_progress_values_from_line(line)
                if extracted_values:
                    for extracted in extracted_values:
                        self._emit(extracted, f"Processing MRI: {max(1, min(95, extracted))}%")
                    return

                if self.callback and self._last_progress < 95 and line != self._last_hint:
                    self._last_hint = line
                    self._emit(self._last_progress + 1, "Processing MRI...")

        _ensure_onnx_model_aliases()
        os.environ.setdefault("FASTSURFER_DISABLE_ONNX", "1")
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
            "threads": 1,
            "batch_size": 1,
            "device": "cpu",
            "viewagg_device": "cpu",
            "lut": self.lut_path,
        }

        if progress_callback:
            progress_callback(1, "Starting MRI processing")

        progress_capture = _ProgressCapture(progress_callback)

        try:
            with contextlib.redirect_stdout(progress_capture), contextlib.redirect_stderr(progress_capture):
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

        if progress_callback:
            progress_callback(100, "MRI processing completed")

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
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict[str, Any]:
        resolved_input = Path(str(input_path)).expanduser().resolve()
        if not resolved_input.exists():
            raise FileNotFoundError(f"input_path does not exist: {resolved_input}")

        resolved_output = Path(str(output_dir)).expanduser().resolve()
        resolved_output.mkdir(parents=True, exist_ok=True)

        run_result, seg_path = self.run_prediction(
            input_path=resolved_input,
            output_dir=resolved_output,
            progress_callback=progress_callback,
        )

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
