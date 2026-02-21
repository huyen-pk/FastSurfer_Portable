from __future__ import annotations

import base64
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
        from FastSurferCNN.run_prediction import main as run_main

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
