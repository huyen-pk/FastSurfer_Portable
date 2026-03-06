# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# IMPORTS
import argparse
import sys
import warnings
from collections.abc import Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy import typing as npt

from FastSurferCNN.data_loader.conform import orientation_to_ornts, to_target_orientation
from FastSurferCNN.utils.arg_types import OrientationType, VoxSizeOption
from FastSurferCNN.utils.checkpoint import get_checkpoints, get_config_file, load_checkpoint_config_defaults
from FastSurferCNN.utils.common import SubjectList
from FastSurferCNN.utils.parallel import SerialExecutor
from FastSurferCNN.utils.parser_defaults import SubjectDirectoryConfig
from FastSurferCNN.run_prediction import RunModelOnData, make_parser
from torch.utils.data import DataLoader
from torchvision import transforms
from FastSurferCNN.data_loader.augmentation import ToTensorTest
from FastSurferCNN.data_loader.dataset import MultiScaleOrigDataThickSlices
from FastSurferCNN.utils import logging


LOGGER = logging.getLogger(__name__)

##
# Input array preparation
##


class RunModelOnDataAndConvert:
    """
    Run the model prediction on given data and convert to different formats.
    Adapted from run_prediction.py.

    Attributes
    ----------

    Methods
    -------
    __init__()
        Construct object.
    convert2onnx()
        Convert model to ONNX format.
    """

    def __init__(
            self,
            runner: RunModelOnData
    ):
        """
        Construct RunModelOnData object.

        Parameters
        ----------
        viewagg_device : str, default="auto"
            Device to run viewagg on. Can be auto, cuda or cpu.
        """
        # TODO Fix docstring of RunModelOnData.__init__
        self.runner = runner

    @property
    def pool(self) -> Executor:
        """
        Return, and maybe create the objects executor object (with the number of threads
        specified in __init__).
        """
        if not hasattr(self, "_pool"):
            self._pool = ThreadPoolExecutor(self._threads) if self._async_io else SerialExecutor()
        return self._pool

    def __del__(self):
        """Class destructor."""
        if hasattr(self, "_pool"):
            # only wait on futures, if we specifically ask (see end of the script, so we
            # do not wait if we encounter a fail case)
            self._pool.shutdown(True)

    def data_prep(self, orig_data: np.ndarray, zoom: np.ndarray | Sequence[int], affine: npt.NDArray[float]):
            if not np.allclose(_zoom := np.asarray(zoom), np.mean(zoom), atol=1e-4, rtol=1e-3):
                msg = "FastSurfer support for anisotropic images is experimental, we detected the following voxel sizes"
                LOGGER.warning(f"{msg}: {np.round(_zoom, decimals=4).tolist()}!")

            orig_in_lia, _ = to_target_orientation(orig_data, affine, target_orientation="LIA")
            _ornt_transform, _ = orientation_to_ornts(affine, target_orientation="LIA")
            _zoom = _zoom[_ornt_transform[:, 0]]
            return orig_in_lia, _zoom
    
    def convert2onnx(
        self, orig_data: np.ndarray, zoom: np.ndarray | Sequence[int], affine: npt.NDArray[float], batch_size: int | None = None
    ) -> np.ndarray:
        """
        Run and get prediction.

        Parameters
        ----------
        orig_data : np.ndarray
            Original image data.
        zoom : np.ndarray, tuple
            Original zoom.
        affine : npt.NDArray[float]
            Original affine.
        batch_size : int, optional
        """

        orig_in_lia, _zoom = self.data_prep(orig_data, zoom, affine)

        # inference and view aggregation, export model for each plane
        for plane, inference_engine in self.runner.models.items():
            LOGGER.info(f"Run {plane} prediction")
            self.runner.set_model(plane)

            # Set up DataLoader
            test_dataset = MultiScaleOrigDataThickSlices(
                orig_in_lia,
                _zoom,
                inference_engine.cfg,
                transforms=transforms.Compose([ToTensorTest()]),
            )
            test_data_loader = DataLoader(
                dataset=test_dataset,
                shuffle=False,
                batch_size=inference_engine.cfg.TEST.BATCH_SIZE if batch_size is None else batch_size,
            )
            images = next(iter(test_data_loader))
            inputs, scale_factors = images["image"].to(inference_engine.device), images["scale_factor"].to(inference_engine.device)
            # Define symbolic dimensions
            from torch.export import Dim
            batch = Dim("batch")
            height = Dim("height")
            width = Dim("width")
            # Create the dynamic_shapes object
            dynamic_shapes = {
                "x": {0: batch, 2: height, 3: width},
                "scale_factor": {0: batch},
            }
            from torch.export import draft_export
            exported_program = draft_export(
                inference_engine.model,
                args=(inputs, scale_factors), 
                dynamic_shapes=dynamic_shapes,
                strict=False
            )
            model_name = f"FastSurferVINN_{plane}.onnx"
            # Export to ONNX
            torch.onnx.export(
                exported_program, 
                (inputs, scale_factors),
                model_name,
                export_params=True,        # Store the trained parameter weights inside the model file
                opset_version=18,          # Standard opset for broad compatibility
                do_constant_folding=True,  # Optimization
            )
            LOGGER.info(
                f"Model plane {plane} converted to {model_name}"
            )

def main(
        *,
        orig_name: Path | str,
        out_dir: Path,
        pred_name: str,
        ckpt_ax: Path,
        ckpt_sag: Path,
        ckpt_cor: Path,
        cfg_ax: Path,
        cfg_sag: Path,
        cfg_cor: Path,
        conf_name: str = "mri/orig.mgz",
        in_dir: Path | None = None,
        sid: str | None = None,
        search_tag: str | None = None,
        csv_file: str | Path | None = None,
        lut: Path | str | None = None,
        remove_suffix: str = "",
        brainmask_name: str = "mri/mask.mgz",
        vox_size: VoxSizeOption = "min",
        device: str = "auto",
        viewagg_device: str = "auto",
        batch_size: int = 1,
        orientation: OrientationType = "lia",
        image_size: bool = True,
        async_io: bool = True,
        threads: int = -1,
        conform_to_1mm_threshold: float = 0.95,
        **kwargs,
) -> Literal[0] | str:

    if len(kwargs) > 0:
        LOGGER.warning(f"Unknown arguments {list(kwargs.keys())} in {__file__}:main.")

    # Download checkpoints if they do not exist
    # see utils/checkpoint.py for default paths
    LOGGER.info("Checking or downloading default checkpoints ...")

    config_file = get_config_file("FastSurferCNN")

    urls = load_checkpoint_config_defaults("url", filename=config_file)

    get_checkpoints(ckpt_ax, ckpt_cor, ckpt_sag, urls=urls)

    config = SubjectDirectoryConfig(
        orig_name=orig_name,
        pred_name=pred_name,
        conf_name=conf_name,
        in_dir=in_dir,
        csv_file=csv_file,
        sid=sid,
        search_tag=search_tag,
        brainmask_name=brainmask_name,
        remove_suffix=remove_suffix,
        out_dir=out_dir,
    )
    slist_kwargs = {"segfile": "pred_name"}
    if out_dir is not None and out_dir != Path(""):
        config.copy_orig_name = "mri/orig/001.mgz"
        slist_kwargs["copy_orig_name"] = "copy_orig_name"

    try:
        # Get all subjects of interest
        subjects = SubjectList(config, **slist_kwargs)
        subjects.make_subjects_dir()

        # Set Up Model
        runner = RunModelOnData(
            lut=lut,
            ckpt_ax=ckpt_ax,
            ckpt_sag=ckpt_sag,
            ckpt_cor=ckpt_cor,
            cfg_ax=cfg_ax,
            cfg_sag=cfg_sag,
            cfg_cor=cfg_cor,
            device=device,
            viewagg_device=viewagg_device,
            threads=threads,
            batch_size=batch_size,
            vox_size=vox_size,
            orientation=orientation,
            image_size=image_size,
            async_io=async_io,
            conform_to_1mm_threshold=conform_to_1mm_threshold,
        )
    except RuntimeError as e:
        return e.args[0]


    iter_subjects = runner.pipeline_conform_and_save_orig(subjects)
    converter = RunModelOnDataAndConvert(runner)
    _, (orig_img, data_array) = next(iter_subjects)
    converter.convert2onnx(data_array, orig_img.header.get_zooms(), orig_img.affine)
            

if __name__ == "__main__":
    parser = make_parser()
    _args = parser.parse_args()

    # Set up logging
    from FastSurferCNN.utils.logging import setup_logging
    setup_logging(_args.log_name)

    sys.exit(main(**vars(_args)))
