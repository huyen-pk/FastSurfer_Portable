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

import os

# IMPORTS
import time
from typing import Optional

import numpy as np
import torch
import yacs.config
from numpy import typing as npt
from pandas import DataFrame
from torch.utils.data import DataLoader
from torchvision import transforms

from FastSurferCNN.data_loader.augmentation import ToTensorTest
from FastSurferCNN.data_loader.data_utils import map_prediction_sagittal2full
from FastSurferCNN.data_loader.dataset import MultiScaleOrigDataThickSlices
from FastSurferCNN.inference import Inference
from FastSurferCNN.models.networks import build_model
from FastSurferCNN.utils import logging

logger = logging.getLogger(__name__)


class InferenceONNX(Inference):
    """Model evaluation class to run inference using FastSurferCNN.

    Attributes
    ----------
    permute_order : Dict[str, Tuple[int, int, int, int]]
        Permutation order for axial, coronal, and sagittal
    device : Optional[torch.device])
        Device specification for distributed computation usage.
    default_device : torch.device
        Default device specification for distributed computation usage.
    cfg : yacs.config.CfgNode
        Configuration Node
    model_parallel : bool
        Option for parallel run
    model : torch.nn.Module
        Neural network model
    model_name : str
        Name of the model
    alpha : Dict[str, float]
        Alpha values for different planes.
    post_prediction_mapping_hook
        Hook for post prediction mapping.

    Methods
    -------
    setup_model
        Set up the initial model
    set_cfg
        Set configuration node
    to
        Moves and/or casts the parameters and buffers.
    load_checkpoint
        Load the checkpoint
    eval
        Evaluate predictions
    run
        Run the loaded model
    """

    permute_order: dict[str, tuple[int, int, int, int]]
    device: torch.device | None
    default_device: torch.device
    ONNX_FOLDER: str

    def __init__(
        self,
        cfg: yacs.config.CfgNode,
        device: torch.device = None,
        ckpt: str = "",
        lut: None | str | np.ndarray | DataFrame = None,
    ):
        """
        Construct Inference object.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            Configuration Node.
        device : torch.device
            Device specification for distributed computation usage.
        ckpt : str
            String or os.PathLike object containing the name to the checkpoint file (Default value = "").
        lut : str, np.ndarray, DataFrame, optional
             Lookup table for mapping.
        """
        # Set random seed from configs.
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        self.set_cfg(cfg)

        # Switch on denormal flushing for faster CPU processing
        # seems to have less of an effect on VINN than old CNN
        torch.set_flush_denormal(True)

        self.default_device = device

        # Options for parallel run
        self.model_parallel = (
            torch.cuda.device_count() > 1
            and self.default_device.type == "cuda"
            and self.default_device.index is None
        )

        # Initial model setup
        self.model_name = self.cfg.MODEL.MODEL_NAME
        self.ONNX_FOLDER = self.cfg.ONNX_FOLDER
        self.setup_model(cfg, device=self.default_device)

        self.alpha = {"sagittal": 0.2}
        self.permute_order = {
            "axial": (3, 0, 2, 1),
            "coronal": (2, 3, 0, 1),
            "sagittal": (0, 3, 2, 1),
        }
        self.lut = lut

    def setup_model(self, cfg=None, device: torch.device = None):
        """
        Set up the model.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            Configuration Node (Default value = None).
        device : torch.device
            Device specification for distributed computation usage. (Default value = None).
        """
        if device is None:
            self.device = self.default_device
        else:
            self.device = device

        # Load the ONNX model and create an inference session
        import onnxruntime as ort
        # Providers: Use ['CUDAExecutionProvider', 'CPUExecutionProvider'] for GPU
        model_name = f"{self.model_name}_{self.cfg.DATA.PLANE.capitalize()}.onnx"
        model_path = os.path.join(self.ONNX_FOLDER, model_name)
        self.model  = ort.InferenceSession(model_path)
        
    def set_cfg(self, cfg: yacs.config.CfgNode):
        """
        Set the configuration node.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            Configuration node.
        """
        self.cfg = cfg
        self.ONNX_FOLDER = cfg.ONNX_FOLDER

    def to(self, device: torch.device | None = None):
        """
        Move and/or cast the parameters and buffers.

        Parameters
        ----------
        device : Optional[torch.device]
            The desired device of the parameters and buffers in this module (Default value = None).
        """
        if self.model_parallel:
            raise RuntimeError("Moving the model to other devices is not supported for multi-device models.")
        _device = self.default_device if device is None else device
        self.device = _device
        self.model.to(device=_device)

    def get_modelname(self) -> str:
        """
        Return the model name.

        Returns
        -------
        str
            The name of the model.
        """
        return self.model_name

    def get_cfg(self) -> yacs.config.CfgNode:
        """
        Return the configurations.

        Returns
        -------
        yacs.config.CfgNode
            Configuration node.
        """
        return self.cfg

    def get_num_classes(self) -> int:
        """
        Return the number of classes.

        Returns
        -------
        int
            The number of classes.
        """
        return self.cfg.MODEL.NUM_CLASSES

    def get_plane(self) -> str:
        """
        Return the plane.

        Returns
        -------
        str
            The plane used in the model.
        """
        return self.cfg.DATA.PLANE

    def get_model_height(self) -> int:
        """
        Return the model height.

        Returns
        -------
        int
            The height of the model.
        """
        return self.cfg.MODEL.HEIGHT

    def get_model_width(self) -> int:
        """
        Return the model width.

        Returns
        -------
        int
            The width of the model.
        """
        return self.cfg.MODEL.WIDTH

    def get_device(self) -> torch.device:
        """
        Return the device.

        Returns
        -------
        torch.device
            The device used for computation.
        """
        return self.device

    @torch.no_grad()
    def eval(
        self,
        init_pred: torch.Tensor,
        val_loader: DataLoader,
        *,
        out_scale: Optional = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform prediction and inplace-aggregate views into pred_prob.

        Parameters
        ----------
        init_pred : torch.Tensor
            Initial prediction.
        val_loader : DataLoader
            Validation loader.
        out_scale : Optional
            Output scale (Default value = None).
        out : torch.Tensor, optional
            Previous prediction tensor (Default value = None).

        Returns
        -------
        torch.Tensor
            Prediction probability tensor.
        """
        # we should check here, whether the DataLoader is a Random or a SequentialSampler, but we cannot easily.
        if not isinstance(val_loader.sampler, torch.utils.data.SequentialSampler):
            logger.warning(
                "The Validation loader seems to not use the SequentialSampler. This might interfere with "
                "the assumed sorting of batches."
            )

        start_index = 0
        plane = self.cfg.DATA.PLANE
        index_of_current_plane = self.permute_order[plane].index(0)
        target_shape = init_pred.shape
        ii = [slice(None) for _ in range(4)]
        pred_ii = tuple(slice(i) for i in target_shape[:3])

        from tqdm import tqdm
        from tqdm.contrib.logging import logging_redirect_tqdm

        if out is None:
            out = init_pred.detach().clone()
        log_batch_idx = None
        logger.info("Inference with ONNX model")
        logger.info("predii shape: %s", pred_ii)
        logger.info("Target shape: %s", target_shape)
        with logging_redirect_tqdm():
            try:
                
                for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), unit="batch"):
                    log_batch_idx = batch_idx
                    # move data to the model device
                    images, scale_factors = batch["image"].to(self.device), batch["scale_factor"].to(self.device)
                    # predict the current batch, outputs logits
                    session = self.model
                    input_data = {
                        "x": images.detach().cpu().numpy(),
                        "scale_factor": scale_factors.detach().cpu().numpy(),
                        "scale_factor_out": out_scale
                    }
                    # Run prediction
                    outputs = session.run(None, input_data)
                    # 'outputs' is a list of results matching your output_names
                    logits = outputs[0]
                    # print("Original Prediction shape:", out.shape)
                    pred = torch.tensor(logits).to(self.device)
                    logger.info("Prediction shape: %s", pred.shape)
                    logger.info("Logits shape: %s", logits.shape)
            
                    batch_size = pred.shape[0]
                    end_index = start_index + batch_size

                    # check if we need a special mapping (e.g. as for sagittal)
                    if self.get_plane() == "sagittal":
                        pred = map_prediction_sagittal2full(pred, num_classes=self.get_num_classes(), lut=self.lut)

                    # permute the prediction into the out slice order
                    # a.k.a rearrange the output dimensions of 2D slices so that they match their position on the original 3D volume.
                    pred = pred.permute(*self.permute_order[plane]).to(out.device)  # the to-operation is implicit

                    # cut prediction to the image size
                    pred = pred[pred_ii]

                    # add prediction logits into the output (same as multiplying probabilities)
                    ii[index_of_current_plane] = slice(start_index, end_index) # indexing the current batch (slab) in the output tensor
                    # select a slab of the output tensor and add the current batch prediction to it
                    out[tuple(ii)].add_(pred.half(), alpha=self.alpha.get(plane, 0.4)) # pred.half() => halving precision
                    start_index = end_index

            except:
                logger.error(f"Exception in batch {log_batch_idx + 1} of {plane} inference.")
                raise
            else:
                logger.info(f"Inference on {log_batch_idx + 1} batches for {plane} successful")

        return out

    @torch.no_grad()
    def run(
        self,
        init_pred: torch.Tensor,
        img_filename: str,
        orig_data: npt.NDArray,
        orig_zoom: npt.NDArray,
        out: torch.Tensor | None = None,
        out_res: int | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """
        Run the loaded model on the data (T1) from orig_data and
        img_filename (for messages only) with scale factors orig_zoom.

        Parameters
        ----------
        init_pred : torch.Tensor
            Initial prediction.
        img_filename : str
            Original image filename.
        orig_data : npt.NDArray
            Original image data.
        orig_zoom : npt.NDArray
            Original zoom.
        out : Optional[torch.Tensor]
            Updated output tensor (Default = None).
        out_res : Optional[int]
            Output resolution (Default value = None).
        batch_size : int, optional
            Batch size.

        Returns
        -------
        torch.Tensor
            Prediction probability tensor.
        """
        # Set up DataLoader
        test_dataset = MultiScaleOrigDataThickSlices(
            orig_data,
            orig_zoom,
            self.cfg,
            transforms=transforms.Compose([ToTensorTest()]),
        )

        test_data_loader = DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=self.cfg.TEST.BATCH_SIZE if batch_size is None else batch_size,
        )

        # Run evaluation
        start = time.time()
        out = self.eval(init_pred, test_data_loader, out=out, out_scale=out_res)
        time_delta = time.time() - start
        logger.info(
            f"{self.cfg.DATA.PLANE.capitalize()} inference on {img_filename} finished in {time_delta:0.4f} seconds"
        )

        return out
