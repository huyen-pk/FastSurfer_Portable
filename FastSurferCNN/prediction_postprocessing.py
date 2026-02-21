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

"""
Arithmetic and post-processing operations for model predictions.

This module separates the arithmetic/post-processing operations from the core
inference logic in run_prediction.py.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from numpy import typing as npt

from FastSurferCNN.data_loader import data_utils as du
from FastSurferCNN.utils import logging

LOGGER = logging.getLogger(__name__)


def init_pred_prob(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    requires_grad: bool = False,
) -> torch.Tensor:
    """
    Initialize the prediction probability tensor with zeros.

    Parameters
    ----------
    shape : tuple of int
        Shape of the prediction tensor (height, width, depth, num_classes).
    device : torch.device
        Device to allocate the tensor on.
    dtype : torch.dtype, default=torch.float16
        Data type for the tensor.
    requires_grad : bool, default=False
        Whether to track gradients.

    Returns
    -------
    torch.Tensor
        Zero-initialized prediction probability tensor.
    """
    return torch.zeros(shape, device=device, dtype=dtype, requires_grad=requires_grad)


def postprocess_pred(
    pred_prob: torch.Tensor,
    back_to_native: Callable[[torch.Tensor], torch.Tensor],
    labels: npt.NDArray[Any],
) -> np.ndarray:
    """
    Apply arithmetic post-processing to model prediction probabilities.

    Converts raw prediction probabilities to final hard label assignments by:
    1. Computing argmax to get hard predictions.
    2. Reordering from LIA orientation back to the native orientation.
    3. Mapping network output indices to FreeSurfer label space.
    4. Splitting cortex labels.

    Parameters
    ----------
    pred_prob : torch.Tensor
        Aggregated prediction probability tensor (H x W x D x num_classes).
    back_to_native : callable
        Function to reorder the prediction tensor from LIA orientation back to
        the native image orientation.
    labels : npt.NDArray
        Array of label IDs used to map network output to FreeSurfer label space.

    Returns
    -------
    np.ndarray
        Final predicted segmentation labels in FreeSurfer label space.
    """
    # Get hard predictions via argmax over class dimension
    pred_classes = torch.argmax(pred_prob, 3)
    del pred_prob
    # Reorder from LIA orientation back to native orientation
    pred_classes = back_to_native(pred_classes)
    # Map network output indices to FreeSurfer label space
    pred_classes = du.map_label2aparc_aseg(pred_classes, labels)
    # Convert to numpy array and split cortex labels
    # TODO: split_cortex_labels requires a numpy ndarray input, maybe we can also use Mapper here
    pred_classes = du.split_cortex_labels(pred_classes.cpu().numpy())
    return pred_classes
