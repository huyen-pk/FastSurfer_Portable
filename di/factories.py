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

"""Factory classes for dependency injection."""


from typing import Literal
import logging
import os
import torch
import yacs.config
from torch.utils.data import DataLoader

from FastSurferCNN.inference import Inference
try:
    from di.wrappers import FastSurferCNN_FL_Trainer, TrainerBase
except ImportError:
    from di.wrappers import TrainerBase

    FastSurferCNN_FL_Trainer = None

from FastSurferCNN.data_loader import loader
from FastSurferCNN.models import losses, networks               

LOGGER = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory for creating neural network models.
    This factory encapsulates model creation logic and allows for
    dependency injection of model instances.

    Methods
    -------
    create_model(cfg: yacs.config.CfgNode) -> torch.nn.Module
        Creates a model based on the configuration.
    """
    def __init__(self, cfg: yacs.config.CfgNode):
        self.set_config(cfg)

    def set_config(self, cfg: yacs.config.CfgNode):
        """
        Set the configuration for the model factory.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            Configuration node to set for the factory.
        """
        self.__cfg = cfg

    def create_model(self) -> torch.nn.Module:
        """
        Create a model based on configuration.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            Configuration node containing model specifications.

        Returns
        -------
        torch.nn.Module
            The created neural network model.
        """
        return networks.build_model(self.__cfg)

class InferenceEngineFactory:
    """
    Factory for creating inference engine instances.

    This factory encapsulates inference engine creation logic and allows
    for dependency injection of inference engine instances.

    Methods
    -------
    create_inference_engine(cfg: yacs.config.CfgNode) -> Optional[InferenceONNX]
        Creates an inference engine instance based on the configuration.
    """

    def create_inference_engine(self, cfg: yacs.config.CfgNode) -> torch.nn.Module | None:
        """
        Create an inference engine instance based on configuration.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            The configuration node to retrieve settings from.

        Returns
        -------
        torch.nn.Module | None
             The inference engine instance if enabled, otherwise None.
        """
        disable_onnx = os.getenv("FASTSURFER_DISABLE_ONNX", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if disable_onnx:
            return Inference(cfg=cfg, device=torch.device("cpu"))

        if cfg.ONNX_FOLDER is not None:
            from FastSurferCNN.inference_onnx import InferenceONNX
            try:
                return InferenceONNX(cfg=cfg)
            except Exception as exc:
                LOGGER.warning(
                    "Falling back to PyTorch inference engine because ONNX initialization failed: %s",
                    exc,
                )
        return Inference(cfg=cfg, device=torch.device("cpu"))
class LossFunctionFactory:
    """
    Factory for creating loss functions.

    This factory encapsulates loss function creation logic and allows
    for dependency injection of loss function instances.

    Methods
    -------
    create_loss_function(cfg: yacs.config.CfgNode) -> torch.nn.Module
        Creates a loss function based on the configuration.
    """

    def __init__(self, cfg: yacs.config.CfgNode):
        self.set_config(cfg)

    def set_config(self, cfg: yacs.config.CfgNode):
        """
        Set the configuration for the loss function.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            Configuration node to set for the factory.
        """
        self.__cfg = cfg

    def create_loss_function(self) -> torch.nn.Module:
        """
        Create a loss function based on configuration.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            Configuration node containing loss function specifications.

        Returns
        -------
        torch.nn.Module
            The created loss function.
        """
        return losses.get_loss_func(self.__cfg)


class DataLoaderFactory:
    """
    Factory for creating data loaders.

    This factory encapsulates data loader creation logic and allows
    for dependency injection of data loader instances.

    Methods
    -------
    create_dataloader(cfg: yacs.config.CfgNode, mode: str) -> DataLoader
        Creates a data loader based on the configuration.
    """

    def __init__(self, cfg: yacs.config.CfgNode):
        self.set_config(cfg)

    def set_config(self, cfg: yacs.config.CfgNode):
        """
        Set the configuration for the data loader factory.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            Configuration node to set for the factory.
        """
        self.__cfg = cfg

    def create_dataloader(
        self, mode: str
    ) -> DataLoader:
        """
        Create a data loader based on configuration.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            Configuration node containing data loader specifications.
        mode : str
            The mode for the data loader ('train', 'val', or 'test').

        Returns
        -------
        DataLoader
            The created data loader.
        """
        return loader.get_dataloader(self.__cfg, mode)

class ModelTrainerFactory:
    """
    Factory for creating model trainers.

    This factory encapsulates model trainer creation logic and allows
    for dependency injection of model trainer instances.

    Methods
    -------
    create_model_trainer(cfg: yacs.config.CfgNode) -> Trainer
        Creates a model trainer based on the configuration.
    """

    def __init__(self, cfg: yacs.config.CfgNode):
        self.set_config(cfg)

    def set_config(self, cfg: yacs.config.CfgNode):
        """
        Set the configuration for the model trainer factory.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            Configuration node to set for the factory.
        """
        self.__cfg = cfg

    def create_model_trainer(
            self, 
            model_name: Literal["FastSurferCNN", "HypVINN", "CorpusCallosum", "CerebNet" ]) -> TrainerBase | None:
        """
        Create a model trainer based on configuration.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            Configuration node containing model trainer specifications.

        Returns
        -------
        Trainer
            The created model trainer.
        """
        trainer: TrainerBase = None
        if(model_name == "FastSurferCNN"):
            if FastSurferCNN_FL_Trainer is None:
                raise RuntimeError(
                    "FastSurferCNN_FL_Trainer is unavailable. Install optional federated learning dependencies to use training wrappers."
                )
            trainer = FastSurferCNN_FL_Trainer(self.__cfg)

        return trainer