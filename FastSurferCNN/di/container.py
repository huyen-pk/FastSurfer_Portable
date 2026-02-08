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

"""Dependency injection container using the injector framework."""

import torch
import yacs.config
from injector import Injector, Module, provider, singleton

from FastSurferCNN.di.config_loader import ConfigLoader
from FastSurferCNN.di.factories import (
    DataLoaderFactory,
    LossFunctionFactory,
    ModelFactory
)


class FastSurferModule(Module):
    """
    Dependency injection module for FastSurfer components.

    This module configures the bindings for dependency injection,
    defining how components should be created and their lifecycles.
    """

    @singleton
    @provider
    def provide_config_loader(self) -> ConfigLoader:
        """
        Provide a singleton ConfigLoader instance.

        Returns
        -------
        ConfigLoader
            The configuration loader instance.
        """
        return ConfigLoader()

    @provider
    def provide_model_factory(self) -> ModelFactory:
        """
        Provide a ModelFactory instance.

        Returns
        -------
        ModelFactory
            The model factory instance.
        """
        return ModelFactory()

    @provider
    def provide_loss_function_factory(self) -> LossFunctionFactory:
        """
        Provide a LossFunctionFactory instance.

        Returns
        -------
        LossFunctionFactory
            The loss function factory instance.
        """
        return LossFunctionFactory()

    @provider
    def provide_dataloader_factory(self) -> DataLoaderFactory:
        """
        Provide a DataLoaderFactory instance.

        Returns
        -------
        DataLoaderFactory
            The data loader factory instance.
        """
        return DataLoaderFactory()


def create_injector(
) -> Injector:
    """
    Create and configure a dependency injector for FastSurfer.

    Parameters
    ----------
    cfg : yacs.config.CfgNode, optional
        Configuration to use.
    device : torch.device, optional
        Device to use.

    Returns
    -------
    Injector
        Configured dependency injector.

    Examples
    --------
    >>> injector = create_injector(cfg)
    >>> device_manager = injector.get(DeviceManager)
    >>> model_factory = injector.get(ModelFactory)
    """
    return Injector([FastSurferModule()])
