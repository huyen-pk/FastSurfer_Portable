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

from typing import TYPE_CHECKING

import yacs.config
from injector import Injector, Module, provider, singleton

from FastSurferCNN.inference import Inference as FastSurferCNNInference
from di.config_loader import ConfigLoader, FLConfigLoader
from di.factories import (
    DataLoaderFactory,
    InferenceEngineFactory,
    LossFunctionFactory,
    ModelFactory
)
from di.federated_factories import FederatedBackendFactory
from fl import FederatedClientAPI
from fl.backends import FederatedBackend

if TYPE_CHECKING:
    pass


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

    @singleton
    @provider
    def provide_fl_config_loader(self) -> FLConfigLoader:
        """Provide a singleton FLConfigLoader instance for node mode."""
        return FLConfigLoader(mode="node")

    @provider
    def provide_federated_learning_config(self, fl_config_loader: FLConfigLoader) -> yacs.config.CfgNode:
        """Provide federated learning config node."""
        return fl_config_loader.load_config()
    
    @provider
    def provide_inference_engine_factory(self) -> InferenceEngineFactory:
        """
        Provide an InferenceEngineFactory instance.

        Returns
        -------
        InferenceEngineFactory
            The inference engine factory instance.

        """
        return InferenceEngineFactory()

    @provider
    def provide_model_factory(self, config_loader: ConfigLoader) -> ModelFactory:
        """
        Provide a ModelFactory instance.

        Returns
        -------
        ModelFactory
            The model factory instance.
        """
        return ModelFactory(cfg=config_loader.load_config())

    @provider
    def provide_loss_function_factory(self, config_loader: ConfigLoader) -> LossFunctionFactory:
        """
        Provide a LossFunctionFactory instance.

        Returns
        -------
        LossFunctionFactory
            The loss function factory instance.
        """
        return LossFunctionFactory(cfg=config_loader.load_config())

    @provider
    def provide_dataloader_factory(self, config_loader: ConfigLoader) -> DataLoaderFactory:
        """
        Provide a DataLoaderFactory instance.

        Returns
        -------
        DataLoaderFactory
            The data loader factory instance.
        """
        return DataLoaderFactory(cfg=config_loader.load_config())

     
    @provider
    def provide_federated_backend_factory(
        self, config_loader: ConfigLoader, fl_config_loader: FLConfigLoader) -> FederatedBackendFactory:
        """
        Provide a FederatedBackendFactory instance.

        Returns
        -------
        FederatedBackendFactory
            The federated backend factory instance.
        """
        return FederatedBackendFactory(config_loader=config_loader, fl_config_loader=fl_config_loader)
    
    @provider
    def provide_federated_backend(
        self, 
        backend_factory: FederatedBackendFactory) -> FederatedBackend:

        return backend_factory.resolve_backend()

    @provider
    def provide_federated_client(
        self, 
        federated_learning_config: yacs.config.CfgNode,
        backend: FederatedBackend) -> FederatedClientAPI:

        return FederatedClientAPI(federated_learning_config, backend)


def create_injector() -> Injector:
    """
    Create and configure a dependency injector for FastSurfer.

    Returns
    -------
    Injector
        Configured dependency injector.

    Examples
    --------
    >>> injector = create_injector()
    >>> model_factory = injector.get(ModelFactory)
    """
    return Injector([FastSurferModule()])
