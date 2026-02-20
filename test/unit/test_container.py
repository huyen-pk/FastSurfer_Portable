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

"""Unit tests for DI container."""

import unittest
from unittest.mock import MagicMock, patch

import yacs.config

from FastSurferCNN.di import (
    ConfigLoader,
    DataLoaderFactory,
    DeviceManager,
    LossFunctionFactory,
    ModelFactory,
    OptimizerFactory,
    SchedulerFactory,
    create_injector,
)


class TestDIContainer(unittest.TestCase):
    """Test cases for dependency injection container."""

    @patch("FastSurferCNN.di.container.DeviceManager")
    def test_create_injector(self, mock_device_manager_class):
        """Test creating an injector."""
        mock_cfg = MagicMock(spec=yacs.config.CfgNode)
        
        injector = create_injector(cfg=mock_cfg)
        
        self.assertIsNotNone(injector)

    @patch("FastSurferCNN.di.container.DeviceManager")
    def test_injector_provides_device_manager(self, mock_device_manager_class):
        """Test that injector provides DeviceManager."""
        mock_device_manager = MagicMock()
        mock_device_manager_class.return_value = mock_device_manager
        
        injector = create_injector()
        device_manager = injector.get(DeviceManager)
        
        self.assertIsNotNone(device_manager)

    @patch("FastSurferCNN.di.container.ModelFactory")
    def test_injector_provides_model_factory(self, mock_model_factory_class):
        """Test that injector provides ModelFactory."""
        mock_factory = MagicMock()
        mock_model_factory_class.return_value = mock_factory
        
        injector = create_injector()
        factory = injector.get(ModelFactory)
        
        self.assertIsNotNone(factory)

    @patch("FastSurferCNN.di.container.LossFunctionFactory")
    def test_injector_provides_loss_factory(self, mock_loss_factory_class):
        """Test that injector provides LossFunctionFactory."""
        mock_factory = MagicMock()
        mock_loss_factory_class.return_value = mock_factory
        
        injector = create_injector()
        factory = injector.get(LossFunctionFactory)
        
        self.assertIsNotNone(factory)

    @patch("FastSurferCNN.di.container.OptimizerFactory")
    def test_injector_provides_optimizer_factory(self, mock_optimizer_factory_class):
        """Test that injector provides OptimizerFactory."""
        mock_factory = MagicMock()
        mock_optimizer_factory_class.return_value = mock_factory
        
        injector = create_injector()
        factory = injector.get(OptimizerFactory)
        
        self.assertIsNotNone(factory)

    @patch("FastSurferCNN.di.container.DataLoaderFactory")
    def test_injector_provides_dataloader_factory(self, mock_dataloader_factory_class):
        """Test that injector provides DataLoaderFactory."""
        mock_factory = MagicMock()
        mock_dataloader_factory_class.return_value = mock_factory
        
        injector = create_injector()
        factory = injector.get(DataLoaderFactory)
        
        self.assertIsNotNone(factory)

    @patch("FastSurferCNN.di.container.SchedulerFactory")
    def test_injector_provides_scheduler_factory(self, mock_scheduler_factory_class):
        """Test that injector provides SchedulerFactory."""
        mock_factory = MagicMock()
        mock_scheduler_factory_class.return_value = mock_factory
        
        injector = create_injector()
        factory = injector.get(SchedulerFactory)
        
        self.assertIsNotNone(factory)

    @patch("FastSurferCNN.di.container.ConfigLoader")
    def test_injector_provides_config_loader(self, mock_config_loader_class):
        """Test that injector provides ConfigLoader."""
        mock_loader = MagicMock()
        mock_config_loader_class.return_value = mock_loader
        
        injector = create_injector()
        loader = injector.get(ConfigLoader)
        
        self.assertIsNotNone(loader)

    @patch("FastSurferCNN.di.container.DeviceManager")
    def test_device_manager_singleton(self, mock_device_manager_class):
        """Test that DeviceManager is a singleton."""
        mock_device_manager = MagicMock()
        mock_device_manager_class.return_value = mock_device_manager
        
        injector = create_injector()
        dm1 = injector.get(DeviceManager)
        dm2 = injector.get(DeviceManager)
        
        # Should be the same instance
        self.assertIs(dm1, dm2)

    @patch("FastSurferCNN.di.container.ConfigLoader")
    def test_config_loader_singleton(self, mock_config_loader_class):
        """Test that ConfigLoader is a singleton."""
        mock_loader = MagicMock()
        mock_config_loader_class.return_value = mock_loader
        
        injector = create_injector()
        loader1 = injector.get(ConfigLoader)
        loader2 = injector.get(ConfigLoader)
        
        # Should be the same instance
        self.assertIs(loader1, loader2)


if __name__ == "__main__":
    unittest.main()
