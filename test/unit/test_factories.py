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

"""Unit tests for DI factories."""

import unittest
from unittest.mock import MagicMock, patch

import torch
import yacs.config

from FastSurferCNN.di import (
    DataLoaderFactory,
    LossFunctionFactory,
    ModelFactory,
    OptimizerFactory,
    SchedulerFactory,
)


class TestModelFactory(unittest.TestCase):
    """Test cases for ModelFactory."""

    @patch("FastSurferCNN.models.networks.build_model")
    def test_create_model(self, mock_build_model):
        """Test model creation."""
        mock_model = MagicMock()
        mock_build_model.return_value = mock_model
        
        cfg = MagicMock(spec=yacs.config.CfgNode)
        factory = ModelFactory()
        model = factory.create_model(cfg)
        
        mock_build_model.assert_called_once_with(cfg)
        self.assertEqual(model, mock_model)


class TestLossFunctionFactory(unittest.TestCase):
    """Test cases for LossFunctionFactory."""

    @patch("FastSurferCNN.models.losses.get_loss_func")
    def test_create_loss_function(self, mock_get_loss_func):
        """Test loss function creation."""
        mock_loss = MagicMock()
        mock_get_loss_func.return_value = mock_loss
        
        cfg = MagicMock(spec=yacs.config.CfgNode)
        factory = LossFunctionFactory()
        loss = factory.create_loss_function(cfg)
        
        mock_get_loss_func.assert_called_once_with(cfg)
        self.assertEqual(loss, mock_loss)


class TestOptimizerFactory(unittest.TestCase):
    """Test cases for OptimizerFactory."""

    @patch("FastSurferCNN.models.optimizer.get_optimizer")
    def test_create_optimizer(self, mock_get_optimizer):
        """Test optimizer creation."""
        mock_optimizer = MagicMock()
        mock_get_optimizer.return_value = mock_optimizer
        
        model = MagicMock(spec=torch.nn.Module)
        cfg = MagicMock(spec=yacs.config.CfgNode)
        factory = OptimizerFactory()
        optimizer = factory.create_optimizer(model, cfg)
        
        mock_get_optimizer.assert_called_once_with(model, cfg)
        self.assertEqual(optimizer, mock_optimizer)


class TestDataLoaderFactory(unittest.TestCase):
    """Test cases for DataLoaderFactory."""

    @patch("FastSurferCNN.data_loader.loader.get_dataloader")
    def test_create_dataloader(self, mock_get_dataloader):
        """Test data loader creation."""
        mock_dataloader = MagicMock()
        mock_get_dataloader.return_value = mock_dataloader
        
        cfg = MagicMock(spec=yacs.config.CfgNode)
        factory = DataLoaderFactory()
        dataloader = factory.create_dataloader(cfg, "train")
        
        mock_get_dataloader.assert_called_once_with(cfg, "train")
        self.assertEqual(dataloader, mock_dataloader)


class TestSchedulerFactory(unittest.TestCase):
    """Test cases for SchedulerFactory."""

    @patch("FastSurferCNN.utils.lr_scheduler.get_lr_scheduler")
    def test_create_scheduler(self, mock_get_scheduler):
        """Test scheduler creation."""
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler
        
        optimizer = MagicMock(spec=torch.optim.Optimizer)
        cfg = MagicMock(spec=yacs.config.CfgNode)
        factory = SchedulerFactory()
        scheduler = factory.create_scheduler(optimizer, cfg)
        
        mock_get_scheduler.assert_called_once_with(optimizer, cfg)
        self.assertEqual(scheduler, mock_scheduler)


if __name__ == "__main__":
    unittest.main()
