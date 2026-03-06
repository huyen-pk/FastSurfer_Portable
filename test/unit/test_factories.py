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

import yacs.config

from di.factories import DataLoaderFactory, LossFunctionFactory, ModelFactory, ModelTrainerFactory


class TestModelFactory(unittest.TestCase):
    """Test cases for ModelFactory."""

    @patch("FastSurferCNN.models.networks.build_model")
    def test_create_model(self, mock_build_model):
        """Test model creation."""
        mock_model = MagicMock()
        mock_build_model.return_value = mock_model

        cfg = MagicMock(spec=yacs.config.CfgNode)
        factory = ModelFactory(cfg=cfg)
        model = factory.create_model()

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
        factory = LossFunctionFactory(cfg=cfg)
        loss = factory.create_loss_function()

        mock_get_loss_func.assert_called_once_with(cfg)
        self.assertEqual(loss, mock_loss)


class TestDataLoaderFactory(unittest.TestCase):
    """Test cases for DataLoaderFactory."""

    @patch("FastSurferCNN.data_loader.loader.get_dataloader")
    def test_create_dataloader(self, mock_get_dataloader):
        """Test data loader creation."""
        mock_dataloader = MagicMock()
        mock_get_dataloader.return_value = mock_dataloader

        cfg = MagicMock(spec=yacs.config.CfgNode)
        factory = DataLoaderFactory(cfg=cfg)
        dataloader = factory.create_dataloader("train")

        mock_get_dataloader.assert_called_once_with(cfg, "train")
        self.assertEqual(dataloader, mock_dataloader)


class TestModelTrainerFactory(unittest.TestCase):
    """Test cases for ModelTrainerFactory."""

    @patch("di.factories.FastSurferCNN_FL_Trainer")
    def test_create_model_trainer_fastsurfer(self, mock_trainer_class):
        """Test FastSurfer trainer creation when FL wrapper is available."""
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        cfg = MagicMock(spec=yacs.config.CfgNode)
        factory = ModelTrainerFactory(cfg=cfg)
        trainer = factory.create_model_trainer("FastSurferCNN")

        mock_trainer_class.assert_called_once_with(cfg)
        self.assertEqual(trainer, mock_trainer)


if __name__ == "__main__":
    unittest.main()
