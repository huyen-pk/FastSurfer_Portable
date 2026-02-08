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

"""Unit tests for DI config loader."""

import argparse
import unittest
from unittest.mock import MagicMock, patch

import yacs.config

from FastSurferCNN.di import ConfigLoader


class TestConfigLoader(unittest.TestCase):
    """Test cases for ConfigLoader."""

    @patch("FastSurferCNN.di.config_loader.get_cfg_defaults")
    def test_get_default_config(self, mock_get_defaults):
        """Test getting default configuration."""
        mock_cfg = MagicMock(spec=yacs.config.CfgNode)
        mock_get_defaults.return_value = mock_cfg
        
        loader = ConfigLoader()
        cfg = loader.get_default_config()
        
        mock_get_defaults.assert_called_once()
        self.assertEqual(cfg, mock_cfg)

    @patch("FastSurferCNN.di.config_loader.get_cfg_defaults")
    def test_load_from_file(self, mock_get_defaults):
        """Test loading configuration from file."""
        mock_cfg = MagicMock(spec=yacs.config.CfgNode)
        mock_get_defaults.return_value = mock_cfg
        
        loader = ConfigLoader()
        cfg = loader.load_from_file("/path/to/config.yaml")
        
        mock_cfg.merge_from_file.assert_called_once_with("/path/to/config.yaml")
        self.assertEqual(cfg, mock_cfg)

    @patch("FastSurferCNN.di.config_loader.get_cfg_defaults")
    def test_load_config_with_file(self, mock_get_defaults):
        """Test loading configuration with file argument."""
        mock_cfg = MagicMock(spec=yacs.config.CfgNode)
        mock_get_defaults.return_value = mock_cfg
        
        args = argparse.Namespace(cfg_file="/path/to/config.yaml", opts=None)
        
        loader = ConfigLoader()
        cfg = loader.load_config_from_args(args)
        
        mock_cfg.merge_from_file.assert_called_once_with("/path/to/config.yaml")
        self.assertEqual(cfg, mock_cfg)

    @patch("FastSurferCNN.di.config_loader.get_cfg_defaults")
    def test_load_config_with_opts(self, mock_get_defaults):
        """Test loading configuration with opts argument."""
        mock_cfg = MagicMock(spec=yacs.config.CfgNode)
        mock_get_defaults.return_value = mock_cfg
        
        args = argparse.Namespace(cfg_file=None, opts=["MODEL.NUM_CLASSES", "50"])
        
        loader = ConfigLoader()
        cfg = loader.load_config_from_args(args)
        
        mock_cfg.merge_from_list.assert_called_once_with(["MODEL.NUM_CLASSES", "50"])
        self.assertEqual(cfg, mock_cfg)

    @patch("FastSurferCNN.di.config_loader.get_cfg_defaults")
    def test_load_config_with_file_and_opts(self, mock_get_defaults):
        """Test loading configuration with both file and opts."""
        mock_cfg = MagicMock(spec=yacs.config.CfgNode)
        mock_get_defaults.return_value = mock_cfg
        
        args = argparse.Namespace(
            cfg_file="/path/to/config.yaml",
            opts=["MODEL.NUM_CLASSES", "50"]
        )
        
        loader = ConfigLoader()
        cfg = loader.load_config_from_args(args)
        
        mock_cfg.merge_from_file.assert_called_once_with("/path/to/config.yaml")
        mock_cfg.merge_from_list.assert_called_once_with(["MODEL.NUM_CLASSES", "50"])
        self.assertEqual(cfg, mock_cfg)

    @patch("FastSurferCNN.di.config_loader.get_cfg_defaults")
    def test_merge_from_args_with_rng_seed(self, mock_get_defaults):
        """Test merging RNG seed from args."""
        mock_cfg = MagicMock(spec=yacs.config.CfgNode)
        mock_get_defaults.return_value = mock_cfg
        
        args = argparse.Namespace(rng_seed=42)
        
        loader = ConfigLoader()
        loader.merge_from_args(mock_cfg, args)
        
        self.assertEqual(mock_cfg.RNG_SEED, 42)

    @patch("FastSurferCNN.di.config_loader.get_cfg_defaults")
    def test_merge_from_args_with_output_dir(self, mock_get_defaults):
        """Test merging output directory from args."""
        mock_cfg = MagicMock(spec=yacs.config.CfgNode)
        mock_get_defaults.return_value = mock_cfg
        
        args = argparse.Namespace(output_dir="/path/to/output")
        
        loader = ConfigLoader()
        loader.merge_from_args(mock_cfg, args)
        
        self.assertEqual(mock_cfg.LOG_DIR, "/path/to/output")

    @patch("FastSurferCNN.di.config_loader.get_cfg_defaults")
    def test_create_config_with_file(self, mock_get_defaults):
        """Test creating configuration with file."""
        mock_cfg = MagicMock(spec=yacs.config.CfgNode)
        mock_get_defaults.return_value = mock_cfg
        
        loader = ConfigLoader()
        cfg = loader.create_config(cfg_file="/path/to/config.yaml")
        
        mock_cfg.merge_from_file.assert_called_once_with("/path/to/config.yaml")
        self.assertEqual(cfg, mock_cfg)

    @patch("FastSurferCNN.di.config_loader.get_cfg_defaults")
    def test_create_config_with_opts(self, mock_get_defaults):
        """Test creating configuration with opts."""
        mock_cfg = MagicMock(spec=yacs.config.CfgNode)
        mock_get_defaults.return_value = mock_cfg
        
        loader = ConfigLoader()
        cfg = loader.create_config(opts=["MODEL.NUM_CLASSES", "50"])
        
        mock_cfg.merge_from_list.assert_called_once_with(["MODEL.NUM_CLASSES", "50"])
        self.assertEqual(cfg, mock_cfg)

    @patch("FastSurferCNN.di.config_loader.get_cfg_defaults")
    def test_create_config_with_kwargs(self, mock_get_defaults):
        """Test creating configuration with kwargs."""
        mock_cfg = MagicMock(spec=yacs.config.CfgNode)
        mock_cfg.RNG_SEED = None
        mock_get_defaults.return_value = mock_cfg
        
        loader = ConfigLoader()
        loader.create_config(rng_seed=42)
        
        self.assertEqual(mock_cfg.RNG_SEED, 42)


if __name__ == "__main__":
    unittest.main()
