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

import unittest
from unittest.mock import MagicMock, patch

import yacs.config

from di.config_loader import ConfigLoader


class TestConfigLoader(unittest.TestCase):
    """Test cases for ConfigLoader."""

    @patch("di.config_loader.get_cfg_defaults")
    def test_get_default_config(self, mock_get_defaults):
        """Test getting default configuration."""
        mock_cfg = MagicMock(spec=yacs.config.CfgNode)
        mock_get_defaults.return_value = mock_cfg

        loader = ConfigLoader()
        cfg = loader.get_default_config()

        mock_get_defaults.assert_called_once()
        self.assertEqual(cfg, mock_cfg)

    @patch("di.config_loader.get_cfg_defaults")
    def test_create_config_with_file_and_opts(self, mock_get_defaults):
        """Test create_config merges file and options."""
        mock_cfg = MagicMock(spec=yacs.config.CfgNode)
        mock_get_defaults.return_value = mock_cfg

        loader = ConfigLoader()
        cfg = loader.create_config(
            cfg_file="/path/to/config.yaml",
            opts=["MODEL.NUM_CLASSES", "50"],
        )

        mock_cfg.merge_from_file.assert_called_once_with("/path/to/config.yaml")
        mock_cfg.merge_from_list.assert_called_once_with(["MODEL.NUM_CLASSES", "50"])
        self.assertEqual(cfg, mock_cfg)

    @patch("di.config_loader.get_cfg_defaults")
    def test_create_config_with_kwargs(self, mock_get_defaults):
        """Test create_config applies supported kwargs."""
        mock_cfg = MagicMock(spec=yacs.config.CfgNode)
        mock_cfg.RNG_SEED = None
        mock_get_defaults.return_value = mock_cfg

        loader = ConfigLoader()
        loader.create_config(rng_seed=42)

        self.assertEqual(mock_cfg.RNG_SEED, 42)


if __name__ == "__main__":
    unittest.main()
