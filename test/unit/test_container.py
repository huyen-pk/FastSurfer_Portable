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

from di import ConfigLoader, FLConfigLoader, create_injector
from fl.api import FederatedClientAPI
from fl.backends import FederatedBackend


class TestDIContainer(unittest.TestCase):
    """Test cases for dependency injection container."""

    def test_create_injector(self):
        """Test creating an injector."""
        injector = create_injector()
        self.assertIsNotNone(injector)

    def test_injector_provides_config_loader_singleton(self):
        """Test ConfigLoader is provided as singleton."""
        injector = create_injector()
        loader1 = injector.get(ConfigLoader)
        loader2 = injector.get(ConfigLoader)

        self.assertIsInstance(loader1, ConfigLoader)
        self.assertIs(loader1, loader2)

    def test_injector_provides_fl_config_loader_singleton(self):
        """Test FLConfigLoader is provided as singleton."""
        injector = create_injector()
        loader1 = injector.get(FLConfigLoader)
        loader2 = injector.get(FLConfigLoader)

        self.assertIsInstance(loader1, FLConfigLoader)
        self.assertIs(loader1, loader2)

    @patch("di.container.ConfigLoader.load_config")
    @patch("di.container.FLConfigLoader.load_config")
    def test_injector_provides_federated_backend_and_client(self, mock_fl_load_config, mock_load_config):
        """Test federated backend/client bindings are resolvable."""
        app_cfg = MagicMock(spec=yacs.config.CfgNode)
        fl_cfg = MagicMock(spec=yacs.config.CfgNode)
        fl_cfg.BACKEND = "internal_ewma"
        fl_cfg.ENABLED = False
        fl_cfg.AGGREGATION = "turbo_aggregate"
        fl_cfg.TOPOLOGY = "asynchronous_decentralized_parallel_sgd"
        fl_cfg.SILO_ID = ""

        mock_load_config.return_value = app_cfg
        mock_fl_load_config.return_value = fl_cfg

        injector = create_injector()
        backend = injector.get(FederatedBackend)
        client = injector.get(FederatedClientAPI)

        self.assertIsInstance(backend, FederatedBackend)
        self.assertIsInstance(client, FederatedClientAPI)


if __name__ == "__main__":
    unittest.main()
