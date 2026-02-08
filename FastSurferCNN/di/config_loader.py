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

"""Configuration loader for dependency injection."""
import os

import yacs.config

from FastSurferCNN.config.defaults import get_cfg_defaults


class ConfigLoader:
    """
    Manages configuration loading and initialization.

    This class provides a centralized way to handle configuration loading,
    supporting both file-based and programmatic configuration.

    Methods
    -------
    load_config(args: argparse.Namespace) -> yacs.config.CfgNode
        Loads configuration from arguments.
    get_default_config() -> yacs.config.CfgNode
        Returns the default configuration.
    load_from_file(cfg_file: str) -> yacs.config.CfgNode
        Loads configuration from a YAML file.
    merge_from_args(cfg: yacs.config.CfgNode, args: argparse.Namespace) -> yacs.config.CfgNode
        Merges configuration with command-line arguments.
    """

    def load_config(self)-> yacs.config.CfgNode:
        """
        Load and initialize configuration.

        This method is a placeholder for loading configuration from various sources.
        It can be extended to support different loading mechanisms as needed.

        Returns
        -------
        yacs.config.CfgNode
            Configuration node.
        """
         # Setup cfg with defaults
        cfg = self.get_default_config()

        # Load config from file if provided
        cfg.merge_from_file(os.environ.get("FASTSURFERCNN_CONFIG", "default_config.yaml"))
        
        # Merge config from environment variables
        for key, _ in cfg.items():
            if os.environ.get(key) is not None:
                try:
                    cfg[key] = os.environ[key]
                except Exception as e:
                    print(f"Could not merge environment variable {key}: {e}")

        return cfg
       

    def get_default_config(self) -> yacs.config.CfgNode:
        """
        Get the default configuration.

        Returns
        -------
        yacs.config.CfgNode
            Default configuration node.
        """
        return get_cfg_defaults()

    def load_from_file(self, cfg_file: str) -> yacs.config.CfgNode:
        """
        Load configuration from a YAML file.

        Parameters
        ----------
        cfg_file : str
            Path to the configuration file.

        Returns
        -------
        yacs.config.CfgNode
            Configuration node loaded from file.
        """
        cfg = self.get_default_config()
        cfg.merge_from_file(cfg_file)
        return cfg

    def create_config(
        self,
        cfg_file: str | None = None,
        opts: list | None = None,
        **kwargs
    ) -> yacs.config.CfgNode:
        """
        Create configuration from various sources.

        Parameters
        ----------
        cfg_file : str, optional
            Path to configuration file.
        opts : list, optional
            List of configuration options to override.
        **kwargs
            Additional keyword arguments to set in configuration.

        Returns
        -------
        yacs.config.CfgNode
            Configuration node.
        """
        cfg = self.get_default_config()

        if cfg_file is not None:
            cfg.merge_from_file(cfg_file)

        if opts is not None:
            cfg.merge_from_list(opts)

        # Apply kwargs
        for key, value in kwargs.items():
            if hasattr(cfg, key.upper()):
                setattr(cfg, key.upper(), value)

        return cfg
