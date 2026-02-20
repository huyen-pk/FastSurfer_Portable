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
from abc import ABC
import os
from typing import Literal

import yacs.config

from FastSurferCNN.config.defaults import get_cfg_defaults
from fl.config.federated_defaults import get_federated_cfg_defaults
from FastSurferCNN.utils import misc

class IConfigLoader(ABC):
    """
    Interface for configuration loading.

    This interface defines the contract for any configuration loader implementation,
    ensuring consistency and flexibility in how configurations are loaded and managed.
    """

    def load_config(self) -> yacs.config.CfgNode:
        """
        Load and initialize configuration.

        Returns
        -------
        yacs.config.CfgNode
            Configuration node.
        """
        raise NotImplementedError("load_config method must be implemented by subclasses.")
    
    def get_default_config(self) -> yacs.config.CfgNode:
        """
        Get the default configuration.

        Returns
        -------
        yacs.config.CfgNode
            Default configuration node.
        """
        raise NotImplementedError("get_default_config method must be implemented by subclasses.")
    
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
        raise NotImplementedError("load_from_file method must be implemented by subclasses.")
    
    def create_config(self, cfg_file: str | None = None, opts: list | None = None, **kwargs) -> yacs.config.CfgNode:
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
        raise NotImplementedError("create_config method must be implemented by subclasses.")
class ConfigLoader(IConfigLoader):
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

        # Merge common federation settings, then config file if provided
        cfg.merge_from_file(os.environ.get("FASTSURFERCNN_CONFIG", "default_config.yaml"))
        
        # Merge config from environment variables
        for key, _ in cfg.items():
            if os.environ.get(key) is not None:
                try:
                    cfg[key] = os.environ[key]
                except Exception as e:
                    print(f"Could not merge environment variable {key}: {e}")

        summary_path = misc.check_path(os.path.join(cfg.LOG_DIR, "summary"))
        if cfg.EXPR_NUM == "Default":
            cfg.EXPR_NUM = str(
                misc.find_latest_experiment(os.path.join(cfg.LOG_DIR, "summary")) + 1
            )

        # These are used to set up checkpoint and log dir
        if cfg.TRAIN.RESUME and cfg.TRAIN.RESUME_EXPR_NUM != "Default":
            cfg.EXPR_NUM = cfg.TRAIN.RESUME_EXPR_NUM

        cfg.SUMMARY_PATH = misc.check_path(os.path.join(summary_path, f"{cfg.EXPR_NUM}"))
        cfg.CONFIG_LOG_PATH = misc.check_path(
            os.path.join(cfg.LOG_DIR, "config", f"{cfg.EXPR_NUM}")
        )

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
        
        summary_path = misc.check_path(os.path.join(cfg.LOG_DIR, "summary"))
        if cfg.EXPR_NUM == "Default":
            cfg.EXPR_NUM = str(
                misc.find_latest_experiment(os.path.join(cfg.LOG_DIR, "summary")) + 1
            )

        # These are used to set up checkpoint and log dir
        if cfg.TRAIN.RESUME and cfg.TRAIN.RESUME_EXPR_NUM != "Default":
            cfg.EXPR_NUM = cfg.TRAIN.RESUME_EXPR_NUM

        cfg.SUMMARY_PATH = misc.check_path(os.path.join(summary_path, f"{cfg.EXPR_NUM}"))
        cfg.CONFIG_LOG_PATH = misc.check_path(
            os.path.join(cfg.LOG_DIR, "config", f"{cfg.EXPR_NUM}")
        )

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


class FLConfigLoader(IConfigLoader):
    """Configuration loader for federated learning settings."""

    def __init__(self, mode: Literal["node", "orchestrator"]):
        self.mode = mode

    def load_config(self) -> yacs.config.CfgNode:
        """Load and initialize configuration for federated learning."""
        cfg = self.get_default_config()
        cfg_dir = cfg.CONFIG_DIR
        cfg_path = os.path.join(cfg_dir, f"fl_{self.mode}.yaml")
        cfg.merge_from_file(cfg_path)
        return cfg

    def get_default_config(self) -> yacs.config.CfgNode:
        """Get the default configuration for federated learning."""
        return get_federated_cfg_defaults()
    
    def load_from_file(self, cfg_file: str) -> yacs.config.CfgNode:
        """Load configuration from a YAML file for federated learning."""
        cfg = self.get_default_config()
        cfg.merge_from_file(cfg_file)
        return cfg
    
    def create_config(self, cfg_file: str | None = None, opts: list | None = None, **kwargs) -> yacs.config.CfgNode:
        """Create configuration for federated learning from various sources."""
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
    
