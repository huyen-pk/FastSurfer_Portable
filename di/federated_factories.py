"""Factories for federated learning components."""
import logging
from typing import Literal, get_args
import yacs.config

from fl.config.federated_defaults import get_federated_cfg_defaults
from di.config_loader import ConfigLoader, FLConfigLoader
from fl.backends import FedMLBackend, FederatedBackend, InternalEWMABackend

logger = logging.getLogger(__name__)
SUPPORTED_BACKENDS = Literal["fedml", "internal_ewma"]

class FederatedBackendFactory:
    """
    Factory class responsible for creating and configuring federated backends 
    used in distributed machine learning systems. It provides a mechanism to resolve and instantiate 
    specific backend implementations based on the provided configuration.

    Methods
    -------
    __init__(cfg: yacs.config.CfgNode)
        Initializes the factory with the given configuration and sets default values.
    set_config(cfg: yacs.config.CfgNode)
        Sets the configuration for the federated backend factory.
    resolve_backend() -> FederatedBackend
        Resolves and returns an instance of the federated backend based on the configuration.
    """

    def __init__(self, config_loader: ConfigLoader, fl_config_loader: FLConfigLoader):
        """
        Initializes the FederatedBackendFactory with the provided configuration.
        
        Parameters
        ----------
        cfg : yacs.config.CfgNode
            Configuration node containing settings for the federated backend factory.
        """
        self.set_config(config_loader, fl_config_loader)
        defaults = get_federated_cfg_defaults()
        self.DEFAULT_AGGREGATION = defaults.AGGREGATION
        self.DEFAULT_TOPOLOGY = defaults.TOPOLOGY
        self.DEFAULT_BACKEND = defaults.BACKEND
        self.TURBO_AGGREGATE_PREVIOUS_WEIGHT = defaults.AGGREGATION_PREVIOUS_WEIGHT
        self.TURBO_AGGREGATE_CURRENT_WEIGHT = defaults.AGGREGATION_CURRENT_WEIGHT
        # Small tolerance to account for floating point drift when validating weight sums
        self.WEIGHT_SUM_TOLERANCE = defaults.WEIGHT_SUM_TOLERANCE

    def set_config(self, config_loader: ConfigLoader, fl_config_loader: FLConfigLoader):
        """
        Set the configuration for the federated backend factory.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            Configuration node to set for the factory.
        """
        self.__cfg = config_loader.load_config()
        self.fl_config = fl_config_loader.load_config()

    def resolve_backend(self) -> FederatedBackend:
        """Return a backend instance by name."""
        # TODO: check for availability and compatibility between requested backend and aggregation/topology choices
        name = self.fl_config.BACKEND
        normalized = name.lower()
        if normalized not in get_args(SUPPORTED_BACKENDS):
            logger.warning(
            "Unknown federated backend '%s' specified; falling back to internal EWMA.",
            name,
            )
            return InternalEWMABackend()
        if normalized == "fedml":
            return FedMLBackend()
        return InternalEWMABackend()
        
