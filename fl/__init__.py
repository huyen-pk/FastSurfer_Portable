"""Federated learning package for Tensor Opera AI integration."""

from fl.config.federated_defaults import get_federated_cfg_defaults
from fl.api import (
    FederatedClientAPI,
    FederatedLearningConfig,
    FederatedOrchestrator,
)
from fl.backends import (
    DEFAULT_AGGREGATION,
    DEFAULT_BACKEND,
    DEFAULT_TOPOLOGY,
    TURBO_AGGREGATE_CURRENT_WEIGHT,
    TURBO_AGGREGATE_PREVIOUS_WEIGHT,
    resolve_backend,
)

__all__ = [
    "FederatedClientAPI",
    "FederatedServerAPI",
    "FederatedLearningConfig",
    "resolve_backend",
    "DEFAULT_AGGREGATION",
    "DEFAULT_TOPOLOGY",
    "DEFAULT_BACKEND",
    "TURBO_AGGREGATE_PREVIOUS_WEIGHT",
    "TURBO_AGGREGATE_CURRENT_WEIGHT",
    "get_federated_cfg_defaults",
]
