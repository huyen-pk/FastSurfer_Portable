"""Federated learning package for Tensor Opera AI integration."""

from fl.config.federated_defaults import get_federated_cfg_defaults

FL_IMPORT_ERROR: Exception | None = None

try:
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
except Exception as exc:
    FL_IMPORT_ERROR = exc

    FederatedClientAPI = None
    FederatedLearningConfig = None
    FederatedOrchestrator = None

    DEFAULT_AGGREGATION = None
    DEFAULT_BACKEND = None
    DEFAULT_TOPOLOGY = None
    TURBO_AGGREGATE_CURRENT_WEIGHT = None
    TURBO_AGGREGATE_PREVIOUS_WEIGHT = None

    def resolve_backend(*args, **kwargs):
        raise RuntimeError(
            "Federated learning backend is unavailable in this environment"
        ) from FL_IMPORT_ERROR

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
