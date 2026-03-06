"""Federated learning package for Tensor Opera AI integration."""

from fl.config.federated_defaults import get_federated_cfg_defaults
from fl.backends import FedMLBackend, FlowerFLBackend, InternalEWMABackend

FL_IMPORT_ERROR: Exception | None = None

_defaults = get_federated_cfg_defaults()
DEFAULT_AGGREGATION = _defaults.AGGREGATION
DEFAULT_BACKEND = _defaults.BACKEND
DEFAULT_TOPOLOGY = _defaults.TOPOLOGY
TURBO_AGGREGATE_CURRENT_WEIGHT = _defaults.AGGREGATION_CURRENT_WEIGHT
TURBO_AGGREGATE_PREVIOUS_WEIGHT = _defaults.AGGREGATION_PREVIOUS_WEIGHT


def resolve_backend(name: str | None = None):
    """Return federated backend instance by name."""
    normalized = (name or DEFAULT_BACKEND).lower()
    if normalized in {"fedml"}:
        return FedMLBackend()
    if normalized in {"flower", "flowerfl"}:
        return FlowerFLBackend()
    return InternalEWMABackend()


try:
    from fl.api import FederatedClientAPI
except Exception as exc:
    FL_IMPORT_ERROR = exc

    FederatedClientAPI = None

__all__ = [
    "FederatedClientAPI",
    "resolve_backend",
    "DEFAULT_AGGREGATION",
    "DEFAULT_TOPOLOGY",
    "DEFAULT_BACKEND",
    "TURBO_AGGREGATE_PREVIOUS_WEIGHT",
    "TURBO_AGGREGATE_CURRENT_WEIGHT",
    "get_federated_cfg_defaults",
]
