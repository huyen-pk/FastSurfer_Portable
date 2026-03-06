from pytest import approx

from FastSurferCNN.config.defaults import get_cfg_defaults
from FastSurferCNN.federated import (
    DEFAULT_AGGREGATION,
    DEFAULT_BACKEND,
    DEFAULT_TOPOLOGY,
    TURBO_AGGREGATE_CURRENT_WEIGHT,
    TURBO_AGGREGATE_PREVIOUS_WEIGHT,
    FederatedClientAPI,
)


def test_federation_defaults_present():
    cfg = get_cfg_defaults()
    assert hasattr(cfg, "FEDERATION")
    assert cfg.FEDERATION.AGGREGATION == DEFAULT_AGGREGATION
    assert cfg.FEDERATION.TOPOLOGY == DEFAULT_TOPOLOGY
    assert cfg.FEDERATION.AGGREGATION_PREVIOUS_WEIGHT == TURBO_AGGREGATE_PREVIOUS_WEIGHT
    assert cfg.FEDERATION.AGGREGATION_CURRENT_WEIGHT == TURBO_AGGREGATE_CURRENT_WEIGHT
    assert cfg.FEDERATION.BACKEND == DEFAULT_BACKEND


def test_tensor_opera_client_safe_without_torch(monkeypatch):
    cfg = get_cfg_defaults()
    cfg.FEDERATION.ENABLED = True
    cfg.FEDERATION.SYNC_INTERVAL = 1
    cfg.FEDERATION.BACKEND = "fedml"

    client = FederatedClientAPI.from_cfg(cfg)
    # FedML not installed in test env, so backend reports unavailable.
    assert not client.backend.is_available()

    class DummyModel:
        def state_dict(self):
            return {}

        def load_state_dict(self, state, strict=False):
            self.last_loaded_state = (state, strict)

    dummy = DummyModel()

    # Ensure torch import fails so we exercise the guarded path.
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    client.maybe_sync(dummy, epoch=0)

    # No state is applied when torch is unavailable, but no exception should occur.
    assert getattr(dummy, "last_loaded_state", None) is None


def test_tensor_opera_client_aggregates_with_mock_torch(monkeypatch):
    class FakeTensor:
        def __init__(self, value: float):
            self.value = value

        def detach(self):
            return self

        def clone(self):
            return FakeTensor(self.value)

        def mul(self, other):
            return FakeTensor(self.value * other)

        def add(self, other):
            other_val = other.value if isinstance(other, FakeTensor) else other
            return FakeTensor(self.value + other_val)

    class FakeTorch:
        Tensor = FakeTensor

    cfg = get_cfg_defaults()
    cfg.FEDERATION.ENABLED = True
    cfg.FEDERATION.SYNC_INTERVAL = 1
    cfg.FEDERATION.BACKEND = "internal"

    client = FederatedClientAPI.from_cfg(cfg)

    class DummyModel:
        def __init__(self, value):
            self.value = value
            self.last_loaded_state = None

        def state_dict(self):
            return {"w": FakeTensor(self.value)}

        def load_state_dict(self, state, strict=False):
            self.last_loaded_state = (state, strict)

    import sys

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    first = DummyModel(1.0)
    second = DummyModel(3.0)

    client.maybe_sync(first, epoch=0)
    client.maybe_sync(second, epoch=1)

    # Sequential EWMA aggregation seeds with 1.0 then blends in 3.0
    # (1.0 * TURBO_AGGREGATE_PREVIOUS_WEIGHT + 3.0 * TURBO_AGGREGATE_CURRENT_WEIGHT = 2.4),
    # applying 30% weight to the previous state and 70% weight to the new update.
    assert isinstance(client._global_state["w"], FakeTensor)
    assert client._global_state["w"].value == approx(2.4)
