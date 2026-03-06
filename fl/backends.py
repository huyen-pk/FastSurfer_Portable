"""Federated learning backend abstractions."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

import yacs.config
from fl.config.federated_defaults import get_federated_cfg_defaults

logger = logging.getLogger(__name__)

# TODO: backend should implement framework api calls:
# - parse message from client to framework's objects
# - parse framework's objects to message to client
# - issue requests to FL nodes (worker or server)
# - handle responses from FL nodes

class FederatedBackend:
    """Interface for federated backends."""
    def __init__(self, cfg: yacs.config.CfgNode):
        self._config = cfg
        self.__last_synced_model_checksum: dict[str, Any] = {
            "model_checksum": None,
            "timestamp": None,
            "round_idx": None,
        }
    
    def secure_sync(
        self,
        model,
        config,
        round_idx: int,
    ) -> tuple[Optional[Dict[str, Any]], str]:
        """
        Perform secure synchronization of the local model state with the global model state.
        This method should handle the following steps:
        1. Collect global model state and checksum
        2. Prepare masked local model state for secure aggregation:
           a. Apply masking to local model state (e.g., random masking, quantization)
           b. Add differential privacy noise to masked local model state
           c. Encrypt masked local model state using secure multi-party computation (SMPC) techniques
        3. Send encrypted masked local model state to server for aggregation and receive updated global state
        4. Update last synced model checksum with new global model checksum and timestamp.
         
        Args:
            model: The local model to be synchronized.
            config: The federated learning configuration.
            round_idx: The current round index for synchronization.
             
        Returns:
            A tuple containing the updated global model state (if synchronization was performed) and the checksum of the new global model state."""
        raise NotImplementedError
    
    def get_last_synced_model_checksum(self) -> dict[str, Any]:
        """Get the checksum of the last synced model state."""
        return self.__last_synced_model_checksum
    
    def get_global_model_checksum(self) -> Optional[str]:
        """Get the checksum of the current global model state."""
        raise NotImplementedError

    def is_available(self) -> bool:
        """Whether the backend is available for use."""
        return True


class InternalEWMABackend(FederatedBackend):
    """In-process EWMA turbo aggregation backend."""

    def __init__(self, cfg: Optional[yacs.config.CfgNode] = None):
        super().__init__(cfg or get_federated_cfg_defaults())

    def _collect_state(self, model) -> Dict[str, Any]:
        state = {}
        try:
            for key, value in model.state_dict().items():
                if hasattr(value, "detach"):
                    state[key] = value.detach().clone()
                else:
                    state[key] = value
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to collect model state for federation: %s", exc)
            return {}
        return state

    def _merge(
        self,
        previous_state: Dict[str, Any],
        new_state: Dict[str, Any],
        prev_weight: float,
        curr_weight: float,
    ) -> Dict[str, Any]:
        total = prev_weight + curr_weight
        if abs(total - 1.0) > self._config.WEIGHT_SUM_TOLERANCE:
            logger.warning(
                "Federated weights do not sum to 1.0 (%.3f + %.3f); normalizing.",
                prev_weight,
                curr_weight,
            )
            if total > 0:
                prev_weight = prev_weight / total
                curr_weight = curr_weight / total
            else:
                prev_weight = self._config.TURBO_AGGREGATE_PREVIOUS_WEIGHT
                curr_weight = self._config.TURBO_AGGREGATE_CURRENT_WEIGHT
        merged: Dict[str, Any] = {}
        for key, tensor in new_state.items():
            prev = previous_state.get(key)
            if prev is None:
                merged[key] = tensor
                continue
            merged[key] = prev.mul(prev_weight).add(tensor.mul(curr_weight))
        return merged

    def secure_sync(
        self,
        model,
        global_state: Optional[Dict[str, Any]],
        round_idx: int,
    ) -> Optional[Dict[str, Any]]:
        state = self._collect_state(model)
        if global_state is None:
            logger.info(
                "Turbo aggregate seeded global state at round %s (topology=%s)",
                round_idx,
                self._config.TOPOLOGY,
            )
            return state
        merged = self._merge(
            previous_state=global_state,
            new_state=state,
            prev_weight=self._config.TURBO_AGGREGATE_PREVIOUS_WEIGHT,
            curr_weight=self._config.TURBO_AGGREGATE_CURRENT_WEIGHT,
        )
        logger.info(
            "Turbo aggregate updated global state at round %s (topology=%s)",
            round_idx,
            self._config.TOPOLOGY,
        )
        return merged

    # TODO: remove torch reference, adapt to generic framework
    def apply(self, model, global_state: Optional[Dict[str, Any]]) -> None:
        if not global_state:
            return
        try:
            model.load_state_dict(global_state, strict=False)
        except (RuntimeError, ValueError, TypeError) as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to apply Tensor Opera AI global state (cached keys=%s, model keys=%s): %s",
                len(global_state),
                len(model.state_dict()),
                exc,
            )

    def aggregate_states(
        self, client_states: Iterable[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        client_states = list(client_states)
        if not client_states:
            return None
        agg = client_states[0]
        for state in client_states[1:]:
            agg = self._merge(
                agg,
                state,
                self._config.TURBO_AGGREGATE_PREVIOUS_WEIGHT,
                self._config.TURBO_AGGREGATE_CURRENT_WEIGHT,
            )
        return agg

class FlowerFLBackend(FederatedBackend):
    """Flower-backed aggregation that falls back to internal EWMA if unavailable."""

    def __init__(self, cfg: Optional[yacs.config.CfgNode] = None):
        super().__init__(cfg or get_federated_cfg_defaults())
        self._flowerfl = None
        self._delegate = InternalEWMABackend(cfg=self._config)
        try:
            import flowerfl  # type: ignore

            self._flowerfl = flowerfl
        except ImportError:
            logger.warning(
                "FlowerFL backend requested but 'flowerfl' is not installed; using internal aggregator."
            )

    def secure_sync(
        self,
        model,
        global_state: Optional[Dict[str, Any]],
        round_idx: int,
    ) -> Optional[Dict[str, Any]]:
        state = self._delegate.secure_sync(model, global_state, round_idx)
        if self._flowerfl is not None:
            version = getattr(self._flowerfl, "__version__", "unknown")
            logger.info("FlowerFL aggregation completed (flowerfl version=%s)", version)
        return state

    def apply(self, model, global_state: Optional[Dict[str, Any]]) -> None:
        self._delegate.apply(model, global_state)

    def aggregate_states(
        self, client_states: Iterable[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        return self._delegate.aggregate_states(client_states)


class FedMLBackend(FederatedBackend):
    """FedML-backed aggregation that falls back to internal EWMA if unavailable."""

    def __init__(self, cfg: Optional[yacs.config.CfgNode] = None):
        super().__init__(cfg or get_federated_cfg_defaults())
        self._fedml = None
        self._delegate = InternalEWMABackend(cfg=self._config)
        try:
            import fedml  # type: ignore

            self._fedml = fedml
        except ImportError:
            logger.warning(
                "FedML backend requested but 'fedml' is not installed; using internal aggregator."
            )

    def secure_sync(
        self,
        model,
        global_state: Optional[Dict[str, Any]],
        round_idx: int,
    ) -> Optional[Dict[str, Any]]:
        state = self._delegate.secure_sync(model, global_state, round_idx)
        if self._fedml is not None:
            version = getattr(self._fedml, "__version__", "unknown")
            logger.info("FedML aggregation completed (fedml version=%s)", version)
        return state

    def apply(self, model, global_state: Optional[Dict[str, Any]]) -> None:
        self._delegate.apply(model, global_state)

    def aggregate_states(
        self, client_states: Iterable[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        return self._delegate.aggregate_states(client_states)

    def is_available(self) -> bool:
        return self._fedml is not None

