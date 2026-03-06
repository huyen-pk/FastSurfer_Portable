"""Federated learning client/server APIs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
import time
from typing import Any, Optional

from yacs.config import CfgNode

from fl.backends import (
    FederatedBackend
)

logger = logging.getLogger(__name__)

class FederatedClientAPI:
    """Client-side abstraction for federated learning."""

    def __init__(self, config: CfgNode, backend: FederatedBackend):
        self.config = config
        self.backend = backend
        self._current_state = None
        self._round = 0
        if self.config.ENABLED:
            logger.info(
                "Federated mode enabled (backend=%s, strategy=%s, topology=%s, silo=%s)",
                self.config.BACKEND,
                self.config.AGGREGATION,
                self.config.TOPOLOGY,
                self.config.SILO_ID or "unspecified",
            )

    # TODO: handle race condition
    def update_internal_state(self, new_state: dict[str, Any], checksum: str) -> None:
        """Set internal state for the client API."""
        self._current_state = new_state
        self._global_state_checksum = checksum

    def get_internal_state(self) -> dict[str, Any]:
        """Get the current global state.
        This should ping a node (peer or central server) in the network to get the latest global state if the client is not up to date."""
        return self._current_state

    def maybe_sync(self, model, epoch: int) -> bool:
        """Perform sync if enabled and on schedule.
        
        Returns:
            bool: True if sync was performed, False otherwise.
        """
        if not self.config.ENABLED:
            return False

        if self.config.SYNC_INTERVAL <= 0:
            logger.debug("Federated sync skipped because SYNC_INTERVAL <= 0")
            return False

        if (epoch + 1) % self.config.SYNC_INTERVAL != 0:
            return False

        self._round += 1
        updated_state, checksum = self.backend.secure_sync(
            model, self.config, self._round
        )
        self.update_internal_state(new_state=updated_state, checksum=checksum)
        return True

    def report_node_sync_status(self) -> bool:
        """Report sync state with global model of the current node. 
        It's like a heartbeat to the server to indicate the client is alive and synced.
        
        Returns:
            bool: True if the client is considered synced, False if the client is considered out of sync.
        """
        last_sync_state = self.backend.get_last_synced_model_checksum()
        timelapse_since_last_sync = (time.time() - last_sync_state.get("timestamp")) / 60
        heartbeat_interval = self.config.get("HEARTBEAT_INTERVAL_MINUTES", 5)
        if timelapse_since_last_sync > heartbeat_interval:
            return False
        
        # Get global model checksum from server to determine sync status
        global_model_checksum = self.backend.get_global_model_checksum()
        return last_sync_state.get("checksum") == global_model_checksum

