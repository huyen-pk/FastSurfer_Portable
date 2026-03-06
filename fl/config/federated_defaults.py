"""Federated learning configuration defaults."""

from yacs.config import CfgNode as CN


def get_federated_cfg_defaults() -> CN:
    """Return federated learning defaults as a standalone config node."""
    cfg = CN()
    cfg.CONFIG_DIR = "configs"
    cfg.ENABLED = False
    cfg.BACKEND = "internal"
    cfg.AGGREGATION = "turbo_aggregate"
    cfg.TOPOLOGY = "asynchronous_decentralized_parallel_sgd"
    cfg.SILO_ID = ""
    cfg.SYNC_INTERVAL = 1
    cfg.AGGREGATION_PREVIOUS_WEIGHT = 0.3
    cfg.AGGREGATION_CURRENT_WEIGHT = 0.7
    cfg.WEIGHT_SUM_TOLERANCE = 1e-6
    return cfg
