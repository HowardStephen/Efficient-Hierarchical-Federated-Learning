# @PATH: fl_system/aggregation/__init__.py
# @DATE: 03.19.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn
#
# Aggregation module: aggregators, fafa_r, and utilities.

from .aggregation_factory import get_aggregator
from .base import Aggregator
from .fedavg import FedAvgAggregator
from .fedprox import FedProxAggregator
from .hierarchical_fedavg import HierarchicalFedAvgAggregator
from .scaffold import SCAFFOLDAggregator
from .fafa_r import edge_aggregate, cloud_aggregate, apply_global_update, compute_similarity

__all__ = [
    "Aggregator",
    "get_aggregator",
    "FedAvgAggregator",
    "HierarchicalFedAvgAggregator",
    "FedProxAggregator",
    "SCAFFOLDAggregator",
    "edge_aggregate",
    "cloud_aggregate",
    "apply_global_update",
    "compute_similarity",
]
