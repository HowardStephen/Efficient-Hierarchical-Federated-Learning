# @PATH: fl_system/aggregation/aggregation_factory.py
# @DATE: 03.19.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn
#
# Aggregator factory: returns aggregation algorithm instances by name.

"""
Aggregator factory: returns the corresponding aggregation algorithm instance by name.
"""

from .base import Aggregator
from .fedavg import FedAvgAggregator
from .fedprox import FedProxAggregator
from .hierarchical_fedavg import HierarchicalFedAvgAggregator
from .scaffold import SCAFFOLDAggregator

# Algorithm name -> aggregator class
_AGGREGATOR_REGISTRY: dict[str, type[Aggregator]] = {
    "fedavg": FedAvgAggregator,
    "hierarchical_fedavg": HierarchicalFedAvgAggregator,
    "fedprox": FedProxAggregator,
    "scaffold": SCAFFOLDAggregator,
}


def get_aggregator(name: str, **kwargs) -> Aggregator:
    """
    Get aggregator instance by name.

    Args:
        name: Algorithm name; supported: fedavg, hierarchical_fedavg, fedprox, scaffold
        **kwargs: Arguments passed to aggregator constructor (e.g., mu, lambda_decay, num_edges)

    Returns:
        Aggregator instance

    Raises:
        ValueError: Unknown algorithm name

    Example:
        >>> agg = get_aggregator("fedavg")
        >>> agg = get_aggregator("fedprox", mu=0.01)
        >>> agg = get_aggregator("hierarchical_fedavg", num_edges=4)
    """
    key = name.strip().lower()
    if key not in _AGGREGATOR_REGISTRY:
        raise ValueError(
            f"Unknown aggregation algorithm: {name}. Supported: {', '.join(_AGGREGATOR_REGISTRY.keys())}"
        )
    cls = _AGGREGATOR_REGISTRY[key]
    return cls(**kwargs)
