# @PATH: fl_system/aggregation/hierarchical_fedavg.py
# @DATE: 03.19.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn
#
# Hierarchical FedAvg: two-tier federated averaging.
from __future__ import annotations

"""
Hierarchical FedAvg: two-tier aggregation.

1. Edge tier: each edge server runs FedAvg over its clients, yielding edge models
2. Cloud tier: FedAvg over edge models, yielding global model

client_updates and edge_ids are 1:1; clients with same edge_id are aggregated first.
"""

from typing import Any

import torch

from ..utils.model_utils import state_dict_weighted_sum

from .base import Aggregator
from .fedavg import FedAvgAggregator


class HierarchicalFedAvgAggregator(Aggregator):
    """Hierarchical FedAvg: aggregate by edge first, then cloud."""

    def __init__(self, num_edges: int | None = None):
        """
        Args:
            num_edges: Number of edge servers; inferred from edge_ids when None
        """
        self.num_edges = num_edges
        self._fedavg = FedAvgAggregator()

    def aggregate(
        self,
        client_updates: list[dict[str, torch.Tensor]],
        sample_counts: list[int] | None = None,
        edge_ids: list[int] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Perform hierarchical FedAvg aggregation.

        Args:
            client_updates: List of client model parameters
            sample_counts: Per-client sample counts; equal-weighted when None
            edge_ids: Edge ID for each client, 1:1 with client_updates.
                      Falls back to single-tier FedAvg when None

        Returns:
            Aggregated global model parameters
        """
        if not client_updates:
            raise ValueError("client_updates cannot be empty")

        # Fall back to plain FedAvg when edge_ids absent
        if edge_ids is None or len(set(edge_ids)) <= 1:
            return self._fedavg.aggregate(client_updates, sample_counts, **kwargs)

        if len(edge_ids) != len(client_updates):
            raise ValueError("edge_ids length must match client_updates")

        n = len(client_updates)
        weights = [1.0 / n] * n if sample_counts is None else [c / sum(sample_counts) for c in sample_counts]

        # Group by edge
        from collections import defaultdict
        edge_updates: dict[int, list[dict[str, torch.Tensor]]] = defaultdict(list)
        edge_weights: dict[int, list[float]] = defaultdict(list)
        for i, (sd, eid) in enumerate(zip(client_updates, edge_ids)):
            edge_updates[eid].append(sd)
            edge_weights[eid].append(weights[i])

        # Edge-tier aggregation: weighted average within each edge
        edge_models: list[dict[str, torch.Tensor]] = []
        edge_totals: list[float] = []
        for eid in sorted(edge_updates.keys()):
            w_list = edge_weights[eid]
            total_w = sum(w_list)
            # Normalize weights within this edge
            norm_w = [w / total_w for w in w_list]
            agg = state_dict_weighted_sum(edge_updates[eid], norm_w)
            edge_models.append(agg)
            edge_totals.append(total_w)

        # Cloud-tier aggregation: weight by edge total
        total = sum(edge_totals)
        edge_weights_final = [w / total for w in edge_totals]
        return state_dict_weighted_sum(edge_models, edge_weights_final)
