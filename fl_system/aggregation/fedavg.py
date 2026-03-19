# @PATH: fl_system/aggregation/fedavg.py
# @DATE: 03.19.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn
#
# FedAvg: Federated Averaging aggregation.
from __future__ import annotations

"""
FedAvg: Federated Averaging.

McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", 2017.
Aggregation: w_global = sum(n_i / N) * w_i, where n_i is client sample count, N is total.
If sample counts not provided: equal-weighted average w_global = (1/K) * sum(w_i).
"""

from typing import Any

import torch

from ..utils.model_utils import state_dict_weighted_sum

from .base import Aggregator


class FedAvgAggregator(Aggregator):
    """FedAvg aggregator: sample-weighted average; equal-weighted when sample counts absent."""

    def aggregate(
        self,
        client_updates: list[dict[str, torch.Tensor]],
        sample_counts: list[int] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Perform FedAvg aggregation.

        Args:
            client_updates: List of client model parameters
            sample_counts: Per-client local sample counts; equal-weighted when None

        Returns:
            Aggregated global model parameters
        """
        if not client_updates:
            raise ValueError("client_updates cannot be empty")

        n = len(client_updates)
        if sample_counts is not None:
            if len(sample_counts) != n:
                raise ValueError("sample_counts length must match client_updates")
            total = sum(sample_counts)
            if total <= 0:
                raise ValueError("sample_counts total must be > 0")
            weights = [c / total for c in sample_counts]
        else:
            weights = [1.0 / n] * n

        return state_dict_weighted_sum(client_updates, weights)
