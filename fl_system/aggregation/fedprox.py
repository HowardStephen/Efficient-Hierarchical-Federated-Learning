# @PATH: fl_system/aggregation/fedprox.py
# @DATE: 03.19.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn
#
# FedProx aggregation algorithm.
from __future__ import annotations

"""
FedProx: Federated Optimization in Heterogeneous Networks.

Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020.
FedProx adds proximal regularization to client loss: F_i(w) + (mu/2)||w - w_global||^2.
Server-side aggregation is identical to FedAvg (weighted average).
"""

from typing import Any

import torch

from .base import Aggregator
from .fedavg import FedAvgAggregator


class FedProxAggregator(Aggregator):
    """
    FedProx aggregator.

    Aggregation rule matches FedAvg; proximal term is applied during client-side local training.
    """

    def __init__(self, mu: float = 0.01):
        """
        Args:
            mu: Proximal regularization coefficient, used in client training (stored here; aggregation logic same as FedAvg)
        """
        self.mu = mu
        self._fedavg = FedAvgAggregator()

    def aggregate(
        self,
        client_updates: list[dict[str, torch.Tensor]],
        sample_counts: list[int] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Perform FedProx aggregation (same as FedAvg).

        Args:
            client_updates: List of client model parameters
            sample_counts: Per-client sample counts; equal-weighted when None

        Returns:
            Aggregated global model parameters
        """
        return self._fedavg.aggregate(client_updates, sample_counts, **kwargs)
