# @PATH: fl_system/aggregation/scaffold.py
# @DATE: 03.19.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn
#
# SCAFFOLD aggregation algorithm.
from __future__ import annotations

"""
SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning", ICML 2020.
Server-side: simple average of model parameters w_global = (1/K) * sum(w_i).
Control variates (c_i, c_global) are maintained client/server-side; this module implements model aggregation only.
"""

from typing import Any

import torch

from ..utils.model_utils import state_dict_weighted_sum

from .base import Aggregator


class SCAFFOLDAggregator(Aggregator):
    """SCAFFOLD aggregator: equal-weighted average of client model parameters."""

    def aggregate(
        self,
        client_updates: list[dict[str, torch.Tensor]],
        sample_counts: list[int] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Perform SCAFFOLD aggregation (equal-weighted average).

        SCAFFOLD uses equal-weighted averaging on the server, not sample-weighted.

        Args:
            client_updates: List of client model parameters
            sample_counts: Ignored (SCAFFOLD uses equal weights)

        Returns:
            Aggregated global model parameters
        """
        if not client_updates:
            raise ValueError("client_updates cannot be empty")
        n = len(client_updates)
        weights = [1.0 / n] * n
        return state_dict_weighted_sum(client_updates, weights)
