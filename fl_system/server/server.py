# @PATH: fl_system/server/server.py
# @DATE: 03.18.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn
#
# Cloud parameter server. Implements FAFA-R algorithm lines 21-22.
# Maintains w and g_bar (previous round global average update) for client s_i computation.
from __future__ import annotations

"""
Parameter server: maintains w, g_bar (previous round global average update) for client s_i computation.
"""

from typing import Any, Optional

import torch

from ..aggregation import cloud_aggregate, apply_global_update
from ..config import AggregationConfig


class ParameterServer:
    """Parameter server. Maintains w^(t), g_bar^(t) (saved after each round update)."""

    def __init__(
        self,
        initial_state: dict[str, torch.Tensor],
        aggregation_config: AggregationConfig,
    ):
        self.global_state = {k: v.clone() for k, v in initial_state.items()}
        self.agg_config = aggregation_config
        # g_bar^(t-1): previous round cloud-aggregated global average update; None for first round
        self._g_bar_prev: Optional[dict[str, torch.Tensor]] = None

    def get_global_state(self) -> dict[str, torch.Tensor]:
        """Broadcast w^(t-1)."""
        return {k: v.clone() for k, v in self.global_state.items()}

    def get_g_bar_prev(self) -> Optional[dict[str, torch.Tensor]]:
        """Broadcast g_bar^(t-1) for client s_i computation. Returns None for first round."""
        if self._g_bar_prev is None:
            return None
        return {k: v.clone() for k, v in self._g_bar_prev.items()}

    def update(
        self,
        edge_updates: list[dict[str, torch.Tensor]],
        kappa_list: list[float],
        r_list: list[float],
    ) -> None:
        """
        Algorithm lines 21-22: cloud aggregation, update w, save g_bar^(t) for next round.
        """
        g_global = cloud_aggregate(edge_updates, kappa_list, r_list)
        self.global_state = apply_global_update(
            self.global_state,
            g_global,
            self.agg_config.global_lr,
        )
        self._g_bar_prev = {k: v.clone() for k, v in g_global.items()}
