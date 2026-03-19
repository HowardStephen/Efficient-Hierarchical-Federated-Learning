# @PATH: fl_system/edge/edge.py
# @DATE: 03.18.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn
#
# Edge server module. Implements FAFA-R algorithm lines 10-19 + ablation.
# Receives (g_i, tau_i, s_i), performs aggregation; supports ablation switches.
from __future__ import annotations

"""
Edge server: receives (g_i, tau_i, s_i), performs aggregation (lines 11-19).
Supports ablation switches; returns per-round weight/reputation statistics.
"""

from typing import Any, Dict, Optional

import torch

from ..aggregation import edge_aggregate
from ..config import AggregationConfig


class EdgeServer:
    """Edge server; maintains client reputation r_i."""

    def __init__(
        self,
        edge_id: int,
        client_ids: list[int],
        aggregation_config: AggregationConfig,
    ):
        self.edge_id = edge_id
        self.client_ids = client_ids
        self.agg_config = aggregation_config
        self.reputations: dict[int, float] = {i: 1.0 for i in client_ids}

    def aggregate(
        self,
        updates: dict[int, dict[str, torch.Tensor]],
        tau: dict[int, float],
        s: dict[int, float],
    ) -> tuple[dict[str, torch.Tensor], float, float, Optional[Dict[str, float]]]:
        """
        Algorithm lines 10-19. s_i is uploaded by clients (or computed by edge in deviation_normalization).

        Returns:
            (g_e, kappa_e, r_e, stats)  # stats for ablation experiments
        """
        valid_ids = [i for i in self.client_ids if i in updates and i in tau and i in s]
        if not valid_ids:
            raise ValueError(f"Edge {self.edge_id}: no valid client updates")

        updates_list = [updates[i] for i in valid_ids]
        tau_list = [tau[i] for i in valid_ids]
        s_list = [s[i] for i in valid_ids]
        reputations = [self.reputations.get(i, 1.0) for i in valid_ids]

        fedavg_like = getattr(self.agg_config, "algorithm", "yourmethod").lower() in (
            "fedavg", "hierarchical_fedavg", "fedprox", "scaffold"
        )
        baseline_mode = getattr(self.agg_config, "baseline_mode", False)
        use_equal = fedavg_like or baseline_mode

        g_e, kappa_e, r_e, r_new, stats = edge_aggregate(
            updates_list,
            tau_list,
            s_list,
            reputations,
            lambda_decay=self.agg_config.reputation_decay,
            use_equal_weights=use_equal,
            weight_clipping=getattr(self.agg_config, "weight_clipping", False),
            weight_clip_epsilon=getattr(self.agg_config, "weight_clip_epsilon", 0.01),
            trust_mixing=getattr(self.agg_config, "trust_mixing", False),
            trust_mixing_alpha=getattr(self.agg_config, "trust_mixing_alpha", 1.0),
            deviation_normalization=getattr(self.agg_config, "deviation_normalization", False),
            gamma=getattr(self.agg_config, "temperature", 1.0),
        )

        for i, cid in enumerate(valid_ids):
            self.reputations[cid] = r_new[i]

        if fedavg_like or baseline_mode:
            kappa_e, r_e = 1.0, 1.0

        return g_e, kappa_e, r_e, stats
