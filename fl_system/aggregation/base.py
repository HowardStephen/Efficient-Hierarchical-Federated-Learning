# @PATH: fl_system/aggregation/base.py
# @DATE: 03.19.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn
#
# Aggregator base class and unified interface.
from __future__ import annotations

"""
Unified aggregator interface: all aggregation algorithms implement this interface.
"""

from typing import Any

import torch


class Aggregator:
    """Aggregator base class: unified interface."""

    def aggregate(
        self,
        client_updates: list[dict[str, torch.Tensor]],
        sample_counts: list[int] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Aggregate client updates into global model parameters.

        Args:
            client_updates: List of client model parameters, each element is a state_dict
            sample_counts: Optional, per-client sample counts for weighted averaging
            **kwargs: Additional parameters for specific algorithms (e.g., edge_ids)

        Returns:
            Aggregated global model parameters (state_dict)
        """
        raise NotImplementedError
