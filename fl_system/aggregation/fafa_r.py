# @PATH: fl_system/aggregation/fafa_r.py
# @DATE: 03.19.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn
#
# FAFA-R (Fast Adaptive Federated Aggregation with Reputation) algorithm.
# Implements similarity, reputation, system capability, and adaptive aggregation weights.
# Supports ablation switches: weight_clipping, trust_mixing, deviation_normalization, baseline_mode.

"""
FAFA-R: Fast Adaptive Federated Aggregation with Reputation.

Core formulas:
  (1) Similarity:     s_i = exp(-gamma * ||g_i - g_bar^(t-1)||^2)
  (2) Reputation:     r_i^(t) = lambda * r_i^(t-1) + (1 - lambda) * s_i
  (3) Capability:     kappa_i = 1 / (1 + tau_i / tau_avg)
  (4) Aggregation:    beta_i ∝ r_i * kappa_i

g_bar^(t-1) is the previous round's global average update (broadcast by cloud).
Supports ablation: weight_clipping, trust_mixing, deviation_normalization, baseline_mode.
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch

from ..utils.model_utils import (
    state_dict_weighted_sum,
    state_dict_scale,
    state_dict_subtract,
    flatten_state_dict,
)

logger = logging.getLogger(__name__)

_EPS = 1e-8
_MAX_WEIGHT = 10.0
_MIN_WEIGHT = 1e-6


def _compute_deviation_normalized_scores(
    updates: list[dict[str, torch.Tensor]],
    gamma: float,
) -> list[float]:
    """
    Deviation-normalized similarity (ablation variant).
    Compute s_i at edge from g_mean instead of g_bar_prev from cloud.

    sigma_i = ||g_i - g_mean||^2 / (||g_mean||^2 + epsilon)
    s_i = exp(-gamma * sigma_i)

    Used when deviation_normalization=True; g_mean is the mean of client updates at this edge.
    """
    n = len(updates)
    if n == 0:
        return []

    g_mean = state_dict_weighted_sum(updates, [1.0 / n] * n)
    g_mean_norm_sq = float((flatten_state_dict(g_mean) @ flatten_state_dict(g_mean)).item()) + _EPS

    scores = []
    for g_i in updates:
        diff = state_dict_subtract(g_i, g_mean)
        diff_flat = flatten_state_dict(diff)
        sq_norm = float((diff_flat @ diff_flat).item())
        sigma_i = sq_norm / g_mean_norm_sq
        s_i = math.exp(-gamma * sigma_i)
        scores.append(s_i)
    return scores


def edge_aggregate(
    updates: list[dict[str, torch.Tensor]],
    tau_list: list[float],
    s_list: list[float],
    reputations: list[float],
    lambda_decay: float,
    use_equal_weights: bool = False,
    weight_clipping: bool = False,
    weight_clip_epsilon: float = 0.01,
    trust_mixing: bool = False,
    trust_mixing_alpha: float = 1.0,
    deviation_normalization: bool = False,
    gamma: float = 1.0,
) -> Tuple[dict[str, torch.Tensor], float, float, list[float], Dict[str, float]]:
    """
    Edge-layer aggregation (Algorithm lines 10-19).

    Computes reputation update r_new, capability kappa, aggregation weights beta,
    and weighted sum g_e of client updates. Supports ablation switches.

    Args:
        updates: Client gradient updates (g_i)
        tau_list: Per-client round latency (tau_i)
        s_list: Per-client similarity scores (s_i), or recomputed if deviation_normalization
        reputations: Previous round reputations (r_i^(t-1))
        lambda_decay: Reputation smoothing factor (lambda)
        use_equal_weights: If True, use FedAvg-like equal weights (baseline_mode)
        weight_clipping: (a) Enforce beta_i >= epsilon
        weight_clip_epsilon: Minimum weight in clipping
        trust_mixing: (b) beta_i = (1-alpha)*(1/N) + alpha*beta_i
        trust_mixing_alpha: Mixing coefficient
        deviation_normalization: (d) Compute s_i from edge g_mean instead of g_bar_prev
        gamma: Temperature for similarity

    Returns:
        (g_e, kappa_e, r_e, r_new, stats)
        g_e: Aggregated gradient at edge; kappa_e, r_e: edge-level capability/reputation
        r_new: Updated per-client reputations; stats: beta_min, beta_max, beta_var, r_mean, r_var
    """
    n = len(updates)
    if n == 0:
        raise ValueError("No valid client updates")

    # (d) deviation_normalization: compute s_i at edge from g_mean instead of g_bar_prev
    if deviation_normalization:
        s_list = _compute_deviation_normalized_scores(updates, gamma)

    tau_e = sum(tau_list) / n

    # Capability: kappa_i = 1 / (1 + tau_i / tau_avg)
    kappas = [
        1.0 / (1.0 + tau_i / (tau_e + _EPS))
        for tau_i in tau_list
    ]

    # Reputation update: r_i^(t) = lambda * r_i^(t-1) + (1 - lambda) * s_i
    r_new = [
        lambda_decay * r_old + (1 - lambda_decay) * s
        for r_old, s in zip(reputations, s_list)
    ]

    # Aggregation weight: beta_i ∝ r_i * kappa_i (normalized, clipped)
    if use_equal_weights:
        betas = [1.0 / n] * n
    else:
        weights = [r * k for r, k in zip(r_new, kappas)]
        total = sum(weights) + _EPS
        betas = [max(_MIN_WEIGHT, min(_MAX_WEIGHT, w / total)) for w in weights]
        total_beta = sum(betas)
        betas = [b / total_beta for b in betas]

    # (a) Ablation: weight_clipping — enforce beta_i >= epsilon
    if weight_clipping and not use_equal_weights:
        eps = max(weight_clip_epsilon, _EPS)
        betas = [max(eps, b) for b in betas]
        total_beta = sum(betas)
        betas = [b / total_beta for b in betas]

    # (b) Ablation: trust_mixing — beta_i = (1-alpha)*(1/N) + alpha*beta_i
    if trust_mixing and not use_equal_weights:
        alpha = max(0.0, min(1.0, trust_mixing_alpha))
        betas = [(1.0 - alpha) * (1.0 / n) + alpha * b for b in betas]
        total_beta = sum(betas)
        betas = [b / total_beta for b in betas]

    g_e = state_dict_weighted_sum(updates, betas)
    kappa_e = sum(kappas) / n
    r_e = sum(r_new) / n

    # Stats for ablation / logging
    import statistics
    stats = {
        "beta_min": min(betas),
        "beta_max": max(betas),
        "beta_var": statistics.variance(betas) if len(betas) > 1 else 0.0,
        "r_mean": sum(r_new) / n,
        "r_var": statistics.variance(r_new) if len(r_new) > 1 else 0.0,
    }

    if logger.isEnabledFor(logging.DEBUG):
        g_norm = sum(v.pow(2).sum().item() for v in g_e.values()) ** 0.5
        logger.debug(
            "edge_agg: n=%d | s=%s | κ=%s | r_new=%s | β=%s | ||g_e||=%.6f",
            n,
            [f"{s:.4f}" for s in s_list],
            [f"{k:.4f}" for k in kappas],
            [f"{r:.4f}" for r in r_new],
            [f"{b:.4f}" for b in betas],
            g_norm,
        )

    return g_e, kappa_e, r_e, r_new, stats


def cloud_aggregate(
    edge_updates: list[dict[str, torch.Tensor]],
    kappa_list: list[float],
    r_list: list[float],
) -> dict[str, torch.Tensor]:
    """
    Cloud-layer aggregation (Algorithm lines 21-22).

    Weighted sum of edge updates: alpha_e ∝ kappa_e * r_e.
    g_global = sum_e alpha_e * g_e (normalized, clipped).
    """
    n = len(edge_updates)
    if n == 0:
        raise ValueError("No valid edge updates")

    weights = [k * r for k, r in zip(kappa_list, r_list)]
    total = sum(weights) + _EPS
    alphas = [max(_MIN_WEIGHT, min(_MAX_WEIGHT, w / total)) for w in weights]
    total_alpha = sum(alphas)
    alphas = [a / total_alpha for a in alphas]

    g_global = state_dict_weighted_sum(edge_updates, alphas)
    if logger.isEnabledFor(logging.DEBUG):
        g_norm = sum(v.pow(2).sum().item() for v in g_global.values()) ** 0.5
        logger.debug(
            "cloud_agg: n=%d | α=%s | ||g_global||=%.6f",
            n,
            [f"{a:.4f}" for a in alphas],
            g_norm,
        )

    return g_global


def apply_global_update(
    current_state: dict[str, torch.Tensor],
    aggregate_gradient: dict[str, torch.Tensor],
    lr: float,
) -> dict[str, torch.Tensor]:
    """
    Line 22: Apply global update.
    w^(t) = w^(t-1) - eta * g_global, where g_global = sum_e alpha_e * g_e
    """
    delta = state_dict_scale(aggregate_gradient, lr)
    return state_dict_subtract(current_state, delta)


def compute_similarity(
    g_i: dict[str, torch.Tensor],
    g_bar_prev: Optional[dict[str, torch.Tensor]],
    gamma: float,
) -> float:
    """
    Line 6: Similarity score s_i = exp(-gamma * ||g_i - g_bar^(t-1)||^2).

    Measures how close client i's gradient is to the previous global average.
    Returns 1.0 when g_bar_prev is None (first round, no reference).
    """
    if g_bar_prev is None:
        return 1.0
    diff = state_dict_subtract(g_i, g_bar_prev)
    diff_flat = flatten_state_dict(diff)
    sq_norm = float((diff_flat @ diff_flat).item())
    return math.exp(-gamma * sq_norm)
