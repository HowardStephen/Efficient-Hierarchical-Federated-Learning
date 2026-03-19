# @PATH: fl_system/config/config.py
# @DATE: 03.19.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn
#
# Experiment configuration: client count, edge count, data, aggregation, etc.
from __future__ import annotations

"""
Federated learning experiment configuration.
Client count, edge server count, and other parameters are configured here.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ClientConfig:
    """Client-related configuration."""

    # Number of clients (core config)
    num_clients: int = 100

    # Clients participating per round (None = all)
    clients_per_round: Optional[int] = None

    # Local training epochs
    local_epochs: int = 1

    # Local batch size
    batch_size: int = 32

    # Learning rate
    learning_rate: float = 0.01


@dataclass
class HeterogeneityConfig:
    """System heterogeneity simulation config."""

    # Compute power range [min, max]; training time = actual_time / compute_power
    compute_power_range: Tuple[float, float] = (0.5, 2.0)

    # Bandwidth range [min, max] Mbps; comm delay = model_size / bandwidth
    bandwidth_range: Tuple[float, float] = (1.0, 10.0)

    # Enable heterogeneity simulation
    enabled: bool = True


@dataclass
class EdgeConfig:
    """Edge server configuration."""

    # Number of edge servers
    num_edges: int = 10

    # Client ID list per edge (assigned by main from num_clients/num_edges)
    # If None, system auto-assigns by num_clients // num_edges
    clients_per_edge: Optional[List[List[int]]] = None


@dataclass
class DataConfig:
    """Data configuration."""

    # Dataset: cifar10 | cifar100 | femnist
    dataset: str = "cifar10"

    # Dirichlet Non-IID parameter (for CIFAR)
    alpha: float = 0.5

    # Data root (relative to project root)
    data_root: str = "data"

    # Partition path. None = auto-generate; absolute path = use as-is
    # CIFAR: data/processed/cifar10_dirichlet/alpha0.5/
    # FEMNIST: data/processed/femnist_clients/
    partition_path: Optional[str] = None


@dataclass
class SecurityConfig:
    """Security config: verifiable edge, dropout tolerance."""

    # Enable edge integrity verification (commit = hash(g_e))
    verifiable_edge: bool = False

    # Dropout tolerance: per-round client dropout probability (0=no simulation, >0 simulates dropout)
    dropout_prob: float = 0.0


@dataclass
class PrivacyConfig:
    """Differential privacy and secret sharing (Algorithm 1 step 8)."""

    # Enable differential privacy
    dp_enabled: bool = False

    # Gradient clip bound C (L2 norm upper bound)
    clip_norm: float = 1.0

    # Gaussian noise scale sigma
    noise_scale: float = 0.01

    # Privacy budget epsilon (total, used to compute per-round sigma)
    epsilon: float = 1.0

    # Enable secret sharing (placeholder, to be implemented)
    secret_sharing_enabled: bool = False


@dataclass
class AggregationConfig:
    """Adaptive aggregation algorithm config (Algorithm 1)."""

    # Aggregation algorithm (synced with ExperimentConfig.algorithm)
    algorithm: str = "fafa_r"

    # Reputation decay factor lambda in (0, 1)
    reputation_decay: float = 0.9

    # Temperature gamma > 0; similarity s_i = exp(-gamma * ||g_i - nabla F_i||^2)
    temperature: float = 1.0

    # Global learning rate eta_t (extensible to scheduler)
    # Effective lr approx global_lr * client_lr; FedAvg suggests 1.0 with client_lr=0.01
    global_lr: float = 1.0

    # Compute similarity scores (fedavg etc. can disable for speed)
    compute_similarity: bool = True

    # --- Ablation study switches ---
    # (a) weight_clipping: enforce beta_i >= epsilon
    weight_clipping: bool = False
    weight_clip_epsilon: float = 0.01

    # (b) trust_mixing: beta_i = (1 - alpha) * (1/N) + alpha * beta_i
    trust_mixing: bool = False
    trust_mixing_alpha: float = 1.0  # e.g. 0.3, 0.5, 0.7, 1.0

    # (c) reputation_decay: lambda in r_i update (already above)

    # (d) deviation_normalization: sigma_i = ||g_i - g_mean||^2 / (||g_mean||^2 + epsilon)
    deviation_normalization: bool = False

    # (e) baseline_mode: FedAvg (uniform weights)
    baseline_mode: bool = False


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    # Global training rounds
    global_rounds: int = 100

    # Random seed
    seed: int = 42

    # Device
    device: str = "cuda"

    # Algorithm name (for result path results/dataset/model/alpha/algorithm/)
    # fedavg, hierarchical_fedavg, fedprox, scaffold, fafa_r
    algorithm: str = "fafa_r"

    # Result path overrides (optional)
    results_model_dir: Optional[str] = None  # e.g. "vgglite", overrides default model dir name
    results_alpha_format: str = "default"  # "default" | "alpha_prefix" (alpha0.5)
    results_experiment_name: Optional[str] = None  # Ablation: override algorithm as dir name

    # Fairness: evaluate global model on each client's local data per round, compute accuracy_variance
    record_fairness_metrics: bool = False

    # Sub-configs
    client: ClientConfig = field(default_factory=ClientConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    heterogeneity: HeterogeneityConfig = field(default_factory=HeterogeneityConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    def __post_init__(self):
        """Set default partition_path and sync algorithm to aggregation."""
        self.aggregation.algorithm = self.algorithm
        if self.data.partition_path is None:
            if self.data.dataset == "femnist":
                self.data.partition_path = f"{self.data.data_root}/processed/femnist_clients"
            else:
                self.data.partition_path = (
                    f"{self.data.data_root}/processed/"
                    f"{self.data.dataset}_dirichlet/alpha{self.data.alpha}"
                )


# Default config instance for direct import
default_config = ExperimentConfig()
