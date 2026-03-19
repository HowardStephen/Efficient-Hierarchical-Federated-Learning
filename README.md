<!--
@PATH: README.md
@DATE: 03.17.2026
@AUTHOR: Howard
@E-MAIL: QSX20251439@student.fjnu.edu.cn

Project root README. Overview, fl_system, experimental setup, data.
-->

# Efficient-Hierarchical-Federated-Learning

A three-tier hierarchical federated learning system (Client → Edge Server → Cloud) for improved convergence under system heterogeneity and non-IID data.

---

## fl_system

The core federated learning framework lives in [`fl_system/`](fl_system/). It implements:

| Component | Description |
|-----------|-------------|
| **Core algorithm** | fafa_r (Fast Adaptive Federated Aggregation with Reputation) |
| **Baselines** | FedAvg, Hierarchical FedAvg, FedProx, SCAFFOLD |
| **Architecture** | Client, Edge Server, Cloud (Parameter Server) |
| **Datasets** | CIFAR-10, CIFAR-100, FEMNIST |

**Quick start:**
```bash
./run.sh                                    # Create env + prepare data
conda activate ehfl_env
python fl_system/main.py --config config/experiment.yaml --algorithm fafa_r
```

See [fl_system/README.md](fl_system/README.md) for full documentation (architecture, configuration, ablation study, etc.).

---

## Experimental Setup

### Conda Environment

The project uses a conda environment named `ehfl_env` with Python 3.9.

**GPU (CUDA 11.8):**
```bash
conda env create -f environment.yml
conda activate ehfl_env
```
- PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118
- numpy, matplotlib, pyyaml

**CPU-only:**
```bash
conda env create -f environment-cpu.yml
conda activate ehfl_env
```

Alternatively, run `./run.sh` to create the environment and prepare data in one step.

---

### Data

The [`data/`](data/) directory is gitignored (no data in repo). Run `./run.sh` or `scripts/download_and_partition.sh` to populate it. See [data/README.md](data/README.md) for details.

#### Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| **CIFAR-10** | [torchvision](https://pytorch.org/vision/stable/datasets.html#cifar) | 60K 32×32 color images, 10 classes (airplane, automobile, bird, etc.) |
| **CIFAR-100** | [torchvision](https://pytorch.org/vision/stable/datasets.html#cifar) | 60K 32×32 color images, 100 fine-grained classes |
| **FEMNIST** | [LEAF](https://github.com/TalwalkarLab/leaf) (NIST SD19) | Federated Extended MNIST; 62 classes (digits + letters), one client per writer |

Raw data is stored in `data/raw/`; processed/partitioned data in `data/processed/`.

#### Data Partitioning

**CIFAR-10 / CIFAR-100** — Dirichlet Non-IID partitioning over class labels:
- For each class k, sample proportions p ~ Dir(alpha, ..., alpha) and assign class-k samples to clients accordingly.
- Lower alpha → more heterogeneous (client-skewed); higher alpha → more IID-like.
- Config: `config/partition_cifar.yaml`
  - `num_clients`: 100 (default)
  - `cifar10_alphas`: [0.1, 0.5, 1.0]
  - `cifar100_alphas`: [0.1, 0.5]
- Output: `data/processed/cifar10_dirichlet/alpha{alpha}/`, `data/processed/cifar100_dirichlet/alpha{alpha}/`
  - Per client: `client_{i}.pt`; shared test: `test.pt`; metadata: `meta.json`

**FEMNIST** — Natural partition by writer:
- One client per writer (no synthetic partitioning).
- Output: `data/processed/femnist_clients/` with `all_data`, `train`, `test` splits.
