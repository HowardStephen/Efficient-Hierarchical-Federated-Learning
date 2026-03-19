<!--
@PATH: data/README.md
@DATE: 03.19.2026
@AUTHOR: Howard
@E-MAIL: QSX20251439@student.fjnu.edu.cn

Data directory. Download/partition scripts, directory structure, fl_system integration.
-->

# Data Directory

**Data is not included in this repository.** The `data/` directory is gitignored; you must download and prepare datasets locally before running experiments.

---

## How to Obtain Data

Run the download and partition scripts from the project root:

```bash
# One-click: create conda env + download + partition
./run.sh
```

Or step by step:

```bash
conda activate ehfl_env

# 1. Download CIFAR-10/100 to data/raw/
python scripts/prepare_cifar.py

# 2. Partition CIFAR (Dirichlet Non-IID)
python scripts/partition_cifar.py

# 3. Download and prepare FEMNIST (via LEAF)
python scripts/prepare_femnist.py
```

Alternatively, run `scripts/download_and_partition.sh` to execute steps 1–3 in sequence.

---

## Directory Structure (after preparation)

```
data/
├── raw/                              # Raw datasets (downloaded)
│   ├── cifar-10-batches-py/          # CIFAR-10 (from prepare_cifar.py)
│   ├── cifar-100-python/             # CIFAR-100 (from prepare_cifar.py)
│   └── femnist/                      # FEMNIST (from prepare_femnist.py)
│       └── leaf/                     # LEAF repo + preprocessed JSON
│
└── processed/                         # Partitioned data (for federated learning)
    ├── cifar10_dirichlet/
    │   ├── alpha0.1/                 # client_{i}.pt, test.pt, meta.json
    │   ├── alpha0.5/
    │   └── alpha1.0/
    ├── cifar100_dirichlet/
    │   ├── alpha0.1/
    │   └── alpha0.5/
    └── femnist_clients/              # all_data/, train/, test/
```

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/prepare_cifar.py` | Download CIFAR-10/100 via torchvision → `data/raw/` |
| `scripts/partition_cifar.py` | Dirichlet Non-IID partition → `data/processed/cifar*_dirichlet/` |
| `scripts/prepare_femnist.py` | Clone LEAF, run preprocess → `data/processed/femnist_clients/` |
| `scripts/download_and_partition.sh` | Run all three scripts |

CIFAR partitioning is configured in `config/partition_cifar.yaml` (num_clients, alphas, seed).

---

## fl_system Integration

The federated learning framework reads data from `data/processed/` via the following components:

| File | Role |
|------|------|
| **`fl_system/data/dataset_loader.py`** | Loads client and test data. CIFAR: reads `client_{i}.pt`, `test.pt`; FEMNIST: reads `train/*.json`, `test/*.json`. |
| **`fl_system/config/config.py`** | Sets `partition_path` (defaults: `data/processed/cifar10_dirichlet/alpha{alpha}/` or `data/processed/femnist_clients/`). |
| **`fl_system/data/__init__.py`** | Exports `load_client_data`, `load_test_data`, `load_all_clients`, etc. |

**Data flow:**
```
data/processed/...  →  fl_system/data/dataset_loader.py  →  TensorDataset  →  FederatedClient
```

`main.py` (or the training entry point) passes `partition_path` from `ExperimentConfig.data.partition_path` to the loader. The path is auto-generated in `config.py` based on `dataset` and `alpha`, or can be overridden.
