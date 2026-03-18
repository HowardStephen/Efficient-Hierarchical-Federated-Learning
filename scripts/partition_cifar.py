# @PATH: scripts/partition_cifar.py
# @DATE: 03.18.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn

"""
CIFAR Dirichlet Non-IID partitioning script.

This script partitions CIFAR-10 and CIFAR-100 into client datasets using the
Dirichlet distribution over class labels. Lower alpha yields more heterogeneous
(client-skewed) data; higher alpha yields more IID-like splits.

It:
  1. Loads raw CIFAR data from data/raw/ (run prepare_cifar.py first)
  2. Partitions train set by Dirichlet(alpha) per class
  3. Saves each client's data as client_{i}.pt and shared test.pt
  4. Writes meta.json with partition statistics

Output: data/processed/cifar10_dirichlet/alpha{alpha}/ and
        data/processed/cifar100_dirichlet/alpha{alpha}/
        Default alphas: CIFAR-10 {0.1, 0.5, 1.0}, CIFAR-100 {0.1, 0.5}

Usage: python scripts/partition_cifar.py [--config config/partition_cifar.yaml]
       [--num_clients N] [--seed N] [--cifar10_only | --cifar100_only]
       (CLI args override config file)
"""

import argparse
import json
import os
import numpy as np
import yaml
import torch
from torchvision import datasets, transforms

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "partition_cifar.yaml")


def dirichlet_partition(labels, num_clients, alpha, seed):
    """
    Partition indices by Dirichlet distribution over labels.
    For each class k, sample p ~ Dir(alpha, ..., alpha) and assign class-k samples to clients by p.

    Args:
        labels: array of shape (N,) with class labels
        num_clients: number of clients
        alpha: Dirichlet concentration parameter
        seed: random seed

    Returns:
        client_indices: list of arrays, client_indices[i] = indices for client i
    """
    np.random.seed(seed)
    num_samples = len(labels)
    num_classes = len(np.unique(labels))

    # Group indices by class
    class_indices = {k: np.where(labels == k)[0] for k in range(num_classes)}

    # Initialize client index lists
    client_indices = [[] for _ in range(num_clients)]

    # For each class, sample Dirichlet and assign
    for k in range(num_classes):
        idx = class_indices[k]
        np.random.shuffle(idx)
        n_k = len(idx)

        if n_k == 0:
            continue

        # p[i] = proportion of class k going to client i
        p = np.random.dirichlet(alpha * np.ones(num_clients))
        # Convert to counts (ensure sum = n_k)
        counts = (np.cumsum(p) * n_k).astype(int)
        counts = np.diff(np.concatenate([[0], counts]))
        # Fix rounding
        counts[-1] = n_k - np.sum(counts[:-1])

        start = 0
        for i in range(num_clients):
            end = start + counts[i]
            client_indices[i].extend(idx[start:end].tolist())
            start = end

    # Shuffle each client's indices
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return [np.array(ci) for ci in client_indices]


def partition_and_save(dataset_name, alpha, num_clients, seed):
    """Load CIFAR, partition by Dirichlet, save to disk."""
    to_tensor = transforms.ToTensor()
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(root=DATA_RAW, train=True, download=False)
        num_classes = 10
    elif dataset_name == "cifar100":
        dataset = datasets.CIFAR100(root=DATA_RAW, train=True, download=False)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Test set (shared, not partitioned)
    if dataset_name == "cifar10":
        test_dataset = datasets.CIFAR10(root=DATA_RAW, train=False, download=False)
    else:
        test_dataset = datasets.CIFAR100(root=DATA_RAW, train=False, download=False)

    # Load train: PIL -> tensor (N, C, H, W)
    data = torch.stack([to_tensor(x) for x, _ in dataset])
    labels = np.array([y for _, y in dataset])
    test_data = torch.stack([to_tensor(x) for x, _ in test_dataset])
    test_labels = torch.tensor([y for _, y in test_dataset])

    # Partition
    client_indices = dirichlet_partition(labels, num_clients, alpha, seed)

    # Output dir: data/processed/cifar10_dirichlet/alpha0.1/
    out_dir = os.path.join(DATA_PROCESSED, f"{dataset_name}_dirichlet", f"alpha{alpha}")
    os.makedirs(out_dir, exist_ok=True)

    # Save each client's train data (as tensors, shape NCHW)
    for i, indices in enumerate(client_indices):
        if len(indices) == 0:
            continue
        x = data[indices]
        y = torch.tensor(labels[indices], dtype=torch.long)
        torch.save({"x": x, "y": y}, os.path.join(out_dir, f"client_{i}.pt"))

    # Save shared test set
    torch.save({"x": test_data, "y": test_labels}, os.path.join(out_dir, "test.pt"))

    # Meta
    meta = {
        "dataset": dataset_name,
        "alpha": alpha,
        "num_clients": num_clients,
        "num_classes": num_classes,
        "seed": seed,
        "train_samples_per_client": [len(ci) for ci in client_indices],
        "test_samples": len(test_labels),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  {dataset_name} alpha={alpha}: {num_clients} clients, "
          f"train samples {[len(ci) for ci in client_indices]}, test {len(test_labels)}")


def load_config(config_path):
    """Load config from YAML file with defaults."""
    defaults = {
        "num_clients": 100,
        "seed": 42,
        "cifar10_alphas": [0.1, 0.5, 1.0],
        "cifar100_alphas": [0.1, 0.5],
        "cifar10_only": False,
        "cifar100_only": False,
    }
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return {**defaults, **cfg}
    return defaults


def main():
    parser = argparse.ArgumentParser(description="Dirichlet partition CIFAR-10/100")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH,
                        help="Path to config YAML (default: config/partition_cifar.yaml)")
    parser.add_argument("--num_clients", type=int, default=None, help="Override num_clients from config")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config")
    parser.add_argument("--cifar10_only", action="store_true", help="Only partition CIFAR-10")
    parser.add_argument("--cifar100_only", action="store_true", help="Only partition CIFAR-100")
    args = parser.parse_args()

    cfg = load_config(args.config)
    num_clients = args.num_clients if args.num_clients is not None else cfg["num_clients"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    cifar10_alphas = cfg["cifar10_alphas"]
    cifar100_alphas = cfg["cifar100_alphas"]
    cifar10_only = args.cifar10_only or cfg.get("cifar10_only", False)
    cifar100_only = args.cifar100_only or cfg.get("cifar100_only", False)

    print("Dirichlet partitioning CIFAR...")
    print(f"  config={args.config}, num_clients={num_clients}, seed={seed}")

    if not cifar100_only:
        for alpha in cifar10_alphas:
            partition_and_save("cifar10", alpha, num_clients, seed)

    if not cifar10_only:
        for alpha in cifar100_alphas:
            partition_and_save("cifar100", alpha, num_clients, seed)

    print(f"Done. Saved to {DATA_PROCESSED}/")


if __name__ == "__main__":
    main()
