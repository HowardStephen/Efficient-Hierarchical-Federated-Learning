# @PATH: fl_system/data/dataset_loader.py
# @DATE: 03.18.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn
#
# Load CIFAR (pt format) and FEMNIST (LEAF JSON format) client data.
from __future__ import annotations

"""
CIFAR: partition_cifar.py produces client_{i}.pt, test.pt
FEMNIST: LEAF prepare_femnist produces train/*.json, test/*.json
"""

import json
import os
from typing import Dict

import torch
from torch.utils.data import TensorDataset


# ---------------------------------------------------------------------------
# CIFAR loading (pt format)
# ---------------------------------------------------------------------------

# CIFAR10/100 standard normalization (training set statistics)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _normalize_cifar(x: torch.Tensor, dataset: str) -> torch.Tensor:
    """Channel-wise normalization for CIFAR images."""
    if "cifar100" in dataset.lower():
        mean = torch.tensor(CIFAR100_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR100_STD).view(1, 3, 1, 1)
    else:
        mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return (x.float() - mean) / std


def load_cifar_client_data(partition_path: str, client_id: int) -> TensorDataset:
    """Load CIFAR single client data client_{i}.pt (with normalization)."""
    path = os.path.join(partition_path, f"client_{client_id}.pt")
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    x, y = data["x"], data["y"]
    x = _normalize_cifar(x, partition_path)
    return TensorDataset(x, y)


def load_cifar_test_data(partition_path: str) -> TensorDataset:
    """Load CIFAR shared test set test.pt (with normalization)."""
    path = os.path.join(partition_path, "test.pt")
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    x, y = data["x"], data["y"]
    x = _normalize_cifar(x, partition_path)
    return TensorDataset(x, y)


def load_cifar_all_clients(
    partition_path: str,
    num_clients: int,
) -> Dict[int, TensorDataset]:
    """Load CIFAR data for all clients."""
    result = {}
    for i in range(num_clients):
        result[i] = load_cifar_client_data(partition_path, i)
    return result


# ---------------------------------------------------------------------------
# FEMNIST loading (LEAF JSON format)
# ---------------------------------------------------------------------------


def _load_femnist_json(json_path: str) -> dict:
    """Read LEAF JSON: {users, num_samples, user_data: {user_id: {x, y}}}."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _femnist_user_to_tensor_dataset(user_data: dict) -> TensorDataset:
    """Convert user_data {x: [[...]], y: [...]} to TensorDataset; x reshaped to (N,1,28,28)."""
    x_list = user_data["x"]
    y_list = user_data["y"]
    x = torch.tensor(x_list, dtype=torch.float32)
    # LEAF: flattened 784, reshape to (N, 1, 28, 28)
    if x.dim() == 2 and x.size(1) == 784:
        x = x.view(-1, 1, 28, 28)
    y = torch.tensor(y_list, dtype=torch.long)
    return TensorDataset(x, y)


def load_femnist_all_clients(partition_path: str) -> Dict[int, TensorDataset]:
    """
    Load FEMNIST data for all clients.
    Prefer train/; fallback to all_data/.
    Also try data/raw/femnist/leaf/data/femnist/data/ (LEAF raw output).
    Client IDs follow JSON user order.
    """
    result = {}
    client_id = 0

    # Candidate paths: processed and LEAF raw path
    candidates = [
        partition_path,
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(partition_path))), "raw", "femnist", "leaf", "data", "femnist"),
    ]
    for base in candidates:
        for subdir in ["train", "all_data"]:
            src_dir = os.path.join(base, subdir)
            if not os.path.isdir(src_dir):
                continue
            for fname in sorted(os.listdir(src_dir)):
                if not fname.endswith(".json"):
                    continue
                path = os.path.join(src_dir, fname)
                data = _load_femnist_json(path)
                users = data.get("users", [])
                user_data = data.get("user_data", {})
                for uid in users:
                    if uid not in user_data:
                        continue
                    result[client_id] = _femnist_user_to_tensor_dataset(user_data[uid])
                    client_id += 1
            if result:
                return result

    return result


def load_femnist_test_data(partition_path: str) -> TensorDataset:
    """
    Load FEMNIST test set.
    Aggregate test data from all users in test/*.json.
    """
    x_list, y_list = [], []
    candidates = [
        partition_path,
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(partition_path))), "raw", "femnist", "leaf", "data", "femnist"),
    ]
    test_dir = None
    for base in candidates:
        td = os.path.join(base, "test")
        if os.path.isdir(td):
            test_dir = td
            break
    if test_dir is None:
        raise FileNotFoundError(f"FEMNIST test dir not found. Tried: {candidates}")

    for fname in sorted(os.listdir(test_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(test_dir, fname)
        data = _load_femnist_json(path)
        user_data = data.get("user_data", {})
        for uid, ud in user_data.items():
            x_list.extend(ud["x"])
            y_list.extend(ud["y"])

    if not x_list:
        raise FileNotFoundError(f"No FEMNIST test data in {test_dir}")

    x = torch.tensor(x_list, dtype=torch.float32)
    if x.dim() == 2 and x.size(1) == 784:
        x = x.view(-1, 1, 28, 28)
    y = torch.tensor(y_list, dtype=torch.long)
    return TensorDataset(x, y)


def load_femnist_client_data(
    partition_path: str,
    client_id: int,
    client_datasets: Dict[int, TensorDataset] | None = None,
) -> TensorDataset:
    """Load FEMNIST single client (call load_femnist_all_clients first for mapping)."""
    if client_datasets is None:
        client_datasets = load_femnist_all_clients(partition_path)
    if client_id not in client_datasets:
        raise KeyError(f"FEMNIST client {client_id} not found")
    return client_datasets[client_id]


# ---------------------------------------------------------------------------
# Unified interface (select loader by dataset type)
# ---------------------------------------------------------------------------


def load_client_data(
    partition_path: str,
    client_id: int,
    dataset: str = "cifar10",
    femnist_cache: Dict[int, TensorDataset] | None = None,
) -> TensorDataset:
    """Unified interface: load single client data."""
    if dataset == "femnist":
        return load_femnist_client_data(partition_path, client_id, femnist_cache)
    return load_cifar_client_data(partition_path, client_id)


def load_test_data(partition_path: str, dataset: str = "cifar10") -> TensorDataset:
    """Unified interface: load test set."""
    if dataset == "femnist":
        return load_femnist_test_data(partition_path)
    return load_cifar_test_data(partition_path)


def load_all_clients(
    partition_path: str,
    num_clients: int,
    dataset: str = "cifar10",
) -> Dict[int, TensorDataset]:
    """
    Unified interface: load all client data.
    FEMNIST: num_clients ignored, returns all LEAF users; CIFAR: loads by num_clients.
    """
    if dataset == "femnist":
        return load_femnist_all_clients(partition_path)
    return load_cifar_all_clients(partition_path, num_clients)
