# @PATH: scripts/prepare_cifar.py
# @DATE: 03.18.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn

"""
CIFAR dataset download script.

This script downloads CIFAR-10 and CIFAR-100 datasets via torchvision and saves
them to data/raw/. The raw data is used by partition_cifar.py for Non-IID
client partitioning (e.g., Dirichlet).

Output: data/raw/cifar-10-batches-py/ and data/raw/cifar-100-python/

Usage: python scripts/prepare_cifar.py
"""

import os
import torch
from torchvision import datasets, transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "raw")

def download_cifar(dataset_name="cifar10"):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    os.makedirs(DATA_ROOT, exist_ok=True)
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root=DATA_ROOT,
            train=True,
            download=True,
            transform=transform
        )
    elif dataset_name == "cifar100":
        dataset = datasets.CIFAR100(
            root=DATA_ROOT,
            train=True,
            download=True,
            transform=transform
        )
    else:
        raise ValueError("Unknown dataset name")

    print(f"{dataset_name} downloaded. Total samples: {len(dataset)}")

if __name__ == "__main__":
    download_cifar("cifar10")
    download_cifar("cifar100")
