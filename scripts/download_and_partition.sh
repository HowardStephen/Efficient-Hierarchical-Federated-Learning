#!/bin/bash
# @PATH: scripts/download_and_partition.sh
# @DATE: 03.18.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn
#
# Download datasets and perform data partitioning.
# 1. Download CIFAR-10/100 to data/raw/
# 2. Partition CIFAR (Dirichlet Non-IID)
# 3. Download and prepare FEMNIST
#
# Usage: ./scripts/download_and_partition.sh
#        (run from project root, or call from run.sh)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ========== Step 1: Download CIFAR-10/100 to data/raw/ ==========
echo ""
echo "=== Step 1: Preparing CIFAR data ==="
conda run -n ehfl_env python scripts/prepare_cifar.py

# ========== Step 2: Dirichlet partition CIFAR (depends on Step 1) ==========
echo ""
echo "=== Step 2: Partitioning CIFAR (Dirichlet Non-IID) ==="
conda run -n ehfl_env python scripts/partition_cifar.py

# ========== Step 3: Prepare FEMNIST data ==========
echo ""
echo "=== Step 3: Preparing FEMNIST data ==="
conda run -n ehfl_env python scripts/prepare_femnist.py

echo ""
echo "=== Download and partition done! ==="
