#!/bin/bash
# @PATH: run.sh
# @DATE: 03.18.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn
#
# One-click run script for the entire project.
# 1. Create conda environment (ehfl_env)
# 2. Run scripts/download_and_partition.sh (download + partition data)

set -e

cd "$(dirname "$0")"

# ========== Create conda environment ==========
if ! conda env list | grep -q "ehfl_env"; then
    echo ">>> Creating conda environment ehfl_env (environment.yml, GPU)..."
    conda env create -f environment.yml
    echo ">>> Environment created"
else
    echo ">>> Conda environment ehfl_env already exists, skipping"
fi

# For CPU-only: use conda env create -f environment-cpu.yml

# ========== Download and partition data ==========
bash scripts/download_and_partition.sh

echo ""
echo "=== All done! ==="
