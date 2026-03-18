# @PATH: scripts/prepare_femnist.py
# @DATE: 03.18.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn

"""
FEMNIST dataset preparation script.

This script downloads and preprocesses the FEMNIST (Federated Extended MNIST)
dataset using the LEAF benchmark pipeline. It:
  1. Clones the LEAF repository into data/raw/femnist/leaf/
  2. Runs LEAF's preprocess.sh to download NIST SD19 and convert to JSON
  3. Copies the processed client data to data/processed/femnist_clients/

Output: data/processed/femnist_clients/ with all_data, train, and test splits
        (one client per writer for federated learning).

Usage: python scripts/prepare_femnist.py
"""

import os
import subprocess
import sys

# Resolve paths relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
FEMNIST_DIR = os.path.join(DATA_RAW, "femnist")
LEAF_REPO_URL = "https://github.com/TalwalkarLab/leaf.git"
LEAF_FEMNIST_PATH = os.path.join(FEMNIST_DIR, "leaf", "data", "femnist")


def run_cmd(cmd, cwd=None, shell=True):
    """Run shell command and raise on failure."""
    result = subprocess.run(cmd, cwd=cwd, shell=shell)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {cmd}")


def download_femnist():
    """Clone LEAF repo and run FEMNIST preprocessing pipeline."""
    os.makedirs(FEMNIST_DIR, exist_ok=True)

    leaf_dir = os.path.join(FEMNIST_DIR, "leaf")
    leaf_git = os.path.join(leaf_dir, ".git")
    leaf_femnist_exists = os.path.exists(LEAF_FEMNIST_PATH)

    if os.path.exists(leaf_git):
        print("LEAF repository already exists (has .git), skipping clone.")
    elif leaf_femnist_exists:
        print("LEAF FEMNIST path already exists, skipping clone.")
    else:
        # When leaf dir exists but is not a git repo, git clone fails; clone to temp then replace
        import shutil
        leaf_temp = os.path.join(FEMNIST_DIR, "leaf_tmp_clone")
        if os.path.exists(leaf_temp):
            shutil.rmtree(leaf_temp)
        print("Cloning LEAF repository...")
        run_cmd(f"git clone {LEAF_REPO_URL} leaf_tmp_clone", cwd=FEMNIST_DIR)
        if os.path.exists(leaf_dir):
            shutil.rmtree(leaf_dir)
        os.rename(leaf_temp, leaf_dir)
        print("LEAF repository cloned.")

    if not os.path.exists(LEAF_FEMNIST_PATH):
        raise RuntimeError(f"Expected path not found: {LEAF_FEMNIST_PATH}")

    preprocess_sh = os.path.join(LEAF_FEMNIST_PATH, "preprocess.sh")
    if not os.path.isfile(preprocess_sh):
        raise RuntimeError(f"preprocess.sh not found at {preprocess_sh}")

    print("Running LEAF FEMNIST preprocess.sh (downloads NIST SD19, converts to JSON)...")
    run_cmd("./preprocess.sh", cwd=LEAF_FEMNIST_PATH)
    print("FEMNIST preprocessing completed.")

    # Copy/convert to project's processed format
    _copy_to_processed()


def _copy_to_processed():
    """Copy LEAF data/ to data/processed/femnist_clients/."""
    import shutil

    leaf_data_dir = os.path.join(LEAF_FEMNIST_PATH, "data")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed", "femnist_clients")

    if not os.path.isdir(leaf_data_dir):
        print("Warning: LEAF data/ not found, skipping copy to processed.")
        return

    os.makedirs(processed_dir, exist_ok=True)
    # Copy all_data, train, test (LEAF output structure)
    for subdir in ["all_data", "train", "test"]:
        src = os.path.join(leaf_data_dir, subdir)
        if os.path.isdir(src) and os.listdir(src):
            dst = os.path.join(processed_dir, subdir)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"Copied {subdir}/ to femnist_clients/")
    # If only all_data exists (no train/test split)
    if not os.listdir(processed_dir):
        for item in os.listdir(leaf_data_dir):
            s = os.path.join(leaf_data_dir, item)
            d = os.path.join(processed_dir, item)
            if os.path.isfile(s):
                shutil.copy2(s, d)
                print(f"Copied {item} to femnist_clients/")
            elif os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
                print(f"Copied {item}/ to femnist_clients/")

    print(f"Processed FEMNIST clients saved to: {processed_dir}")


if __name__ == "__main__":
    try:
        download_femnist()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
