#!/usr/bin/env bash

# Stop the script if an error occurs
set -e

# 1. Get the absolute path of the directory where this script resides (MP/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# # Define project root (one level up from MP/)
ROOT_DIR="$(cd "$SCRIPT_DIR/" && pwd)"

echo "Initializing environment in project root: $ROOT_DIR"

# 2. Create the auxiliary directories configs/ and test_data/ in root
mkdir -p "$ROOT_DIR/configs"
mkdir -p "$ROOT_DIR/test_data"
echo "Directories 'configs/' and 'test_data/' created successfully."

# 3. Create the .venv virtual environment in root if it does not exist
if [ ! -d "$ROOT_DIR/.venv" ]; then
    echo "Creating virtual environment (.venv)..."
    python3 -m venv "$ROOT_DIR/.venv"
else
    echo "The virtual environment (.venv) already exists."
fi

# 4. Activate virtual environment and install the package from project root
echo "Installing the MP package in editable mode..."
source "$ROOT_DIR/.venv/bin/activate"

# Upgrade pip and install the package using the root directory where pyproject.toml lives
pip install --upgrade pip
pip install -e "$ROOT_DIR"

echo "Environment initialized successfully!"
echo "To activate the environment in your terminal, run: source .venv/bin/activate"