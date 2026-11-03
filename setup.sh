
#!/usr/bin/env bash

# CMSC 672 Playing Card Detection - Project Setup
# Windows (Git Bash), macOS, and Linux

set -e

# Always run from the project directory.
cd "$(dirname "$0")"

echo "========================================"
echo "  CMSC 672 Playing Card Detection Setup"
echo "========================================"

# Find an existing virtual environment.
if [ -f ".venv/Scripts/python.exe" ]; then
    VENV_PYTHON=".venv/Scripts/python.exe"
elif [ -f ".venv/bin/python" ]; then
    VENV_PYTHON=".venv/bin/python"
else
    echo ""
    echo "Creating Python virtual environment..."

    # Find an available Python installation.
    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON="python3.12"
    elif command -v python >/dev/null 2>&1; then
        PYTHON="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON="python3"
    else
        echo "ERROR: Python not found. Install Python 3.12."
        exit 1
    fi

    "$PYTHON" -m venv .venv

    # Locate the virtual environment's Python executable.
    if [ -f ".venv/Scripts/python.exe" ]; then
        VENV_PYTHON=".venv/Scripts/python.exe"
    elif [ -f ".venv/bin/python" ]; then
        VENV_PYTHON=".venv/bin/python"
    else
        echo "ERROR: Could not create virtual environment."
        exit 1
    fi
fi

echo ""
echo "Using Python:"
"$VENV_PYTHON" --version

# Install all project dependencies.
echo ""
echo "Installing Python dependencies..."

"$VENV_PYTHON" -m pip install -r backend/requirements.txt

# Create the local dataset directories.
echo ""
echo "Creating dataset directories..."

mkdir -p "data/raw/Blackjack"
mkdir -p "data/raw/Baccarat"
mkdir -p "data/raw/Pai-Gow-Poker"

# Download the shared dataset.
echo ""
echo "Synchronizing Google Drive dataset..."

if [ -f "scripts/download_dataset.py" ]; then
    "$VENV_PYTHON" scripts/download_dataset.py
else
    echo "ERROR: scripts/download_dataset.py not found."
    exit 1
fi

echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "Dataset location: data/raw/"
echo ""
