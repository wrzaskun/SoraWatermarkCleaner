#!/bin/bash

# I have no proof—is this enough!

echo "=== SoraWatermarkCleaner — Setup for WSL ==="
echo "I have no proof—is this enough?"

# --- CONFIG ---
PYTHON_BIN=python3.10
VENV_DIR=env

echo "[1/10] Removing old environment (if exists)..."
rm -rf "$VENV_DIR"

echo "[2/10] Creating virtual environment..."
$PYTHON_BIN -m venv "$VENV_DIR"

echo "[3/10] Activating environment..."
source "$VENV_DIR/bin/activate"

echo "[4/10] Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "[5/10] Installing NumPy < 2.0..."
pip install "numpy<2"

echo "[6/10] Installing PyTorch 2.1.2 + CUDA 12.1..."
pip install torch==2.1.2+cu121 torchvision==0.16.1+cu121 --index-url https://download.pytorch.org/whl/cu121

echo "[7/10] Installing MMCV (compatible)..."
pip install mmcv==1.7.1

echo "[8/10] Installing diffusers, transformers, accelerate..."
pip install diffusers==0.25.0 transformers==4.36 accelerate==0.26

echo "[9/10] Installing remaining project requirements..."
pip install -r requirements.txt --no-deps

echo "[10/10] Setup complete!"
echo
echo "Start server with:"
echo "    source env/bin/activate"
echo "    ./run.sh"

echo "=== DONE ==="
