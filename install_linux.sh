#!/usr/bin/env bash
set -e
echo "========================================================================"
echo " STEELGUARD AI — CPU-Only Portable Environment Setup (Linux / Raspberry Pi)"
echo "========================================================================"

if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Please install Python 3.10-3.12."
    exit 1
fi

echo "[1/3] Creating virtual environment (.venv)..."
python3 -m venv .venv

echo "[2/3] Activating venv and upgrading pip..."
source .venv/bin/activate
pip install --upgrade pip

echo "[3/3] Installing PyTorch (CPU) and core inference dependencies..."
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements-cpu.txt

echo ""
echo "========================================================================"
echo " Installation Complete! Run: ./run_demo_cpu.sh"
echo "========================================================================"
