#!/usr/bin/env bash
set -e
echo "========================================================================"
echo " STEELGUARD AI — Launching CPU-Only Industrial Inspection Dashboard"
echo " Binding: http://127.0.0.1:8080/ (Loopback-only)"
echo "========================================================================"

if [ ! -f ".venv/bin/activate" ]; then
    echo "[ERROR] .venv not found. Run ./install_linux.sh first."
    exit 1
fi

source .venv/bin/activate
export STEELGUARD_DEVICE=cpu
export CUDA_VISIBLE_DEVICES=""

python3 steelguard_app/run_dashboard.py --cpu --demo --port 8080 --host 127.0.0.1 "$@"
