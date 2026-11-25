@echo off
setlocal
echo ========================================================================
echo  STEELGUARD AI — Launching CPU-Only Industrial Inspection Dashboard
echo  Binding: http://127.0.0.1:8080/ (Loopback-only)
echo ========================================================================

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Please run install_windows.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
set STEELGUARD_DEVICE=cpu
set CUDA_VISIBLE_DEVICES=

echo Starting server on http://127.0.0.1:8080/ ...
python steelguard_app\run_dashboard.py --cpu --demo --port 8080 --host 127.0.0.1 %*
pause
