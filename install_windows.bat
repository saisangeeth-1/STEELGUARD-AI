@echo off
setlocal
echo ========================================================================
echo  STEELGUARD AI — CPU-Only Portable Environment Setup (Windows)
echo ========================================================================
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.10-3.12.
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment (.venv)...
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [2/3] Installing CPU-only PyTorch and TorchVision...
call .venv\Scripts\activate.bat
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

echo [3/3] Installing SteelGuard core inference dependencies...
pip install -r requirements-cpu.txt

echo.
echo ========================================================================
echo  Installation Successful!
echo  Launch the demo using: run_demo_cpu.bat
echo ========================================================================
pause
