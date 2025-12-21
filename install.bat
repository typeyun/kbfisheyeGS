@echo off
chcp 65001 >nul
echo ========================================
echo Installing CUDA PyTorch for 3D Gaussian Splatting
echo ========================================

cd /d %~dp0

echo.
echo Step 1: Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 3: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 4: Installing CUDA PyTorch...
echo This may take 5-10 minutes depending on your internet speed...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch
    pause
    exit /b 1
)

echo.
echo Step 5: Installing other dependencies...
pip install plyfile tqdm
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Step 6: Verifying installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Next steps:
echo 1. Compile CUDA extension:
echo    venv\Scripts\activate
echo    cd submodules\diff-gaussian-rasterization
echo    python setup.py install
echo.
echo 2. Run tests:
echo    python quick_check.py
echo.
pause
