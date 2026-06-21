@echo off
echo ============================================
echo   GlaucoScan v2 - Windows Setup
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install
    pause
    exit /b 1
)

echo [1/4] Python found. Creating virtual environment...
python -m venv venv

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing dependencies (this takes 3-5 minutes)...
pip install flask flask-cors opencv-python-headless Pillow numpy reportlab werkzeug gunicorn

echo [4/4] Installing PyTorch (CPU version)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo ============================================
echo   Setup Complete!
echo   Now place your trained weights:
echo   - models/generator.pth
echo   - models/classifier.pth
echo ============================================
echo.
pause
