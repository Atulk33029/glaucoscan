@echo off
echo ============================================
echo   GlaucoScan v2 - Starting Server
echo ============================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if weights exist
if not exist "models\generator.pth" (
    echo WARNING: models\generator.pth not found!
    echo App will run but enhancement will use untrained model.
    echo Download weights from Google Drive and place in models\ folder.
    echo.
)

echo Starting Flask server...
echo.
echo ============================================
echo   Open this URL in your browser:
echo   http://localhost:5000
echo ============================================
echo.
echo Press Ctrl+C to stop the server.
echo.

python app.py
pause
