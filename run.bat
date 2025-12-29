@echo off
echo ============================================
echo Python Installation and Setup
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Python is already installed!
    python --version
    goto :install_packages
)

echo Python not found. Installing from Microsoft Store...
echo.
echo PLEASE: Click 'Get' or 'Install' when the Microsoft Store opens
echo.
start ms-windows-store://pdp/?ProductId=9NRWMJP3717K
timeout /t 5
echo.
echo Waiting for Python installation...
echo After installation completes, press any key to continue...
pause

:install_packages
echo.
echo ============================================
echo Installing Required Packages
echo ============================================
echo.

python -m pip install --upgrade pip
python -m pip install streamlit yfinance pandas numpy matplotlib statsmodels scikit-learn

echo.
echo ============================================
echo Starting Stock Prediction App
echo ============================================
echo.
echo The app will open in your browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

python -m streamlit run app.py

pause
