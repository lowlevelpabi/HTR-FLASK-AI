@echo off
echo ===========================================
echo   HTR-FLASK-AI Setup Script
echo ===========================================

REM Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

REM Create Virtual Environment if it doesn't exist
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
) else (
    echo [INFO] Virtual environment already exists.
)

REM Activate Virtual Environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install Dependencies
if exist "requirements.txt" (
    echo [INFO] Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo [WARNING] requirements.txt not found! Skipping dependency installation.
)

echo ===========================================
echo   Setup Complete!
echo ===========================================
pause
