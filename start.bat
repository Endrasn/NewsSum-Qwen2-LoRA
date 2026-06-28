@echo off
setlocal
cd /d "%~dp0"

echo [1/4] Preparing Python virtual environment...
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 (
    echo Failed to create virtual environment. Please make sure Python 3.10+ is installed.
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"

echo [2/4] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed. Check your network or install requirements manually.
  pause
  exit /b 1
)

echo [3/4] Opening browser...
start "" "http://127.0.0.1:7860"

echo [4/4] Starting local server...
python app.py
pause
