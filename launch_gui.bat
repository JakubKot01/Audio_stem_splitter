@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo The project virtual environment was not found.
    echo Create .venv and install requirements.txt first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo The .venv environment points to a Python installation that is no longer available.
    echo Reinstall Python 3.10 or 3.11, delete and recreate .venv, then install requirements.txt.
    pause
    exit /b 1
)

start "" ".venv\Scripts\pythonw.exe" -m app.gui
