@echo off
setlocal
cd /d "%~dp0"
title STEP 4 - Anti-Choking Module Test Center

echo ============================================================
echo   Anti-Choking Real-time Monitor - Module Test Center
echo ------------------------------------------------------------
echo   Launching interactive Python menu...
echo ============================================================
echo.

set "VENV_DIR=%~dp0.venv"
set "VPY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VPY%" (
    set "VPY=python"
)

set PYTHONIOENCODING=utf-8
"%VPY%" "%~dp0test_menu.py"

pause
