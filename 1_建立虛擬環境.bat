@echo off
setlocal
cd /d "%~dp0"
title STEP 1 - Create Virtual Environment

echo ============================================================
echo   STEP 1 of 4 : create the Python virtual environment (.venv)
echo ------------------------------------------------------------
echo   Makes an isolated Python just for this app, so installing
echo   packages won't touch your system Python. Do this ONCE.
echo ============================================================
echo.

set "VENV_DIR=%~dp0.venv"
set "VPY=%VENV_DIR%\Scripts\python.exe"

if exist "%VPY%" (
    echo [OK] .venv already exists - nothing to do.
    echo      NEXT: run  "2_install packages"  (STEP 2)
    pause
    exit /b 0
)

REM prefer Python 3.12 / 3.11 / 3.10 (tensorflow/mediapipe have no 3.13/3.14 wheels)
set "BASEPY="
for %%V in (3.12 3.11 3.10) do (
    if not defined BASEPY (
        py -%%V -c "import sys" >nul 2>nul && set "BASEPY=py -%%V"
    )
)
if not defined BASEPY ( where py >nul 2>nul && set "BASEPY=py -3" )
if not defined BASEPY ( where python >nul 2>nul && set "BASEPY=python" )
if not defined BASEPY (
    echo [ERROR] Python not found. Install from https://www.python.org
    echo         and tick "Add Python to PATH", then run STEP 1 again.
    pause
    exit /b 1
)

echo [..] creating .venv using: %BASEPY%
%BASEPY% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo [ERROR] failed to create the virtual environment.
    pause
    exit /b 1
)
"%VPY%" -m pip install --upgrade pip

echo.
echo [DONE] STEP 1 finished.
echo        NEXT: run  "2_install packages"  (STEP 2)
pause
