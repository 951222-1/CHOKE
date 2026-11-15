@echo off
setlocal
cd /d "%~dp0"
title STEP 2 - Install Packages

echo ============================================================
echo   STEP 2 of 4 : install the packages the app needs
echo ------------------------------------------------------------
echo   Core = camera + face detection (needed).
echo   Audio = sound choking detection (big; tensorflow ~350MB).
echo   Slow network is normal here - just let it run.
echo ============================================================
echo.

set "VENV_DIR=%~dp0.venv"
set "VPY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VPY%" (
    echo [WARN] .venv not found. Run STEP 1 first ("1_create venv").
    pause
    exit /b 1
)

echo [1/2] installing CORE packages (mediapipe / opencv / numpy) ...
"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install --retries 10 --timeout 120 -r "%~dp0requirements-core.txt"
if errorlevel 1 (
    echo [ERROR] core install failed. Check your network and run STEP 2 again.
    pause
    exit /b 1
)
echo [OK] core packages installed.
echo.

echo [2/2] installing AUDIO packages (tensorflow ~350MB - be patient) ...
"%VPY%" -m pip install --retries 10 --timeout 120 -r "%~dp0requirements-audio.txt"
if errorlevel 1 (
    echo.
    echo [WARN] audio install failed/interrupted. The app STILL runs in
    echo        vision-only mode. You can re-run STEP 2 later to add sound.
) else (
    echo [OK] audio packages installed - full sound+vision mode ready.
)

echo.
echo [DONE] STEP 2 finished.
echo        NEXT: run  "3_start system"  (STEP 3)
pause
