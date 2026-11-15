@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title L4 Sensitivity Test - TEST ONLY

set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo [ERROR] Test-copy Python environment is missing.
    echo Run the setup steps in this test folder first.
    pause
    exit /b 1
)

"%VPY%" -c "import cv2, mediapipe, numpy, requests" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Required vision packages are not ready.
    pause
    exit /b 1
)

REM Prefer the existing model assets at repository root; fall back to local assets.
set "MODEL_ROOT=%~dp0..\.."
if not exist "%MODEL_ROOT%\face_landmarker.task" set "MODEL_ROOT=%~dp0"
set "ANTICHOKE_FACE_MODEL=%MODEL_ROOT%\face_landmarker.task"
set "ANTICHOKE_HAND_MODEL=%MODEL_ROOT%\hand_landmarker.task"
REM Suppress external LINE notifications in this test copy.
set "LINE_TOKEN="
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "LOG_FILE=%~dp0sensitivity-test.log"
if exist "%LOG_FILE%" del "%LOG_FILE%"
set "TARGET="
for %%F in ("%~dp0*4_2.py") do set "TARGET=%%~fF"
if not defined TARGET (
    echo [ERROR] Main program not found.
    pause
    exit /b 1
)

"%VPY%" -u "%TARGET%" >"%LOG_FILE%" 2>&1
set "APP_EXIT=%ERRORLEVEL%"
echo.
echo [TEST EXIT CODE] %APP_EXIT%
echo -------- sensitivity-test.log --------
type "%LOG_FILE%"
echo ---------------------------------------
pause
