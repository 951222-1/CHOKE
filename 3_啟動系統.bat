@echo off
setlocal
cd /d "%~dp0"
title STEP 3 - Start the Anti-Choking Monitor

echo ============================================================
echo   STEP 3 of 4 : start the program (opens your camera)
echo ------------------------------------------------------------
echo   A window opens showing the camera. First run downloads the
echo   face model + the sound model, so give it a few seconds.
echo.
echo   STEP 4 = just USE it (see below), and press  q  to quit.
echo ============================================================
echo.

set "VENV_DIR=%~dp0.venv"
set "VPY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VPY%" (
    echo [ERROR] .venv not found. Do STEP 1 to create venv then STEP 2 to install.
    pause
    exit /b 1
)

"%VPY%" -c "import cv2, mediapipe, numpy, requests" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] packages missing. Run STEP 2 to install them first.
    pause
    exit /b 1
)

REM find the main .py by its ASCII ending (its real name has Chinese in it)
set "TARGET="
for %%F in ("%~dp0*4_2.py") do set "TARGET=%%F"
if not defined TARGET (
    echo [ERROR] main program "*4_2.py" not found in this folder.
    pause
    exit /b 1
)

echo [RUN] starting...
echo ------------------------------------------------------------
echo   STEP 4 - how to use it:
echo    - look at the camera; green dots should appear on your face
echo    - watch the "Audio Choke" bar move when you cough near the mic
echo    - put a hand near your throat for ~1s to test the choke gesture
echo    - press  q  in the camera window to close
echo ============================================================
set PYTHONIOENCODING=utf-8
set ANTICHOKE_FACE_MODEL=C:\Users\ph090\face_landmarker.task
set ANTICHOKE_HAND_MODEL=C:\Users\ph090\hand_landmarker.task
"%VPY%" "%TARGET%"

echo.
echo [DONE] program closed.
pause
