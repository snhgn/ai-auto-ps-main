@echo off
setlocal

cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [INFO] Python virtual environment not found. Creating .venv ...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo Please install Python 3.10+ and ensure py or python is available in PATH.
        pause
        exit /b 1
    )
)

echo [INFO] Checking required packages ...
"%VENV_PY%" -c "import numpy, PIL, cv2, gradio, transformers" >nul 2>nul
if errorlevel 1 (
    echo [INFO] Installing dependencies from requirements.txt ...
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
)

if "%AI_AUTO_PS_HOST%"=="" set "AI_AUTO_PS_HOST=127.0.0.1"
if "%AI_AUTO_PS_PORT%"=="" set "AI_AUTO_PS_PORT=7860"
if "%AI_AUTO_PS_OPEN_BROWSER%"=="" set "AI_AUTO_PS_OPEN_BROWSER=1"

set "DISPLAY_HOST=%AI_AUTO_PS_HOST%"
if "%DISPLAY_HOST%"=="0.0.0.0" set "DISPLAY_HOST=127.0.0.1"

echo [INFO] Starting AI Auto PS UI ...
echo [INFO] Browser will open after server is ready.
echo [INFO] Expected URL: http://%DISPLAY_HOST%:%AI_AUTO_PS_PORT%

"%VENV_PY%" ai_auto_ps.py

if errorlevel 1 (
    echo [ERROR] Program exited with an error.
    pause
    exit /b 1
)

echo [INFO] Program exited normally.
pause
