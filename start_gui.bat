@echo off
setlocal

cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
set "DEPS_STAMP=.venv\.deps_ready"

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

if "%AI_AUTO_PS_FORCE_PIP%"=="1" (
    echo [INFO] Force dependency refresh enabled.
    goto install_deps
)

if exist "%DEPS_STAMP%" (
    echo [INFO] Fast startup: dependency check skipped.
    goto deps_ready
)

echo [INFO] First-time dependency check ...
"%VENV_PY%" -c "import importlib.util, sys; mods=('numpy','PIL','cv2','gradio','transformers'); sys.exit(0 if all(importlib.util.find_spec(m) is not None for m in mods) else 1)" >nul 2>nul
if errorlevel 1 (
    goto install_deps
)

echo [INFO] Dependencies already ready.
> "%DEPS_STAMP%" echo ready
goto deps_ready

:install_deps
echo [INFO] Installing dependencies from requirements.txt ...
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)
> "%DEPS_STAMP%" echo ready

:deps_ready

if "%AI_AUTO_PS_VERIFY_DEPS%"=="1" (
    echo [INFO] Verifying dependency specs ...
    "%VENV_PY%" -c "import importlib.util, sys; mods=('numpy','PIL','cv2','gradio','transformers'); sys.exit(0 if all(importlib.util.find_spec(m) is not None for m in mods) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo [INFO] Required package specs not found. Reinstalling dependencies ...
        goto install_deps
    )
) else (
    echo [INFO] Fast startup: skip per-run dependency verification.
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
