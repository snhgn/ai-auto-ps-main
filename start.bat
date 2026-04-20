@echo off
REM AI Auto PS - 统一启动脚本 (支持正常模式和调试模式)
setlocal enabledelayedexpansion

REM 检查命令行参数
set "DEBUG_MODE=0"
if "%1"=="debug" set "DEBUG_MODE=1"
if "%1"=="--debug" set "DEBUG_MODE=1"
if "%1"=="/debug" set "DEBUG_MODE=1"

if "%DEBUG_MODE%"=="1" (
    title AI Auto PS - Debug Console
    echo ========================================
    echo AI Auto PS - DEBUG MODE
    echo ========================================
    echo.
) else (
    title AI Auto PS
    echo [INFO] AI Auto PS 启动中...
    echo.
)

REM 切换到脚本目录
cd /d "%~dp0"
if errorlevel 1 (
    echo [ERROR] 无法切换到脚本目录
    pause
    exit /b 1
)

set "SCRIPT_DIR=%cd%"

REM 检查 requirements.txt
if not exist "requirements.txt" (
    echo [ERROR] 找不到 requirements.txt
    pause
    exit /b 1
)

set "VENV_PY=.venv\Scripts\python.exe"

REM 检查虚拟环境是否存在
if not exist "%VENV_PY%" (
    echo [INFO] 虚拟环境不存在，正在创建...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 (
        echo [ERROR] 创建虚拟环境失败
        echo 请确保已安装 Python 3.10+
        pause
        exit /b 1
    )
    echo [SUCCESS] 虚拟环境创建完毕
)

REM 检查依赖是否已安装
if "%AI_AUTO_PS_FORCE_PIP%"=="1" (
    echo [INFO] 强制刷新依赖...
    goto install_deps
)

if not exist ".venv\.deps_ready" (
    goto install_deps
) else (
    goto run_app
)

:install_deps
echo [INFO] 检查并安装依赖...
REM 默认跳过 pip/setuptools/wheel 升级以减少冷启动耗时；仅在需要时设置 AI_AUTO_PS_BOOTSTRAP_PIP=1
if "%AI_AUTO_PS_BOOTSTRAP_PIP%"=="1" (
    echo [INFO] 升级 pip/setuptools/wheel...
    "%VENV_PY%" -m pip install --upgrade pip setuptools wheel --disable-pip-version-check -q
    if errorlevel 1 (
        echo [ERROR] pip升级失败
        pause
        exit /b 1
    )
)

"%VENV_PY%" -m pip install -r requirements.txt --disable-pip-version-check --prefer-binary
if errorlevel 1 (
    echo [ERROR] 依赖安装失败
    pause
    exit /b 1
)

echo [SUCCESS] 依赖安装完毕
echo. > ".venv\.deps_ready"

:run_app
echo.
echo [INFO] 启动应用...
echo.

REM 设置环境变量
set "PYTHONIOENCODING=utf-8"
if "%AI_AUTO_PS_HOST%"=="" set "AI_AUTO_PS_HOST=127.0.0.1"
if "%AI_AUTO_PS_PORT%"=="" set "AI_AUTO_PS_PORT=7860"
if "%AI_AUTO_PS_OPEN_BROWSER%"=="" set "AI_AUTO_PS_OPEN_BROWSER=1"

set "DISPLAY_HOST=%AI_AUTO_PS_HOST%"
if "%DISPLAY_HOST%"=="0.0.0.0" set "DISPLAY_HOST=127.0.0.1"

if "%DEBUG_MODE%"=="1" (
    echo [DEBUG] 主机: %AI_AUTO_PS_HOST%
    echo [DEBUG] 端口: %AI_AUTO_PS_PORT%
    echo [DEBUG] 自动打开浏览器: %AI_AUTO_PS_OPEN_BROWSER%
    echo [DEBUG] URL: http://%DISPLAY_HOST%:%AI_AUTO_PS_PORT%
    echo.
)

REM 运行应用
"%VENV_PY%" -u ai_auto_ps.py

if %ERRORLEVEL% equ 0 (
    if "%DEBUG_MODE%"=="1" (
        echo [DEBUG] 应用正常退出
    )
) else (
    echo [ERROR] 应用异常退出 (code: %ERRORLEVEL%)
    if "%DEBUG_MODE%"=="1" (
        pause
    )
    exit /b %ERRORLEVEL%
)

endlocal
