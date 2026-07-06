@echo off
setlocal
set "APP_DIR=%~dp0"
set "PYTHON=%APP_DIR%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    set "PYTHON=python"
)

"%PYTHON%" -c "import numpy; import PIL" >nul 2>nul
if errorlevel 1 (
    echo 必要なPythonライブラリが見つかりません。
    echo 初回のみ setup.bat を実行してください。
    echo.
    pause
    exit /b 1
)

start "" "%PYTHON%" "%APP_DIR%lts_analysis_app.py"
