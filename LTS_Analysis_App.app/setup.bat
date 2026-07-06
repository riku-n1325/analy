@echo off
setlocal
set "APP_DIR=%~dp0"
set "VENV_DIR=%APP_DIR%.venv"

where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m venv "%VENV_DIR%"
) else (
    python -m venv "%VENV_DIR%"
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Python 3 が見つからないため、セットアップできませんでした。
    echo https://www.python.org/downloads/ から Python 3 をインストールしてください。
    echo インストール時は Add python.exe to PATH にチェックを入れてください。
    pause
    exit /b 1
)

"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV_DIR%\Scripts\python.exe" -m pip install -r "%APP_DIR%requirements.txt"

echo.
echo セットアップが完了しました。次回からは 起動.bat をダブルクリックしてください。
pause
