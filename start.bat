@echo off
setlocal enabledelayedexpansion

echo "========================================================"
echo " [ Minhwa Studio ] Launcher"
echo "========================================================"

if not exist "venv" (
    echo "[ERROR] Installation not found."
    echo "Please run 'install_auto.bat' first."
    pause
    exit /b
)

echo " - Activating environment..."
call venv\Scripts\activate

echo " - Starting Minhwa Studio..."
python main_launcher.py

if !errorlevel! neq 0 (
    echo.
    echo "[ERROR] The program crashed or was stopped (Code: !errorlevel!)"
    pause
)

endlocal