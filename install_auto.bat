@echo off
setlocal enabledelayedexpansion

echo =================================================================================================================================================
echo [ Minhwa Studio ] Auto Installer (Safe Mode via GOTO)
echo =================================================================================================================================================

REM ==================================================================================================================================================================
REM [CONFIG] Version & URL Management
REM ==================================================================================================================================================================
set DIFFUSERS_URL=git+https://github.com/huggingface/diffusers.git
set SAM2_URL=git+https://github.com/facebookresearch/segment-anything-2.git

REM Nunchaku Wheels
set NUNCHAKU_T26=https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.1/nunchaku-1.0.1+torch2.6-cp311-cp311-win_amd64.whl
set NUNCHAKU_XPU=https://github.com/nunchaku-tech/nunchaku/releases/download/v1.1.0dev20251111/nunchaku-1.1.0.dev20251111+torch2.10-cp311-cp311-win_amd64.whl

REM ==================================================================================================================================================================
REM [STEP 0] Check Git
REM ==================================================================================================================================================================
git --version > nul 2>&1
if %errorlevel% neq 0 goto ERROR_GIT
goto STEP_VENV

:ERROR_GIT
echo [ERROR] Git is not installed!
pause
exit /b

REM ==================================================================================================================================================================
REM [STEP 1] Setup Virtual Environment
REM ==================================================================================================================================================================
:STEP_VENV
echo.
echo ### [1/6] Setting up Virtual Environment ###
set "PYCMD="

REM --- 1) Try Python 3.11 via the "py" launcher (works even if other versions are also installed) ---
py -3.11 --version > nul 2>&1
if %errorlevel% equ 0 set "PYCMD=py -3.11"
if not "%PYCMD%"=="" goto FOUND_PYTHON

REM --- 2) Fallback: check if the default "python" command is already 3.11.x ---
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
echo %PYVER% | findstr /r "^3\.11\." > nul
if %errorlevel% equ 0 set "PYCMD=python"
if not "%PYCMD%"=="" goto FOUND_PYTHON

goto ERROR_PYTHON_VERSION

:ERROR_PYTHON_VERSION
echo.
echo [ERROR] Python 3.11.x was not found on this system.
echo         This project requires Python 3.11.x (e.g. 3.11.9).
echo.
echo         1^) Install Python 3.11.x from https://www.python.org/downloads/
echo            ^(check "Add python.exe to PATH" during install^)
echo         2^) If another Python version is already installed, you do NOT
echo            need to remove it - just install 3.11.x alongside it.
echo         3^) Run "py -0" to confirm 3.11 appears in the list, then
echo            re-run this installer.
echo.
pause
exit /b

:FOUND_PYTHON
echo - Using Python: %PYCMD%
if exist "venv" rmdir /s /q "venv"
%PYCMD% -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
goto STEP_HARDWARE

REM ==================================================================================================================================================================
REM [STEP 2] Hardware Detection
REM ==================================================================================================================================================================
:STEP_HARDWARE
echo.
echo ### [2/6] Detecting Hardware... ###
set "GPU_TYPE=CPU"

REM Check NVIDIA
powershell -command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name" | findstr /i "NVIDIA" > nul
if %errorlevel% equ 0 goto DETECTED_NVIDIA

REM Check INTEL (Only if NVIDIA not found)
powershell -command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name" | findstr /i "Intel" > nul
if %errorlevel% equ 0 goto DETECTED_INTEL

REM Default to CPU
goto DETECTED_CPU

:DETECTED_NVIDIA
set "GPU_TYPE=NVIDIA"
echo - Detected Mode: NVIDIA
goto STEP_INSTALL_TORCH

:DETECTED_INTEL
set "GPU_TYPE=INTEL"
echo - Detected Mode: INTEL
goto STEP_INSTALL_TORCH

:DETECTED_CPU
set "GPU_TYPE=CPU"
echo - Detected Mode: CPU (Standard)
goto STEP_INSTALL_TORCH

REM ==================================================================================================================================================================
REM [STEP 3] Install PyTorch (Branching Logic)
REM ==================================================================================================================================================================
:STEP_INSTALL_TORCH
echo.
echo ### [3/6] Installing PyTorch Core ###

if "%GPU_TYPE%"=="NVIDIA" goto INSTALL_TORCH_NVIDIA
if "%GPU_TYPE%"=="INTEL"  goto INSTALL_TORCH_INTEL
goto INSTALL_TORCH_CPU

:INSTALL_TORCH_NVIDIA
echo [NVIDIA] Installing Torch 2.6.0 (CUDA 12.4)
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
set "TARGET_NUNCHAKU=%NUNCHAKU_T26%"
goto STEP_INSTALL_REQ

:INSTALL_TORCH_INTEL
echo [INTEL] Installing Torch (XPU Nightly)
pip uninstall -y torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
pip install intel-extension-for-pytorch
set "TARGET_NUNCHAKU=%NUNCHAKU_XPU%"
goto STEP_INSTALL_REQ

:INSTALL_TORCH_CPU
echo [CPU] Installing Standard Torch
pip install torch torchvision torchaudio
set "TARGET_NUNCHAKU="
goto STEP_INSTALL_REQ

REM ==================================================================================================================================================================
REM [STEP 4] Install Requirements
REM ==================================================================================================================================================================
:STEP_INSTALL_REQ
echo.
echo ### [4/6] Installing Base Requirements ###
if exist requirements.txt goto RUN_REQ_INSTALL
echo [WARNING] requirements.txt not found. Skipping.
goto STEP_INSTALL_EXTRA

:RUN_REQ_INSTALL
pip install -r requirements.txt
goto STEP_INSTALL_EXTRA

REM ==================================================================================================================================================================
REM [STEP 5] Install Special Packages
REM ==================================================================================================================================================================
:STEP_INSTALL_EXTRA
echo.
echo ### [5/6] Installing Special Packages ###

echo - Installing Diffusers (Dev)
pip install %DIFFUSERS_URL%

echo - Installing SAM2
pip install %SAM2_URL%

REM Check if NUNCHAKU is needed
if "%TARGET_NUNCHAKU%"=="" goto SKIP_NUNCHAKU
goto INSTALL_NUNCHAKU

:INSTALL_NUNCHAKU
echo - Installing Nunchaku (Optimization)
pip install ninja
pip install "%TARGET_NUNCHAKU%"
goto FINISH

:SKIP_NUNCHAKU
echo - Skipping Nunchaku (Not supported on CPU)
goto FINISH

REM ==================================================================================================================================================================
REM [STEP 6] Finish
REM ==================================================================================================================================================================
:FINISH
echo.
echo =================================================================================================================================================
echo ### [6/6] Installation Complete! ###
echo =================================================================================================================================================
pause