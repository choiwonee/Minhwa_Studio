@echo off
setlocal enabledelayedexpansion

echo =================================================================================================================================================
echo  [ Minhwa Studio ] Dependency Updater (Safe Mode via GOTO)
echo  Updates libraries while protecting PyTorch/CUDA
echo =================================================================================================================================================

REM ==================================================================================================================================================================
REM [CONFIG] Version & URL Management Variables
REM ==================================================================================================================================================================
set DIFFUSERS_URL=git+https://github.com/huggingface/diffusers.git
set SAM2_URL=git+https://github.com/facebookresearch/segment-anything-2.git

REM Nunchaku Wheels
set NUNCHAKU_T26=https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.1/nunchaku-1.0.1+torch2.6-cp311-cp311-win_amd64.whl
set NUNCHAKU_XPU=https://github.com/nunchaku-tech/nunchaku/releases/download/v1.1.0dev20251111/nunchaku-1.1.0.dev20251111+torch2.10-cp311-cp311-win_amd64.whl

REM ==================================================================================================================================================================
REM [STEP 1] Virtual Environment Check
REM ==================================================================================================================================================================
:STEP_VENV
if exist "venv" goto ACTIVATE_VENV
goto ERROR_VENV

:ERROR_VENV
echo [ERROR] Virtual environment 'venv' not found!
echo Please run 'install_auto.bat' first.
pause
exit /b

:ACTIVATE_VENV
echo ### [1/5] Activating virtual environment... ###
call venv\Scripts\activate
goto STEP_HARDWARE

REM ==================================================================================================================================================================
REM [STEP 2] Hardware Detection
REM ==================================================================================================================================================================
:STEP_HARDWARE
echo ### [2/5] Detecting Hardware for Component Selection... ###
set "GPU_TYPE=CPU"

REM Check NVIDIA
powershell -command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name" | findstr /i "NVIDIA" > nul
if %errorlevel% equ 0 goto SET_NVIDIA

REM Check INTEL
powershell -command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name" | findstr /i "Intel" > nul
if %errorlevel% equ 0 goto SET_INTEL

REM Default CPU
goto SET_CPU

:SET_NVIDIA
set "GPU_TYPE=NVIDIA"
set "TARGET_NUNCHAKU=%NUNCHAKU_T26%"
echo  - Mode: NVIDIA
goto STEP_UPDATE_COMMON

:SET_INTEL
set "GPU_TYPE=INTEL"
set "TARGET_NUNCHAKU=%NUNCHAKU_XPU%"
echo  - Mode: INTEL
goto STEP_UPDATE_COMMON

:SET_CPU
set "GPU_TYPE=CPU"
set "TARGET_NUNCHAKU="
echo  - Mode: CPU
goto STEP_UPDATE_COMMON

REM ==================================================================================================================================================================
REM [STEP 3] Common Libraries Update
REM ==================================================================================================================================================================
:STEP_UPDATE_COMMON
echo ### [3/5] Updating Common Libraries... ###

if exist requirements.txt goto UPDATE_REQ
echo [WARNING] requirements.txt not found.
goto STEP_UPDATE_SPECIAL

:UPDATE_REQ
REM Preserve PyTorch/CUDA by excluding them from upgrade (pip handles dependencies)
pip install -r requirements.txt --upgrade
goto STEP_UPDATE_SPECIAL

REM ==================================================================================================================================================================
REM [STEP 4] Special Libraries Update
REM ==================================================================================================================================================================
:STEP_UPDATE_SPECIAL
echo ### [4/5] Updating Special Components... ###

REM --- 4-1. Diffusers ---
echo - Updating Diffusers (Latest Dev Build)...
pip install --upgrade %DIFFUSERS_URL%

REM --- 4-2. SAM2 ---
echo - Checking SAM2...
python -c "import sam2" > nul 2>&1
if %errorlevel% neq 0 goto INSTALL_SAM2
goto SKIP_SAM2

:INSTALL_SAM2
echo Installing SAM2...
pip install %SAM2_URL%
goto CHECK_NUNCHAKU

:SKIP_SAM2
echo SAM2 is present. (Skipping reinstall to save time)
goto CHECK_NUNCHAKU

REM --- 4-3. Nunchaku ---
:CHECK_NUNCHAKU
if "%TARGET_NUNCHAKU%"=="" goto SKIP_NUNCHAKU
goto UPDATE_NUNCHAKU

:UPDATE_NUNCHAKU
echo - Updating Nunchaku (%GPU_TYPE%)...
pip install "%TARGET_NUNCHAKU%"
goto FINISH

:SKIP_NUNCHAKU
REM Nunchaku not needed for CPU or unspecified hardware
goto FINISH

REM ==================================================================================================================================================================
REM [FINISH]
REM ==================================================================================================================================================================
:FINISH
echo =================================================================================================================================================
echo  ### [5/5] Update Complete! ###
echo      PyTorch core was preserved.
echo =================================================================================================================================================
pause