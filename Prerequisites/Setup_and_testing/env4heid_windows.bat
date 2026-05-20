@echo off
setlocal enabledelayedexpansion

# ===== Configuration =====
set "ENV_NAME=chemtorch"
# =========================

REM Get current script directory
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Resolve to the Heid folder
set "HEID_DIR=%SCRIPT_DIR%\..\Sessions\Heid"
if not exist "%HEID_DIR%" set "HEID_DIR=%SCRIPT_DIR%\..\..\Sessions\Heid"

REM Get absolute path for HEID_DIR
pushd "%HEID_DIR%"
set "HEID_DIR=%CD%"
popd

set "CHEMTORCH_DIR=%HEID_DIR%\chemtorch"

echo ====================================================
echo Script directory: %SCRIPT_DIR%
echo Heid directory:   %HEID_DIR%
echo ====================================================

REM 0) Fix the missing/broken chemtorch folder
set "NEED_CLONE=0"
if not exist "%CHEMTORCH_DIR%" set NEED_CLONE=1

if exist "%CHEMTORCH_DIR%" (
    dir /b /a "%CHEMTORCH_DIR%" | findstr . >nul
    if errorlevel 1 set NEED_CLONE=1
    if not exist "%CHEMTORCH_DIR%\setup.py" if not exist "%CHEMTORCH_DIR%\pyproject.toml" set NEED_CLONE=1
)

if "%NEED_CLONE%"=="1" (
    echo ⚠️ 'chemtorch' folder is empty, missing, or a broken link.
    echo --^> Cloning a fresh copy of chemtorch directly into %CHEMTORCH_DIR%...
    
    if exist "%CHEMTORCH_DIR%" rmdir /s /q "%CHEMTORCH_DIR%"
    
    call git clone https://github.com/heid-lab/ChemTorch.git "%CHEMTORCH_DIR%"
    if errorlevel 1 (
        echo ❌ Git clone failed. Aborting.
        exit /b 1
    )
)

REM 1) Create the conda environment if it does not exist
set "ENV_EXISTS="
for /f "skip=2 tokens=1" %%E in ('call conda env list') do (
    if "%%E"=="%ENV_NAME%" set "ENV_EXISTS=1"
)

if defined ENV_EXISTS (
    echo --^> Conda environment "%ENV_NAME%" already exists. Skipping creation.
) else (
    echo --^> Creating conda environment "%ENV_NAME%" from environment.yml...
    pushd "%HEID_DIR%"
    call conda env create -f environment.yml
    if errorlevel 1 (
        echo ❌ Failed to create conda environment. Aborting.
        popd
        exit /b 1
    )
    popd
)

REM 2) Install Python packages inside the environment
echo --^> Installing core Python packages in "%ENV_NAME%"...
call conda run -n "%ENV_NAME%" pip install ^
    rdkit ^
    numpy==1.26.4 ^
    scikit-learn ^
    pandas ^
    torch==2.10.0 ^
    hydra-core ^
    wandb ^
    ipykernel ^
    jupyter ^
    matplotlib
if errorlevel 1 exit /b 1

echo --^> Installing PyTorch Geometric dependencies...
call conda run -n "%ENV_NAME%" pip install ^
    torch_scatter ^
    torch_sparse ^
    torch_cluster ^
    torch_spline_conv ^
    -f https://data.pyg.org/whl/torch-2.10.0+cpu.html
if errorlevel 1 exit /b 1

REM 3) Local editable chemtorch install
echo --^> Installing local chemtorch in editable mode...
pushd "%HEID_DIR%"
call conda run -n "%ENV_NAME%" pip install -e chemtorch
popd
if errorlevel 1 exit /b 1

REM 4) Install chemprop
echo --^> Installing chemprop...
call conda run -n "%ENV_NAME%" pip install chemprop
if errorlevel 1 exit /b 1

echo ====================================================
echo Success! Environment "%ENV_NAME%" is ready.
echo ====================================================

endlocal
pause
