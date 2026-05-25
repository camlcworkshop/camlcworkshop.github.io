@echo off
setlocal enabledelayedexpansion

REM ===== Configuration =====
set "ENV_NAME=chemtorch"
REM =========================

REM Get current script directory
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Resolve to the Heid folder
set "HEID_DIR=%SCRIPT_DIR%\Sessions\Heid"

if not exist "%HEID_DIR%\environment.yml" (
    set "HEID_DIR=%SCRIPT_DIR%\..\..\Sessions\Heid"
)

if not exist "%HEID_DIR%\environment.yml" (
    echo ❌ Could not find environment.yml inside Sessions\Heid
    pause
    exit /b 1
)

REM Convert to absolute path
for %%I in ("%HEID_DIR%") do set "HEID_DIR=%%~fI"

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
    echo --^> This might take a few minutes...

    call conda env create -f "%HEID_DIR%\environment.yml"
    if errorlevel 1 (
        echo ❌ Failed to create conda environment. Aborting.
        exit /b 1
    )
)

REM 2) Install Python packages inside the environment
echo --^> Installing core Python packages in "%ENV_NAME%"...
echo --^> This might take a few minutes...
call conda run -n "%ENV_NAME%" pip install ^
    torch==2.10.0 ^
    hydra-core ^
    wandb ^
    ipykernel
if errorlevel 1 exit /b 1

echo --^> Installing PyTorch Geometric dependencies...
echo --^> This might take a few minutes...
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
