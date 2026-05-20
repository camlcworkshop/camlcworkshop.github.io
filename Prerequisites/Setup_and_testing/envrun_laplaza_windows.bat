@echo off
setlocal enabledelayedexpansion

REM ===== Configuration =====
set "ENV_NAME=gm-tutorial"
REM =========================

REM Directory of this script (Prerequisites\Setup_and_testing\)
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Laplaza directory: go up two levels, then Sessions\Laplaza
set "LAPLAZA_DIR=%SCRIPT_DIR%\..\..\Sessions\Laplaza"

echo Script directory: %SCRIPT_DIR%
echo Laplaza directory: %LAPLAZA_DIR%

REM 1) Check if environment exists
set "ENV_EXISTS="
for /f "skip=2 tokens=1" %%E in ('conda env list') do (
    if "%%E"=="%ENV_NAME%" set "ENV_EXISTS=1"
)

if defined ENV_EXISTS (
    echo Conda environment "%ENV_NAME%" already exists. Skipping creation.
) else (
    echo Creating conda environment "%ENV_NAME%" from environment.yaml...
    pushd "%LAPLAZA_DIR%"
    conda env create -f environment.yaml
    if errorlevel 1 (
        echo Failed to create environment. Aborting.
        popd
        exit /b 1
    )
    popd
)

REM 2) Execute the notebook using the environment
echo Running gm.ipynb using conda environment "%ENV_NAME%"...

conda run -n "%ENV_NAME%" python -m nbconvert ^
    --to notebook ^
    --execute "%LAPLAZA_DIR%\gm.ipynb" ^
    --output "%LAPLAZA_DIR%\gm.executed.ipynb"

if errorlevel 1 (
    echo Notebook execution failed.
    exit /b 1
)

echo Done. Executed notebook saved as gm.executed.ipynb in:
echo   %LAPLAZA_DIR%

endlocal

