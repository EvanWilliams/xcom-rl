@echo off
setlocal enabledelayedexpansion
title XCOM RL - Visual Training

set "ROOT=%~dp0"
set "PYDIR=%ROOT%python"
set "CKPTDIR=%PYDIR%\checkpoints"
set "BUILDDIR=%ROOT%engine\openxcom\build"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars32.bat"

echo.
echo  XCOM Reinforcement Learning - Fully Visualized Training
echo  ========================================================
echo  Action space  : Discrete(23)  mouse clicks + keyboard keys
echo  Reward signal : click +0.002  battle-score x1.0  research x1.0
echo  Architecture  : XcomExtractor (economy/time/mission) -^> PPO
echo  ========================================================
echo.

:: ── Step 1: Kill any leftover processes from a previous run ─────────────────
echo  [1/4] Cleaning up leftover processes...
taskkill /IM openxcom.exe /F >nul 2>&1
powershell -NoProfile -Command ^
  "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*train_ppo*' -or $_.CommandLine -like '*xcom_gym*' -or $_.CommandLine -like '*xcom_rl*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1
echo  Done.
echo.

:: ── Step 2: Rebuild the engine ───────────────────────────────────────────────
echo  [2/4] Building engine...
if not exist "%VCVARS%" (
    echo  ERROR: Visual Studio Build Tools not found at:
    echo    %VCVARS%
    echo  Install MSVC 2022 Build Tools and re-run.
    pause
    exit /b 1
)
call "%VCVARS%" >nul 2>&1
pushd "%BUILDDIR%"
nmake /S
set "BUILD_ERR=%ERRORLEVEL%"
popd
if %BUILD_ERR% NEQ 0 (
    echo.
    echo  ERROR: Engine build failed ^(exit code %BUILD_ERR%^).
    echo  Fix the errors above then re-run.
    echo.
    pause
    exit /b 1
)
echo  Engine OK.
echo.

:: ── Step 3: Find latest checkpoint ──────────────────────────────────────────
echo  [3/4] Checking for checkpoints in:
echo    %CKPTDIR%
set "LATEST="
for /f "delims=" %%F in ('dir /b /o-d "%CKPTDIR%\xcom_ppo_*.zip" 2^>nul') do (
    if not defined LATEST set "LATEST=%%F"
)
if defined LATEST (
    echo  Latest checkpoint : !LATEST!
    echo  Mode              : RESUME
) else (
    echo  No checkpoint found.
    echo  Mode              : FRESH start
)
echo.

:: ── Step 4: TensorBoard + training ──────────────────────────────────────────
echo  [4/4] Starting TensorBoard at http://localhost:6006
start "TensorBoard" /D "%PYDIR%" cmd /k tensorboard --logdir runs/
timeout /t 3 /nobreak >nul

echo  Launching 4 fully-visual envs for 1,000,000 timesteps...
echo.
cd /d "%PYDIR%"

if defined LATEST (
    py -3 -m xcom_rl.train_ppo --n-envs 4 --all-visual --timesteps 1000000 --run-name visual_run --resume
) else (
    py -3 -m xcom_rl.train_ppo --n-envs 4 --all-visual --timesteps 1000000 --run-name visual_run
)

echo.
if %ERRORLEVEL% NEQ 0 (
    echo  ERROR: Training exited with code %ERRORLEVEL%
) else (
    echo  Training complete.
)
echo.
pause
endlocal
