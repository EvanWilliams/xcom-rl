@echo off
setlocal enabledelayedexpansion
title XCOM RL - Visual Training

set "ROOT=%~dp0"
set "PYDIR=%ROOT%python"
set "CKPTDIR=%PYDIR%\checkpoints"

echo.
echo  XCOM Reinforcement Learning - Fully Visualized Training
echo  ========================================================
echo  Action space  : Discrete(23)  mouse clicks + keyboard keys
echo  Reward signal : click +0.002  battle-score x1.0  research x1.0
echo  Architecture  : XcomExtractor (economy/time/mission) -> PPO
echo  ========================================================
echo.

:: Kill any leftover engine processes from a previous run
taskkill /IM openxcom.exe /F >nul 2>&1

:: Find the most recent xcom_ppo_*.zip (newest file by date, skip bc_init)
set "LATEST="
for /f "delims=" %%F in ('dir /b /o-d "%CKPTDIR%\xcom_ppo_*.zip" 2^>nul') do (
    if not defined LATEST set "LATEST=%%F"
)

if defined LATEST (
    echo  Latest checkpoint : !LATEST!
    echo  Mode              : RESUME from checkpoint
) else (
    echo  No checkpoint found.
    echo  Mode              : FRESH start
)
echo.

:: Start TensorBoard in a separate window
echo  Starting TensorBoard at http://localhost:6006
start "TensorBoard" cmd /k "cd /d "%PYDIR%" && tensorboard --logdir runs/"
timeout /t 3 /nobreak >nul

:: Launch training
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
