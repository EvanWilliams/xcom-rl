@echo off
title XCOM RL — Visual Training

:: ── locate this script's directory (project root) ────────────────────────────
set "ROOT=%~dp0"
set "PYTHON_DIR=%ROOT%python"
set "CKPT_DIR=%PYTHON_DIR%\checkpoints"

echo.
echo ============================================================
echo   XCOM Reinforcement Learning — Fully Visualized Training
echo ============================================================
echo.

:: ── find the most recent checkpoint (newest .zip that isn't bc_init) ─────────
set "LATEST_CKPT="
for /f "delims=" %%F in ('dir /b /o-d "%CKPT_DIR%\xcom_ppo_*.zip" 2^>nul') do (
    if not defined LATEST_CKPT set "LATEST_CKPT=%%F"
)

if defined LATEST_CKPT (
    echo   Latest checkpoint : %LATEST_CKPT%
    echo   Mode              : RESUME from checkpoint
    set "RESUME_FLAG=--resume"
) else (
    echo   No checkpoint found.
    echo   Mode              : FRESH start
    set "RESUME_FLAG="
)
echo.

:: ── start TensorBoard in a separate window ────────────────────────────────────
echo Starting TensorBoard at http://localhost:6006 ...
start "TensorBoard" cmd /k "cd /d "%PYTHON_DIR%" && py -3 -m tensorboard.main --logdir runs/"
timeout /t 3 /nobreak >nul

:: ── launch training ───────────────────────────────────────────────────────────
echo Launching training  (4 visible envs, 1 000 000 timesteps) ...
echo.
cd /d "%PYTHON_DIR%"
py -3 -m xcom_rl.train_ppo ^
    --n-envs 4 ^
    --all-visual ^
    --timesteps 1000000 ^
    --run-name visual_run ^
    %RESUME_FLAG%

:: ── keep window open so errors are visible ───────────────────────────────────
echo.
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Training exited with code %ERRORLEVEL%
) else (
    echo Training complete.
)
echo.
pause
