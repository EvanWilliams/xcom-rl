@echo off
REM XCOM RL Training launcher
REM Run from this directory (python\) or double-click.
REM
REM Quick options:
REM   train.bat                  -- default: 1 visible + 8 headless, 9M steps
REM   train.bat allvisual        -- all 4 windows visible, slower (~6-12 hrs for 327k)
REM   train.bat resume           -- resume from latest checkpoint (default mode)
REM   train.bat watch            -- live per-env reward monitor (run in separate terminal)
REM   train.bat help             -- show full CLI help

setlocal
cd /d "%~dp0"

if "%1"=="help" goto help
if "%1"=="allvisual" goto allvisual
if "%1"=="resume" goto resume
if "%1"=="watch" goto watch

REM ── DEFAULT: fast headless run ────────────────────────────────────────────
echo Starting training: 9 envs (1 visible + 8 headless), 9M steps
py -3 -m xcom_rl.train_ppo --n-envs 9 --timesteps 9000000 --run-name "run_%date:~-4%%date:~4,2%%date:~7,2%"
goto end

:allvisual
REM ── ALL VISUAL: see all 4 windows ─────────────────────────────────────────
echo Starting training: 4 envs ALL visible, 327k steps (slower)
py -3 -m xcom_rl.train_ppo --n-envs 4 --all-visual --max-steps 3000 --timesteps 327000 --run-name "allvisual_%date:~-4%%date:~4,2%%date:~7,2%"
goto end

:resume
REM ── RESUME from latest checkpoint ─────────────────────────────────────────
echo Resuming from latest checkpoint...
for /f "delims=" %%f in ('dir /b /od checkpoints\*.zip 2^>nul') do set CKPT=checkpoints\%%f
if not defined CKPT (echo No checkpoint found. & goto end)
echo Resuming: %CKPT%
py -3 -m xcom_rl.train_ppo --n-envs 4 --timesteps 500000 --load-model "%CKPT%" --run-name "resume_%date:~-4%%date:~4,2%%date:~7,2%"
goto end

:watch
REM ── LIVE MONITOR: per-env instantaneous reward display ────────────────────
echo Launching live reward monitor (run this in a separate terminal alongside train.bat)...
py -3 -m xcom_rl.watch
goto end

:help
py -3 -m xcom_rl.train_ppo --help

:end
endlocal
pause
