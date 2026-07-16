@echo off
title Kill XCOM RL

echo.
echo  Stopping all XCOM RL processes...
echo.

:: ── 1. Engine instances ──────────────────────────────────────────────────────
echo  [1/3] Killing openxcom.exe...
taskkill /IM openxcom.exe /F >nul 2>&1

:: ── 2. Python processes (training, envs, demo collector, bc pretraining) ────
echo  [2/3] Killing Python processes...
powershell -NoProfile -Command ^
  "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*train_ppo*' -or $_.CommandLine -like '*xcom_gym*' -or $_.CommandLine -like '*xcom_rl*' -or $_.CommandLine -like '*demo_collect*' -or $_.CommandLine -like '*bc_pretrain*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

:: ── 3. TensorBoard (process + window) ───────────────────────────────────────
echo  [3/3] Killing TensorBoard...
powershell -NoProfile -Command ^
  "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*tensorboard*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
taskkill /FI "WINDOWTITLE eq TensorBoard" /F >nul 2>&1

echo.
echo  Done.
echo.
timeout /t 2 /nobreak >nul
