@echo off
title Kill XCOM RL

echo.
echo  Stopping all XCOM RL processes...
echo.

:: Kill engine instances
echo  [1/4] Killing openxcom.exe instances...
taskkill /IM openxcom.exe /F >nul 2>&1

:: Kill Python training and environment processes
echo  [2/4] Killing Python training processes...
wmic process where "commandline like '%%train_ppo%%'"  delete >nul 2>&1
wmic process where "commandline like '%%xcom_gym%%'"   delete >nul 2>&1
wmic process where "commandline like '%%xcom_rl%%'"    delete >nul 2>&1
wmic process where "commandline like '%%demo_collect%%'" delete >nul 2>&1
wmic process where "commandline like '%%bc_pretrain%%'"  delete >nul 2>&1

:: Kill TensorBoard
echo  [3/4] Killing TensorBoard...
wmic process where "commandline like '%%tensorboard%%'" delete >nul 2>&1

:: Close the TensorBoard cmd window by its title
echo  [4/4] Closing TensorBoard window...
taskkill /FI "WINDOWTITLE eq TensorBoard" /F >nul 2>&1

echo.
echo  Done. All XCOM RL processes stopped.
echo.
timeout /t 2 /nobreak >nul
