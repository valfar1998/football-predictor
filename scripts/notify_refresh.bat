@echo off
chcp 65001 >nul
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" exit /b 1

if not exist "data\processed" mkdir "data\processed"

set "LOG=data\processed\notify_refresh.log"
echo ===== %DATE% %TIME% =====>> "%LOG%"

".venv\Scripts\python.exe" -X utf8 main.py --notify-refresh >> "%LOG%" 2>&1
