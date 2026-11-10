@echo off
rem Start ClearTalk on Windows: double-click this file. Keep the window open; Ctrl+C stops the app.
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=.venv-train\Scripts\python.exe"
if not exist "%PY%" (
  echo ClearTalk environment is missing. See README.md for installation.
  pause
  exit /b 1
)
"%PY%" scripts\launch.py
if errorlevel 1 pause
