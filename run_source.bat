@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
set "APP_PYTHON="
if exist "%~dp0.venv\Scripts\python.exe" set "APP_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined APP_PYTHON where py >nul 2>nul && set "APP_PYTHON=py -3"
if not defined APP_PYTHON where python >nul 2>nul && set "APP_PYTHON=python"
if not defined APP_PYTHON if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" set "APP_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not defined APP_PYTHON (
  echo Python was not found. Install Python 3.11 or 3.12 and run this file again.
  pause
  exit /b 1
)
%APP_PYTHON% -m coat_analyzer
if errorlevel 1 pause
