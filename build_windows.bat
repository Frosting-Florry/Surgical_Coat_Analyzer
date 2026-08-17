@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"

set "BUILD_PYTHON="
if exist "%~dp0.build-env\Scripts\python.exe" set "BUILD_PYTHON=%~dp0.build-env\Scripts\python.exe"
if not defined BUILD_PYTHON if exist "%~dp0.venv\Scripts\python.exe" set "BUILD_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined BUILD_PYTHON if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" set "BUILD_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if not defined BUILD_PYTHON (
  where py >nul 2>nul
  if not errorlevel 1 py -3 -m venv .build-env
  if exist "%~dp0.build-env\Scripts\python.exe" set "BUILD_PYTHON=%~dp0.build-env\Scripts\python.exe"
)

if not defined BUILD_PYTHON (
  where python >nul 2>nul
  if not errorlevel 1 python -m venv .build-env
  if exist "%~dp0.build-env\Scripts\python.exe" set "BUILD_PYTHON=%~dp0.build-env\Scripts\python.exe"
)

if not defined BUILD_PYTHON (
  echo No Python 3.11 or 3.12 installation was found.
  echo Install Python from https://www.python.org/downloads/windows/
  echo During installation, select "Add python.exe to PATH", then run this file again.
  goto :error
)

echo Using Python: %BUILD_PYTHON%
"%BUILD_PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] in [(3,11),(3,12)] else 1)"
if errorlevel 1 (
  echo Python 3.11 or 3.12 is required.
  goto :error
)

"%BUILD_PYTHON%" -m pip install setuptools wheel pyinstaller numpy Pillow
if errorlevel 1 goto :error
"%BUILD_PYTHON%" -m PyInstaller --noconfirm --clean --onefile --windowed --name Surgical_Coat_Analyzer --paths src --add-data "src\coat_analyzer\web;coat_analyzer\web" packaging\launcher.py
if errorlevel 1 goto :error
echo Build completed: dist\Surgical_Coat_Analyzer.exe
exit /b 0
:error
echo Build failed.
pause
exit /b 1
