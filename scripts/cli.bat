@echo off
REM Chinminator CLI — пакетная обработка без Photoshop.
REM Пример: cli.bat C:\photos C:\photos_fixed --mode auto
setlocal
set ROOT=%~dp0..
set SERVER=%ROOT%\server
set VENV_PY=%SERVER%\.venv\Scripts\python.exe
if not exist "%VENV_PY%" (
    echo Сначала выполните setup_windows.ps1 & pause & exit /b 1
)
cd /d "%SERVER%"
"%VENV_PY%" cli.py %*
