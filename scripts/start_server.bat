@echo off
setlocal
"%~dp0..\server\.venv\Scripts\python.exe" -X utf8 "%~dp0launch.py"
if errorlevel 1 pause
