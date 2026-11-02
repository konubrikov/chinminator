@echo off
REM ============================================================
REM  Chinminator — запуск сервера Z-Image Edit (Windows)
REM  Обычный режим: uvicorn на http://127.0.0.1:8765 + ComfyUI
REM  Portable-режим:  start_server.bat --portable
REM    (то же самое, но дополнительно открывает папку с готовыми
REM     файлами-заглушками для переноса; окружение целиком внутри
REM     проекта — .venv и vendor\ComfyUI, ничего в систему не ставится)
REM ============================================================
setlocal
set ROOT=%~dp0..
set SERVER=%ROOT%\server
set VENV_PY=%SERVER%\.venv\Scripts\python.exe
set COMFY=%ROOT%\vendor\ComfyUI

if not exist "%VENV_PY%" (
    echo [ERROR] venv не найден. Сначала выполните:
    echo   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
    pause & exit /b 1
)

REM --- Поднимаем ComfyUI (если установлен), ждём порт 8188 -----------------
if exist "%COMFY%\main.py" (
    echo [1/2] Запуск ComfyUI (Z-Image-Turbo GGUF)...
    start "Chinminator ComfyUI" /min cmd /c ""%VENV_PY%" "%COMFY%\main.py" --port 8188 --listen 127.0.0.1"
) else (
    echo [WARN] ComfyUI не найден в %COMFY% — шаг 4 будет недоступен,
    echo        анализ (шаги 1-3) работает.
)

REM --- Chinminator API --------------------------------------------------------
echo [2/2] Запуск Chinminator API на http://127.0.0.1:8765 ...
cd /d "%SERVER%"
if "%~1"=="--portable" (
    echo Portable-режим: все пути и модели остаются внутри папки проекта.
    set DC_MODELS_DIR=%SERVER%\models
    set ZIMAGE_MLX_DIR=%SERVER%\models\z-image-turbo-mlx
)
"%VENV_PY%" -m uvicorn app:app --host 127.0.0.1 --port 8765
pause
