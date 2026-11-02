# ============================================================================
#  Chinminator — подготовка окружения Windows (CUDA >= 12 ГБ VRAM)
#  Запуск:  правый клик -> "Выполнить с помощью PowerShell", либо:
#           powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
#  Флаги:   -SkipModels   не скачивать модели (ComfyUI GGUF + FaceMesh)
#           -Force        пересоздать .venv с нуля
# ============================================================================
param(
    [switch]$SkipModels,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root    = Split-Path -Parent $PSScriptRoot
$Server  = Join-Path $Root "server"
$Venv    = Join-Path $Server ".venv"
$Py      = Join-Path $Venv "Scripts\python.exe"

Write-Host "=== Chinminator: настройка Windows ===" -ForegroundColor Cyan
Write-Host "Корень проекта: $Root"

# --- 1. Python 3.10+ -------------------------------------------------------
$candidates = @()
try { $candidates += (& py -3.12 -c "import sys;print(sys.executable)" 2>$null) } catch {}
try { $candidates += (& py -3.11 -c "import sys;print(sys.executable)" 2>$null) } catch {}
try { $candidates += (& python  -c "import sys;print(sys.executable)" 2>$null) } catch {}
$basePy = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $basePy) {
    Write-Host "НЕ НАЙДЕН Python 3.10+. Установите с https://www.python.org/downloads/ (галочка Add to PATH)." -ForegroundColor Red
    exit 1
}
Write-Host "Python: $basePy"

# --- 2. venv (портативный, внутри проекта) ---------------------------------
if ((Test-Path $Venv) -and $Force) { Remove-Item -Recurse -Force $Venv }
if (-not (Test-Path $Venv)) {
    & $basePy -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "Не удалось создать venv" }
}
Write-Host "venv: $Venv"

# --- 3. Зависимости ядра ----------------------------------------------------
& $Py -m pip install --upgrade pip
& $Py -m pip install -r (Join-Path $Server "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt завершился с ошибкой" }

# --- 4. PyTorch CUDA (для опционального diffusers-бэкенда) ------------------
Write-Host "Установка PyTorch cu124 (опционально, но рекомендуется)..." -ForegroundColor Yellow
& $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyTorch CUDA не установлен — основной путь (ComfyUI) это не блокирует." -ForegroundColor Yellow
}

# --- 5. ComfyUI (бэкенд Z-Image-Turbo GGUF) --------------------------------
$ComfyDir = Join-Path $Root "vendor\ComfyUI"
if (-not (Test-Path (Join-Path $ComfyDir ".git"))) {
    Write-Host "Клонирование ComfyUI в $ComfyDir ..."
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git $ComfyDir
    if ($LASTEXITCODE -ne 0) { throw "git clone ComfyUI не удался (нужен Git for Windows)" }
    Push-Location $ComfyDir
    & $Py -m pip install -r requirements.txt
    Pop-Location
} else {
    Write-Host "ComfyUI уже установлен: $ComfyDir"
}

# GGUF-загрузчик для Z-Image (ComfyUI-GGUF)
& $Py -m pip install comfyui-frontend-package
$ggufRepo = Join-Path $ComfyDir "custom_nodes\ComfyUI-GGUF"
if (-not (Test-Path $ggufRepo)) {
    git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git $ggufRepo
    & $Py -m pip install -r (Join-Path $ggufRepo "requirements.txt")
}

# --- 6. Модели --------------------------------------------------------------
if (-not $SkipModels) {
    # 6a. FaceMesh landmarker (~3.6 МБ, обязателен для шага 1)
    $faceModel = Join-Path $Server "models\face_landmarker.task"
    New-Item -ItemType Directory -Force -Path (Split-Path $faceModel) | Out-Null
    if (-not (Test-Path $faceModel)) {
        Write-Host "Скачивание face_landmarker.task ..."
        Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" `
                          -OutFile $faceModel
    }

    # 6b. Z-Image-Turbo GGUF Q8 из HuggingFace (несколько ГБ) -> в модели ComfyUI
    Write-Host ""
    Write-Host "Z-Image-Turbo: скачиваем gguf Q8 с HuggingFace (Tongzean/Z-Image-Turbo-ComfyUI)." -ForegroundColor Yellow
    Write-Host "Если модель gated — сначала выполните: huggingface-cli login"
    & $Py -m pip install -U "huggingface_hub[cli]"
    $modelsDir = Join-Path $ComfyDir "models\diffusion_models"
    New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
    try {
        & $Py -m huggingface_hub.commands.huggingface_cli download Tongzean/Z-Image-Turbo-ComfyUI `
            --local-dir $modelsDir
        Write-Host "Модель Z-Image-Turbo скачана в $modelsDir" -ForegroundColor Green
    } catch {
        Write-Host "Автоматическая загрузка модели не удалась. Скачайте вручную из" -ForegroundColor Red
        Write-Host "  https://huggingface.co/Tongzean/Z-Image-Turbo-ComfyUI"
        Write-Host "и положите .gguf/.safetensors в: $modelsDir"
        Write-Host "(шаги 1-3 работают без неё; шаг 4 потребует модель при первом реальном рендере)"
    }
}

# --- 7. Итог ----------------------------------------------------------------
Write-Host ""
Write-Host "Готово! Запуск:" -ForegroundColor Green
Write-Host "  сервер для Photoshop :  scripts\start_server.bat"
Write-Host "  CLI-пакетная обработка:  scripts\cli.bat <input> <output> --mode auto"
Write-Host "  portable-режим (без установки Ps-связи): запустите start_server.bat и укажите URL в плагине"
