# ============================================================================
#  Chinminator — сборка PORTABLE-архива для Windows (.zip)
#  Архив самодостаточен: переносите и запускаете на другой Win-машине с CUDA>=12ГБ.
#  Содержимое: server/ + plugin/ + scripts/ + .venv + vendor\ComfyUI + модели.
#  Запуск:  powershell -ExecutionPolicy Bypass -File scripts\make_portable.ps1
#           (-IncludeModels  — включить скачанные модели в архив; по умолчанию да,
#            если они уже есть на диске)
# ============================================================================
param(
    [switch]$ExcludeModels,
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$Root   = Split-Path -Parent $PSScriptRoot
$Stage  = Join-Path $env:TEMP ("chinminator_portable_" + [guid]::NewGuid().ToString("N").Substring(0,8))
$Dist   = if ($OutDir) { $OutDir } else { Join-Path $Root "dist" }
$Zip    = Join-Path $Dist ("Chinminator-portable-win-" + (Get-Date -Format "yyyyMMdd") + ".zip")

Write-Host "=== Сборка portable-архива ===" -ForegroundColor Cyan

if (-not (Test-Path (Join-Path $Root "server\.venv"))) {
    throw "Нет server\.venv — сначала выполните setup_windows.ps1"
}

New-Item -ItemType Directory -Force -Path $Stage | Out-Null
$Inner = Join-Path $Stage "Chinminator"
New-Item -ItemType Directory -Force -Path $Inner | Out-Null

function Copy-Tree($src, $dst) {
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    Get-ChildItem -Recurse -Force $src | ForEach-Object {
        $rel = $_.FullName.Substring($src.Length).TrimStart('\')
        # вычищаем мусор из переносимого образа
        if ($rel -match '\\__pycache__(\\|$)' -or $rel -match '\.pyc$' -or
            $rel -match '^comfyui_user\\' -or $rel -match '^out\\') { return }
        $target = Join-Path $dst $rel
        if ($_.PSIsContainer) { New-Item -ItemType Directory -Force -Path $target | Out-Null }
        else { Copy-Item -Force $_.FullName $target }
    }
}

foreach ($part in @("server", "plugin", "scripts")) {
    Write-Host "  + $part"
    Copy-Tree (Join-Path $Root $part) (Join-Path $Inner $part)
}
if (Test-Path (Join-Path $Root "vendor\ComfyUI")) {
    Write-Host "  + vendor\ComfyUI (с кастомными нодами)"
    Copy-Tree (Join-Path $Root "vendor\ComfyUI") (Join-Path $Inner "vendor\ComfyUI")
}

if ($ExcludeModels) {
    foreach ($m in @(
        (Join-Path $Inner "vendor\ComfyUI\models"),
        (Join-Path $Inner "server\models\z-image-turbo-mlx"))) {
        if (Test-Path $m) { Remove-Item -Recurse -Force $m }
    }
    Write-Host "  модели ИСКЛЮЧЕНЫ из архива (-ExcludeModels)"
}

# README и файл-маркер portable
Set-Content -Path (Join-Path $Inner "PORTABLE.txt") -Encoding UTF8 -Value @"
Chinminator portable build $(Get-Date)
Запуск сервера: scripts\start_server.bat --portable
Подключение плагина: Photoshop -> Plug-ins -> Chinni... (см. plugin\README.md)
"@
Copy-Item (Join-Path $Root "README.md") (Join-Path $Inner "README.md") -Force

New-Item -ItemType Directory -Force -Path $Dist | Out-Null
if (Test-Path $Zip) { Remove-Item -Force $Zip }
Write-Host "Упаковка в $Zip ..."
Compress-Archive -Path $Inner -DestinationPath $Zip -CompressionLevel Optimal
Remove-Item -Recurse -Force $Stage

$sizeGB = [math]::Round((Get-Item $Zip).Length / 1GB, 2)
Write-Host ""
Write-Host "Готово: $Zip (${sizeGB} ГБ)" -ForegroundColor Green
Write-Host "Перенесите архив на целевую машину, распакуйте в любую папку,"
Write-Host "запустите scripts\start_server.bat — установка не требуется."
