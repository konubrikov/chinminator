$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Bootstrap = Join-Path $Root ".bootstrap"
New-Item -ItemType Directory -Force $Bootstrap | Out-Null
$Zip = Join-Path $Bootstrap "uv.zip"
$Url = "https://github.com/astral-sh/uv/releases/download/0.13.6/uv-x86_64-pc-windows-msvc.zip"
$Expected = "a2a8820d5eea32f08ef467ef3b7f66883a9b076d5737a2fc2fbb5f458d1bb50f"
try {
    $Existing = Join-Path $Root "server\.venv\Scripts\python.exe"
    $Marker = Join-Path $Root ".installed-version"
    if ((Test-Path -LiteralPath $Existing) -and (Test-Path -LiteralPath $Marker) -and ((Get-Content -LiteralPath $Marker -Raw).Trim() -in @("1.6.0", "1.7.0"))) {
        & $Existing -X utf8 (Join-Path $PSScriptRoot "launch.py")
        exit $LASTEXITCODE
    }
    Write-Host "Downloading verified Python bootstrap..."
    Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing
    if ((Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToLower() -ne $Expected) { throw "uv checksum mismatch" }
    Expand-Archive -LiteralPath $Zip -DestinationPath $Bootstrap -Force
    $Uv = Join-Path $Bootstrap "uv.exe"
    if (-not (Test-Path $Uv)) { throw "uv.exe missing from archive" }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Root ".python"
    $env:UV_PYTHON_BIN_DIR = Join-Path $Root ".bootstrap\bin"
    $env:UV_CACHE_DIR = Join-Path $Root ".uv-cache"
    & $Uv python install 3.12
    if ($LASTEXITCODE -ne 0) { throw "Python download failed" }
    $Python = (& $Uv python find --managed-python 3.12).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Python lookup failed" }
    & $Python -X utf8 (Join-Path $PSScriptRoot "install.py")
    if ($LASTEXITCODE -ne 0) { throw "Installation failed; inspect messages above" }
} catch { Write-Error $_; exit 1 }
