$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Write-Host "Chinminator local GPU installer"
Write-Host "Internet access is needed; models and dependencies are downloaded separately."
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Install Python 3.12 (python.org, include Python launcher), then run Install.cmd again."
}
& py -3.12 -c "import sys; assert sys.version_info[:2] == (3,12)"
if ($LASTEXITCODE -ne 0) { throw "Python 3.12 is required" }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Install Git for Windows (git-scm.com), reopen installer, then retry."
}
& (Join-Path $PSScriptRoot "setup_windows.ps1")
if ($LASTEXITCODE -ne 0) { throw "Installation failed; inspect the error above" }
Write-Host "Ready. Run Start.cmd. Load plugin/manifest.json in Adobe UXP Developer Tool."
