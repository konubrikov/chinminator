param([switch]$SkipModels)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root "server\.venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { py -3.12 -m venv (Join-Path $Root "server\.venv") }
if (-not (Test-Path $Py)) { throw "Python 3.12 environment creation failed" }
$Comfy = Join-Path $Root "vendor\ComfyUI"
if (-not (Test-Path (Join-Path $Comfy "main.py"))) {
    git clone https://github.com/comfyanonymous/ComfyUI.git $Comfy
    if ($LASTEXITCODE -ne 0) { throw "ComfyUI clone failed" }
    git -C $Comfy checkout (Get-Content (Join-Path $PSScriptRoot "comfy_revision.txt")).Trim()
    if ($LASTEXITCODE -ne 0) { throw "ComfyUI revision checkout failed" }
}
& $Py -m pip install -r (Join-Path $Root "server\requirements.txt") -r (Join-Path $Comfy "requirements.txt") huggingface_hub
if ($LASTEXITCODE -ne 0) { throw "Dependencies installation failed" }
& $Py -m pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed" }
& $Py -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name())"
if ($LASTEXITCODE -ne 0) { throw "CUDA verification failed" }
if (-not $SkipModels) {
    & $Py (Join-Path $PSScriptRoot "download_http.py")
    if ($LASTEXITCODE -ne 0) { throw "Qwen-Image 2.1 download failed" }
}
Write-Host "Setup complete. Run scripts/start_server.bat"
