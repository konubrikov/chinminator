"""Check the interpreter and GPU before downloading large model files."""
import platform, sys
import torch
print('Python:',sys.version)
print('OS:',platform.platform(),'CPU:',platform.machine(),'PyTorch:',torch.__version__)
if sys.platform == 'darwin':
    assert platform.machine() == 'arm64', 'Native Apple Silicon Python required'
    assert torch.backends.mps.is_available(), 'MPS unavailable; check macOS and PyTorch'
    device='mps'
else:
    assert torch.cuda.is_available(), 'CUDA GPU unavailable; install NVIDIA driver'
    device='cuda'
x=torch.ones((32,32),device=device)
assert (x@x).cpu()[0,0].item()==32
print('GPU matrix test passed:',device)
print('This checks GPU availability, not Qwen model compatibility or image quality.')
