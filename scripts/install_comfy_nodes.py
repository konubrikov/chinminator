"""Install the project's Comfy node before starting ComfyUI."""
from pathlib import Path
import shutil

def install(root):
    root=Path(root)
    target=root/'vendor/ComfyUI/custom_nodes/chinminator_qwen.py'
    source=root/'server/comfy_nodes/chinminator_qwen.py'
    if not target.exists() or target.read_bytes()!=source.read_bytes():
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)

if __name__=='__main__':install(Path(__file__).resolve().parents[1])
