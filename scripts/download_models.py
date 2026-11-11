from huggingface_hub import hf_hub_download
from pathlib import Path
root=Path(__file__).resolve().parents[1] / 'vendor' / 'ComfyUI' / 'models'
for name in ['diffusion_models/qwen_image_2.1_int8_convrot.safetensors','text_encoders/qwen3vl_8b_w4a8.safetensors','vae/qwen_image_2.1_vae_bf16.safetensors']:
 print('Downloading',name,flush=True)
 hf_hub_download('Comfy-Org/Qwen-Image-2.1',name,local_dir=str(root))
 print('Complete',name,flush=True)
