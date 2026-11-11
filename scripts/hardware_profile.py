"""Conservative resource-based installation profiles, independently testable."""
import json,sys
from pathlib import Path
from model_sources import QWEN_ADAPTERS
DIT8='diffusion_models/qwen_image_2.1_int8_convrot.safetensors'
DIT16='diffusion_models/qwen_image_2.1_bf16.safetensors'
TE4='text_encoders/qwen3vl_8b_w4a8.safetensors'
TE8='text_encoders/qwen3vl_8b_int8_convrot.safetensors'
TE16='text_encoders/qwen3vl_8b_bf16.safetensors'
VAE='vae/qwen_image_2.1_vae_bf16.safetensors'
DIT4='diffusion_models/qwen_image_2.1_Q4_K_M.gguf'

def choose_profile(platform,ram_gb,vram_gb=0,variant='standard',engine='qwen2'):
    if variant not in ('standard','q4'):raise ValueError('Unknown model variant')
    if engine not in ('qwen2','flux3', *QWEN_ADAPTERS):raise ValueError('Unknown model engine')
    if engine!='qwen2' and variant!='standard':raise ValueError('Q4 is available for standard Qwen only')
    profile=_standard_profile(platform,ram_gb,vram_gb)
    profile['engine']=engine
    profile['variant']=variant
    if engine in QWEN_ADAPTERS:
        adapter = QWEN_ADAPTERS[engine]
        profile['name'] = engine + '-' + profile['name']
        profile['models'].append(adapter['file'])
        if not adapter['cache']:
            profile['cache_device'] = 'off'
    if engine=='flux3':
        full=(ram_gb>=60 and (platform=='darwin' or vram_gb>=23))
        precision='bf16' if full else 'int8-convrot'
        encoder='bf16' if full else 'int8_convrot'
        upsampler='int8_convrot' if platform=='darwin' else 'w4a8'
        profile['models']=[f'diffusion_models/flux3-image-klein-{precision}.safetensors',
            f'text_encoders/qwen3vl_4b_{encoder}.safetensors','vae/flux2-vae.safetensors',
            f'text_encoders/qwen3.5_9b_flux3_prompt_upsampler_{upsampler}.safetensors']
        profile['name']='flux3-'+profile['name']
        return profile
    if variant=='q4':
        profile['name']+='-q4'
        profile['models'][0]=DIT4
        profile['models'][1]=TE8 if platform=='darwin' else TE4
        profile['cache_device']='off'
    return profile

def _standard_profile(platform,ram_gb,vram_gb=0):
    if platform=='darwin':
        if ram_gb<23:raise RuntimeError('This local Mac build requires at least 24 GB unified memory')
        full=ram_gb>=60
        return dict(name='mac-full' if full else 'mac-compact',models=[DIT16 if full else DIT8,TE16 if full else TE8,VAE],
            auto_max_side=1536 if full else (1024 if ram_gb>=30 else 768),quality_max_side=1536 if ram_gb>=30 else 1024,
            text_device='cpu',cache_device='off',flags=['--lowvram','--force-fp16','--fp16-text-enc','--cpu-vae','--cache-none','--reserve-vram','6'])
    if vram_gb<7.5:raise RuntimeError('At least 8 GB NVIDIA VRAM is required')
    if ram_gb<15:raise RuntimeError('At least 16 GB system RAM is required; 32 GB recommended for 8 GB VRAM')
    full=vram_gb>=23 and ram_gb>=60
    medium=vram_gb>=15 and ram_gb>=30
    low=vram_gb<11.5
    return dict(name='cuda-full' if full else ('cuda-balanced' if medium else 'cuda-compact'),
        models=[DIT16 if full else DIT8,TE16 if full else (TE8 if medium else TE4),VAE],
        auto_max_side=768 if low else (1024 if vram_gb<15 else 1536),quality_max_side=1024 if low else 1536,
        text_device='cpu' if low else 'default',cache_device='off' if low else 'auto',
        flags=['--lowvram','--cpu-vae','--reserve-vram','1'] if low else ['--lowvram'])
def detect(variant='standard',engine='qwen2'):
    import torch,psutil
    ram=psutil.virtual_memory().total/2**30
    if sys.platform=='darwin':
        if not torch.backends.mps.is_available():raise RuntimeError('Apple GPU / MPS unavailable')
        profile=choose_profile('darwin',ram,variant=variant,engine=engine)
    else:
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; install a current NVIDIA driver and retry')
        if torch.cuda.get_device_capability(0)<(7,5):raise RuntimeError('CUDA 13 build requires NVIDIA Turing or newer, in addition to 8 GB VRAM')
        profile=choose_profile('win32',ram,torch.cuda.get_device_properties(0).total_memory/2**30,variant,engine)
    return profile
if __name__=='__main__':print(json.dumps(detect(),indent=2))
