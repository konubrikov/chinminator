"""FLUX 3 distilled masked editing: 16 steps, native / 1.5x detail."""
import hashlib
import json
import math
import os
from pathlib import Path
import urllib.request
import cv2
import numpy as np
from edit_qwen import BaseBackend
from backends.comfyui_backend import ComfyUIBackend
from backends.comfy_common import comfy_is_up, post_prompt, wait_done, fetch_image
from backends.flux_klein_backend import flux_graph

SYSTEM = (Path(__file__).resolve().parents[1]/'workflows/flux3_upsampler_system.txt').read_text('utf-8')
DEFAULT_MODELS = ['diffusion_models/flux3-image-klein-int8-convrot.safetensors',
                  'text_encoders/qwen3vl_4b_int8_convrot.safetensors',
                  'vae/flux2-vae.safetensors',
                  'text_encoders/qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors']

def settings():
    from model_selection import PROFILE_DATA
    return PROFILE_DATA if PROFILE_DATA.get('engine') == 'flux3' else dict(models=DEFAULT_MODELS,text_device='default')

def make_graph(image, mask, prompt, width, height, seed, profile):
    models = [Path(p).name for p in profile['models']]
    graph = flux_graph(image,mask,prompt,width,height,seed)
    graph['2']['inputs']['unet_name'] = models[0]
    graph['3']['inputs'].update(clip_name=models[1],type='flux3',device=profile['text_device'])
    graph['4']['inputs']['vae_name'] = models[2]
    def node(kind,**kw): return dict(class_type=kind,inputs=kw)
    graph['up_clip'] = node('CLIPLoader',clip_name=models[3],type='flux3',device=profile['text_device'])
    graph['expand'] = node('TextGenerate',clip=['up_clip',0],image=['1',0],
        prompt='Reference images follow in order. User request: '+prompt+' Output the caption.',
        system_prompt=SYSTEM,max_length=3584,sampling_mode='on',thinking=False,use_default_template=True,mtp='off',
        **{'sampling_mode.temperature':1.,'sampling_mode.top_k':1,'sampling_mode.top_p':.95,
           'sampling_mode.min_p':0.,'sampling_mode.repetition_penalty':1.,'sampling_mode.seed':seed,'sampling_mode.presence_penalty':0.})
    graph['5']['inputs']['text'] = ['expand',0]
    graph['shift'] = node('ModelSamplingFlux',model=['2',0],max_shift=math.log(16),base_shift=math.log(16),width=width,height=height)
    graph['16'] = node('KSampler',model=['shift',0],positive=['7',0],negative=['9',0],latent_image=['15',0],
        seed=seed,steps=16,cfg=1.,sampler_name='euler',scheduler='simple',denoise=1.)
    for key in ('10','11','12','13'): del graph[key]
    graph['18']['inputs']['filename_prefix'] = 'chinminator/flux3'
    return graph

class Flux3Backend(BaseBackend):
    name = 'comfyui-flux3-image-klein'

    @staticmethod
    def available():
        if not comfy_is_up(): return False
        with urllib.request.urlopen(os.environ.get('COMFY_URL','http://127.0.0.1:8188')+'/object_info',timeout=10) as response:
            info=json.load(response)
        graph=make_graph('source.png','mask.png','test',32,32,42,settings())
        for node in graph.values():
            kind=node['class_type']
            if kind not in info: raise RuntimeError('ComfyUI node missing: '+kind)
            for key in ('unet_name','clip_name','vae_name'):
                if key in node['inputs'] and node['inputs'][key] not in info[kind]['input']['required'][key][0]:
                    raise RuntimeError('Model not installed: '+node['inputs'][key])
        return True

    def inpaint(self,img_bgr,mask01,prompt,p,strength,seed):
        if mask01.shape != img_bgr.shape[:2]: raise ValueError('Mask dimensions differ from image')
        original_h,original_w=img_bgr.shape[:2]
        # UI Maximum Details is explicit; automatic mode always keeps native pixels.
        scale=1.5 if not p.auto_quality and p.steps>=40 else 1.0
        image=img_bgr
        if scale!=1:
            image=cv2.resize(image,(round(original_w*scale),round(original_h*scale)),interpolation=cv2.INTER_LANCZOS4)
            mask01=cv2.resize(mask01,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_LINEAR)
        scaled_h,scaled_w=image.shape[:2]
        image=cv2.copyMakeBorder(image,0,(-scaled_h)%32,0,(-scaled_w)%32,cv2.BORDER_REPLICATE)
        mask01=np.pad(mask01,((0,(-scaled_h)%32),(0,(-scaled_w)%32)))
        mask=(mask01>0).astype(np.uint8)*255
        mask[:16]=mask[-16:]=0;mask[:,:16]=mask[:,-16:]=0
        if not mask.any(): raise ValueError('Empty FLUX3 generation mask')
        def upload(array,prefix):
            digest=hashlib.sha256(array.tobytes()+str(array.shape).encode()).hexdigest()[:24]
            return ComfyUIBackend._upload(array,prefix+digest+'.png')
        graph=make_graph(upload(image,'flux3_'),upload(mask,'flux3_mask_'),prompt,image.shape[1],image.shape[0],seed,settings())
        output=fetch_image(wait_done(post_prompt(graph)))
        if output.shape!=image.shape: raise RuntimeError('FLUX3 returned unexpected dimensions')
        output=output[:scaled_h,:scaled_w]
        if scale!=1: output=cv2.resize(output,(original_w,original_h),interpolation=cv2.INTER_AREA)
        return output
