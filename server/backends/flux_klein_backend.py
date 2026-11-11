"""FLUX.2 klein comparison backend: native-size masked sampling, 4 distilled steps."""
import hashlib
import numpy as np
from edit_qwen import BaseBackend
from backends.comfyui_backend import ComfyUIBackend
from backends.comfy_common import post_prompt,wait_done,fetch_image


def flux_graph(image,mask,prompt,width,height,seed,reference=None):
    def node(kind,**inputs):return {'class_type':kind,'inputs':inputs}
    graph = {
      '1':node('LoadImage',image=image),
      '2':node('UNETLoader',unet_name='flux-2-klein-4b.safetensors',weight_dtype='default'),
      '3':node('CLIPLoader',clip_name='qwen_3_4b.safetensors',type='flux2',device='default'),
      '4':node('VAELoader',vae_name='flux2-vae.safetensors'),
      '5':node('CLIPTextEncode',clip=['3',0],text=prompt),
      '6':node('VAEEncode',pixels=['1',0],vae=['4',0]),
      '7':node('ReferenceLatent',conditioning=['5',0],latent=['6',0]),
      '8':node('ConditioningZeroOut',conditioning=['5',0]),
      '9':node('ReferenceLatent',conditioning=['8',0],latent=['6',0]),
      '10':node('CFGGuider',model=['2',0],positive=['7',0],negative=['9',0],cfg=1.0),
      '11':node('KSamplerSelect',sampler_name='euler'),
      '12':node('RandomNoise',noise_seed=seed),
      '13':node('Flux2Scheduler',steps=4,width=width,height=height),
      '14':node('LoadImageMask',image=mask,channel='red'),
      '15':node('SetLatentNoiseMask',samples=['6',0],mask=['14',0]),
      '16':node('SamplerCustomAdvanced',noise=['12',0],guider=['10',0],sampler=['11',0],sigmas=['13',0],latent_image=['15',0]),
      '17':node('VAEDecode',samples=['16',0],vae=['4',0]),
      '18':node('SaveImage',images=['17',0],filename_prefix='chinminator/flux2-klein')}
    if reference is not None:
        graph['reference_image']=node('LoadImage',image=reference)
        graph['reference_latent']=node('VAEEncode',pixels=['reference_image',0],vae=['4',0])
        for key in ('7','9'):
            graph[key]['inputs']['latent']=['reference_latent',0]
    return graph


def skin_reference(image,mask,labels):
    """Hide obsolete skin geometry in conditioning; keep clothes and context.

    The sampler still receives the original latent and its independent hard mask.
    This reference is never composited into the output.
    """
    if labels.shape != mask.shape or image.shape[:2] != mask.shape:
        raise ValueError('Reference labels and mask must match native dimensions')
    reference=image.copy()
    reference[(mask>0)&np.isin(labels,(2,3))]=127
    return reference


class FluxKleinBackend(BaseBackend):
    name='comfyui-flux2-klein-4b'
    def inpaint(self,img_bgr,mask01,prompt,p,strength,seed):
        h,w=img_bgr.shape[:2]
        if h%32 or w%32:raise ValueError('Native input must be padded to 32')
        digest=hashlib.sha256(img_bgr.tobytes()).hexdigest()[:24]
        image=ComfyUIBackend._upload(img_bgr,'flux_'+digest+'.png')
        mask=(mask01>0).astype(np.uint8)*255
        border=min(16,min(h,w)//4)
        mask[:border]=mask[-border:]=0;mask[:,:border]=mask[:,-border:]=0
        if not mask.any():raise ValueError('Empty generation mask')
        name=ComfyUIBackend._upload(mask,'fluxmask_'+hashlib.sha256(mask.tobytes()).hexdigest()[:24]+'.png')
        from skin_segment import segment_skin
        reference=skin_reference(img_bgr,mask,segment_skin(img_bgr))
        refname=ComfyUIBackend._upload(reference,'fluxref_'+hashlib.sha256(reference.tobytes()).hexdigest()[:24]+'.png')
        result=fetch_image(wait_done(post_prompt(flux_graph(image,name,prompt,w,h,seed,reference=refname))))
        if result.shape!=img_bgr.shape:raise RuntimeError('FLUX changed native dimensions')
        return result
