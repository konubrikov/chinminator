"""Native Qwen conditioning with shared reference/initial latent and CFG=1 shortcut."""



class _PositiveOnlyClip:
    def __init__(self, clip):
        self.clip=clip
        self.conditioning=None
    def tokenize(self, *args, **kwargs):
        if self.conditioning is not None:
            return None
        return self.clip.tokenize(*args, **kwargs)
    def encode_from_tokens_scheduled(self, tokens):
        if tokens is not None:
            self.conditioning=self.clip.encode_from_tokens_scheduled(tokens)
        return self.conditioning


class _CaptureVAE:
    def __init__(self, vae):
        self.vae=vae
        self.samples=None
    def encode(self, pixels):
        if self.samples is not None:
            raise RuntimeError('Shared Qwen node accepts exactly one reference')
        self.samples=self.vae.encode(pixels)
        return self.samples


class ChinminatorQwenCondition:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'clip':('CLIP',),'vae':('VAE',),'image':('IMAGE',),
            'prompt':('STRING',{'multiline':True}),'negative_prompt':('STRING',{'multiline':True}),
            'skip_negative':('BOOLEAN',{'default':True})}}
    RETURN_TYPES=('CONDITIONING','CONDITIONING','LATENT')
    FUNCTION='encode'
    CATEGORY='Chinminator'
    def encode(self, clip, vae, image, prompt, negative_prompt, skip_negative):
        # Padding is performed upstream. Never let the native reference encoder rescale.
        if image.shape[1]%32 or image.shape[2]%32:
            raise ValueError('Chinminator reference must be padded to 32 pixels')
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21
        capture=_CaptureVAE(vae)
        encoded=TextEncodeQwenImage21.execute(clip=_PositiveOnlyClip(clip) if skip_negative else clip,
            vae=capture,prompt=prompt,negative_prompt=negative_prompt,resolution=0,images={'image_1':image})
        return encoded[0],encoded[1],{'samples':capture.samples}


NODE_CLASS_MAPPINGS={'ChinminatorQwenCondition':ChinminatorQwenCondition}
NODE_DISPLAY_NAME_MAPPINGS={'ChinminatorQwenCondition':'Chinminator Qwen Shared Encode'}
