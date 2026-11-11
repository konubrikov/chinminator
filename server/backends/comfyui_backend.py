"""Qwen-Image 2.1 through the local ComfyUI API."""
from __future__ import annotations

import json
import os
import sys
import hashlib
import math
from pathlib import Path
import urllib.request

import cv2
import numpy as np

from backends.comfy_common import (comfy_is_up, fetch_image, post_prompt,
                                   set_node_field, wait_done)
from config import NEGATIVE_PROMPT, QualityParams
from edit_qwen import BaseBackend
from model_selection import ADAPTER, ENGINE

_INSTALLED_WORKFLOW = os.path.join(os.path.dirname(__file__), "..", "workflows", "qwen21_installed.json")
WORKFLOW_PATH = os.environ.get("CHINMINATOR_WORKFLOW") or (
    _INSTALLED_WORKFLOW if os.path.exists(_INSTALLED_WORKFLOW) else os.path.join(
        os.path.dirname(__file__), "..", "workflows",
        "qwen21_edit_macos.json" if sys.platform == "darwin" else "qwen21_edit_quality.json"))


class ComfyUIBackend(BaseBackend):
    name = "comfyui-qwen-image-2.1"

    adapter = None

    @classmethod
    def available(cls) -> bool:
        if not comfy_is_up() or not os.path.exists(WORKFLOW_PATH):
            return False
        with open(WORKFLOW_PATH, encoding="utf-8") as f:
            graph = json.load(f)
        url = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
        with urllib.request.urlopen(url + "/object_info", timeout=10) as response:
            info = json.load(response)
        if cls.adapter:
            graph['adapter_check'] = {'class_type': 'LoraLoaderModelOnly',
                                      'inputs': {'lora_name': Path(cls.adapter['file']).name}}
        for node in graph.values():
            kind = node["class_type"]
            if kind not in info:
                raise RuntimeError(f"ComfyUI: отсутствует узел {kind}; обновите ComfyUI")
            for field in ("unet_name", "clip_name", "vae_name", "lora_name"):
                if field in node["inputs"]:
                    options = info[kind]["input"]["required"][field][0]
                    if node["inputs"][field] not in options:
                        raise RuntimeError(f"Не установлена модель: {node['inputs'][field]}")
        return True

    def inpaint(self, img_bgr: np.ndarray, mask01: np.ndarray, prompt: str,
                p: QualityParams, strength: float, seed: int) -> np.ndarray:
        self.name = type(self).name
        with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
            graph = json.load(f)

        # Загрузка входного изображения и маски через /upload/image
        img_path = self._upload(img_bgr, "chinminator_" + hashlib.sha256(img_bgr.tobytes() + str(img_bgr.shape).encode()).hexdigest()[:24] + ".png")
        set_node_field(graph, "LoadImage", "image", img_path)
        if mask01.shape != img_bgr.shape[:2]:
            raise ValueError("Inpainting mask must match the reference image")
        # Binary generation region includes the final compositing feather.
        # Black = original latent restored at every step; white = generate.
        generation_mask = (mask01 > 0).astype(np.uint8) * 255
        border = min(16, min(generation_mask.shape) // 4)
        generation_mask[:border] = generation_mask[-border:] = 0
        generation_mask[:, :border] = generation_mask[:, -border:] = 0
        if not np.any(generation_mask):
            raise ValueError("Inpainting mask is empty after protecting the crop frame")
        mask_path = self._upload(generation_mask, "chinminator_mask_" + hashlib.sha256(generation_mask.tobytes() + str(generation_mask.shape).encode()).hexdigest()[:24] + ".png")
        set_node_field(graph, "LoadImageMask", "image", mask_path)
        set_node_field(graph, "TextEncodeQwenImage21", "prompt", "Edit <image1>. " + prompt)
        set_node_field(graph, "KSampler", "seed", seed)
        set_node_field(graph, "KSampler", "steps", p.steps)
        set_node_field(graph, "KSampler", "cfg", p.cfg)
        set_node_field(graph, "TextEncodeQwenImage21", "negative_prompt", NEGATIVE_PROMPT)
        # Sample from the encoded original with a noise mask, not an empty latent.
        # Comfy's KSamplerX0Inpaint restores unmasked original latents every step.
        # Final pixel compositing additionally removes VAE reconstruction drift.

        # Custom node preserves the native positive conditioning and shares its VAE output.
        # Old Comfy servers remain usable until restarted with the installed project node.
        url = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
        with urllib.request.urlopen(url + "/object_info/ChinminatorQwenCondition", timeout=10) as response:
            optimized = bool(json.load(response))
        if optimized and os.environ.get("CHINMINATOR_QWEN_OPTIMIZE", "1") != "0":
            graph = optimize_qwen_graph(graph, p.cfg)
        default_cache = 'auto'
        if self.adapter:
            default_cache = next(node['inputs']['device'] for node in graph.values()
                                 if node['class_type'] == 'QwenImage21Cache')
        cache_device = os.environ.get("CHINMINATOR_QWEN_CACHE", default_cache)
        if self.adapter and not self.adapter['cache']:
            cache_device = 'off'
        if cache_device not in ("auto", "cpu", "gpu", "off"):
            raise ValueError("Invalid Qwen cache device")
        set_node_field(graph, "QwenImage21Cache", "device", cache_device)
        if self.adapter:
            height, width = img_bgr.shape[:2]
            # Qwen 2.1 uses unpatched VAE latents at 1/16 scale.
            tokens = (width // 16) * (height // 16)
            shift = math.exp(0.5 + 0.4 * (tokens - 256) / (8192 - 256))
            sigmas = [shift / (shift + 1 / sigma - 1) for sigma in self.adapter['sigmas']]
            graph = turbo_qwen_graph(graph, seed, lora_name=Path(self.adapter['file']).name,
                                     sigmas=', '.join(map(str, [*sigmas, 0.0])))
        elif os.environ.get("CHINMINATOR_QWEN_TURBO_FULL") == "1":
            graph = turbo_qwen_graph(graph, seed, full_model=True)
            self.name = "comfyui-qwen-image-2.1-turbo"
        elif os.environ.get("CHINMINATOR_QWEN_TURBO_LORA") == "1":
            graph = turbo_qwen_graph(graph, seed)
            self.name = "comfyui-qwen-image-2.1-turbo-lora"
        pid = post_prompt(graph)
        outputs = wait_done(pid)
        out = fetch_image(outputs)
        if out.shape != img_bgr.shape:
            raise RuntimeError('ComfyUI changed the image dimensions; refusing to stretch the result')
        return out

    @staticmethod
    def _upload(arr: np.ndarray, filename: str) -> str:
        import urllib.request
        from io import BytesIO
        ok, buf = cv2.imencode(".png", arr)
        assert ok
        boundary = "----dcboundary"
        # Stable content-addressed filenames allow ComfyUI to reuse reference
        # encoding when retrying the same photo. Overwrite avoids renamed copies.
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode() + buf.tobytes() + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            os.environ.get("COMFY_URL", "http://127.0.0.1:8188") + "/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["name"]


def mask01_to_u8(m: np.ndarray) -> np.ndarray:
    """Comfy LoadImageMask читает одноканальный PNG."""
    g = cv2.cvtColor((m * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) \
        if m.ndim == 3 else (m * 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def optimize_qwen_graph(graph, cfg):
    import copy
    graph=copy.deepcopy(graph)
    matches=[(key,node) for key,node in graph.items() if node['class_type']=='TextEncodeQwenImage21']
    if len(matches)!=1:
        raise ValueError('Expected one native Qwen reference encoder')
    key,node=matches[0];inputs=node['inputs']
    if inputs.get('resolution',0)!=0 or any(k.startswith('images.image_') and k!='images.image_1' for k in inputs):
        raise ValueError('Shared encoder requires one native-resolution reference')
    reference=inputs['images.image_1']
    node['class_type']='ChinminatorQwenCondition'
    node['inputs']={k:inputs[k] for k in ('clip','vae','prompt','negative_prompt')}
    node['inputs'].update(image=reference,skip_negative=(cfg==1.0))
    for encode_key,encode_node in list(graph.items()):
        if encode_node['class_type']=='VAEEncode' and encode_node['inputs']=={'pixels':reference,'vae':inputs['vae']}:
            for consumer in graph.values():
                for field,value in consumer['inputs'].items():
                    if value==[encode_key,0]:consumer['inputs'][field]=[key,2]
            del graph[encode_key]
    return graph


class DistilledQwenBackend(ComfyUIBackend):
    name = f'comfyui-{ENGINE}'
    adapter = ADAPTER


TURBO_SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"


def turbo_qwen_graph(graph, seed, full_model=False, *,
                     lora_name='qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors',
                     sigmas=TURBO_SIGMAS):
    """Use a distilled checkpoint or LoRA with its explicit Euler schedule."""
    import copy
    graph=copy.deepcopy(graph)
    def node(kind,**inputs):return {'class_type':kind,'inputs':inputs}
    sampler_key,sampler=next((k,n) for k,n in graph.items() if n['class_type']=='KSampler')
    original=sampler['inputs']
    if original['cfg']!=1.0:raise ValueError('Distilled Qwen requires CFG=1')
    unet_key=next(k for k,n in graph.items() if n['class_type']=='UNETLoader')
    if full_model:
        graph[unet_key]['inputs']['unet_name']='qwen_image_2.1_turbo_int8_convrot.safetensors'
    else:
        for n in graph.values():
            for field,value in n['inputs'].items():
                if value==[unet_key,0]:n['inputs'][field]=['turbo_lora',0]
        graph['turbo_lora']=node('LoraLoaderModelOnly',model=[unet_key,0],
            lora_name=lora_name,strength_model=1.0)
    graph['turbo_noise']=node('RandomNoise',noise_seed=seed)
    graph['turbo_guider']=node('CFGGuider',model=original['model'],positive=original['positive'],negative=original['negative'],cfg=1.0)
    graph['turbo_sampler']=node('KSamplerSelect',sampler_name='euler')
    graph['turbo_sigmas']=node('ManualSigmas',sigmas=sigmas)
    graph[sampler_key]=node('SamplerCustomAdvanced',noise=['turbo_noise',0],guider=['turbo_guider',0],
        sampler=['turbo_sampler',0],sigmas=['turbo_sigmas',0],latent_image=original['latent_image'])
    return graph
