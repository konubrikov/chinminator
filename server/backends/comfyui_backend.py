"""Бэкенд Z-Image-Turbo через локальный ComfyUI (Windows + CUDA ≥12 ГБ VRAM).

Workflow (API-формат) лежит в workflows/zimage_inpaint_quality.json и собран
под ноды стандартного ComfyUI + GGUF-загрузчик для Tongzean Q8 кванта.
Если у вас другой набор нод — отредактируйте JSON, код подставляет только
поля: image/mask/prompt/steps/cfg/denoise/seed.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from backends.comfy_common import (comfy_is_up, fetch_image, post_prompt,
                                   set_node_field, wait_done)
from config import NEGATIVE_PROMPT, QualityParams
from edit_zimage import BaseBackend

WORKFLOW_PATH = os.path.join(os.path.dirname(__file__), "..",
                             "workflows", "zimage_inpaint_quality.json")


class ComfyUIBackend(BaseBackend):
    name = "comfyui-z-image-turbo-q8"

    @staticmethod
    def available() -> bool:
        return comfy_is_up() and os.path.exists(WORKFLOW_PATH)

    def inpaint(self, img_bgr: np.ndarray, mask01: np.ndarray, prompt: str,
                p: QualityParams, strength: float, seed: int) -> np.ndarray:
        with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
            graph = json.load(f)

        # Загрузка входного изображения и маски через /upload/image
        img_path = self._upload(img_bgr, "dc_input.png")
        msk_path = self._upload(mask01_to_u8(mask01), "dc_mask.png")

        set_node_field(graph, "LoadImage", "image", img_path)
        set_node_field(graph, "LoadImageMask", "image", msk_path)
        set_node_field(graph, "CLIPTextEncode", "text", prompt)   # positive
        # второй CLIPTextEncode — negative
        neg_set = False
        for node in graph.values():
            if node.get("class_type") == "CLIPTextEncode":
                if not neg_set:
                    neg_set = True
                    continue
                node["inputs"]["text"] = NEGATIVE_PROMPT
        set_node_field(graph, "KSampler", "seed", seed)
        set_node_field(graph, "KSampler", "steps", p.steps)
        set_node_field(graph, "KSampler", "cfg", p.cfg)
        set_node_field(graph, "KSampler", "denoise", strength)

        pid = post_prompt(graph)
        outputs = wait_done(pid)
        out = fetch_image(outputs)
        return cv2.resize(out, (img_bgr.shape[1], img_bgr.shape[0]))

    @staticmethod
    def _upload(arr: np.ndarray, filename: str) -> str:
        import urllib.request
        from io import BytesIO
        ok, buf = cv2.imencode(".png", arr)
        assert ok
        boundary = "----dcboundary"
        body = (
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
