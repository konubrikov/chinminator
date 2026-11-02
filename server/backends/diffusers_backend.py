"""Нативный diffusers-бэкенд (опционально, Windows/macOS).

Z-Image архитектурно совместим с Qwen-Image (Tongzean/Qwen-Image-Z-Image-Turbo).
Если у вас стоит diffusers>=0.36 с поддержкой QwenImageImg2ImgPipeline — этот
бэкенд работает без ComfyUI. При отсутствии поддержки — available() вернёт False
и пайплайн выберет другой бэкенд.
"""
from __future__ import annotations

import os

import numpy as np

from config import NEGATIVE_PROMPT, QualityParams
from edit_zimage import BaseBackend

REPO_ID = os.environ.get("ZIMAGE_DIFFUSERS_REPO", "Tongzean/Z-Image-Turbo-diffusers")


class DiffusersBackend(BaseBackend):
    name = "diffusers-z-image-turbo"

    def __init__(self) -> None:
        import torch
        from diffusers import QwenImageImg2ImgPipeline  # type: ignore

        dtype = torch.bfloat16
        device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        if device == "cpu":
            raise RuntimeError("diffusers-бэкенду нужен CUDA или MPS")
        self.torch = torch
        self.pipe = QwenImageImg2ImgPipeline.from_pretrained(
            REPO_ID, torch_dtype=dtype).to(device)
        try:
            self.pipe.enable_model_cpu_offload()   # для 12 ГБ VRAM
        except Exception:
            pass

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401
            from diffusers import QwenImageImg2ImgPipeline  # noqa: F401,type: ignore
        except Exception:
            return False
        return torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

    def inpaint(self, img_bgr: np.ndarray, mask01: np.ndarray, prompt: str,
                p: QualityParams, strength: float, seed: int) -> np.ndarray:
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        m8 = (np.clip(mask01, 0, 1) * 255).astype(np.uint8)
        gen = self.torch.Generator(device="cpu").manual_seed(seed)
        out = self.pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=Image.fromarray(rgb),
            mask_image=Image.fromarray(m8),
            strength=strength,
            num_inference_steps=p.steps,
            guidance_scale=p.cfg,
            generator=gen,
        ).images[0]
        return cv2.cvtColor(np.array(out.convert("RGB")), cv2.COLOR_RGB2BGR)
