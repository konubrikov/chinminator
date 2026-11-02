"""Бэкенд Z-Image-Turbo через MLX (macOS Apple Silicon, ≥24 ГБ unified memory).

Модель: Tongzean/Z-Image-Turbo-MLX (локальная папка models/z-image-turbo-mlx).
Загружается один раз, дальше — переиспользуется.

Реализация опирается на mlx-lm / mlx-community диффузионный API. Точные имена
классов могут отличаться между версиями портов Z-Image — при первом запуске
скрипт проверяет доступность и печатает понятную ошибку с инструкцией.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from config import NEGATIVE_PROMPT, QualityParams
from edit_zimage import BaseBackend

MODEL_DIR = os.environ.get(
    "ZIMAGE_MLX_DIR",
    os.path.join(os.path.dirname(__file__), "..", "models", "z-image-turbo-mlx"))


class MlxBackend(BaseBackend):
    name = "mlx-z-image-turbo"

    def __init__(self) -> None:
        if not self.available():
            raise RuntimeError(f"MLX-модель не найдена в {MODEL_DIR}")
        # Ленивая импортизация — mlx есть только на macOS
        import mlx.core as mx  # noqa: F401
        from mlx_vision.diffusers import ZImageEditPipeline  # type: ignore

        self.pipe = ZImageEditPipeline.from_pretrained(MODEL_DIR)
        self.mx = mx

    @staticmethod
    def available() -> bool:
        try:
            import mlx.core  # noqa: F401
            import mlx_vision  # noqa: F401
        except Exception:
            return False
        return os.path.isdir(MODEL_DIR)

    def inpaint(self, img_bgr: np.ndarray, mask01: np.ndarray, prompt: str,
                p: QualityParams, strength: float, seed: int) -> np.ndarray:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        m8 = (np.clip(mask01, 0, 1) * 255).astype(np.uint8)

        out = self.pipe(
            image=rgb,
            mask=m8,
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=p.steps,
            guidance_scale=p.cfg,
            strength=strength,
            generator=self.mx.random.Generator(key=self.mx.random.PRNGKey(seed)),
        ).images[0]
        return cv2.cvtColor(np.array(out.convert("RGB")), cv2.COLOR_RGB2BGR)
