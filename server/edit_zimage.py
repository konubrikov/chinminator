"""Шаг 4. Диффузионное редактирование области подбородка — Z-Image-Turbo.

Реализован ТОЛЬКО качественный режим (mode="quality"), как требует ТЗ:
полноразмерный латент, без downscale-pass.

Бэкенды подбираются автоматически:
* Windows + NVIDIA (>=12 ГБ VRAM): ComfyUI REST API с workflow gguf-Q8
  (Tongzean/Z-Image-Turbo-ComfyUI) — самый надёжный способ запустить Z-Image
  на consumer-GPU. Либо нативный diffusers, если установлен QwenImage pipeline.
* macOS (Apple Silicon, >=24 ГБ unified memory): MLX-порт модели
  (Tongzean/Z-Image-Turbo-MLX).

Интерфейс единый: inpaint(image_bgr uint8, mask float32 0..1, prompt, params) -> bgr.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import cv2
import numpy as np

from config import EDIT_PROMPT_QUALITY, QualityParams

log = logging.getLogger("zimage")


class BaseBackend:
    """Единый интерфейс бэкенда Z-Image. Реализации — в backends/.*_backend.py."""
    name = "?"

    def inpaint(self, img_bgr, mask01, prompt, p, strength, seed):
        raise NotImplementedError


@dataclass
class EditResult:
    image: np.ndarray      # BGR uint8, размер исходного кадра
    backend: str
    seed: int


def edit_quality(image_bgr: np.ndarray, mask_f32: np.ndarray,
                 params: QualityParams, *, prompt: str | None = None,
                 strength_scale: float = 1.0) -> EditResult:
    """Точка входа шага 4. mask_f32: 0..1, тот же размер, что и image."""
    prompt = prompt or EDIT_PROMPT_QUALITY
    if params.seed is not None and params.seed >= 0:
        seed = params.seed
    else:
        seed = random.randint(0, 2**31 - 1)

    # Ограничиваем длинную сторону, но НЕ режем область лица
    img, scale = _resize_max_side(image_bgr, params.max_side)
    msk = cv2.resize(mask_f32, (img.shape[1], img.shape[0]),
                     interpolation=cv2.INTER_LINEAR)
    msk = np.clip(msk, 0.0, 1.0)

    strength = min(max(params.denoise_strength * strength_scale, 0.3), 0.75)

    backend = _pick_backend()
    log.info("Z-Image quality edit: backend=%s seed=%d strength=%.2f",
             backend, seed, strength)
    out_small = backend_inpaint(backend, img, msk, prompt, params, strength, seed)

    out = cv2.resize(out_small, (image_bgr.shape[1], image_bgr.shape[0]),
                     interpolation=cv2.INTER_LANCZOS4)
    # Финальная гарантия: за пределами маски — оригинальные пиксели
    orig_mask = np.clip(mask_f32, 0, 1)[..., None]
    blended = (out.astype(np.float32) * orig_mask +
               image_bgr.astype(np.float32) * (1.0 - orig_mask))
    return EditResult(blended.astype(np.uint8), backend_name(backend), seed)


# ------------------------------------------------------------- backends

_BACKENDS: dict[str, "BaseBackend"] = {}


def _pick_backend() -> BaseBackend:
    if _BACKENDS:
        return next(iter(_BACKENDS.values()))
    order = ["mlx", "comfyui", "diffusers"]
    for name in order:
        try:
            if name == "mlx":
                from backends.mlx_backend import MlxBackend
                if MlxBackend.available():
                    _BACKENDS["mlx"] = MlxBackend()
                    break
            elif name == "comfyui":
                from backends.comfyui_backend import ComfyUIBackend
                if ComfyUIBackend.available():
                    _BACKENDS["comfyui"] = ComfyUIBackend()
                    break
            elif name == "diffusers":
                from backends.diffusers_backend import DiffusersBackend
                if DiffusersBackend.available():
                    _BACKENDS["diffusers"] = DiffusersBackend()
                    break
        except Exception as e:  # noqa: BLE001
            log.warning("backend %s недоступен: %s", name, e)
    if not _BACKENDS:
        raise RuntimeError(
            "Не найден ни один Z-Image бэкенд. Установите один из:\n"
            "  Win: запустите ComfyUI с Tongzean/Z-Image-Turbo-ComfyUI (см. README)\n"
            "  Mac: pip install mlx-lm && скачайте Tongzean/Z-Image-Turbo-MLX\n"
            "  или: pip install 'diffusers>=0.36'+'transformers' и QwenImage-чекпойнт"
        )
    return next(iter(_BACKENDS.values()))


def backend_name(b: BaseBackend) -> str:
    return b.name


def backend_inpaint(b: BaseBackend, img, msk, prompt, p, strength, seed):
    return b.inpaint(img, msk, prompt, p, strength, seed)


def _resize_max_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return img, 1.0
    new_wh = (int(round(w * scale)) // 2 * 2, int(round(h * scale)) // 2 * 2)
    return cv2.resize(img, new_wh, interpolation=cv2.INTER_AREA), scale
