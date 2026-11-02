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
import threading
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
                 strength_scale: float = 1.0,
                 region_bbox: tuple[int, int, int, int] | None = None) -> EditResult:
    """Точка входа шага 4. mask_f32: 0..1, тот же размер, что и image.

    Диффузия гоняется не по всему кадру, а по кропу вокруг маски с запасом
    контекста (CROP_CONTEXT). Это в разы быстрее, а подбородок
    обрабатывается в большем разрешении, чем при даунскейле всего кадра.
    За пределами маски возвращаются исходные пиксели.
    """
    prompt = prompt or EDIT_PROMPT_QUALITY
    if params.seed is not None and params.seed >= 0:
        seed = params.seed
    else:
        seed = random.randint(0, 2**31 - 1)

    H, W = image_bgr.shape[:2]
    if region_bbox is None:
        ys, xs = np.where(mask_f32 > 0.03)
        region_bbox = ((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                       if xs.size else (0, 0, W - 1, H - 1))
    cx1, cy1, cx2, cy2 = _context_crop(region_bbox, W, H)
    crop = image_bgr[cy1:cy2, cx1:cx2]
    crop_mask = mask_f32[cy1:cy2, cx1:cx2]

    # Ограничиваем длинную сторону кропа (VRAM), стороны кратны 16 (латент VAE)
    img = _resize_for_model(crop, params.max_side)
    msk = np.clip(cv2.resize(crop_mask, (img.shape[1], img.shape[0]),
                             interpolation=cv2.INTER_LINEAR), 0.0, 1.0)

    strength = min(max(params.denoise_strength * strength_scale, 0.3), 0.75)

    backend = _pick_backend()
    log.info("Z-Image quality edit: backend=%s seed=%d strength=%.2f crop=%dx%d",
             backend.name, seed, strength, img.shape[1], img.shape[0])
    out_small = backend.inpaint(img, msk, prompt, params, strength, seed)

    out_crop = cv2.resize(out_small, (crop.shape[1], crop.shape[0]),
                          interpolation=cv2.INTER_LANCZOS4)
    # Финальная гарантия: за пределами маски — оригинальные пиксели
    a = np.clip(crop_mask, 0, 1)[..., None]
    blended = out_crop.astype(np.float32) * a + crop.astype(np.float32) * (1.0 - a)
    result = image_bgr.copy()
    result[cy1:cy2, cx1:cx2] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
    return EditResult(result, backend.name, seed)


CROP_CONTEXT = 0.6   # запас контекста вокруг маски (доля от её размера)


def _context_crop(bbox, W: int, H: int) -> tuple[int, int, int, int]:
    """Квадратоподобный кроп вокруг bbox маски с запасом контекста."""
    x1, y1, x2, y2 = bbox
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    side = int(max(bw, bh) * (1 + 2 * CROP_CONTEXT))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nx1 = int(max(0, cx - side / 2)); ny1 = int(max(0, cy - side / 2))
    nx2 = int(min(W, nx1 + side));    ny2 = int(min(H, ny1 + side))
    nx1 = int(max(0, nx2 - side));    ny1 = int(max(0, ny2 - side))
    return nx1, ny1, nx2, ny2


# ------------------------------------------------------------- backends

_BACKENDS: dict[str, "BaseBackend"] = {}
_PICK_LOCK = threading.Lock()


def _pick_backend() -> BaseBackend:
    if _BACKENDS:
        return next(iter(_BACKENDS.values()))
    with _PICK_LOCK:       # /health и /process не должны грузить модель дважды
        if _BACKENDS:
            return next(iter(_BACKENDS.values()))
        return _load_backend()


def _load_backend() -> BaseBackend:
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


def _resize_for_model(img: np.ndarray, max_side: int) -> np.ndarray:
    """Длинная сторона ≤ max_side, обе стороны кратны 16 (иначе бэкенды
    сами обрезают/паддят латент и результат съезжает на пару пикселей)."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    nw = max(16, int(round(w * scale / 16)) * 16)
    nh = max(16, int(round(h * scale / 16)) * 16)
    if (nw, nh) == (w, h):
        return img
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (nw, nh), interpolation=interp)
