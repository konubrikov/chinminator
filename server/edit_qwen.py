"""Qwen-Image 2.1 reference editing with local masked compositing."""
from __future__ import annotations

import logging
import os
import random
import threading
from dataclasses import dataclass, replace

import cv2
import numpy as np

from config import EDIT_PROMPT_QUALITY, QualityParams
from lighting import match_context_light

log = logging.getLogger("qwen21")


class BaseBackend:
    """Единый интерфейс бэкенда Qwen-Image 2.1. Реализации — в backends/.*_backend.py."""
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
    from model_selection import DEFAULT_PROMPT
    prompt = prompt or DEFAULT_PROMPT
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

    params = select_crop_quality(params, crop.shape[1], crop.shape[0])
    log.info("Crop quality: source=%dx%d steps=%d native=%s", crop.shape[1], crop.shape[0], params.steps, bool(params.native_resolution))
    if params.native_resolution:
        bottom, right = (-crop.shape[0]) % 32, (-crop.shape[1]) % 32
        img = cv2.copyMakeBorder(crop, 0, bottom, 0, right, cv2.BORDER_REPLICATE)
        msk = np.pad(crop_mask, ((0,bottom),(0,right)), constant_values=0)
    else:
        img = _resize_for_model(crop, params.max_side)
        msk = np.clip(cv2.resize(crop_mask, (img.shape[1], img.shape[0]),
                                 interpolation=cv2.INTER_LINEAR), 0.0, 1.0)

    strength = 1.0  # full replacement; the transition exists only outside the opaque skin core

    backend = _pick_backend()
    log.info("Qwen-Image 2.1 quality edit: backend=%s seed=%d strength=%.2f crop=%dx%d",
             backend.name, seed, strength, img.shape[1], img.shape[0])
    out_small = backend.inpaint(img, msk, prompt, params, strength, seed)

    if out_small.shape != img.shape:
        raise RuntimeError('Model returned a different image size; refusing to rescale native pixels')
    out_crop = out_small[:crop.shape[0], :crop.shape[1]] if params.native_resolution else cv2.resize(
        out_small, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    out_crop = match_context_light(crop, out_crop, crop_mask)
    # Финальная гарантия: за пределами маски — оригинальные пиксели
    a = np.clip(crop_mask, 0, 1)[..., None]
    # Keep the crop perimeter exactly equal to the original even when skin
    # reaches the image edge. Transition inward rather than moving the seam.
    yy, xx = np.ogrid[:crop.shape[0], :crop.shape[1]]
    edge_distance = np.minimum(np.minimum(yy, crop.shape[0]-1-yy),
                               np.minimum(xx, crop.shape[1]-1-xx))
    edge_t = np.clip(edge_distance / max(2, round(0.02 * min(crop.shape[:2]))), 0, 1)
    a *= (edge_t * edge_t * (3 - 2 * edge_t))[..., None]
    blended = out_crop.astype(np.float32) * a + crop.astype(np.float32) * (1.0 - a)
    result = image_bgr.copy()
    result[cy1:cy2, cx1:cx2] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
    return EditResult(result, backend.name, seed)


def select_crop_quality(params: QualityParams, width: int, height: int) -> QualityParams:
    """Choose from the actual padded crop, never from the whole photo size."""
    selected = params
    if params.auto_quality:
        selected = replace(params, steps=28, max_side=1024) if max(width, height) <= 2048 else replace(params, steps=40, max_side=1536)
    if params.native_resolution:
        return selected
    cap_key = "CHINMINATOR_AUTO_MAX_SIDE" if params.auto_quality else "CHINMINATOR_MAX_SIDE"
    cap = int(os.environ.get(cap_key, os.environ.get("CHINMINATOR_MAX_SIDE", "4096")))
    if cap < 32 or cap > 4096:
        raise ValueError("CHINMINATOR_MAX_SIDE must be between 32 and 4096")
    return replace(selected, max_side=min(selected.max_side, cap))


CROP_CONTEXT = 1.0   # запас контекста вокруг маски (доля от её размера)


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
    from model_selection import backend_class, ENGINE
    selected = backend_class()
    if not selected.available():
        raise RuntimeError("Запустите ComfyUI на 127.0.0.1:8188 и установите выбранную модель")
    backend = selected()
    _BACKENDS[ENGINE] = backend
    return backend


def _resize_for_model(img: np.ndarray, max_side: int) -> np.ndarray:
    """Длинная сторона ≤ max_side, обе стороны кратны 16 (иначе бэкенды
    сами обрезают/паддят латент и результат съезжает на пару пикселей)."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    # The Qwen reference encoder uses multiples of 32. Match its canvas here
    # so it cannot silently reshape the reference relative to the mask.
    nw = max(32, int(w * scale // 32) * 32)
    nh = max(32, int(h * scale // 32) * 32)
    if (nw, nh) == (w, h):
        return img
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (nw, nh), interpolation=interp)
