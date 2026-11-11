"""HTTP-сервер для Photoshop UXP-плагина.

Эндпоинты:
  GET  /health   — статус/бэкенд
  POST /analyze  — шаги 1–3 + превью анализа (JSON + base64 PNG)
  POST /process  — полный пайплайн
                   ?sidebyside=1 — before|after одним PNG
                   ?format=raw   — сырые RGB-байты (для imaging.putPixels в PS),
                                   размеры в заголовках X-Image-Width/Height
Запуск: uvicorn app:app --host 127.0.0.1 --port 8765

Эндпоинты объявлены обычными `def`: FastAPI выполняет их в пуле потоков,
поэтому долгая диффузия не блокирует event loop и /health продолжает отвечать.
"""
from __future__ import annotations

import base64
import json
import logging
import math
from dataclasses import replace

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from config import DEFAULT_THRESHOLDS, QUALITY, QualityParams, Thresholds
from pipeline import process_image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

from contextlib import asynccontextmanager
from automation_bridge import start_worker, save_settings

@asynccontextmanager
async def lifespan(app):
    stop = start_worker()
    try:
        yield
    finally:
        stop.set()

app = FastAPI(title="Chinminator", version="1.7.0", lifespan=lifespan)
# UXP fetch из Ps приходит с origin "psrp://" или localhost —
# разрешаем всё на loopback-сервере (он слушает только 127.0.0.1).
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["*"])


def _decode(img_bytes: bytes) -> np.ndarray:
    if not img_bytes:
        raise HTTPException(400, "Пустое изображение")
    if img_bytes.startswith(b"CHINRAW1"):
        import struct
        if len(img_bytes) < 16: raise HTTPException(400, "Invalid raw header")
        width, height = struct.unpack_from("<II", img_bytes, 8)
        if not width or not height or len(img_bytes) != 16 + width * height * 3:
            raise HTTPException(400, "Invalid raw dimensions")
        rgb = np.frombuffer(img_bytes, np.uint8, offset=16).reshape(height,width,3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    arr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Не удалось декодировать изображение")
    return bgr


def _parse_params(params: str | None) -> tuple[Thresholds, QualityParams, str | None]:
    if not params:
        return DEFAULT_THRESHOLDS, QUALITY, None
    try:
        pj = json.loads(params)
        th = replace(DEFAULT_THRESHOLDS, **pj.get("thresholds", {}))
        quality_options = dict(pj.get("quality", {}))
        if "auto_quality" not in quality_options and any(k in quality_options for k in ("steps", "max_side")):
            quality_options["auto_quality"] = 0
        q = replace(QUALITY, **quality_options)
        if q.auto_quality not in (0, 1):
            raise ValueError("auto_quality: требуется 0 или 1")
        if q.native_resolution not in (0, 1):
            raise ValueError("native_resolution: требуется 0 или 1")
        for obj in (th, q):
            for key, value in vars(obj).items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{key}: требуется конечное число")
        if not 0 < th.submental_ratio <= 1 or not 0 <= th.jaw_collar_gap <= 1 or not 0 <= th.neck_shadow_depth <= 1 or th.min_face_px < 1:
            raise ValueError("Недопустимые пороги")
        if not 1 <= q.steps <= 100 or not 0 <= q.cfg <= 30 or not 0 < q.denoise_strength <= 1 or not 32 <= q.max_side <= 4096 or not 0 <= q.mask_blur_px <= 256:
            raise ValueError("Недопустимые параметры качества")
        for key in ("steps", "max_side", "mask_blur_px", "seed"):
            if not isinstance(getattr(q, key), int):
                raise ValueError(f"{key}: требуется целое число")
        if pj.get("prompt") is not None and not isinstance(pj["prompt"], str):
            raise ValueError("prompt: требуется строка")
        return th, q, pj.get("prompt")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"bad params: {e}")


def _run(bgr, mode, **kw):
    try:
        return process_image(bgr, mode=mode, **kw)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.get("/health")
def health():
    from model_selection import backend_class, ENGINE, MODEL_LABEL, DEFAULT_PROMPT, ROOT, ADAPTER
    from pipeline import _EDIT_LOCK
    try:
        selected = backend_class()
        backend = selected.name if selected.available() else "UNAVAILABLE: ComfyUI не запущен"
    except Exception as e:  # noqa: BLE001
        backend = f"UNAVAILABLE: {e}"
    return {"status": "ok", "api_version": 5, "capabilities": ["auto_quality", "native_resolution", "raw_input", "jsx_automation", "model_selection"], "ready": not backend.startswith("UNAVAILABLE"), "qwen_backend": backend, "backend": backend, "model": MODEL_LABEL, "engine": ENGINE, "default_prompt": DEFAULT_PROMPT, "installation_root": str(ROOT), "busy": _EDIT_LOCK.locked(), "fixed_steps": len(ADAPTER['sigmas']) if ADAPTER else None}


@app.post("/analyze")
def analyze_only(file: UploadFile = File(...),
                 mode: str = Form("auto"),
                 params: str | None = Form(None)):
    """Только шаги 1–3: Qwen-Image 2.1 НЕ запускается. Учитывает пороги из панели."""
    bgr = _decode(file.file.read())
    th, _, _ = _parse_params(params)
    res = _run(bgr, mode, thresholds=th, analyze_only=True, with_preview=True)
    preview_b64 = (base64.b64encode(res.analysis_preview).decode()
                   if res.analysis_preview else None)
    return {
        "faces_found": res.faces_found,
        "metrics": res.geometry,
        "decision": res.decision,
        "skipped_reason": res.skipped_reason,
        "timings_ms": vars(res.timings),
        "analysis_preview_png_base64": preview_b64,
    }


@app.post("/process")
def process(file: UploadFile = File(...),
            mode: str = Form("auto"),
            sidebyside: bool = Query(False),
            format: str = Query("png", pattern="^(png|raw)$"),
            params: str | None = Form(None)):
    """Полный пайплайн. Ответ: результат (или оригинал, если SKIP)."""
    bgr = _decode(file.file.read())
    th, q, prompt = _parse_params(params)
    res = _run(bgr, mode, thresholds=th, quality=q, prompt=prompt,
               side_by_side=sidebyside)

    out_img = res.image if res.image is not None else bgr
    h, w = out_img.shape[:2]
    headers = {
        "X-Pipeline-Edited": "1" if res.edited else "0",
        "X-Pipeline-Backend": res.backend or "-",
        "X-Pipeline-Seed": str(res.seed),
        "X-Pipeline-Metrics": json.dumps(res.geometry),
        "X-Pipeline-Decision": json.dumps(res.decision),
        "X-Pipeline-Reason": res.skipped_reason or "-",
        "X-Pipeline-Timings": json.dumps(vars(res.timings)),
        "X-Image-Width": str(w),
        "X-Image-Height": str(h),
    }
    if format == "raw":
        # Не гоняем данные через PNG-кодек туда-обратно: плагин кладёт
        # эти байты прямо в слой через imaging.createImageDataFromBuffer.
        rgb = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        return Response(rgb.tobytes(), media_type="application/octet-stream",
                        headers=headers)
    # Быстрое сжатие: PNG level 1 в ~5 раз быстрее default при +10–15% к размеру
    ok, buf = cv2.imencode(".png", out_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise HTTPException(500, "encode failed")
    return Response(buf.tobytes(), media_type="image/png", headers=headers)


@app.post("/automation/settings")
def automation_settings(payload: dict):
    mode = payload.get("mode", "forced")
    if mode not in ("auto", "forced"): raise HTTPException(400, "Invalid mode")
    th, quality, prompt = _parse_params(json.dumps(payload.get("params", {})))
    save_settings(mode, th, quality, prompt)
    return {"saved": True}
