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

app = FastAPI(title="Chinminator", version="1.1")
# UXP fetch из Ps приходит с origin "psrp://" или localhost —
# разрешаем всё на loopback-сервере (он слушает только 127.0.0.1).
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["*"])


def _decode(img_bytes: bytes) -> np.ndarray:
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
        q = replace(QUALITY, **pj.get("quality", {}))
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
    try:
        from edit_zimage import _pick_backend
        backend = _pick_backend().name
    except Exception as e:  # noqa: BLE001
        backend = f"UNAVAILABLE: {e}"
    return {"status": "ok", "zimage_backend": backend}


@app.post("/analyze")
def analyze_only(file: UploadFile = File(...),
                 mode: str = Form("auto"),
                 params: str | None = Form(None)):
    """Только шаги 1–3: Z-Image НЕ запускается. Учитывает пороги из панели."""
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
