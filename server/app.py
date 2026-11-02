"""HTTP-сервер для Photoshop UXP-плагина.

Эндпоинты:
  GET  /health   — статус/бэкенд
  POST /analyze  — шаги 1–3 + превью анализа (JSON + base64 PNG)
  POST /process  — полный пайплайн, ответ — PNG результата
                   (?sidebyside=1 — before|after одним изображением)
Запуск: uvicorn app:app --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import base64
import json
import logging

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from config import DEFAULT_THRESHOLDS, QUALITY
from pipeline import process_image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

app = FastAPI(title="Chinminator", version="1.0")
# UXP websocket/fetch из Ps приходит с origin "psrp://" или localhost —
# разрешаем всё на loopback-сервере (он слушает только 127.0.0.1).
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


def _decode(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Не удалось декодировать изображение")
    return bgr


@app.get("/health")
def health():
    try:
        from edit_zimage import _pick_backend
        backend = _pick_backend().name
    except Exception as e:  # noqa: BLE001
        backend = f"UNAVAILABLE: {e}"
    return {"status": "ok", "zimage_backend": backend}


@app.post("/analyze")
async def analyze_only(file: UploadFile = File(...),
                       mode: str = Form("auto")):
    bgr = _decode(await file.read())
    res = process_image(bgr, mode=mode, quality=None)
    preview_b64 = base64.b64encode(res.analysis_preview).decode() \
        if res.analysis_preview else None
    return JSONResponse({
        "faces_found": res.faces_found,
        "metrics": res.geometry,
        "decision": res.decision,
        "skipped_reason": res.skipped_reason,
        "timings_ms": vars(res.timings),
        "analysis_preview_png_base64": preview_b64,
    })


@app.post("/process")
async def process(file: UploadFile = File(...),
                  mode: str = Form("auto"),
                  sidebyside: bool = Query(False),
                  params: str | None = Form(None)):
    """Полный пайплайн. Ответ: image/png (результат или оригинал, если SKIP)."""
    bgr = _decode(await file.read())
    q = QUALITY
    th = DEFAULT_THRESHOLDS
    prompt = None
    if params:
        try:
            pj = json.loads(params)
            q = type(q)(**{**vars(q), **pj.get("quality", {})})
            th = type(th)(**{**vars(th), **pj.get("thresholds", {})})
            prompt = pj.get("prompt")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"bad params: {e}")

    try:
        res = process_image(bgr, mode=mode, thresholds=th, quality=q,
                            prompt=prompt, side_by_side=sidebyside)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    out_img = res.image if res.image is not None else bgr
    ok, buf = cv2.imencode(".png", out_img)
    if not ok:
        raise HTTPException(500, "encode failed")
    headers = {
        "X-Pipeline-Edited": "1" if res.edited else "0",
        "X-Pipeline-Backend": res.backend or "-",
        "X-Pipeline-Seed": str(res.seed),
        "X-Pipeline-Metrics": json.dumps(res.geometry),
        "X-Pipeline-Decision": json.dumps(res.decision),
        "X-Pipeline-Timings": json.dumps(vars(res.timings)),
    }
    return Response(buf.tobytes(), media_type="image/png", headers=headers)
