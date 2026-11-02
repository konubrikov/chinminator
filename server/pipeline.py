"""Оркестрация пайплайна: Фото → [1 Детекция] → [2 Анализ] → [3 Решение] → [4 Z-Image Edit].

Используется и как Python API (cli.py), и как ядро HTTP-сервера (app.py).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import DEFAULT_THRESHOLDS, QUALITY, Thresholds
from decision import Decision, decide
from edit_zimage import EditResult, edit_quality
from face_detect import detect_faces
from geometry import Geometry, analyze

log = logging.getLogger("pipeline")

# Один GPU/одна модель: параллельные запросы из Photoshop/CLI ставим в очередь,
# иначе два диффузионных прохода одновременно выбивают VRAM (OOM на 12 ГБ).
_EDIT_LOCK = threading.Lock()
# MediaPipe-landmarker не потокобезопасен.
_DETECT_LOCK = threading.Lock()

PREVIEW_MAX_SIDE = 1024   # превью для панели PS — полный кадр там не нужен


@dataclass
class StepTimings:
    detect_ms: float = 0.0
    analyze_ms: float = 0.0
    decide_ms: float = 0.0
    edit_ms: float = 0.0


@dataclass
class PipelineResult:
    image: np.ndarray | None       # BGR результат; None если правка не нужна
    faces_found: int
    geometry: dict = field(default_factory=dict)
    decision: dict = field(default_factory=dict)
    edited: bool = False
    skipped_reason: str = ""
    backend: str = ""
    seed: int = -1
    timings: StepTimings = field(default_factory=StepTimings)
    analysis_preview: bytes | None = None   # PNG с разметкой для превью в PS


def process_image(bgr: np.ndarray, *, mode: str = "auto",
                  thresholds: Thresholds | None = None,
                  quality=None, prompt: str | None = None,
                  side_by_side: bool = False,
                  analyze_only: bool = False,
                  with_preview: bool = False) -> PipelineResult:
    """Один проход пайплайна по изображению.

    mode: "auto" — сплошной поток (решает анализатор);
          "forced" — правка выполняется всегда.
    analyze_only: только шаги 1–3, без Z-Image (превью / dry-run).
    with_preview: построить PNG-превью разметки (нужно только /analyze).
    """
    if mode not in ("auto", "forced"):
        raise ValueError(f"mode must be auto|forced, got {mode!r}")
    th = thresholds or DEFAULT_THRESHOLDS
    q = quality or QUALITY
    t = StepTimings()

    # ---------- 1. Детекция лица ----------
    t0 = time.perf_counter()
    with _DETECT_LOCK:
        faces = detect_faces(bgr)
    t.detect_ms = (time.perf_counter() - t0) * 1000
    if not faces:
        return PipelineResult(None, 0, skipped_reason="no face detected",
                              timings=t)
    face = faces[0]                      # крупнейшее лицо
    if face.face_h < th.min_face_px:
        return PipelineResult(None, len(faces),
                              skipped_reason=f"face too small ({face.face_h:.0f}px)",
                              timings=t)

    # ---------- 2. Анализ геометрии ----------
    t0 = time.perf_counter()
    geo = analyze(bgr, face)
    t.analyze_ms = (time.perf_counter() - t0) * 1000

    # ---------- 3. Принятие решения ----------
    t0 = time.perf_counter()
    dec = decide(geo, forced=(mode == "forced"), th=th)
    t.decide_ms = (time.perf_counter() - t0) * 1000

    preview = _render_analysis(bgr, geo, dec) if with_preview else None
    base = PipelineResult(None, len(faces), geo.as_dict(),
                          {"should_edit": dec.should_edit,
                           "severity": round(dec.severity, 3),
                           "reason": dec.reason},
                          timings=t, analysis_preview=preview)

    if not dec.should_edit:
        base.skipped_reason = dec.reason
        return base
    if analyze_only:
        return base

    # ---------- 4. Редактирование (Z-Image, только quality) ----------
    mask01 = geo.mask.astype(np.float32) / 255.0
    t0 = time.perf_counter()
    with _EDIT_LOCK:
        er: EditResult = edit_quality(bgr, mask01, q, prompt=prompt,
                                      strength_scale=0.7 + 0.6 * dec.severity,
                                      region_bbox=geo.region_bbox)
    t.edit_ms = (time.perf_counter() - t0) * 1000
    base.edited = True
    base.backend = er.backend
    base.seed = er.seed
    if side_by_side:
        base.image = _side_by_side(bgr, er.image)
    else:
        base.image = er.image
    return base


def _render_analysis(bgr: np.ndarray, geo: Geometry, dec: Decision) -> bytes | None:
    """Превью для UI Photoshop: маска красным overlay + точки + вердикт.

    Рисуется на уменьшенной копии (≤ PREVIEW_MAX_SIDE): полноразмерный PNG
    24-Мп кадра весил десятки МБ в base64 и долго кодировался."""
    h, w = bgr.shape[:2]
    s = min(1.0, PREVIEW_MAX_SIDE / max(h, w))
    size = (max(1, int(w * s)), max(1, int(h * s)))
    vis = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA) if s < 1 else bgr.copy()
    m = cv2.resize(geo.mask, size, interpolation=cv2.INTER_AREA) if s < 1 else geo.mask
    a = (m.astype(np.float32) / 255.0 * 0.45)[..., None]
    red = np.array([0, 0, 255], np.float32)
    vis = (vis.astype(np.float32) * (1 - a) + red * a).astype(np.uint8)
    for key in ("chin", "jaw_left", "jaw_right"):
        x, y = geo.debug_points[key]
        cv2.circle(vis, (int(x * s), int(y * s)), 4, (0, 255, 0), -1)
    label = f"EDIT sev={dec.severity:.2f}" if dec.should_edit else "SKIP"
    cv2.putText(vis, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0) if dec.should_edit else (160, 160, 160), 2)
    ok, buf = cv2.imencode(".png", vis)
    return buf.tobytes() if ok else None


def _side_by_side(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    h, w = before.shape[:2]
    sep = np.full((h, 6, 3), 255, np.uint8)
    return np.hstack([before, sep, after])
