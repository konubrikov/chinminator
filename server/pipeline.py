"""Оркестрация пайплайна: Фото → [1 Детекция] → [2 Анализ] → [3 Решение] → [4 Z-Image Edit].

Используется и как Python API (cli.py), и как ядро HTTP-сервера (app.py).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import DEFAULT_THRESHOLDS, QUALITY, Thresholds
from decision import Decision, decide
from edit_zimage import EditResult, edit_quality
from face_detect import detect_faces
from geometry import Geometry, analyze

log = logging.getLogger("pipeline")


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
                  side_by_side: bool = False) -> PipelineResult:
    """Один проход пайплайна по изображению.

    mode: "auto" — сплошной поток (решает анализатор);
          "forced" — правка выполняется всегда.
    """
    assert mode in ("auto", "forced"), f"mode must be auto|forced, got {mode}"
    th = thresholds or DEFAULT_THRESHOLDS
    q = quality or QUALITY
    import time
    t = StepTimings()

    # ---------- 1. Детекция лица ----------
    t0 = time.perf_counter()
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

    preview = _render_analysis(bgr, geo, dec)
    base = PipelineResult(None, len(faces), geo.as_dict(),
                          {"should_edit": dec.should_edit,
                           "severity": round(dec.severity, 3),
                           "reason": dec.reason},
                          timings=t, analysis_preview=preview)

    if not dec.should_edit:
        base.skipped_reason = dec.reason
        return base

    # ---------- 4. Редактирование (Z-Image, только quality) ----------
    mask01 = geo.mask.astype(np.float32) / 255.0
    t0 = time.perf_counter()
    er: EditResult = edit_quality(bgr, mask01, q, prompt=prompt,
                                  strength_scale=0.7 + 0.6 * dec.severity)
    t.edit_ms = (time.perf_counter() - t0) * 1000
    base.edited = True
    base.backend = er.backend
    base.seed = er.seed
    if side_by_side:
        base.image = _side_by_side(bgr, er.image)
    else:
        base.image = er.image
    return base


def _render_analysis(bgr: np.ndarray, geo: Geometry, dec: Decision) -> bytes:
    """Превью для UI Photoshop: маска красным overlay + точки + вердикт."""
    vis = bgr.copy()
    red = np.zeros_like(vis)
    red[:, :, 2] = 255                       # BGR красный
    m = (geo.mask[..., None] / 255.0) * 0.45
    vis = (vis * (1 - m) + red * m).astype(np.uint8)
    for key in ("chin", "jaw_left", "jaw_right"):
        x, y = map(int, geo.debug_points[key])
        cv2.circle(vis, (x, y), 4, (0, 255, 0), -1)
    label = f"EDIT sev={dec.severity:.2f}" if dec.should_edit else "SKIP"
    cv2.putText(vis, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0) if dec.should_edit else (160, 160, 160), 2)
    ok, buf = cv2.imencode(".png", vis)
    return buf.tobytes() if ok else None


def _side_by_side(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    h, w = before.shape[:2]
    sep = np.full((h, 6, 3), 255, np.uint8)
    return np.hstack([before, sep, after])
