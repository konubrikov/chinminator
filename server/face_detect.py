"""Шаг 1. Детекция лица и ключевых точек (468 landmarks FaceMesh).

Кроссплатформенно (Win/Mac) через MediaPipe Tasks API:
* основной путь — `mediapipe.tasks.python.vision.FaceLandmarker`
  (модель face_landmarker.task скачивается автоматически в models/);
* совместимость со старым `mp.solutions.face_mesh` (mediapipe ≤0.10),
  если tasks-модель недоступна.

Оба пути возвращают одни и те же 468 канонических индексов.
"""
from __future__ import annotations

import logging
import math
import os
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("face_detect")

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_DIR = os.environ.get(
    "DC_MODELS_DIR", os.path.join(os.path.dirname(__file__), "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

# Индексы FaceMesh (канонические, одинаковы в solutions и tasks)
IDX_CHIN = 15                  # кончик подбородка
IDX_JAW_LEFT = 172             # левый угол линии челюсти
IDX_JAW_RIGHT = 397            # правый угол линии челюсти
IDX_CHEEK_LEFT = 234           # левая граница овала лица
IDX_CHEEK_RIGHT = 454          # правая граница овала лица
IDX_EYE_LEFT = 263
IDX_EYE_RIGHT = 133
# Дуга нижнего контура лица между углами челюсти через подбородок
SUBMENTAL_ARC = [172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182,
                 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193,
                 194, 195, 196, 197, 198, 200, 425]


@dataclass
class Face:
    """Результат детекции одного лица в пиксельных координатах кадра."""
    landmarks: np.ndarray                # (468, 2) float32, px
    bbox: tuple[int, int, int, int]      # x1, y1, x2, y2 (лицо + запас на шею)
    roll_deg: float                      # наклон головы по линии глаз
    face_h: float                        # высота лица px (глаза→подбородок *2)
    score: float = 1.0


def _ensure_model() -> bool:
    if os.path.exists(MODEL_PATH):
        return True
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        log.info("Скачиваю face_landmarker.task …")
        tmp = MODEL_PATH + ".tmp"
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, MODEL_PATH)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("Не удалось скачать модель FaceLandmarker: %s", e)
        return False


_landmarker = None  # переиспользуем экземпляр (дорогая инициализация)


def detect_faces(bgr: np.ndarray) -> list[Face]:
    """Возвращает список лиц (по убыванию размера). Пусто — если лиц нет."""
    h, w = bgr.shape[:2]
    pts_list = _detect_landmarks(bgr)
    faces = [_make_face(pts, (w, h)) for pts in pts_list]
    faces.sort(key=lambda f: f.face_h, reverse=True)
    return faces


def _detect_landmarks(bgr: np.ndarray) -> list[np.ndarray]:
    global _landmarker

    # ---- путь 1: MediaPipe Tasks ----
    if _landmarker is None and _ensure_model():
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=base, running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=4,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5)
            _landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        except Exception as e:  # noqa: BLE001
            log.warning("Tasks FaceLandmarker недоступен: %s", e)
            _landmarker = False  # не повторяем попытку

    if _landmarker:
        import mediapipe as mp
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = _landmarker.detect(mimg)
        out = []
        for lms in res.face_landmarks:
            pts = np.array([(lm.x * bgr.shape[1], lm.y * bgr.shape[0])
                            for lm in lms], dtype=np.float32)
            out.append(pts)
        return out

    # ---- путь 2: legacy solutions (mediapipe <= 0.10) ----
    try:
        import mediapipe as mp
        with mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=4,
                refine_landmarks=False, min_detection_confidence=0.5) as mesh:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)
            if not res.multi_face_landmarks:
                return []
            return [np.array([(lm.x * bgr.shape[1], lm.y * bgr.shape[0])
                              for lm in fm.landmark], dtype=np.float32)
                    for fm in res.multi_face_landmarks]
    except AttributeError as e:
        raise RuntimeError(
            "MediaPipe недоступен ни через Tasks, ни через solutions. "
            "pip install 'mediapipe>=0.10.14'") from e


def _make_face(pts: np.ndarray, frame_wh: tuple[int, int]) -> Face:
    w, h = frame_wh
    chin = pts[IDX_CHIN]
    eye_l, eye_r = pts[IDX_EYE_LEFT], pts[IDX_EYE_RIGHT]
    roll = math.degrees(math.atan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0]))

    eye_y = float((eye_l[1] + eye_r[1]) / 2)
    face_h = max((chin[1] - eye_y) * 2.0, 8.0)

    x_min = float(min(pts[:, 0].min(), pts[IDX_CHEEK_LEFT][0], pts[IDX_CHEEK_RIGHT][0]))
    x_max = float(max(pts[:, 0].max(), pts[IDX_CHEEK_LEFT][0], pts[IDX_CHEEK_RIGHT][0]))
    y_min = float(pts[:, 1].min())
    # вниз добавляем запас на шею/подбородок — зона интереса пайплайна
    x1 = int(max(0, x_min - 0.10 * face_h))
    x2 = int(min(w - 1, x_max + 0.10 * face_h))
    y1 = int(max(0, y_min - 0.15 * face_h))
    y2 = int(min(h - 1, chin[1] + 0.90 * face_h))
    return Face(landmarks=pts, bbox=(x1, y1, x2, y2),
                roll_deg=roll, face_h=face_h)
