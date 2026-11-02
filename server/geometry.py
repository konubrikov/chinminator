"""Шаг 2. Анализ геометрии подбородка/челюсти + построение инpaint-маски.

Метрики (все нормированы на face_h — устойчивы к масштабу и дистанции):

* submental_ratio — насколько нижний контур лица («дуга» от угла челюсти до
  угла челюсти через подбородок) выпирает ВНИЗ за эталонную линию, проведённую
  между точками JawFront (угол челюсти + 30% к подбородку).Double chin даёт
  дополнительную выпуклость под костным подбородком → ratio растёт.

* jaw_collar_gap  — вертикальный зазор между самой нижней точкой контура лица
  и «воротником» (границей головы/шеи с фоном внизу bbox). При выраженном
  втором подбородке контур лица сливается с шеей/плечами → gap мал или отрицателен.

* neck_shadow_depth — контраст тени сразу под подбородком относительно щеки
  (LAB L-канал). Второй подбородок обычно отбрасывает тень в складку.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from face_detect import (Face, IDX_CHIN, IDX_JAW_LEFT, IDX_JAW_RIGHT,
                         SUBMENTAL_ARC)


@dataclass
class Geometry:
    submental_ratio: float
    jaw_collar_gap: float
    neck_shadow_depth: float
    mask: np.ndarray          # uint8 0..255, размер всего кадра
    region_bbox: tuple[int, int, int, int]  # bbox маски (x1,y1,x2,y2)
    debug_points: dict        # точки для превью анализа

    def as_dict(self) -> dict:
        return {
            "submental_ratio": round(float(self.submental_ratio), 4),
            "jaw_collar_gap": round(float(self.jaw_collar_gap), 4),
            "neck_shadow_depth": round(float(self.neck_shadow_depth), 4),
        }


def analyze(bgr: np.ndarray, face: Face) -> Geometry:
    h, w = bgr.shape[:2]
    pts = face.landmarks
    fh = face.face_h

    # ---------- 1. submental_ratio ----------
    jaw_l, jaw_r, chin = pts[IDX_JAW_LEFT], pts[IDX_JAW_RIGHT], pts[IDX_CHIN]
    arc = pts[SUBMENTAL_ARC]

    # Эталонная «костная» линия: от точек чуть выше углов челюсти к подбородку.
    # Строим её как ломаную jaw_l→chin→jaw_r, смещённую вверх на 0.12*fh,
    # и измеряем среднее расстояние дуги реального контура ДОЛЖНО быть ~0
    # у идеального профиля; выпирание вниз = второй подбородок.
    # Эталонная «костная» дуга: коническая кривая (квадратичная кривая Безье)
    # с концевыми точками в углах челюсти и контрольной точкой чуть выше
    # подбородка. У лица без второго подбородка реальный контур лежит НА этой
    # дуге или ВЫШЕ неё; провисание вниз за её пределы = мягкие ткани.
    # Контрольная точка подобрана так, чтобы кривая проходила через уровень
    # костного подбородка при t=0.5: B(0.5) = 0.25*jl + 0.5*ctrl + 0.25*jr.
    # (Раньше ctrl ставился у подбородка, и вершина кривой оказывалась на
    #  полпути к углам челюсти — почти любое лицо «провисало» ниже эталона.)
    chin_ref = np.array([chin[0], chin[1]], dtype=np.float64)
    ctrl = 2.0 * chin_ref - 0.5 * (np.asarray(jaw_l, np.float64) +
                                   np.asarray(jaw_r, np.float64))

    # Для каждой точки дуги — signed vertical distance ниже соответствующей
    # точки эталонной ломаной (в системе, повёрнутой против roll головы).
    rot = _rotator(-face.roll_deg, center=chin)
    arc_r = rot(arc)
    jl_r, jr_r, ct_r = rot(jaw_l)[0], rot(jaw_r)[0], rot(ctrl)[0]

    t = np.linspace(0.0, 1.0, len(arc_r))[:, None]
    ref = ((1 - t) ** 2) * jl_r + 2 * (1 - t) * t * ct_r + (t ** 2) * jr_r
    d = arc_r[:, 1] - ref[:, 1]          # >0 ⇒ точка НИЖЕ эталона (y вниз)
    below = d[d > 0]
    submental_ratio = float(below.mean()) / fh if below.size else 0.0

    # ---------- 2. jaw_collar_gap ----------
    lowest_arc_y = float(arc[:, 1].max())
    x1, y1, x2, y2 = face.bbox
    collar_y = _detect_collar_line(bgr, face)   # верх плеч/воротника
    jaw_collar_gap = float(collar_y - lowest_arc_y) / fh

    # ---------- 3. neck_shadow_depth ----------
    neck_shadow_depth = _shadow_depth(bgr, pts, fh)

    # ---------- маска зоны редактирования ----------
    mask, region_bbox = build_mask(bgr_shape=(h, w), face=face,
                                   submental_ratio=submental_ratio)

    dbg = {
        "chin": chin.tolist(),
        "jaw_left": jaw_l.tolist(),
        "jaw_right": jaw_r.tolist(),
        "arc_lowest_y": lowest_arc_y,
        "collar_y": float(collar_y),
    }
    return Geometry(submental_ratio, jaw_collar_gap, neck_shadow_depth,
                    mask, region_bbox, dbg)


def build_mask(bgr_shape: tuple[int, int], face: Face,
               submental_ratio: float = 0.0) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Инpaint-маска: подподбородочная зона + часть шеи.

    Форма — «полумесяц» между дугой контура лица и областью ниже,
    расширяется пропорционально силе второго подбородка. Верхняя граница —
    линия jaw_l→chin→jaw_r (с небольшим подъёмом), нижняя — на 0.75*fh ниже
    подбородка. Лицо ВЫШЕ линии челюсти не затрагивается.
    """
    h, w = bgr_shape
    pts = face.landmarks
    fh = face.face_h
    jaw_l, jaw_r, chin = pts[IDX_JAW_LEFT], pts[IDX_JAW_RIGHT], pts[IDX_CHIN]

    ext = 1.0 + min(max(submental_ratio, 0.0), 0.35) * 1.5  # сильнее ЧП → шире зона
    top_lift = 0.04 * fh

    # Полигон: верхняя граница по дуге челюсти (чуть выше),
    # нижняя — горизонтальная линия ниже подбородка.
    arc = pts[SUBMENTAL_ARC]
    upper = [(int(p[0]), int(p[1] - top_lift)) for p in arc]
    y_bot = int(min(h - 1, chin[1] + 0.75 * fh * ext))
    x_lo = int(max(0, min(p[0] for p in upper) - 0.08 * fh))
    x_hi = int(min(w - 1, max(p[0] for p in upper) + 0.08 * fh))

    poly = np.array(upper + [(x_hi, y_bot), (x_lo, y_bot)], dtype=np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly], 255)

    # Мягкое перо по границам, чтобы диффузия плавноBlend-ила результат
    blur = max(5, int(0.06 * fh)) | 1
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)

    ys, xs = np.where(mask > 8)
    if len(xs) == 0:
        return mask, (x_lo, int(chin[1]), x_hi, y_bot)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return mask, bbox


# ---------------------------------------------------------------- helpers

def _rotator(deg: float, center: np.ndarray):
    a = np.deg2rad(deg)
    ca, sa = np.cos(a), np.sin(a)
    c = np.asarray(center, dtype=np.float64).ravel()

    def f(p):
        p = np.atleast_2d(np.asarray(p, dtype=np.float64)) - c
        out = np.empty_like(p)
        out[:, 0] = p[:, 0] * ca - p[:, 1] * sa
        out[:, 1] = p[:, 0] * sa + p[:, 1] * ca
        return out + c
    return f


def _detect_collar_line(bgr: np.ndarray, face: Face) -> float:
    """Оценивает y-координату «воротника» (плечи/шея→фон) сканом яркости
    под лицом: первая снизу строка bbox, где горизонтальный градиент резко
    меняется (граница тело/фон). Fallback — нижняя граница bbox."""
    x1, y1, x2, y2 = face.bbox
    chin_y = face.landmarks[IDX_CHIN][1]
    start = int(min(max(chin_y + 0.15 * face.face_h, y1), y2 - 2))
    region = bgr[start:y2, x1:x2]
    if region.size == 0 or region.shape[0] < 4:
        return float(y2)
    strip = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    if strip.shape[0] < 4:
        return float(y2)
    grad = np.abs(cv2.Sobel(strip, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
    thr = grad.mean() + 1.5 * grad.std()
    hits = np.where(grad > thr)[0]
    if len(hits) == 0:
        return float(y2)
    return float(start + hits[0])


def _shadow_depth(bgr: np.ndarray, pts: np.ndarray, fh: float) -> float:
    """Контраст L-канала LAB: средняя яркость тени под подбородком vs щека.

    В LAB переводятся только три маленьких патча, а не весь кадр
    (на 24-Мп фото это экономит ~100 МБ памяти и сотни миллисекунд)."""
    h, w = bgr.shape[:2]
    chin = pts[IDX_CHIN]
    cheek_l = pts[50]     # левая щека
    cheek_r = pts[280]    # правая щека
    r = max(3, int(0.06 * fh))

    def patch_mean(cx: float, cy: float) -> float:
        x0, x1i = int(cx - r), int(cx + r)
        y0, y1i = int(cy - r), int(cy + r)
        if x0 < 0 or y0 < 0 or x1i > w or y1i > h:
            return float("nan")          # патч вылез за кадр
        patch = cv2.cvtColor(bgr[y0:y1i, x0:x1i], cv2.COLOR_BGR2LAB)
        return float(patch[..., 0].mean()) / 255.0

    shadow = patch_mean(chin[0], chin[1] + 0.28 * fh)
    cheeks = [v for v in (patch_mean(*cheek_l), patch_mean(*cheek_r))
              if not np.isnan(v)]
    if np.isnan(shadow) or not cheeks:
        return 0.0
    skin = float(np.mean(cheeks))
    if skin <= 1e-3:
        return 0.0
    return float(np.clip((skin - shadow) / skin, 0.0, 1.0))
