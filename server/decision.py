"""Шаг 3. Принятие решения: править ли второй подбородок.

Два режима потока:
* mode="auto"   — сплошной поток: решение по метрикам и порогам.
* mode="forced" — поток, где правка точно нужна: вердикт всегда YES
                  (метрики всё равно считаются — они управляют размером маски).
"""
from __future__ import annotations

from dataclasses import dataclass

from config import Thresholds
from geometry import Geometry


@dataclass
class Decision:
    should_edit: bool
    severity: float        # 0..1 — силаsecondого подбородка → сила denoise
    reason: str
    metrics: dict


def decide(geo: Geometry, *, forced: bool, th: Thresholds) -> Decision:
    m = geo.as_dict()
    if forced:
        sev = _severity(geo, th)
        return Decision(True, sev, "forced stream", m)

    primary = geo.submental_ratio >= th.submental_ratio
    support = (geo.jaw_collar_gap <= th.jaw_collar_gap or
               geo.neck_shadow_depth >= th.neck_shadow_depth)
    if primary and support:
        return Decision(True, _severity(geo, th),
                        "submental bulge confirmed by gap/shadow", m)
    return Decision(False, 0.0, "no double chin detected", m)


def _severity(geo: Geometry, th: Thresholds) -> float:
    """Нормируем силу: ratio над порогом + вклад тени."""
    over = max(geo.submental_ratio - th.submental_ratio * 0.5, 0.0)
    sev = over / (th.submental_ratio * 2.0)
    sev += 0.25 * max(geo.neck_shadow_depth - th.neck_shadow_depth, 0.0)
    return float(min(max(sev, 0.15), 1.0))
