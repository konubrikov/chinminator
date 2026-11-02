"""Конфигурация пайплайна Double Chin Corrector."""
from dataclasses import dataclass


@dataclass
class Thresholds:
    """Пороги шага 3 (принятие решения). Значения — доли от размера лица."""
    submental_ratio: float = 0.12   # выпуклость подподбородочной дуги
    jaw_collar_gap: float = 0.05    # касание «воротника»/низа кадра линией челюсти
    neck_shadow_depth: float = 0.22 # глубина тени под подбородком (LAB L)
    min_face_px: int = 90           # faces мельче не анализируем


DEFAULT_THRESHOLDS = Thresholds()

EDIT_PROMPT_QUALITY = (
    "Remove the double chin and submental fat, slim and smooth the under-jaw area, "
    "define a single clean jawline and natural neck contour. Keep the person's identity, "
    "skin texture, lighting, pose, hair, clothing and background exactly unchanged."
)

NEGATIVE_PROMPT = (
    "blurry skin, plastic doll face, changed identity, warped ears, extra chin, "
    "smudged neck, artifacts, oversmoothed, deformed jaw"
)


@dataclass
class QualityParams:
    """Параметры качественного режима Z-Image-Turbo (реализован единственный)."""
    steps: int = 8
    cfg: float = 2.0
    denoise_strength: float = 0.55   # умеренный: сохраняет текстуру кожи
    mask_blur_px: int = 12           # мягкая граница инpaint-зоны
    max_side: int = 1536             # длинная сторона входного кадра
    seed: int = -1                   # -1 → случайный


QUALITY = QualityParams()
