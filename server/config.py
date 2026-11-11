"""Конфигурация пайплайна Chinminator."""
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
    "Rebuild the complete chin and adjoining neck as one anatomically natural contour, with no remaining double chin, old fold or duplicate jaw edge. Preserve the original light source direction, softness, white balance and contrast. Create physically plausible new shadows for the corrected anatomy; do not retain the old submental fold shadow. "
    "Remove the double chin and submental fat, slim and smooth the under-jaw area, "
    "define a single clean jawline and natural neck contour. Keep the person's identity, "
    "skin texture, lighting, pose, hair, clothing and background exactly unchanged."
    " Keep the upper facial outline and all unedited facial features at exactly the same pixel coordinates as the input. Do not translate, rotate, scale, zoom, or reframe the image. Keep the pixels along the entire outer perimeter of the image crop unchanged and perfectly aligned with the input. Change only the lower jaw, under-chin area and neck; connect the corrected contour smoothly to the unchanged upper face without double edges, ghost outlines or seams."
)

NEGATIVE_PROMPT = (
    "blurry skin, plastic doll face, changed identity, warped ears, extra chin, "
    "smudged neck, artifacts, oversmoothed, deformed jaw"
)


@dataclass
class QualityParams:
    """Параметры качественного режима Qwen-Image 2.1 (реализован единственный)."""
    auto_quality: int = 1          # 1: choose settings from the actual context crop
    native_resolution: int = 1     # preserve crop pixels; pad to model grid only
    steps: int = 28
    cfg: float = 1.0
    denoise_strength: float = 0.75   # legacy API field; correction is always maximum
    mask_blur_px: int = 12           # мягкая граница инpaint-зоны
    max_side: int = 1024             # used only with explicit native_resolution=0
    seed: int = -1                   # -1 → случайный


QUALITY = QualityParams()
