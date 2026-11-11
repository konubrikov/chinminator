"""Local semantic skin segmentation (MediaPipe SelfieMulticlass)."""
import os
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite"
MODEL_PATH = Path(os.environ.get("DC_MODELS_DIR", str(Path(__file__).parent / "models"))) / "selfie_multiclass_256x256.tflite"
_LOCK = threading.Lock()
_segmenter = None


def segment_skin(bgr):
    """Return class IDs: 0 background, 1 hair, 2 body skin, 3 face, 4 clothes."""
    global _segmenter
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    with _LOCK:
        try:
            if _segmenter is None:
                if not MODEL_PATH.exists():
                    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                    tmp = MODEL_PATH.with_suffix('.download')
                    with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
                        tmp.write_bytes(response.read())
                    tmp.replace(MODEL_PATH)
                _segmenter = vision.ImageSegmenter.create_from_options(
                    vision.ImageSegmenterOptions(
                        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
                        running_mode=vision.RunningMode.IMAGE,
                        output_category_mask=True, output_confidence_masks=False))
            h,w = bgr.shape[:2]
            scale = min(1.0, 1024 / max(h,w))
            small = cv2.resize(bgr, (max(1,round(w*scale)), max(1,round(h*scale))))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small,cv2.COLOR_BGR2RGB))
            labels = _segmenter.segment(image).category_mask.numpy_view().copy()
            return cv2.resize(labels,(w,h),interpolation=cv2.INTER_NEAREST)
        except Exception as exc:
            raise RuntimeError(f"Не удалось построить маску кожи: {exc}") from exc
