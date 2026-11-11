"""Download and validate the small analysis models during setup, not first retouch."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'server'))
import numpy as np
from face_detect import _ensure_model
from skin_segment import segment_skin
if not _ensure_model():raise RuntimeError('Face landmark model download failed')
segment_skin(np.zeros((64,64,3),np.uint8))
print('Face and skin analysis models are ready.')
