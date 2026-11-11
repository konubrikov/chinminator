"""The model is selected at process startup, never midway through a batch."""
import json
import os
from pathlib import Path
from config import EDIT_PROMPT_QUALITY

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'hardware-profile.json'
PROFILE_DATA = json.loads(PROFILE.read_text('utf-8')) if PROFILE.exists() else {}
ENGINE = os.environ.get('CHINMINATOR_MODEL', PROFILE_DATA.get('engine', 'qwen2'))
ADAPTERS = json.loads(Path(__file__).with_name('qwen_adapters.json').read_text('utf-8'))
ADAPTER = ADAPTERS.get(ENGINE)
if ENGINE not in ('qwen2', 'flux3', *ADAPTERS):
    raise ValueError('Unknown CHINMINATOR_MODEL: ' + ENGINE)
MODEL_LABEL = 'FLUX 3 Image Klein' if ENGINE == 'flux3' else 'Qwen-Image 2.1'
if ADAPTER:
    MODEL_LABEL = ADAPTER['label']
FLUX_PROMPT = ('Remove the entire double chin and submental fat pouch, reconstruct a single lean natural jawline and neck. '
               'Retain the same person and exact pose. Preserve the upper face, mouth position, clothing neckline and jewelry. '
               'Match the existing lighting direction, softness, skin hue and exposure. Keep the framing and background unchanged.')
DEFAULT_PROMPT = FLUX_PROMPT if ENGINE == 'flux3' else EDIT_PROMPT_QUALITY

def backend_class():
    if ENGINE == 'flux3':
        from backends.flux3_backend import Flux3Backend
        return Flux3Backend
    from backends.comfyui_backend import ComfyUIBackend, DistilledQwenBackend
    return DistilledQwenBackend if ADAPTER else ComfyUIBackend
