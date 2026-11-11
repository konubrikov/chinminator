"""File jobs for synchronous Photoshop JSX / Image Processor actions."""
import hashlib
import json
import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path

import cv2
from config import QualityParams, Thresholds
from pipeline import process_image

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'work' / 'automation-jobs'
SETTINGS = ROOT / 'work' / 'automation-settings.json'
log = logging.getLogger(__name__)
_LAST_RENDER = None
REUSE_SECONDS = 120

def atomic_text(path, text):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(text, encoding='utf-8')
    pending.replace(path)

def save_settings(mode, thresholds, quality, prompt):
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(SETTINGS, json.dumps(dict(mode=mode, thresholds=asdict(thresholds),
        quality=asdict(quality), prompt=prompt), ensure_ascii=False))

def process_job(folder):
    """Only fixed names within the local job directory are read or written."""
    global _LAST_RENDER
    started = time.monotonic()
    try:
        settings = json.loads((folder / 'settings.json').read_text(encoding='utf-8-sig'))
        image = cv2.imread(str(folder / 'input.png'))
        if image is None:
            raise ValueError('Cannot read job input.png')
        # Adobe runs the same Action for each output format. Reuse only within
        # the same Photoshop document, with identical decoded pixels/settings.
        scope_file = folder/'document-id.txt'
        scope = scope_file.read_text().strip() if scope_file.exists() else ''
        key = hashlib.sha256(image.tobytes()+str(image.shape).encode()+
            json.dumps(settings,sort_keys=True).encode()+scope.encode()).hexdigest()
        reused = bool(scope and _LAST_RENDER and _LAST_RENDER[0]==key and
            started-_LAST_RENDER[1]<REUSE_SECONDS)
        result = _LAST_RENDER[2] if reused else process_image(image, mode=settings['mode'],
            thresholds=Thresholds(**settings['thresholds']),
            quality=QualityParams(**settings['quality']), prompt=settings['prompt'])
        output = result.image if result.image is not None else image
        if output.shape != image.shape:
            raise RuntimeError('Automation result dimensions changed')
        if not cv2.imwrite(str(folder / 'output.pending.png'), output):
            raise RuntimeError('Cannot encode automation result')
        (folder / 'output.pending.png').replace(folder / 'output.png')
        atomic_text(folder / 'result.json', json.dumps(dict(edited=result.edited,
            skipped_reason=result.skipped_reason, seconds=round(time.monotonic()-started, 3),
            width=output.shape[1], height=output.shape[0], backend=result.backend, reused=reused)))
        atomic_text(folder / 'done.txt', 'edited' if result.edited else 'skipped')
        if scope and not reused and result.edited and output.nbytes <= 128*1024*1024:
            _LAST_RENDER = (key,time.monotonic(),result)
        log.info('JSX automation job %s complete (reused=%s)', folder.name,reused)
    except Exception as exc:
        log.exception('JSX automation job failed: %s', folder.name)
        atomic_text(folder / 'error.txt', str(exc))

def worker(stop):
    JOBS.mkdir(parents=True, exist_ok=True)
    # A server restart must surface a failure rather than leave JSX waiting forever.
    for folder in JOBS.iterdir():
        if folder.is_dir() and (folder/'running.txt').exists() and not (folder/'done.txt').exists():
            atomic_text(folder/'error.txt', 'Server restarted during this job; run the Action again')
    while not stop.wait(0.2):
        for request in sorted(JOBS.glob('*/request.txt')):
            if stop.is_set(): return
            folder = request.parent
            if folder.is_symlink() or folder.resolve().parent != JOBS.resolve(): continue
            try:
                request.rename(folder/'running.txt')
            except OSError:
                continue
            process_job(folder)

def start_worker():
    stop = threading.Event()
    thread = threading.Thread(target=worker, args=(stop,), name='photoshop-jsx-jobs', daemon=True)
    thread.start()
    def heartbeat():
        while not stop.is_set():
            SETTINGS.parent.mkdir(parents=True, exist_ok=True)
            atomic_text(ROOT/'work/automation-heartbeat.txt', str(time.time()))
            stop.wait(2)
    threading.Thread(target=heartbeat, name='jsx-heartbeat', daemon=True).start()
    return stop
