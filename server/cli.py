"""CLI для пакетной обработки папки с фото без Photoshop.

Примеры:
  python cli.py input_dir output_dir --mode auto
  python cli.py photo.jpg out.jpg --mode forced --sidebyside
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from config import DEFAULT_THRESHOLDS, QUALITY
from pipeline import process_image

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


def load(path: Path) -> np.ndarray | None:
    # cv2.imread не читает юникод-пути на Windows — обходим через fromfile
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def save(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix if path.suffix != ".jpg" else ".jpg", img,
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        ok, buf = cv2.imencode(".png", img)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description="Double Chin Corrector CLI")
    ap.add_argument("input", help="файл или папка с фотографиями")
    ap.add_argument("output", help="файл или папка для результатов")
    ap.add_argument("--mode", choices=["auto", "forced"], default="auto",
                    help="auto — сплошной поток (решает анализатор); "
                         "forced — править каждое фото")
    ap.add_argument("--sidebyside", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="только шаги 1–3, без Z-Image")
    ap.add_argument("--report", default=None, help="JSON-отчёт по потоку")
    args = ap.parse_args()

    inp, outp = Path(args.input), Path(args.output)
    files = ([inp] if inp.is_file()
             else sorted(p for p in inp.rglob("*") if p.suffix.lower() in EXTS))
    if not files:
        print("Вход не найден", file=sys.stderr)
        return 2

    report = []
    edited = skipped = failed = 0
    for i, f in enumerate(files, 1):
        img = load(f)
        if img is None:
            print(f"[{i}/{len(files)}] {f.name}: НЕ ЧИТАЕТСЯ"); failed += 1
            continue
        try:
            res = process_image(img, mode=("forced" if args.mode == "forced" else "auto"),
                                thresholds=DEFAULT_THRESHOLDS,
                                quality=None if args.dry_run else QUALITY,
                                side_by_side=args.sidebyside)
        except RuntimeError as e:
            # нет Z-Image бэкенда — при dry-run не критично
            if args.dry_run:
                res = process_image(img, mode=args.mode,
                                    thresholds=DEFAULT_THRESHOLDS, quality=None)
            else:
                print(f"[{i}/{len(files)}] {f.name}: ОШИБКА РЕДАКТИРОВАНИЯ: {e}")
                failed += 1
                continue

        out_file = outp if inp.is_file() else outp / f.name
        if res.edited and res.image is not None:
            save(out_file.with_suffix(".png" if args.sidebyside else out_file.suffix),
                 res.image)
            edited += 1
            tag = f"EDIT sev={res.decision.get('severity')}"
        else:
            if not args.dry_run and inp.is_file():
                pass
            elif not args.dry_run:
                save(out_file, img)          # в поток «как есть»
            skipped += 1
            tag = f"SKIP ({res.skipped_reason or 'no edit'})"
        print(f"[{i}/{len(files)}] {f.name}: {tag}  metrics={res.geometry}")
        report.append({"file": str(f), "edited": res.edited,
                       "metrics": res.geometry, "decision": res.decision,
                       "timings_ms": vars(res.timings)})

    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nГотово: edited={edited} skipped={skipped} failed={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
