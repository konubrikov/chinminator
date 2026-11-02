# Double Chin Corrector — плагин для Photoshop (Win/Mac) + сервер Z-Image Edit

Пайплайн:

```
Фото → [1. Детекция лица] → [2. Анализ геометрии] → [3. Принятие решения]
     → [4. Редактирование области лица (Z-Image Edit, quality mode)] → Результат
```

Два режима потока:
* **auto**  — сплошной поток фотографий: решение «править или нет» принимает анализатор по метрикам.
* **forced**— поток, где правка нужна точно: шаг 3 пропускает порог и всегда отправляет кадр в редактирование.

## Архитектура

```
┌─────────────────────────── Photoshop (Win / macOS) ───────────────────────────┐
│  UXP-плагин (plugin/)                                                          │
│  • File > Automate / фильтр «Double Chin Fix…»                                 │
│  • Open Image dialog → события selection / open                               │
│  • Экспорт временного JPEG → HTTP POST на локальный сервер                     │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │ localhost:8765 (HTTP JSON/multipart)
┌───────────────────────────────────────▼───────────────────────────────────────┐
│  server/ (Python 3.10+, Windows: CUDA ≥12 ГБ VRAM, macOS: Apple Silicon ≥24 ГБ)│
│  pipeline.py — шаги 1–4                                                        │
│   1. face_detect.py   — MediaPipe FaceMesh (468 точек)                         │
│   2. geometry.py      — метрики подбородка/челюсти                             │
│   3. decision.py      — эвристика решения (auto) / forced                      │
│   4. edit_zimage.py   — диффузионное редактирование инpaint-области            │
│  app.py               — FastAPI эндпоинты                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Быстрый старт (сервер)

```bash
cd server
python -m venv .venv && source .venv/bin/activate   # Win: .venv\Scripts\activate
pip install -r requirements.txt

# Windows (CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
huggingface-cli download Tongzean/Z-Image-Turbo-ComfyUI --filename "z-image-turbo-Q8.gguf" --local-dir models

# macOS (MLX, 24+ ГБ unified memory):
pip install mlx-lm
huggingface-cli download Tongzean/Z-Image-Turbo-MLX --local-dir models/z-image-turbo-mlx

uvicorn app:app --host 127.0.0.1 --port 8765
```

Модель: **Tongzean/Z-Image-Turbo** (Apache-2.0). Режим `quality` = 8 шагов, CFG 2.0,
полноразмерный латент (без downscaled pass); режим `fast` оставлен как заглушка —
согласно ТЗ реализован только качественный.

## Плагин Photoshop

`plugin/` — UXP-манифест v5. Установка: Photoshop → Plugins → Development →
загрузить папку (или пакет через UXP Developer Tool). См. `plugin/README.md`.

## Метрики анализа геометрии (шаг 2)

| Метрика | Смысл | Порог default |
|---|---|---|
| `submental_ratio`    | выпуклость подподбородочной дуги относительно линии челюсть→подбородок | 0.12 |
| `jaw_collar_gap`     | зазор между линией челюсти и «воротником» головы (низ кадра)          | 0.05 |
| `neck_shadow_depth`  | глубина тени под подбородком (LAB L-канал)                              | 0.22 |

Решение (auto): правим, если `submental_ratio` выше порога И подтверждается хотя бы
одна из двух остальных метрик. Пороги настраиваются в UI плагина и в `config.py`.

## Эндпоинты

* `GET  /health` — статус, GPU/backend, модель
* `POST /analyze` — шаги 1–3, возвращает метрики + bbox + mask-preview + вердикт
* `POST /process` — полный пайплайн, возвращает PNG с результатом (side-by-side опционально)
