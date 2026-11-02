# Chinminator — плагин для Photoshop (Win/Mac) + сервер Z-Image Edit

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

### Windows — скрипт подготовки окружения (рекомендуется)

Требования: Python 3.10+ (https://python.org), Git for Windows, NVIDIA GPU с CUDA и ≥12 ГБ VRAM.

```powershell
# 1. Настройка окружения (venv + зависимости + PyTorch cu124 + ComfyUI + модели)
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
#    флаги: -SkipModels (не качать Z-Image gguf)   -Force (пересоздать venv)

# 2. Запуск сервера (поднимает ComfyUI на :8188 и Chinminator API на :8765)
scripts\start_server.bat

# 3. Пакетная обработка без Photoshop
scripts\cli.bat C:\photos C:\photos_fixed --mode auto
```

### macOS — скрипт подготовки окружения

Требования: Apple Silicon (M1+), ≥24 ГБ unified RAM, Xcode CLT (`xcode-select --install`).

```bash
bash scripts/setup_macos.sh      # venv + зависимости + MLX + модели (SKIP_MODELS=1 — без моделей)
./scripts/start_server.sh        # Chinminator API на http://127.0.0.1:8765
```

### Portable-режим (Windows)

Вся система лежит внутри папки проекта (`server\.venv`, `vendor\ComfyUI`, модели) —
ничего не пишется в систему, папку можно переносить между машинами.

```powershell
# Собрать самодостаточный zip-архив (со включённым .venv, ComfyUI и моделями):
powershell -ExecutionPolicy Bypass -File scripts\make_portable.ps1
#    -ExcludeModels — лёгкий архив без гигабайтных моделей (скачаются при первом запуске setup)

# На целевой машине: распакуйте в любую папку и запустите
scripts\start_server.bat --portable
```

Ручной запуск плагина из portable-папки: Photoshop → Plug-ins & Automation → загрузить
`plugin\manifest.json` через UXP Developer Tool; URL сервера оставить `http://127.0.0.1:8765`.

<details>
<summary>Ручная установка сервера (без скриптов)</summary>

```bash
cd server
python -m venv .venv && source .venv/bin/activate   # Win: .venv\Scripts\activate
pip install -r requirements.txt

# Windows (CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
huggingface-cli download Tongzean/Z-Image-Turbo-ComfyUI --local-dir ../vendor/ComfyUI/models/diffusion_models

# macOS (MLX, 24+ ГБ unified memory):
pip install mlx mlx-lm
huggingface-cli download Tongzean/Z-Image-Turbo-MLX --local-dir models/z-image-turbo-mlx

uvicorn app:app --host 127.0.0.1 --port 8765
```

</details>

Модель: **Tongzean/Z-Image-Turbo** (Apache-2.0). Режим `quality` = 8 шагов, CFG 2.0,
полноразмерный латент (без downscaled pass); режим `fast` оставлен как заглушка —
согласно ТЗ реализован только качественный.

---

## Установка в Photoshop (плагин)

1. Запустите сервер (`scripts\start_server.bat` или `./scripts/start_server.sh`).
2. Установите **UXP Developer Tool** (https://github.com/Adobe-CEP/UXPTooling).
3. В DevTool: Add Plugin → выберите `plugin/manifest.json` → Load.
   Либо: Photoshop → Plugins → Development Plugins.
4. Панель «Chinminator» появится в Window → Extensions. URL сервера настраивается в панели (по умолчанию http://127.0.0.1:8765).

## Структура проекта

```
chinminator/
├── server/            # Python-ядро: шаги 1–4, FastAPI, бэкенды Z-Image
├── plugin/            # UXP-панель Photoshop v5 (Win/Mac)
└── scripts/           # подготовка и запуск
    ├── setup_windows.ps1    # установка окружения Windows (venv, torch cu124, ComfyUI, модели)
    ├── setup_macos.sh       # установка окружения macOS (venv, MLX, модели)
    ├── start_server.bat     # запуск сервера Win (+ComfyUI), флаг --portable
    ├── start_server.sh      # запуск сервера macOS
    ├── cli.bat              # пакетная обработка без Photoshop
    └── make_portable.ps1    # сборка portable zip-архива для Windows
```

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
