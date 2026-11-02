# Double Chin Fix — UXP-плагин Photoshop (Windows / macOS)

Панель `Plugins → Development → Load` (или через UXP Developer Tool: `Plugin → Add Plugin → выбрать эту папку`).

## Требования
1. Photoshop 2023+ (UXP manifest v5).
2. Локально запущен Python-сервер пайплайна (`../server`, порт 8765) — см. корневой README.
3. Z-Image бэкенд:
   * **Windows**: ComfyUI + Tongzean/Z-Image-Turbo-ComfyUI, GPU с ≥12 ГБ VRAM;
   * **macOS**: Apple Silicon с ≥24 ГБ unified memory + MLX-порт модели.

## Что делает панель
* **Режим потока**: `Авто` (решает анализатор) / `Принудительно` (править всегда).
* **Предпросмотр анализа** — показывает маску и метрики до применения.
* **Применить** — полный пайплайн, результат кладётся новым слоем `DoubleChin Fixed`
  (неразрушающая правка; оригинал не трогаем).
* **Пакет** — прогон всех открытых документов.

## Файлы
* `manifest.json` — манифест UXP v5, права на fetch к 127.0.0.1:8765
* `index.html` / `styles.css` — UI панели
* `main.js` — экспорт активного документа → сервер → вставка результата слоем
* `startup.js` — точка входа execution context
