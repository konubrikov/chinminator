# ============================================================================
#  Chinminator — подготовка окружения macOS (Apple Silicon, >= 24 ГБ unified RAM)
#  Запуск:  bash scripts/setup_macos.sh
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="$ROOT/server"
VENV="$SERVER/.venv"
PY="$VENV/bin/python"
SKIP_MODELS="${SKIP_MODELS:-0}"

echo "=== Chinminator: настройка macOS ==="
echo "Корень проекта: $ROOT"

# 1. Проверка архитектуры / памяти
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "ВНИМАНИЕ: MLX-бэкенд требует Apple Silicon (M1+). На Intel используйте diffusers/CPU (медленно)." >&2
fi

# 2. Python 3.10+
BASE_PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then BASE_PY="$c"; break; fi
done
if [[ -z "$BASE_PY" ]]; then
  echo "НЕ НАЙДЕН Python 3.10+. Установите: brew install python@3.12" >&2
  exit 1
fi
echo "Python: $(command -v $BASE_PY)"

# 3. venv + зависимости ядра
[[ -d "$VENV" ]] || "$BASE_PY" -m venv "$VENV"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$SERVER/requirements.txt"

# 4. MLX-бэкенд Z-Image-Turbo
"$PY" -m pip install mlx mlx-lm "huggingface_hub[cli]" || {
  echo "Не удалось поставить MLX-пакеты; шаг 4 на mac может быть недоступен." >&2
}

# 5. Модели
# 5a. FaceMesh landmarker
mkdir -p "$SERVER/models"
FACE_MODEL="$SERVER/models/face_landmarker.task"
if [[ ! -f "$FACE_MODEL" ]]; then
  echo "Скачивание face_landmarker.task ..."
  curl -L -o "$FACE_MODEL" \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
fi

# 5b. Z-Image-Turbo MLX
if [[ "$SKIP_MODELS" != "1" ]]; then
  echo "Z-Image-Turbo (MLX): скачивание Tongzean/Z-Image-Turbo-MLX (несколько ГБ)..."
  echo "Если модель gated — выполните: huggingface-cli login"
  "$PY" -m huggingface_hub.commands.huggingface_cli download \
    Tongzean/Z-Image-Turbo-MLX --local-dir "$SERVER/models/z-image-turbo-mlx" || {
    echo "Автозагрузка не удалась — скачайте вручную с https://huggingface.co/Tongzean/Z-Image-Turbo-MLX" >&2
    echo "в каталог: $SERVER/models/z-image-turbo-mlx" >&2
  }
fi

echo ""
echo "Готово! Запуск сервера:  ./scripts/start_server.sh   (или .venv: uvicorn app:app в server/)"
echo "Плагин Photoshop подключается к http://127.0.0.1:8765"
