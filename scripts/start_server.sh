#!/usr/bin/env bash
# ============================================================
#  Chinminator — запуск сервера на macOS (Apple Silicon)
#  Использование: ./scripts/start_server.sh [--portable]
# ============================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="$ROOT/server"
PY="$SERVER/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "[ERROR] venv не найден. Сначала: bash scripts/setup_macos.sh" >&2
  exit 1
fi

echo "Запуск Chinminator API на http://127.0.0.1:8765 ..."
cd "$SERVER"
exec "$PY" -m uvicorn app:app --host 127.0.0.1 --port 8765
