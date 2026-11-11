#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/server/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Run Install.command first."; exit 1; }
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec "$PY" -X utf8 "$ROOT/scripts/launch.py"
