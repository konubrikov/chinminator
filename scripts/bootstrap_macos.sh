#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo "Run on Apple Silicon, not under Rosetta."; exit 1; }
if [[ -x "$ROOT/server/.venv/bin/python" && -f "$ROOT/.installed-version" ]]; then
    case "$(cat "$ROOT/.installed-version")" in
        1.6.0|1.7.0) exec "$ROOT/server/.venv/bin/python" -X utf8 "$ROOT/scripts/launch.py" ;;
    esac
fi
mkdir -p "$ROOT/.bootstrap"
ARCHIVE="$ROOT/.bootstrap/uv.tar.gz"
curl -fL --retry 3 -o "$ARCHIVE" https://github.com/astral-sh/uv/releases/download/0.13.6/uv-aarch64-apple-darwin.tar.gz
EXPECTED=60fbd5c8a1ed681845eea28d9d622fcde03c51e91a43465b64a1e17db1ed3472
ACTUAL="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
[[ "$ACTUAL" == "$EXPECTED" ]] || { echo "uv checksum mismatch"; exit 1; }
tar -xzf "$ARCHIVE" -C "$ROOT/.bootstrap"
UV="$ROOT/.bootstrap/uv-aarch64-apple-darwin/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.python"
export UV_PYTHON_BIN_DIR="$ROOT/.bootstrap/bin"
export UV_CACHE_DIR="$ROOT/.uv-cache"
"$UV" python install 3.12
PY="$("$UV" python find --managed-python 3.12)"
"$PY" -X utf8 "$ROOT/scripts/install.py"
