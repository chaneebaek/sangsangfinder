#!/usr/bin/env bash
# Keep the notification worker awake with caffeinate while this process runs.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv311/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3)"
fi

cd "$ROOT_DIR"
exec /usr/bin/caffeinate -dimsu "$PYTHON_BIN" scripts/notice_notification_worker.py --daemon "$@"
