#!/usr/bin/env bash
# Smoke-test Bedrock Function URL + Q/A reconstruction.
#   bash app/ai/smoke.sh
#   ./start --smoke-llm
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)" || { echo "error: python3 is required" >&2; exit 1; }
fi
cd "$ROOT"
exec "$PY" "$ROOT/ai/smoke.py" "$@"
