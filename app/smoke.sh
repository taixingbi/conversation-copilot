#!/usr/bin/env bash
# Test the inference endpoint configured in .env or the shell environment.
set -euo pipefail
exec bash "$(dirname "$0")/backend/llm/smoke.sh" "$@"
