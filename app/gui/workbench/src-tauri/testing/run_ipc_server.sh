#!/usr/bin/env bash
# Wrapper to launch the development Python IPC server using the desired Python
# Use FASTSURFER_PYTHON_BIN to override, otherwise fall back to python3
set -euo pipefail

PY_BIN="${FASTSURFER_PYTHON_BIN:-python3}"

REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
SCRIPT_PATH="$REPO_ROOT/app/backend/ipc_server.py"

# Ensure Python can import local packages during tests
export PYTHONPATH="${FASTSURFER_PYTHONPATH:-$REPO_ROOT}"

exec "$PY_BIN" "$SCRIPT_PATH"
