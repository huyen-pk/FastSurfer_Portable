#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FRONTEND_DIST_DIR="$SCRIPT_DIR/../simple-viewer/dist"
TAURI_NO_SIDECAR_CONFIG="$SCRIPT_DIR/src-tauri/tauri.no-sidecar.conf.json"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required but not found in PATH." >&2
  exit 127
fi

if [[ ! -d "$FRONTEND_DIST_DIR" ]]; then
  echo "Frontend build output missing: $FRONTEND_DIST_DIR" >&2
  echo "Run app/gui/simple-viewer/build.sh first." >&2
  exit 1
fi

if [[ ! -f "$TAURI_NO_SIDECAR_CONFIG" ]]; then
  echo "Missing no-sidecar Tauri config override: $TAURI_NO_SIDECAR_CONFIG" >&2
  exit 1
fi

npm run tauri build -- --config src-tauri/tauri.no-sidecar.conf.json
