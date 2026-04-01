#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FRONTEND_DIST_DIR="$SCRIPT_DIR/../damadian-ui/dist"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required but not found in PATH." >&2
  exit 127
fi

if [[ ! -d "$FRONTEND_DIST_DIR" ]]; then
  echo "Frontend build output missing: $FRONTEND_DIST_DIR" >&2
  echo "Run app/gui/damadian-ui/build.sh first." >&2
  exit 1
fi

if [[ -z "${FASTSURFER_ORT_RUNTIME_DIR:-}" && -z "${FASTSURFER_ORT_DYLIB_PATH:-}" ]]; then
  echo "ORT runtime env not provided; build.rs will try auto-discovery/download." >&2
  echo "Optional override: set FASTSURFER_ORT_RUNTIME_DIR or FASTSURFER_ORT_DYLIB_PATH." >&2
fi

npm run tauri build
