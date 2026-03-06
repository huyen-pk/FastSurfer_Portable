#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TARGET="tauri"

usage() {
  cat <<'EOF'
Usage: ./build.sh [--target tauri|web]

Builds the simple-viewer Vite app for the selected target.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --target" >&2
        exit 2
      fi
      TARGET="${2,,}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$TARGET" != "tauri" && "$TARGET" != "web" ]]; then
  echo "Unsupported target '$TARGET'. Expected 'tauri' or 'web'." >&2
  exit 2
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required but not found in PATH." >&2
  exit 127
fi

export VITE_FASTSURFER_TARGET="$TARGET"

npm install

echo "Building simple-viewer target: $TARGET"
npm run build
