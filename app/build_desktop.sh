#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$APP_DIR/.." && pwd)"

CLEAN_FIRST=false
CLEAN_AGGRESSIVE=false

usage() {
  cat <<'EOF'
Usage: ./app/build_desktop.sh [--clean] [--clean-aggressive]

Build sequence: backend -> frontend (Svelte) -> desktop (Tauri)

Options:
  --clean             Clean regenerable build artifacts before building.
  --clean-aggressive  Clean first and include larger runtime artifacts.
  -h, --help          Show this help message.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --clean)
      CLEAN_FIRST=true
      ;;
    --clean-aggressive)
      CLEAN_FIRST=true
      CLEAN_AGGRESSIVE=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage
      exit 2
      ;;
  esac
done

BACKEND_BUILD_SCRIPT="$APP_DIR/backend/build.sh"
FRONTEND_BUILD_SCRIPT="$APP_DIR/gui/simple-viewer/build.sh"
TAURI_BUILD_SCRIPT="$APP_DIR/gui/desktop/build.sh"
CLEAN_SCRIPT="$APP_DIR/clean_build_artifacts.sh"

if [[ "$CLEAN_FIRST" == true ]]; then
  if [[ ! -x "$CLEAN_SCRIPT" ]]; then
    echo "Cleanup script is missing or not executable: $CLEAN_SCRIPT" >&2
    exit 1
  fi

  echo "[0/3] Cleaning build artifacts..."
  if [[ "$CLEAN_AGGRESSIVE" == true ]]; then
    "$CLEAN_SCRIPT" --aggressive
  else
    "$CLEAN_SCRIPT"
  fi
fi

if [[ ! -x "$BACKEND_BUILD_SCRIPT" ]]; then
  echo "Backend build script is missing or not executable: $BACKEND_BUILD_SCRIPT" >&2
  exit 1
fi

if [[ ! -x "$FRONTEND_BUILD_SCRIPT" ]]; then
  echo "Frontend build script is missing or not executable: $FRONTEND_BUILD_SCRIPT" >&2
  exit 1
fi

if [[ ! -x "$TAURI_BUILD_SCRIPT" ]]; then
  echo "Tauri build script is missing or not executable: $TAURI_BUILD_SCRIPT" >&2
  exit 1
fi

echo "[1/3] Building backend..."
"$BACKEND_BUILD_SCRIPT"

echo "[2/3] Building frontend (Svelte)..."
"$FRONTEND_BUILD_SCRIPT"

echo "[3/3] Building desktop app (Tauri)..."
"$TAURI_BUILD_SCRIPT"

echo "Build sequence completed successfully."
