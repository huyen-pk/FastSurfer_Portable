#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$APP_DIR/.." && pwd)"

CLEAN_FIRST=false
CLEAN_AGGRESSIVE=false
TARGET_ENVIRONMENT="desktop"

usage() {
  cat <<'EOF'
Usage: ./app/build_app.sh [--target desktop|web] [--clean] [--clean-aggressive]

Build sequence:
  desktop: backend -> frontend (Svelte tauri target) -> desktop (Tauri)
  web:     frontend (Svelte web target with PWA)

Options:
  --target            Select target environment: desktop or web.
  --clean             Clean regenerable build artifacts before building.
  --clean-aggressive  Clean first and include larger runtime artifacts.
  -h, --help          Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      CLEAN_FIRST=true
      shift
      ;;
    --clean-aggressive)
      CLEAN_FIRST=true
      CLEAN_AGGRESSIVE=true
      shift
      ;;
    --target)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --target" >&2
        exit 2
      fi
      TARGET_ENVIRONMENT="${2,,}"
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

if [[ "$TARGET_ENVIRONMENT" != "desktop" && "$TARGET_ENVIRONMENT" != "web" ]]; then
  echo "Unsupported --target '$TARGET_ENVIRONMENT'. Expected 'desktop' or 'web'." >&2
  exit 2
fi

export VITE_FASTSURFER_TARGET="$TARGET_ENVIRONMENT"

BACKEND_BUILD_SCRIPT="$APP_DIR/backend/build.sh"
FRONTEND_BUILD_SCRIPT="$APP_DIR/gui/simple-viewer/build.sh"
TAURI_BUILD_SCRIPT="$APP_DIR/gui/desktop/build.sh"
CLEAN_SCRIPT="$APP_DIR/clean_build_artifacts.sh"

if [[ "$CLEAN_FIRST" == true ]]; then
  if [[ ! -x "$CLEAN_SCRIPT" ]]; then
    echo "Cleanup script is missing or not executable: $CLEAN_SCRIPT" >&2
    exit 1
  fi

  echo "Cleaning build artifacts..."
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


echo "Building backend..."
"$BACKEND_BUILD_SCRIPT"

if [[ ! -x "$FRONTEND_BUILD_SCRIPT" ]]; then
    echo "Frontend build script is missing or not executable: $FRONTEND_BUILD_SCRIPT" >&2
    exit 1
fi

if [[ "$TARGET_ENVIRONMENT" == "web" ]]; then
  echo "Building frontend (Svelte web target with PWA)..."
  "$FRONTEND_BUILD_SCRIPT" --target web
  echo "Build frontend completed successfully for target: $TARGET_ENVIRONMENT"
  exit 0
fi

if [[ "$TARGET_ENVIRONMENT" == "desktop" ]]; then
  if [[ -z "${FASTSURFER_ORT_RUNTIME_DIR:-}" && -z "${FASTSURFER_ORT_DYLIB_PATH:-}" ]]; then
    echo "Desktop build: ORT runtime env not provided; build.rs will try auto-download." >&2
    echo "Optional override: set FASTSURFER_ORT_RUNTIME_DIR or FASTSURFER_ORT_DYLIB_PATH." >&2
  fi

  echo "Building frontend (Svelte tauri target)..."
  "$FRONTEND_BUILD_SCRIPT" --target tauri

  if [[ ! -x "$TAURI_BUILD_SCRIPT" ]]; then
    echo "Tauri build script is missing or not executable: $TAURI_BUILD_SCRIPT" >&2
    exit 1
  fi
  echo "Building desktop app (Tauri)..."
  "$TAURI_BUILD_SCRIPT"
fi


echo "Build sequence completed successfully for target: $TARGET_ENVIRONMENT"
