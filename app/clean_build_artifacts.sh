#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=false
AGGRESSIVE=false

usage() {
  cat <<'EOF'
Usage: ./clean_build_artifacts.sh [--dry-run] [--aggressive]

Removes regenerable build artifacts to reclaim disk space before local/CI builds.

Options:
  --dry-run     Show what would be removed without deleting files.
  --aggressive  Also remove larger release/bundle outputs.
  -h, --help    Show this help message.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=true
      ;;
    --aggressive)
      AGGRESSIVE=true
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

PATHS=(
  "./backend/build"
  "./backend/dist"
  "./gui/damadian-ui/dist"
  "./gui/workbench/dist"
  "./gui/workbench/src-tauri/target"
)

if [[ "$AGGRESSIVE" == true ]]; then
  PATHS+=(
    "./gui/workbench/backend"
  )
fi

remove_path() {
  local rel_path="$1"
  local abs_path="$ROOT_DIR/$rel_path"

  if [[ ! -e "$abs_path" ]]; then
    return 0
  fi

  if [[ "$DRY_RUN" == true ]]; then
    echo "[dry-run] rm -rf $rel_path"
    return 0
  fi

  rm -rf "$abs_path"
  echo "Removed: $rel_path"
}

for path in "${PATHS[@]}"; do
  remove_path "$path"
done

echo "Cleanup complete."
if [[ "$DRY_RUN" == true ]]; then
  echo "No files were deleted (--dry-run)."
fi
