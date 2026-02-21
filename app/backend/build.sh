#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -n "${PYINSTALLER_BIN:-}" ]]; then
	PYINSTALLER="$PYINSTALLER_BIN"
elif command -v pyinstaller >/dev/null 2>&1; then
	PYINSTALLER="$(command -v pyinstaller)"
elif [[ -x "$HOME/anaconda3/envs/fastsurfer/bin/pyinstaller" ]]; then
	PYINSTALLER="$HOME/anaconda3/envs/fastsurfer/bin/pyinstaller"
else
	echo "Could not find pyinstaller. Set PYINSTALLER_BIN or activate an environment that provides it." >&2
	exit 127
fi

TARGET_BACKEND_DIR="$SCRIPT_DIR/../gui/desktop/backend"
DIST_PARENT_DIR="$SCRIPT_DIR/../gui/desktop"

# Remove legacy output path from older build flow to avoid duplicated artifacts.
rm -rf "$SCRIPT_DIR/dist"

"$PYINSTALLER" --clean --noconfirm \
	--distpath "$DIST_PARENT_DIR" \
	--workpath "$SCRIPT_DIR/build" \
	main.spec

TARGET_BIN="$TARGET_BACKEND_DIR/main"

if [[ ! -d "$TARGET_BACKEND_DIR" ]]; then
	echo "Build succeeded but backend folder not found: $TARGET_BACKEND_DIR" >&2
	exit 1
fi

if [[ ! -f "$TARGET_BIN" ]]; then
	echo "Backend folder exists but executable is missing: $TARGET_BIN" >&2
	exit 1
fi

chmod +x "$TARGET_BIN"

echo "Built backend runtime folder in place: $TARGET_BACKEND_DIR"

