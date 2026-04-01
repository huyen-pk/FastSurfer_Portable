#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUNDLE_DIR="$SCRIPT_DIR/src-tauri/target/release/bundle/deb"
REPORT_DIR="$BUNDLE_DIR/size_comparison"
TMP_STAGE_DIR="$SCRIPT_DIR/.tmp_bundle_resources"
TMP_CONFIG_REL="src-tauri/tauri.with-sidecar.exclude-data-results.conf.json"
TMP_CONFIG_PATH="$SCRIPT_DIR/$TMP_CONFIG_REL"
BACKEND_SRC_DIR="$SCRIPT_DIR/backend"
BACKEND_STAGED_DIR="$TMP_STAGE_DIR/backend"
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_JSON="$REPORT_DIR/size_delta_${RUN_TS}.json"
REPORT_CSV="$REPORT_DIR/size_delta_${RUN_TS}.csv"

cleanup() {
  rm -rf "$TMP_STAGE_DIR"
  rm -f "$TMP_CONFIG_PATH"
}

trap cleanup EXIT

latest_deb() {
  local latest
  latest="$(ls -1t "$BUNDLE_DIR"/*.deb 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest" ]]; then
    return 1
  fi
  printf '%s\n' "$latest"
}

bytes_of() {
  local path="$1"
  stat -c '%s' "$path"
}

human_size() {
  local bytes="$1"
  numfmt --to=iec-i --suffix=B "$bytes"
}

if ! command -v numfmt >/dev/null 2>&1; then
  echo "numfmt is required but not found in PATH." >&2
  exit 127
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but not found in PATH." >&2
  exit 127
fi

if [[ ! -f "$BACKEND_SRC_DIR/main" ]]; then
  echo "Bundled backend binary is missing: $BACKEND_SRC_DIR/main" >&2
  exit 1
fi

mkdir -p "$REPORT_DIR"

echo "[0/4] Staging backend resources (excluding all data/results directories)..."
rm -rf "$TMP_STAGE_DIR"
mkdir -p "$TMP_STAGE_DIR"

rsync -a \
  --exclude='**/data/***' \
  --exclude='**/results/***' \
  "$BACKEND_SRC_DIR/" "$BACKEND_STAGED_DIR/"

cat > "$TMP_CONFIG_PATH" <<EOF
{
  "bundle": {
    "resources": [
      "../.tmp_bundle_resources/backend/main",
      "../.tmp_bundle_resources/backend/_internal"
    ]
  }
}
EOF

echo "[1/4] Building default package (with sidecar resources)..."
if [[ ! -f "$TMP_CONFIG_PATH" ]]; then
  echo "Expected temporary config not found: $TMP_CONFIG_PATH" >&2
  exit 1
fi
npm run tauri build -- --config "$TMP_CONFIG_REL"

WITH_DEB="$(latest_deb)"
WITH_NAME="$(basename "$WITH_DEB")"
WITH_SNAPSHOT="$REPORT_DIR/${WITH_NAME%.deb}.with-sidecar.deb"
cp -f "$WITH_DEB" "$WITH_SNAPSHOT"

echo "[2/4] Building no-sidecar package..."
"$SCRIPT_DIR/build_no_sidecar.sh"

WITHOUT_DEB="$(latest_deb)"
WITHOUT_NAME="$(basename "$WITHOUT_DEB")"
WITHOUT_SNAPSHOT="$REPORT_DIR/${WITHOUT_NAME%.deb}.no-sidecar.deb"
cp -f "$WITHOUT_DEB" "$WITHOUT_SNAPSHOT"

WITH_SIZE="$(bytes_of "$WITH_SNAPSHOT")"
WITHOUT_SIZE="$(bytes_of "$WITHOUT_SNAPSHOT")"

DELTA_BYTES=$((WITH_SIZE - WITHOUT_SIZE))

PERCENT_DELTA="$(python3 - <<'PY' "$WITH_SIZE" "$WITHOUT_SIZE"
import sys
with_size = int(sys.argv[1])
without_size = int(sys.argv[2])
if with_size == 0:
    print("0.00")
else:
    print(f"{((with_size - without_size) / with_size) * 100:.2f}")
PY
)"

echo "[3/4] Size comparison complete"
echo
printf '%-22s %14s %14s\n' "Variant" "Bytes" "Human"
printf '%-22s %14s %14s\n' "with sidecar" "$WITH_SIZE" "$(human_size "$WITH_SIZE")"
printf '%-22s %14s %14s\n' "without sidecar" "$WITHOUT_SIZE" "$(human_size "$WITHOUT_SIZE")"
printf '%-22s %14s %14s\n' "delta (saved)" "$DELTA_BYTES" "$(human_size "$DELTA_BYTES")"
echo "delta_percent_saved: ${PERCENT_DELTA}%"

cat > "$REPORT_JSON" <<EOF
{
  "created_at_utc": "$RUN_TS",
  "with_sidecar": {
    "path": "$WITH_SNAPSHOT",
    "bytes": $WITH_SIZE,
    "human": "$(human_size "$WITH_SIZE")"
  },
  "without_sidecar": {
    "path": "$WITHOUT_SNAPSHOT",
    "bytes": $WITHOUT_SIZE,
    "human": "$(human_size "$WITHOUT_SIZE")"
  },
  "delta": {
    "bytes_saved": $DELTA_BYTES,
    "human_saved": "$(human_size "$DELTA_BYTES")",
    "percent_saved": $PERCENT_DELTA
  },
  "excluded_paths": [
    "**/data/**",
    "**/results/**"
  ]
}
EOF

cat > "$REPORT_CSV" <<EOF
created_at_utc,with_sidecar_bytes,without_sidecar_bytes,delta_bytes_saved,percent_saved,with_sidecar_path,without_sidecar_path
$RUN_TS,$WITH_SIZE,$WITHOUT_SIZE,$DELTA_BYTES,$PERCENT_DELTA,"$WITH_SNAPSHOT","$WITHOUT_SNAPSHOT"
EOF

echo
echo "[4/4] Snapshot artifacts"
echo "- $WITH_SNAPSHOT"
echo "- $WITHOUT_SNAPSHOT"
echo "- $REPORT_JSON"
echo "- $REPORT_CSV"
