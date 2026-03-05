#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$ROOT_DIR/testing/benchmarks/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/native-bench-$TIMESTAMP.log"
SUMMARY_FILE="$LOG_DIR/native-bench-$TIMESTAMP.summary.txt"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[bench] started at: $(date -Iseconds)"
echo "[bench] root: $ROOT_DIR"
echo "[bench] log: $LOG_FILE"
echo "[bench] summary: $SUMMARY_FILE"
echo "[bench] uname: $(uname -a)"
echo "[bench] rustc: $(rustc --version)"
echo "[bench] cargo: $(cargo --version)"

cd "$ROOT_DIR"

echo ""
echo "== Criterion (safe mode) =="
FASTSURFER_NATIVE_CPU_THREADS="${FASTSURFER_NATIVE_CPU_THREADS:-16}" \
  cargo bench --bench candle_inference_bench -- --noplot

echo ""
echo "== Criterion (forward diagnostic mode, best effort) =="
set +e
timeout 180s env \
  FASTSURFER_BENCH_FORWARD=1 \
  FASTSURFER_NATIVE_SLICES_PER_PLANE=1 \
  FASTSURFER_NATIVE_PLANE_TIMEOUT_SECS=15 \
  FASTSURFER_NATIVE_CPU_THREADS="${FASTSURFER_NATIVE_CPU_THREADS:-16}" \
  cargo bench --bench candle_inference_bench -- --noplot
FORWARD_EXIT=$?
set -e

echo "[bench] forward criterion exit code: $FORWARD_EXIT"

echo ""
echo "== Node hotspot trace (best effort) =="
set +e
timeout 120s env \
  FASTSURFER_NATIVE_SLICES_PER_PLANE=1 \
  FASTSURFER_NATIVE_CPU_THREADS="${FASTSURFER_NATIVE_CPU_THREADS:-16}" \
  FASTSURFER_NATIVE_PLANE_TIMEOUT_SECS=15 \
  FASTSURFER_NATIVE_TRACE_TIMING=0 \
  FASTSURFER_NATIVE_TRACE_NODES=1 \
  FASTSURFER_NATIVE_TRACE_NODE_MS=0 \
  cargo test parity_native_rust_inference_with_golden_files_should_compare_without_python_runtime -- --ignored --nocapture \
  > "$LOG_DIR/native-node-trace-$TIMESTAMP.log" 2>&1
TRACE_EXIT=$?
set -e

echo "[bench] node trace exit code: $TRACE_EXIT"

echo ""
echo "== Hotspot summary (top 25 by elapsed_ms) =="
awk 'BEGIN{FS="elapsed_ms="} /\[trace\]\[candle-node\] end/ {
  split($2,a," ");
  ms=a[1]+0;
  name=""; op="";
  if (match($0, /name='\''[^'\'']+'\''/)) { name=substr($0,RSTART+6,RLENGTH-7) }
  if (match($0, /op='\''[^'\'']+'\''/)) { op=substr($0,RSTART+4,RLENGTH-5) }
  print ms"\t"op"\t"name
}' "$LOG_DIR/native-node-trace-$TIMESTAMP.log" | sort -nr | head -n 25 | tee "$SUMMARY_FILE"

echo ""
echo "[bench] complete: $(date -Iseconds)"
echo "[bench] logs saved:"
echo "  - $LOG_FILE"
echo "  - $SUMMARY_FILE"
echo "  - $LOG_DIR/native-node-trace-$TIMESTAMP.log"
