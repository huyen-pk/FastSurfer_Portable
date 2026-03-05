# FastSurfer Desktop (Tauri)

Desktop UI built with Tauri. The backend command controllers live in `src-tauri/src/lib.rs`.

Shared coding-agent playbook: [../../SKILLS.md](../../SKILLS.md)

## Development

- Install Node dependencies: `npm install`
- Run Tauri app: `npm run tauri dev`

## Rust Testing Pipeline

This project uses a Rust-focused test pipeline targeting controller logic.

### Requirements

- Rust toolchain + Cargo
- Node.js dependencies (`npm install`)
- For Python-backed parity tests: Python environment with `numpy`, `nibabel`, `scipy`, `scikit-image`, and FastSurfer importable from repo root
- Golden fixtures for component parity:
	- `src-tauri/testing/data/.tmp_e2e_output_py/140_orig.native_input.nii.gz`
	- `src-tauri/testing/data/.tmp_e2e_output_py/140_orig.python_pred.nii.gz`
	- Regenerate with: `cd src-tauri && python3 testing/generate_golden_fixtures.py`

### Test Commands

- Local Rust tests: `npm run test:rust`
- CI-style Rust tests (captured logs): `npm run test:rust:ci`
- Run all controller tests directly:
	- `cd src-tauri && cargo test controller_tests:: -- --nocapture`
- Run all parity tests (function names prefixed with `parity_`):
	- Non-ignored: `cd src-tauri && cargo test parity_ -- --nocapture`
	- Ignored: `cd src-tauri && cargo test parity_ -- --ignored --nocapture`
- Run component parity tests only:
	- `cd src-tauri && cargo test parity_data_loading_ -- --nocapture`
	- `cd src-tauri && cargo test parity_preprocessing_ -- --nocapture`
	- `cd src-tauri && cargo test parity_postprocessing_ -- --nocapture`
- Run golden native parity test:
	- `cd src-tauri && cargo test parity_native_rust_inference_with_golden_files_should_compare_without_python_runtime -- --ignored --nocapture`

### Python Binary Configuration (for IPC/e2e tests)

- Set Python binary in `app/gui/desktop/.env`:
	- `FASTSURFER_PYTHON_BIN=/absolute/path/to/python`
	- or `FASTSURFER_PYTHON_BIN=python3`
- `src-tauri/src/lib.rs` loads `.env` on startup and uses `FASTSURFER_PYTHON_BIN` for launching `app/backend/ipc_server.py` in dev.
- The e2e inference test uses test data from `src-tauri/testing/data/Subject140/140_orig.mgz`.

Skip policy:
- CI may skip Python-launch/e2e inference tests when Python runtime dependencies are missing.
- Local runs do **not** skip; they fail fast with setup guidance.

### Layout

- Rust controller tests are in `testing/rust/controller_tests.rs`
- CI test logs are written to `testing/rust/results/cargo-test.log`
- Benchmark runner script is `src-tauri/testing/benchmarks/run_native_benchmarks.sh`

## Benchmarking

### Run Benchmarks

- From desktop root: `cd src-tauri && bash testing/benchmarks/run_native_benchmarks.sh`
- Optional thread override:
	- `cd src-tauri && FASTSURFER_NATIVE_CPU_THREADS=16 bash testing/benchmarks/run_native_benchmarks.sh`

The runner executes:
- Criterion benchmark (`candle_inference_bench`) in safe mode
- Optional forward diagnostic mode
- Node-level hotspot trace from parity golden test

### Evaluate Benchmark Results

After each run, inspect:
- `src-tauri/testing/benchmarks/logs/native-bench-<timestamp>.log`
- `src-tauri/testing/benchmarks/logs/native-bench-<timestamp>.summary.txt`
- `src-tauri/testing/benchmarks/logs/native-node-trace-<timestamp>.log`

Interpretation checklist:
- Criterion section: compare preprocess/load timing trends between runs
- Forward diagnostic exit code: non-zero means diagnostic forward run timed out/failed (best-effort step)
- Hotspot summary: top operators by `elapsed_ms`; prioritize highest entries first
- Validate improvement by comparing top-25 hotspot list and aggregate wall-clock time across baseline vs changed run

## Feature Flags

### Cargo Features

- `src-tauri/Cargo.toml` currently does not define custom `[features]` toggles for inference behavior.
- Build/test with standard Cargo feature flags as needed for dependencies, e.g.:
	- `cd src-tauri && cargo test --no-default-features`
	- `cd src-tauri && cargo test --features <feature_name>`

### Runtime Inference Flags (environment variables)

Use these to control native inference behavior without recompiling:

- `FASTSURFER_NATIVE_DEVICE`: `cpu` or `cuda`
- `FASTSURFER_NATIVE_CPU_THREADS`: thread count used for native path
- `FASTSURFER_NATIVE_SLICES_PER_PLANE`: sparse sampling count per plane
- `FASTSURFER_NATIVE_SPARSE_PARALLEL`: sparse per-plane parallelism toggle (`1`/`0`)
- `FASTSURFER_NATIVE_TRACE_TIMING`: high-level timing trace (`1`/`0`)
- `FASTSURFER_NATIVE_TRACE_NODES`: node-level ONNX trace (`1`/`0`)
- `FASTSURFER_NATIVE_TRACE_NODE_MS`: node trace threshold in milliseconds
- `FASTSURFER_NATIVE_PLANE_TIMEOUT_SECS`: per-plane forward timeout
- `FASTSURFER_NATIVE_TEST_TIMEOUT_SECS`: outer parity-test timeout

Example:
- `cd src-tauri && FASTSURFER_NATIVE_CPU_THREADS=16 FASTSURFER_NATIVE_SLICES_PER_PLANE=1 FASTSURFER_NATIVE_TRACE_TIMING=1 cargo test parity_native_rust_inference_with_golden_files_should_compare_without_python_runtime -- --ignored --nocapture`

### Test Scope

- Validation behavior for controller input paths used by `open_result_in_file_manager`
- Empty path rejection
- Missing path rejection
- Existing file/dir path acceptance
