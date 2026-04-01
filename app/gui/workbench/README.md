# FastSurfer Desktop (Tauri)

Desktop UI built with Tauri. The backend command controllers live in `src-tauri/src/lib.rs`.

Shared coding-agent playbook: [../../SKILLS.md](../../SKILLS.md)

## Development

- Install Node dependencies: `npm install`
- Run Tauri app: `npm run tauri dev`

## Native Runtime Selection (Candle vs ORT)

Rust native inference now supports selecting the ONNX runtime implementation by environment variable.

- Keep existing Candle path (default):
	- `FASTSURFER_INFERENCE_ENGINE=rust-onnx`
	- `FASTSURFER_NATIVE_RUNTIME=candle`
- Switch to ONNX Runtime C++ backend:
	- `FASTSURFER_INFERENCE_ENGINE=rust-onnx`
	- `FASTSURFER_NATIVE_RUNTIME=ort`

Example (dev run with ORT):
- `FASTSURFER_INFERENCE_ENGINE=rust-onnx FASTSURFER_NATIVE_RUNTIME=ort npm run tauri dev`

Example (dev run with Candle):
- `FASTSURFER_INFERENCE_ENGINE=rust-onnx FASTSURFER_NATIVE_RUNTIME=candle npm run tauri dev`

`FASTSURFER_INFERENCE_ENGINE=python-ipc` still uses the Python backend and ignores `FASTSURFER_NATIVE_RUNTIME`.

Native Rust input support:
- Accepted input files in Rust native mode: `.nii`, `.nii.gz`, `.mgz`, `.mgh`.
- `.mgz/.mgh` are converted to temporary NIfTI before inference.
- Conversion order:
	- `mri_convert` (or override binary with `FASTSURFER_MRI_CONVERT_BIN`)
	- fallback: `python3 + nibabel` (or override with `FASTSURFER_PYTHON_BIN`)

Task cancellation / app close behavior:
- Native Rust inference now cooperatively checks cancellation during slice processing.
- Cancelling a task or closing the app during native inference emits `cancelled` status and stops work as soon as possible.

## ORT Dynamic Runtime Bundling (Installer)

The ORT integration uses dynamic loading (`ort` crate `load-dynamic` feature). To bundle ORT runtime libs into the final installer:

1. Optionally provide runtime directory at build time (recommended):
	- `FASTSURFER_ORT_RUNTIME_DIR=/absolute/path/to/onnxruntime/lib`
	- or `FASTSURFER_ORT_DYLIB_PATH=/absolute/path/to/libonnxruntime.so` (or `.dylib` / `.dll`)
	- Directory should contain platform files such as:
		- Linux: `libonnxruntime*.so*`
		- macOS: `libonnxruntime*.dylib`
		- Windows: `onnxruntime*.dll`
2. Build desktop app normally:
	- `FASTSURFER_ORT_RUNTIME_DIR=/path/to/lib ./build.sh`
	- or simply `./build.sh` and let build.rs auto-download runtime if needed

`src-tauri/build.rs` stages matching ORT runtime files into `src-tauri/resources/ort/<platform>/`, and `tauri.conf.json` bundles `resources/ort` into installers.

Automatic fallback:
- If no runtime path is provided and no staged runtime is found, build.rs downloads ONNX Runtime automatically.
- Configure download cache location with `FASTSURFER_ORT_DOWNLOAD_DIR` (default: `src-tauri/.ort-runtime-cache`).
- Configure runtime version with `FASTSURFER_ORT_VERSION` (default: `1.23.2`).
- Override download URL with `FASTSURFER_ORT_DOWNLOAD_URL` when needed.
- Auto-download requires host tools:
	- Linux/macOS: `curl` + `tar`
	- Windows: `curl` + `unzip`

Build still fails if runtime cannot be resolved and auto-download fails.

## Build Variants (size comparison)

- Build desktop app with bundled backend binary/resources (default): `./build.sh`
- Build desktop app without bundled backend binary/resources: `./build_no_sidecar.sh`
- Both commands require frontend assets first (for example: `cd ../damadian-ui && ./build.sh --target tauri`)
- Compare resulting package size (Linux `.deb`): `ls -lh src-tauri/target/release/bundle/deb/*.deb`

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
	- `cd src-tauri && cargo test parity_label_volume_ratio_rust_inference_with_golden_files_should_compare_without_python_runtime -- --ignored --nocapture`
- Run full-volume stage parity+benchmark report (Subject140):
	- `cd src-tauri && cargo test parity_dice_assd_hd95_hdmax_icc_full_volume_pipeline_should_generate_stage_benchmark_report -- --ignored --nocapture`

### Python Binary Configuration (for IPC/e2e tests)

- Set Python binary in `app/gui/workbench/.env`:
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
- Optional runtime override for native parity/test flows:
	- `cd src-tauri && FASTSURFER_NATIVE_RUNTIME=ort bash testing/benchmarks/run_native_benchmarks.sh`
	- `cd src-tauri && FASTSURFER_NATIVE_RUNTIME=candle bash testing/benchmarks/run_native_benchmarks.sh`

The runner executes:
- Criterion benchmark (`candle_inference_bench`) in safe mode
- Optional forward diagnostic mode
- Node-level hotspot trace from parity golden test

### Evaluate Benchmark Results

After each run, inspect:
- `src-tauri/testing/benchmarks/logs/native-bench-<timestamp>.log`
- `src-tauri/testing/benchmarks/logs/native-bench-<timestamp>.summary.txt`
- `src-tauri/testing/benchmarks/logs/native-node-trace-<timestamp>.log`
- Full-volume parity reports (when running the ignored full-volume test):
	- `src-tauri/testing/rust/results/parity_dice_assd_hd95_hdmax_icc_full_volume_subject140_<timestamp>/full_volume_parity_report.json`
	- Stage artifacts are stored in:
		- `.../preprocess/` (Rust + Python preprocess NIfTI volumes)
		- `.../forward_pass/` (Rust + Python forward prediction volumes + inference metrics JSON)
		- `.../postprocess/` (Rust + Python postprocess aseg/brainmask NIfTI volumes)
	- Same folder also keeps visual artifacts (`.mgz`/`.nii.gz`) for both runtimes:
		- `input_subject140.mgz`, `input_subject140.native_input.nii.gz`
		- `rust_pred.nii.gz`, `python_pred.nii.gz`
		- `rust_post_aseg.nii.gz`, `python_post_aseg.nii.gz`
		- `rust_post_brainmask.nii.gz`, `python_post_brainmask.nii.gz`
- One-slice golden parity artifacts are also persisted per run:
	- `src-tauri/testing/rust/results/parity_label_volume_ratio_one_slice_<timestamp>/`
	- Viewer script for latest/selected run: `src-tauri/testing/rust/results/view_nii_overlay.py`

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
- `cd src-tauri && FASTSURFER_NATIVE_CPU_THREADS=16 FASTSURFER_NATIVE_SLICES_PER_PLANE=1 FASTSURFER_NATIVE_TRACE_TIMING=1 cargo test parity_label_volume_ratio_rust_inference_with_golden_files_should_compare_without_python_runtime -- --ignored --nocapture`

### Test Scope

- Validation behavior for controller input paths used by `open_result_in_file_manager`
- Empty path rejection
- Missing path rejection
- Existing file/dir path acceptance
