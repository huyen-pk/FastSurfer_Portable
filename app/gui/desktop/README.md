# FastSurfer Desktop (Tauri)

Desktop UI built with Tauri. The backend command controllers live in `src-tauri/src/lib.rs`.

Shared coding-agent playbook: [../../SKILLS.md](../../SKILLS.md)

## Development

- Install Node dependencies: `npm install`
- Run Tauri app: `npm run tauri dev`

## Rust Testing Pipeline

This project uses a Rust-focused test pipeline targeting controller logic.

### Commands

- Local Rust tests: `npm run test:rust`
- CI-style Rust tests (with captured logs): `npm run test:rust:ci`
- Run data-driven e2e inference test: `cd src-tauri && cargo test run_fastsurfer_inference_with_test_data_should_produce_output_file -- --ignored --nocapture`

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

### Test Scope

- Validation behavior for controller input paths used by `open_result_in_file_manager`
- Empty path rejection
- Missing path rejection
- Existing file/dir path acceptance
