# SKILLS.md

Guide for coding agents working in `app/`.

## Scope

This guide covers:
- Implementing new features
- Building and running apps
- Writing and running tests

Primary targets in this repo:
- `app/gui/simple-viewer` (Svelte + Vite + TypeScript)
- `app/gui/desktop` (Tauri + Rust backend controllers)
- `app/backend` (Python FastAPI + packaged backend runtime)

---

## Core Workflow for Agents

1. **Understand context first**
   - Read current files before editing.
   - Identify the smallest set of files needed for the change.
   - Prefer minimal, focused diffs.

2. **Implement at root cause**
   - Avoid superficial patches.
   - Keep behavior backward-compatible unless requested otherwise.

3. **Validate after edits**
   - Run targeted tests first.
   - Then run broader build/test checks for affected app.

4. **Document material changes**
   - Update app README when commands, layout, or workflows change.

---

## Implementing New Features

### General Rules

- Follow existing architecture and naming patterns.
- Add types for new contracts/data models.
- Keep UI logic and transport/backend logic separated.
- Avoid unrelated refactors in the same change.

### `simple-viewer` Feature Work

- UI components live under `app/gui/simple-viewer/src`.
- Transport abstractions live under `app/gui/simple-viewer/src/transport`.
- Shared inference types live under `app/gui/simple-viewer/src/types`.
- Prefer adding behavior through typed helpers instead of duplicating logic in components.

### `desktop` (Tauri) Feature Work

- Tauri/Rust command logic is in `app/gui/desktop/src-tauri/src/lib.rs`.
- Keep command handlers thin and move testable logic into helper functions.
- If adding controller logic, ensure it is unit-testable without launching full Tauri runtime.
- Rust tests should live in `app/gui/desktop/src-tauri/testing/rust`.

---

## Build & Run

### `simple-viewer`

From `app/gui/simple-viewer`:

- Install deps: `npm install`
- Dev server: `npm run dev`
- Production build: `npm run build`
- Preview build: `npm run preview`

### `desktop`

From `app/gui/desktop`:

- Install deps: `npm install`
- Run Tauri app: `npm run tauri dev`

Notes:
- Tauri backend build may depend on bundled backend binary paths.
- If build fails around backend path resolution, check expected backend binary location under `app/gui/desktop/backend/main`.

### `backend` (Python)

From repository root:

- Install backend deps: `pip install -r app/backend/requirements.txt`
- Build packaged backend runtime: `bash app/backend/build.sh`

Build details:
- Packaging uses PyInstaller via `app/backend/main.spec`.
- Build output is placed under `app/gui/desktop/backend`.
- Expected executable artifact: `app/gui/desktop/backend/main`.

Validation:
- Confirm artifact exists: `ls -l app/gui/desktop/backend/main`
- Confirm executable bit: `test -x app/gui/desktop/backend/main`

---

## Writing Tests

## Test Naming Convention

Use behavior-driven names:
- `action_should_expected_behavior`

One behavior per test case.

### Real Dependencies & Anti-Mocking Policy

- **MANDATORY**: All tests must use **real dependencies** or **containerized services** (e.g., `testcontainers`).
- **PROHIBITED**: Using mock libraries (Vitest `vi.mock`, Python `unittest.mock`, etc.) or custom "Fake" implementations is strictly forbidden. 
- **Reasoning**: To ensure tests validate real system behavior and integration, avoiding the brittleness of mocked interfaces.
- **Local Dev**: Tests should fail (not skip) if environment dependencies are missing.

### `simple-viewer` Tests

Location:
- `app/gui/simple-viewer/testing/vitest` (unit/component)
- `app/gui/simple-viewer/testing/e2e` (Playwright)

Commands (from `app/gui/simple-viewer`):
- Unit/component: `npm run test`
- Unit/component watch: `npm run test:watch`
- E2E: `npm run test:e2e`
- E2E CI/headless: `npm run test:e2e:ci`

Outputs:
- Vitest report: `testing/vitest/results/vitest-report.json`
- Playwright report: `testing/e2e/results/playwright-report`
- Playwright artifacts: `testing/e2e/results/test-results`

### `desktop` Rust Controller Tests

Location:
- `app/gui/desktop/src-tauri/testing/rust/controller_tests.rs`

Commands (from `app/gui/desktop`):
- Local Rust tests: `npm run test:rust`
- CI-style Rust tests (with logs): `npm run test:rust:ci`
- Data-driven e2e inference test: `cd app/gui/desktop/src-tauri && cargo test run_fastsurfer_inference_with_test_data_should_produce_output_file -- --ignored --nocapture`

Python binary config for desktop IPC launch/e2e tests:
- Configure `app/gui/desktop/.env` with `FASTSURFER_PYTHON_BIN`.
- Example: `FASTSURFER_PYTHON_BIN=/home/<user>/anaconda3/envs/fastsurfer/bin/python`
- Fallback value: `FASTSURFER_PYTHON_BIN=python3`

Skip policy for Python-launch/e2e tests:
- In CI, tests may skip when runtime dependencies are unavailable.
- In local development, tests should fail (not skip) to surface setup issues.

Output:
- Rust CI log: `testing/rust/results/cargo-test.log`

---

## Quality Checklist Before Handoff

- [ ] Feature implemented with minimal, focused edits
- [ ] Types added/updated for new interfaces
- [ ] Relevant tests added or updated
- [ ] Local commands run for impacted app(s)
- [ ] README updated if workflow/layout changed
- [ ] No unrelated files modified

---

## Quick Command Matrix

### Simple Viewer
- `cd app/gui/simple-viewer && npm run dev`
- `cd app/gui/simple-viewer && npm run build`
- `cd app/gui/simple-viewer && npm run test`
- `cd app/gui/simple-viewer && npm run test:e2e`

### Desktop
- `cd app/gui/desktop && npm run tauri dev`
- `cd app/gui/desktop && npm run test:rust`
- `cd app/gui/desktop && npm run test:rust:ci`

### Backend
- `pip install -r app/backend/requirements.txt`
- `bash app/backend/build.sh`
- `ls -l app/gui/desktop/backend/main`
