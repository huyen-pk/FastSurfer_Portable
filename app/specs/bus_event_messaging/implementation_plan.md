## Plan: Bus Event Messaging Transport

Refactor the legacy Python IPC transport in the Tauri desktop crate from free functions over a locked backend process into a transport component that owns request routing, response routing, stdout log routing, and shutdown coordination. The design keeps the existing backend protocol, adds concurrent request correlation by request id, bounds streaming to avoid memory growth, and makes backend exit/shutdown wake blocked callers promptly.

**Overview**
- Replace the current direct stdin/stdout access with a `Transport` struct.
- Move stdout reading into a dedicated long-running reader thread.
- Move stdin writing into a dedicated long-running writer thread.
- Add a dedicated IPC request channel and a dedicated parsed IPC response channel.
- Route correlated responses and progress events into per-request bounded channels.
- Route non-IPC stdout into a dedicated bounded log/output channel.

**Flow**
The normal flow is:
- `BackendState` creates one `Transport` when the backend process is created in backend.rs.
- Each request calls `active_transport()`, which only locks briefly to clone the existing `Arc<Transport>`.
- The request then uses that same shared transport instance to enqueue work onto its existing writer/reader/dispatcher threads.

So the per-request cost is just:
- clone one `Arc`
- allocate per-request channels/state for correlation
- send the request into the already-running transport threads

That is intentional and cheap. The expensive parts, the long-lived threads and owned stdio handles, are created once per backend process, not once per request.

A new `Transport` is only created in two cases:
- backend startup in backend.rs
- backend restart in backend.rs

That is why the field is `Mutex<Arc<Transport>>`: the mutex protects replacing the current transport during restart, while the `Arc` lets many requests reuse the same transport concurrently.

If this had been `Arc<Mutex<Transport>>`, it would suggest locking the transport itself for normal request traffic, which is the opposite of what this transport is designed for. The transport already contains its own internal synchronization in transport.rs, so wrapping the whole thing in an outer mutex for every request would add contention without adding value.

Short version:
- requests reuse the same transport
- restart swaps in a new transport
- the outer mutex is for pointer replacement, not per-request execution

**Design**
1. `Transport` owns:
- An outbound request sender for serialized IPC writes.
- A parsed IPC response sender/receiver pair between the reader and dispatcher threads.
- A bounded log/output channel for non-IPC stdout lines.
- An in-flight registry keyed by request id for per-request event delivery.
- Join handles for the reader, dispatcher, and writer threads.
2. `send_request` assigns a unique request id, registers per-request channels, queues the request for the writer thread, and returns a request handle.
3. `read_ipc_response` waits for the queued write result, then consumes bounded per-request events until the correlated final response arrives. Progress events are emitted incrementally via the supplied callback.
4. Process death, malformed JSON on IPC lines, transport shutdown, or channel closure will terminate all waiters promptly with an error.
5. `BackendState` will own the transport and the child lifecycle separately, allowing restart and force-stop without holding blocking I/O locks.

**Key Files**
- `app/gui/desktop/src-tauri/src/transport.rs`
- `app/gui/desktop/src-tauri/src/process_mgmt.rs`
- `app/gui/desktop/src-tauri/src/backend.rs`
- `app/gui/desktop/src-tauri/src/tasks.rs`
- `app/gui/desktop/src-tauri/testing/rust/controller_tests/support.rs`
- `app/gui/desktop/src-tauri/testing/rust/controller_tests/data_loading_tests.rs`
- `app/gui/desktop/src-tauri/testing/rust/controller_tests/progress_tests.rs`

**Acceptance Targets**
- Concurrent in-flight requests are correlated by `id` and return the correct final response.
- Progress events stream without unbounded buffering.
- Non-JSON stdout is routed to the log channel and does not break IPC parsing.
- Backend exit or forced stop wakes blocked callers without deadlocking.
- Restart replaces the active transport only after the new process passes a health check.

**Environment Notes**
- Validation should run in `app/gui/desktop/src-tauri` with the existing Rust controller tests and `cargo clippy --manifest-path app/gui/desktop/src-tauri/Cargo.toml`.
- Real Python-process tests depend on `FASTSURFER_PYTHON_BIN` or a working `python3` in the environment.
