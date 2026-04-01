## Plan: In-Process Bus / Mediator (Typed Pub/Sub)

This document records the implementation choices for the new in-process, typed mediator (bus) that replaced the older ad-hoc pub/sub and per-observer thread patterns used in the desktop crate. The mediator provides a typed, low-latency, in-process event bus with explicit ownership and clear subscriber lifetimes.

**Overview**
- `Mediator` is a typed, in-process pub/sub coordinator exposed via a global accessor `mediator()` in the desktop crate.
- Event typing is static at subscribe/publish time: callers subscribe to `E: IEvent` and publishers publish `Arc<E>`.
- The mediator uses a `TypeId`-keyed registry so different event types are isolated and delivered only to matching subscribers.

**Key Implementation Decisions**
1. Tokio broadcast-based fan-out: each event type owns a `tokio::sync::broadcast::Sender<EventPayload>` (where `EventPayload = Arc<dyn Any + Send + Sync>`). This provides efficient multi-subscriber delivery without spawning a dedicated thread per observer.
2. Explicit shared ownership: publishers pass `Arc<E>` to `Mediator::publish` so events are shared cheaply and no `Clone` bound is required on `IEvent`. `IEvent` is `Any + Send + Sync + 'static`.
3. Per-subscriber cancellation: each subscriber spawns a short-lived task on Tauri's shared application async runtime that listens on a `broadcast::Receiver` and a `watch`-based cancellation signal, allowing immediate unsubscribe/cleanup.
4. RAII helper (`MediatorSubscription`): convenience wrapper that unsubscribes in `Drop`. Callers may continue to use the RAII handle or manage `ObserverId` manually.
5. Shared runtime execution: the mediator schedules subscriber tasks on the existing application Tokio runtime through `tauri::async_runtime::spawn`, avoiding duplicate runtime ownership inside the mediator.

**Public API (summary)**
- `mediator() -> &'static Mediator`
- `Mediator::subscribe::<E, F>(&self, observer: F) -> ObserverId` where `F: Fn(&E) + Send + 'static`
- `Mediator::publish::<E>(&self, event: Arc<E>)`
- `Mediator::unsubscribe::<E>(&self, id: ObserverId) -> bool`
- `MediatorSubscription::new(mediator(), id)` — RAII wrapper that calls `unsubscribe` in `Drop`.

**Delivered Files / Locations**
-- `app/gui/workbench/src-tauri/src/mediator.rs` — mediator implementation, unit tests, and documentation comments.
-- `app/gui/workbench/src-tauri/src/backend/mod.rs` — updated call sites to publish `Arc` events and to use `mediator()` subscribe/unsubscribe directly.
- Tests: mediator unit tests in `mediator.rs` and targeted controller integration tests that exercise the event flow.

**Migration Notes / Caller Impact**
- `publish` now requires `Arc<E>`; callers that previously took `&E` or relied on `Clone` must be updated to `mediator.publish(Arc::new(event.clone()))` (or reuse existing `Arc` allocations when possible).
- `IEvent` no longer requires `Clone`. If a caller relied on `Clone` in other layers, ensure they still have the required ownership semantics.
- The RAII `MediatorSubscription` is optional; code can store `ObserverId` and call `unsubscribe` manually.

**Validation**
1. Run mediator unit tests: `cargo test -p app-gui-desktop-src-tauri -- mediator` (crate-focused invocation may vary).
2. Run desktop controller integration tests that use the mediator event flow: `cargo test --test controller_tests -- --nocapture` (run from the crate directory).
3. Confirm CI linting: `cargo fmt && cargo clippy --quiet --all-targets --message-format short -- -D warnings` within `app/gui/workbench/src-tauri`.

**Acceptance Criteria**
- Typed events are only delivered to subscribers of that concrete type.
- Multiple subscribers receive the same `Arc` instance (no unexpected copies).
- Unsubscribe removes the subscriber quickly and frees associated resources.
- The mediator does not create or own a second Tokio runtime; subscriber tasks run on the application runtime.

**Open Considerations / Future Work**
- Optionally remove the `MediatorSubscription` RAII in favor of explicit `ObserverId` lifecycle management across the crate for simpler semantics.
- Audit other crates for the changed `publish` signature (`Arc<E>`) and update remaining call sites outside the desktop crate if needed.

**Rationale**
- The new broadcast-based mediator reduces thread-per-observer overhead, simplifies lifecycle management via `ObserverId` + `watch` cancellation, and makes event ownership explicit through `Arc`. Removing the `Clone` requirement from `IEvent` reduces impl burden for large event types and encourages explicit sharing of payloads.

## ObserverCancellation and cancellation flow

Brief summary — what it is and what it does:

- `ObserverCancellation` is a type alias: `ObserverCancellation = watch::Sender<bool>` (see `mediator.rs`).
- It’s the per-observer cancellation handle stored in each `EventChannel`'s `cancellations` map; the subscriber task listens on the corresponding `watch::Receiver<bool>` to know when to stop.

Details, concisely:

- `watch` semantics: a `watch` channel holds the latest value and lets receivers await `changed().await` and then read the current value with `borrow()`; senders are clonable and non-consuming.
- Role in the mediator:
	- On `subscribe` the mediator creates `(cancel_tx, cancel_rx)` and stores `cancel_tx` as the `ObserverCancellation`.
	- The subscriber task concurrently awaits the `broadcast::Receiver` and `cancel_rx.changed()`; when `cancel_rx` becomes `true` the task breaks out and exits cleanly.
	- On `unsubscribe`, the mediator calls `cancel_tx.send(true)` to notify that observer to stop.
- Why `bool` / `watch`:
	- `bool` is a simple one-bit flag (not a stream of messages) to express “keep running / please stop”.
	- `watch` is appropriate because it broadcasts a latest-state flag to any receivers without consuming messages and supports repeated notifications.

Practical implications:

- Cancellation is cooperative — a subscriber may still process an in-flight broadcast message before seeing the cancellation.
- The alias makes intent explicit (this sender is for observer cancellation) and keeps the code readable where it’s stored and used.

### Cleanup when all receivers drop

When an event channel has no active receivers we perform a deferred, generation-guarded cleanup to avoid races and premature removal:

- Detection points:
	- `publish()` checks `sender.send(...).is_err()` and `sender.receiver_count() == 0` and will schedule cleanup if those conditions hold.
	- `unsubscribe_by_type_id()` bumps the channel's `cleanup_generation` and schedules cleanup when the last per-observer cancellation handle is removed.
- Deferred cleanup: `schedule_event_channel_cleanup()` spawns a task that sleeps for `channel_ttl` and then calls `remove_event_channel_if_stale()`.
- Safety checks before removal (`remove_event_channel_if_stale`): the channel is removed only if:
	- the stored `cleanup_generation` still matches the one captured when the cleanup was scheduled (prevents races with new subscribers),
	- `cancellations.is_empty()` (no outstanding per-observer cancellation handles), and
	- `sender.receiver_count() == 0` (no active broadcast receivers).
- Effect: removing the `EventChannel` entry drops the last `broadcast::Sender` clone(s) stored in the registry, which allows the broadcast channel to close and frees associated resources.

Notes:
- Cleanup is eventual (TTL) and cooperative: subscribers may still process an in-flight broadcast message before their task exits.
- The generation token prevents a scheduled cleanup from removing a newly re-created/used channel.
