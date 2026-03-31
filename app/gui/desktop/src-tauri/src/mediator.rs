use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tokio::sync::{broadcast, watch};

const MEDIATOR_CHANNEL_CAPACITY: usize = 128;
const MEDIATOR_CHANNEL_TTL: Duration = Duration::from_secs(30);

/// Unique identifier assigned to each registered observer.
/// Returned by `Mediator::subscribe` and used to remove the observer
/// later with `Mediator::unsubscribe`.
pub type ObserverId = u64;

/// Type-erased, shareable event payload forwarded to observers.
///
/// Publishers pass an `Arc<E>` which is erased to `Arc<dyn Any + Send + Sync>`
/// so the same event allocation can be shared across multiple observers
/// without cloning the inner event value.
/// Subscribers recover the concrete type using `Arc::downcast::<E>`.
type EventPayload = Arc<dyn Any + Send + Sync>;

/// Cancellation signal sender for a single observer task.
type ObserverCancellation = watch::Sender<bool>;

/// Per-event broadcast channel plus per-observer cancellation handles.
struct EventChannel {
    sender: broadcast::Sender<EventPayload>,
    cancellations: HashMap<ObserverId, ObserverCancellation>,
    cleanup_generation: u64,
}

impl EventChannel {
    fn new() -> Self {
        let (sender, _) = broadcast::channel(MEDIATOR_CHANNEL_CAPACITY);
        Self {
            sender,
            cancellations: HashMap::new(),
            cleanup_generation: 0,
        }
    }
}

/// Core registry structure used by the mediator.
///
/// - The outer `HashMap` keys are `TypeId` values for concrete event types.
/// - Each value is an `EventChannel` containing one Tokio broadcast sender for
///   that event type plus cancellation handles for active observers.
///
/// This layout enables many-to-many pub/sub without tracking a separate data
/// sender per observer: publishing an event sends once into the broadcast
/// channel, and each subscriber listens on its own broadcast receiver.
type ObserverRegistry = HashMap<TypeId, EventChannel>;

pub trait IEvent: Any + Send + Sync + 'static {}

impl<T> IEvent for T where T: Any + Send + Sync + 'static {}

struct MediatorInner {
    next_observer_id: AtomicU64,
    observers: Mutex<ObserverRegistry>,
    channel_ttl: Duration,
}

impl MediatorInner {
    fn remove_event_channel_if_stale(
        &self,
        event_type: TypeId,
        cleanup_generation: u64,
    ) {
        let Ok(mut observers) = self.observers.lock() else {
            return;
        };

        let should_remove = observers.get(&event_type).is_some_and(|channel| {
            channel.cleanup_generation == cleanup_generation
                && channel.cancellations.is_empty()
                && channel.sender.receiver_count() == 0
        });

        if should_remove {
            let _ = observers.remove(&event_type);
        }
    }
}

#[derive(Clone)]
pub struct Mediator {
    inner: Arc<MediatorInner>,
}

impl Mediator {
    #[must_use]
    pub fn new() -> Self {
        Self::new_with_channel_ttl(MEDIATOR_CHANNEL_TTL)
    }

    #[must_use]
    pub(crate) fn new_with_channel_ttl(channel_ttl: Duration) -> Self {
        Self {
            inner: Arc::new(MediatorInner {
                next_observer_id: AtomicU64::new(1),
                observers: Mutex::new(HashMap::new()),
                channel_ttl,
            }),
        }
    }

    pub fn subscribe<E, F>(&self, observer: F) -> ObserverId
    where
        E: IEvent,
        F: Fn(&E) + Send + 'static,
    {
        let observer_id =
            self.inner.next_observer_id.fetch_add(1, Ordering::SeqCst);
        let event_type = TypeId::of::<E>();
        let mediator = self.clone();
        let subscription =
            if let Ok(mut observers) = self.inner.observers.lock() {
                let channel = observers
                    .entry(event_type)
                    .or_insert_with(EventChannel::new);
                channel.cleanup_generation =
                    channel.cleanup_generation.wrapping_add(1);
                let receiver = channel.sender.subscribe();
                let (cancel_tx, cancel_rx) = watch::channel(false);
                channel.cancellations.insert(observer_id, cancel_tx);
                Some((receiver, cancel_rx))
            } else {
                None
            };

        if let Some((mut receiver, mut cancel_rx)) = subscription {
            tauri::async_runtime::spawn(async move {
                loop {
                    tokio::select! {
                        received = receiver.recv() => {
                            match received {
                                Ok(payload) => {
                                    if let Ok(event) = Arc::downcast::<E>(payload) {
                                        observer(event.as_ref());
                                    }
                                }
                                Err(broadcast::error::RecvError::Lagged(_)) => {}
                                Err(broadcast::error::RecvError::Closed) => break,
                            }
                        }
                        changed = cancel_rx.changed() => {
                            if changed.is_err() || *cancel_rx.borrow() {
                                break;
                            }
                        }
                    }
                }

                let _ =
                    mediator.unsubscribe_by_type_id(event_type, observer_id);
            });
        }

        observer_id
    }

    pub fn publish<E>(&self, event: Arc<E>)
    where
        E: IEvent,
    {
        let event_type = TypeId::of::<E>();
        let Some(sender) = self.sender_for(event_type) else {
            return;
        };

        let payload: EventPayload = event;
        if sender.send(payload).is_err() && sender.receiver_count() == 0 {
            let cleanup_generation = {
                let Ok(observers) = self.inner.observers.lock() else {
                    return;
                };

                observers
                    .get(&event_type)
                    .map(|channel| channel.cleanup_generation)
            };

            if let Some(cleanup_generation) = cleanup_generation {
                self.schedule_event_channel_cleanup(
                    event_type,
                    cleanup_generation,
                );
            }
        }
    }

    #[must_use]
    pub fn unsubscribe<E>(&self, observer_id: ObserverId) -> bool
    where
        E: IEvent,
    {
        self.unsubscribe_by_type_id(TypeId::of::<E>(), observer_id)
    }

    fn sender_for(
        &self,
        event_type: TypeId,
    ) -> Option<broadcast::Sender<EventPayload>> {
        let Ok(observers) = self.inner.observers.lock() else {
            return None;
        };

        observers
            .get(&event_type)
            .map(|channel| channel.sender.clone())
    }

    fn unsubscribe_by_type_id(
        &self,
        event_type: TypeId,
        observer_id: ObserverId,
    ) -> bool {
        let cleanup_generation = {
            let Ok(mut observers) = self.inner.observers.lock() else {
                return false;
            };

            let Some(event_channel) = observers.get_mut(&event_type) else {
                return false;
            };

            let removed = if let Some(cancel_tx) =
                event_channel.cancellations.remove(&observer_id)
            {
                let _ = cancel_tx.send(true);
                true
            } else {
                false
            };

            if !removed {
                return false;
            }

            if event_channel.cancellations.is_empty() {
                event_channel.cleanup_generation =
                    event_channel.cleanup_generation.wrapping_add(1);
                Some(event_channel.cleanup_generation)
            } else {
                None
            }
        };

        if let Some(cleanup_generation) = cleanup_generation {
            self.schedule_event_channel_cleanup(event_type, cleanup_generation);
        }

        true
    }

    fn schedule_event_channel_cleanup(
        &self,
        event_type: TypeId,
        cleanup_generation: u64,
    ) {
        let mediator = self.clone();
        tauri::async_runtime::spawn(async move {
            tokio::time::sleep(mediator.inner.channel_ttl).await;
            mediator
                .inner
                .remove_event_channel_if_stale(event_type, cleanup_generation);
        });
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn has_event_channel(&self, event_type: TypeId) -> bool {
        let Ok(observers) = self.inner.observers.lock() else {
            return false;
        };

        observers.contains_key(&event_type)
    }
}

impl Default for Mediator {
    fn default() -> Self {
        Self::new()
    }
}

static MEDIATOR: OnceLock<Mediator> = OnceLock::new();

pub fn mediator() -> &'static Mediator {
    MEDIATOR.get_or_init(Mediator::new)
}

pub struct MediatorSubscription {
    event_type: TypeId,
    observer_id: Option<ObserverId>,
}

impl MediatorSubscription {
    #[must_use]
    pub fn new<E>(observer_id: ObserverId) -> Self
    where
        E: IEvent,
    {
        Self {
            event_type: TypeId::of::<E>(),
            observer_id: Some(observer_id),
        }
    }

    #[must_use]
    pub fn unsubscribe(mut self) -> bool {
        self.observer_id.take().is_some_and(|observer_id| {
            mediator().unsubscribe_by_type_id(self.event_type, observer_id)
        })
    }
}

impl Drop for MediatorSubscription {
    fn drop(&mut self) {
        if let Some(observer_id) = self.observer_id.take() {
            let _ =
                mediator().unsubscribe_by_type_id(self.event_type, observer_id);
        }
    }
}

#[cfg(test)]
mod tests {
    // Tests moved to the crate test harness under testing/rust/mediator_tests.rs
}
