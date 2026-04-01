use crate::mediator::{IEvent, Mediator};
use std::any::TypeId;
use std::sync::{Arc, mpsc};
use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ProgressEvent {
    value: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StatusEvent {
    message: &'static str,
}

fn recv_value<T>(rx: &mpsc::Receiver<T>) -> T {
    rx.recv_timeout(Duration::from_secs(1))
        .expect("expected mediator event")
}

#[test]
fn publish_should_fan_out_to_all_subscribers() {
    let mediator = Mediator::new();
    let (left_tx, left_rx) = mpsc::channel();
    let (right_tx, right_rx) = mpsc::channel();

    let left_id = mediator.subscribe::<ProgressEvent, _>(move |event| {
        let _ = left_tx.send(event.value);
    });
    let right_id = mediator.subscribe::<ProgressEvent, _>(move |event| {
        let _ = right_tx.send(event.value);
    });

    mediator.publish(Arc::new(ProgressEvent { value: 42 }));

    assert_eq!(recv_value(&left_rx), 42);
    assert_eq!(recv_value(&right_rx), 42);
    assert!(mediator.unsubscribe::<ProgressEvent>(left_id));
    assert!(mediator.unsubscribe::<ProgressEvent>(right_id));
}

#[test]
fn unsubscribe_one_observer_does_not_stop_others() {
    let mediator = Mediator::new_with_channel_ttl(Duration::from_millis(0));
    let (left_tx, left_rx) = mpsc::channel();
    let (right_tx, right_rx) = mpsc::channel();

    let left_id = mediator.subscribe::<ProgressEvent, _>(move |event| {
        let _ = left_tx.send(event.value);
    });
    let right_id = mediator.subscribe::<ProgressEvent, _>(move |event| {
        let _ = right_tx.send(event.value);
    });

    // Prime both observers
    mediator.publish(Arc::new(ProgressEvent { value: 1 }));
    assert_eq!(recv_value(&left_rx), 1);
    assert_eq!(recv_value(&right_rx), 1);

    // Unsubscribe left only, publish again
    assert!(mediator.unsubscribe::<ProgressEvent>(left_id));
    mediator.publish(Arc::new(ProgressEvent { value: 2 }));

    // Right should still get the event
    assert_eq!(recv_value(&right_rx), 2);
    assert!(mediator.unsubscribe::<ProgressEvent>(right_id));
}

#[test]
fn publish_should_only_notify_matching_event_type() {
    let mediator = Mediator::new();
    let (tx, rx) = mpsc::channel();

    let observer_id = mediator.subscribe::<StatusEvent, _>(move |event| {
        let _ = tx.send(event.message);
    });

    mediator.publish(Arc::new(ProgressEvent { value: 9 }));
    mediator.publish(Arc::new(StatusEvent { message: "ok" }));

    assert_eq!(recv_value(&rx), "ok");
    assert!(mediator.unsubscribe::<StatusEvent>(observer_id));
}

fn assert_i_event<T: IEvent>() {}

#[test]
fn event_types_should_implement_i_event_via_blanket_impl() {
    assert_i_event::<ProgressEvent>();
    assert_i_event::<StatusEvent>();
}

#[test]
fn deferred_cleanup_should_be_triggered_when_all_receivers_drop() {
    let ttl = Duration::from_millis(10);
    let mediator = Mediator::new_with_channel_ttl(ttl);
    let (tx, rx) = mpsc::channel();

    let observer_id = mediator.subscribe::<ProgressEvent, _>(move |_event| {
        let _ = tx.send(());
    });

    // ensure the channel exists while the subscriber is active
    assert!(mediator.has_event_channel(TypeId::of::<ProgressEvent>()));

    // unsubscribe the only observer; mediator should still have the channel until TTL
    assert!(mediator.unsubscribe::<ProgressEvent>(observer_id));
    assert!(mediator.has_event_channel(TypeId::of::<ProgressEvent>()));

    // wait longer than the TTL for deferred cleanup to run
    std::thread::sleep(ttl * 3);

    // channel should be removed after TTL
    assert!(!mediator.has_event_channel(TypeId::of::<ProgressEvent>()));

    // publishing after cleanup should not deliver anything
    mediator.publish(Arc::new(ProgressEvent { value: 7 }));
    assert!(rx.recv_timeout(Duration::from_millis(200)).is_err());
}

#[test]
fn unsubscribe_observer_should_receive_no_more_events() {
    // Use TTL=0 so cleanup is scheduled immediately and we can wait
    let mediator = Mediator::new_with_channel_ttl(Duration::from_millis(0));
    let (left_tx, left_rx) = mpsc::channel();
    let (right_tx, right_rx) = mpsc::channel();

    let left_id = mediator.subscribe::<ProgressEvent, _>(move |event| {
        let _ = left_tx.send(event.value);
    });
    let right_id = mediator.subscribe::<ProgressEvent, _>(move |event| {
        let _ = right_tx.send(event.value);
    });

    // Prime both observers
    mediator.publish(Arc::new(ProgressEvent { value: 1 }));
    assert_eq!(recv_value(&left_rx), 1);
    assert_eq!(recv_value(&right_rx), 1);

    // Unsubscribe left only. With TTL==0, cleanup will remove the channel.
    assert!(mediator.unsubscribe::<ProgressEvent>(left_id));

    // Wait for the observer to receive cancellation request and clean up
    std::thread::sleep(Duration::from_secs(5));

    // Now publish; right should receive, left must not.
    mediator.publish(Arc::new(ProgressEvent { value: 2 }));
    assert_eq!(recv_value(&right_rx), 2);
    assert!(left_rx.recv_timeout(Duration::from_millis(50)).is_err());

    assert!(mediator.unsubscribe::<ProgressEvent>(right_id));
}
