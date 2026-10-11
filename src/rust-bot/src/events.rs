//! Event Bus for async communication

use crate::{TelemetrySnapshot, Action, Domain};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub timestamp: DateTime<Utc>,
    pub kind: EventKind,
    pub source: String,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventKind {
    Telemetry,
    Signal,
    Decision,
    Error,
    StateChange,
}

pub struct EventBus {
    queue: VecDeque<Event>,
    capacity: usize,
}

impl EventBus {
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn emit(&mut self, event: Event) {
        if self.queue.len() >= self.capacity {
            self.queue.pop_front();
        }
        self.queue.push_back(event);
    }

    pub fn poll(&mut self) -> Option<Event> {
        self.queue.pop_front()
    }

    pub fn peek(&self) -> Option<&Event> {
        self.queue.front()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}
