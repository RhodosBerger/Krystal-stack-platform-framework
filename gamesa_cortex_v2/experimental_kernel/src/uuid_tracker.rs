//! UUID Tracking Module with Latency Measurement
//! 
//! This module provides high-performance task tracking with nanosecond precision
//! for the Krystal Stack experimental kernel layer.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use dashmap::DashMap;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use pyo3::prelude::*;

/// Task state in the lifecycle
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskState {
    Pending,
    Scheduled,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Task metadata with full lifecycle tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetadata {
    pub uuid: String,
    pub parent_uuid: Option<String>,
    pub task_type: String,
    pub priority: u8,
    pub state: TaskState,
    pub created_at: u128,
    pub started_at: Option<u128>,
    pub completed_at: Option<u128>,
    pub latency_ns: Option<u128>,
    pub execution_time_ns: Option<u128>,
    pub credits_cost: u32,
    pub tags: Vec<String>,
}

/// Latency statistics for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub min_ns: u128,
    pub max_ns: u128,
    pub avg_ns: u128,
    pub p50_ns: u128,
    pub p95_ns: u128,
    pub p99_ns: u128,
    pub count: u64,
}

/// High-performance UUID tracker with lock-free concurrent access
#[pyclass]
pub struct UUIDTracker {
    tasks: Arc<DashMap<String, TaskMetadata>>,
    completed_tasks: Arc<DashMap<String, TaskMetadata>>,
    latency_history: Arc<crossbeam::queue::SegQueue<u128>>,
    start_time: Instant,
}

#[pymethods]
impl UUIDTracker {
    #[new]
    fn new() -> Self {
        UUIDTracker {
            tasks: Arc::new(DashMap::new()),
            completed_tasks: Arc::new(DashMap::new()),
            latency_history: Arc::new(crossbeam::queue::SegQueue::new()),
            start_time: Instant::now(),
        }
    }

    /// Create a new task with automatic UUID generation
    #[pyo3(text_signature = "($self, task_type, priority=128, credits_cost=0, tags=None)")]
    fn create_task(&self, task_type: String, priority: u8, credits_cost: u32, tags: Option<Vec<String>>) -> String {
        let uuid = Uuid::new_v4().to_string();
        let metadata = TaskMetadata {
            uuid: uuid.clone(),
            parent_uuid: None,
            task_type,
            priority,
            state: TaskState::Pending,
            created_at: self.start_time.elapsed().as_nanos(),
            started_at: None,
            completed_at: None,
            latency_ns: None,
            execution_time_ns: None,
            credits_cost,
            tags: tags.unwrap_or_default(),
        };
        self.tasks.insert(uuid.clone(), metadata);
        uuid
    }

    /// Create a child task linked to a parent
    #[pyo3(text_signature = "($self, parent_uuid, task_type, priority=128, credits_cost=0, tags=None)")]
    fn create_child_task(&self, parent_uuid: String, task_type: String, priority: u8, credits_cost: u32, tags: Option<Vec<String>>) -> Option<String> {
        if !self.tasks.contains_key(&parent_uuid) {
            return None;
        }
        
        let uuid = Uuid::new_v4().to_string();
        let metadata = TaskMetadata {
            uuid: uuid.clone(),
            parent_uuid: Some(parent_uuid),
            task_type,
            priority,
            state: TaskState::Pending,
            created_at: self.start_time.elapsed().as_nanos(),
            started_at: None,
            completed_at: None,
            latency_ns: None,
            execution_time_ns: None,
            credits_cost,
            tags: tags.unwrap_or_default(),
        };
        self.tasks.insert(uuid.clone(), metadata);
        Some(uuid)
    }

    /// Mark task as started
    fn start_task(&self, uuid: String) -> bool {
        if let Some(mut task) = self.tasks.get_mut(&uuid) {
            task.state = TaskState::Running;
            task.started_at = Some(self.start_time.elapsed().as_nanos());
            true
        } else {
            false
        }
    }

    /// Mark task as completed with latency calculation
    fn complete_task(&self, uuid: String) -> Option<u128> {
        if let Some(mut task) = self.tasks.get_mut(&uuid) {
            let now = self.start_time.elapsed().as_nanos();
            task.state = TaskState::Completed;
            task.completed_at = Some(now);
            
            // Calculate latency (time from creation to completion)
            if let Some(started) = task.started_at {
                task.execution_time_ns = Some(now - started);
            }
            task.latency_ns = Some(now - task.created_at);
            
            let latency = task.latency_ns.unwrap();
            self.latency_history.push(latency);
            
            // Move to completed tasks
            let task_clone = task.clone();
            drop(task);
            self.tasks.remove(&uuid);
            self.completed_tasks.insert(uuid, task_clone);
            
            Some(latency)
        } else {
            None
        }
    }

    /// Mark task as failed
    fn fail_task(&self, uuid: String, reason: String) -> bool {
        if let Some(mut task) = self.tasks.get_mut(&uuid) {
            task.state = TaskState::Failed;
            task.completed_at = Some(self.start_time.elapsed().as_nanos());
            task.tags.push(format!("fail_reason:{}", reason));
            true
        } else {
            false
        }
    }

    /// Cancel a task
    fn cancel_task(&self, uuid: String) -> bool {
        if let Some(task) = self.tasks.remove(&uuid) {
            let mut cancelled_task = task.1;
            cancelled_task.state = TaskState::Cancelled;
            cancelled_task.completed_at = Some(self.start_time.elapsed().as_nanos());
            self.completed_tasks.insert(cancelled_task.uuid, cancelled_task);
            true
        } else {
            false
        }
    }

    /// Get task metadata
    fn get_task(&self, uuid: String) -> Option<Py<PyAny>> {
        Python::with_gil(|py| {
            if let Some(task) = self.tasks.get(&uuid) {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("uuid", task.uuid.clone()).ok();
                dict.set_item("parent_uuid", task.parent_uuid.clone()).ok();
                dict.set_item("task_type", task.task_type.clone()).ok();
                dict.set_item("priority", task.priority).ok();
                dict.set_item("state", format!("{:?}", task.state)).ok();
                dict.set_item("created_at", task.created_at).ok();
                dict.set_item("started_at", task.started_at).ok();
                dict.set_item("completed_at", task.completed_at).ok();
                dict.set_item("latency_ns", task.latency_ns).ok();
                dict.set_item("execution_time_ns", task.execution_time_ns).ok();
                dict.set_item("credits_cost", task.credits_cost).ok();
                dict.set_item("tags", task.tags.clone()).ok();
                Some(dict.into())
            } else {
                None
            }
        })
    }

    /// Get latency statistics
    #[pyo3(text_signature = "($self)")]
    fn get_latency_stats(&self) -> Py<PyAny> {
        Python::with_gil(|py| {
            let mut latencies: Vec<u128> = Vec::new();
            while let Ok(latency) = self.latency_history.pop() {
                latencies.push(latency);
            }
            
            // Put them back (we're just reading)
            for lat in &latencies {
                self.latency_history.push(*lat);
            }
            
            if latencies.is_empty() {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("min_ns", 0).ok();
                dict.set_item("max_ns", 0).ok();
                dict.set_item("avg_ns", 0).ok();
                dict.set_item("p50_ns", 0).ok();
                dict.set_item("p95_ns", 0).ok();
                dict.set_item("p99_ns", 0).ok();
                dict.set_item("count", 0).ok();
                return dict.into();
            }
            
            latencies.sort();
            let count = latencies.len() as u64;
            let min = *latencies.first().unwrap();
            let max = *latencies.last().unwrap();
            let avg = latencies.iter().sum::<u128>() / count as u128;
            let p50 = latencies[(count as f64 * 0.50) as usize];
            let p95 = latencies[(count as f64 * 0.95) as usize];
            let p99 = latencies[(count as f64 * 0.99) as usize];
            
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("min_ns", min).ok();
            dict.set_item("max_ns", max).ok();
            dict.set_item("avg_ns", avg).ok();
            dict.set_item("p50_ns", p50).ok();
            dict.set_item("p95_ns", p95).ok();
            dict.set_item("p99_ns", p99).ok();
            dict.set_item("count", count).ok();
            dict.into()
        })
    }

    /// Get number of active tasks
    fn active_count(&self) -> usize {
        self.tasks.len()
    }

    /// Get number of completed tasks
    fn completed_count(&self) -> usize {
        self.completed_tasks.len()
    }

    /// Clear completed tasks history
    fn clear_completed(&self) {
        self.completed_tasks.clear()
    }

    /// Export task history to JSON
    fn export_to_json(&self) -> String {
        let completed: Vec<_> = self.completed_tasks.iter().map(|r| r.value().clone()).collect();
        serde_json::to_string_pretty(&completed).unwrap_or_default()
    }
}

/// Task priority levels for the scheduler
pub mod priorities {
    pub const CRITICAL_SAFETY: u8 = 255;
    pub const REALTIME_CONTROL: u8 = 200;
    pub const HIGH_INFERENCE: u8 = 150;
    pub const NORMAL_COMPUTE: u8 = 100;
    pub const BACKGROUND_LOG: u8 = 50;
    pub const LOW_TELEMETRY: u8 = 10;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uuid_tracker() {
        let tracker = UUIDTracker::new();
        let uuid = tracker.create_task("test_task".to_string(), 128, 10, Some(vec!["test".to_string()]));
        
        assert!(tracker.start_task(uuid.clone()));
        std::thread::sleep(Duration::from_millis(10));
        let latency = tracker.complete_task(uuid.clone());
        
        assert!(latency.is_some());
        assert!(latency.unwrap() >= 10_000_000); // At least 10ms in nanoseconds
    }
}
