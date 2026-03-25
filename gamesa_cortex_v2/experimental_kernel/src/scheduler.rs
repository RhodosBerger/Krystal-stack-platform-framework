//! Experimental Kernel Scheduler
//! 
//! Advanced scheduling algorithms for the Krystal Stack including:
//! - Earliest Deadline First (EDF) with dynamic priority boosting
//! - Stochastic scheduling with probability-based task selection
//! - Economic governor with credit-based resource allocation
//! - Real-time kernel bypass for critical safety tasks

use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering as AtomicOrdering};
use crossbeam::queue::SegQueue;
use dashmap::DashMap;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

use crate::uuid_tracker::{UUIDTracker, TaskMetadata, TaskState};

/// Scheduled task with deadline and priority
#[derive(Debug, Clone)]
struct ScheduledTask {
    uuid: String,
    deadline_ns: u128,
    priority: u8,
    credits_cost: u32,
    task_type: String,
    creation_order: u64,
}

impl PartialEq for ScheduledTask {
    fn eq(&self, other: &Self) -> bool {
        self.uuid == other.uuid
    }
}

impl Eq for ScheduledTask {}

impl PartialOrd for ScheduledTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then earlier deadline, then FIFO
        other.priority.cmp(&self.priority)
            .then_with(|| self.deadline_ns.cmp(&other.deadline_ns))
            .then_with(|| other.creation_order.cmp(&self.creation_order))
    }
}

/// Credit budget for economic scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditBudget {
    pub total_credits: u32,
    pub available_credits: u32,
    pub replenishment_rate: u32,
    pub last_replenish: u128,
    pub replenish_interval_ns: u128,
}

/// Scheduling mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SchedulingMode {
    /// Standard EDF scheduling
    Normal,
    /// Real-time mode with priority boosting
    RealTime,
    /// Power-saving mode with task batching
    Eco,
    /// Performance mode with aggressive prefetching
    Performance,
}

/// Kernel scheduler with advanced features
#[pyclass]
pub struct KernelScheduler {
    ready_queue: Arc<SegQueue<ScheduledTask>>,
    priority_queue: Arc<DashMap<u8, BinaryHeap<ScheduledTask>>>,
    uuid_tracker: Arc<UUIDTracker>,
    credit_budget: Arc<dashmap::DashMap<String, CreditBudget>>,
    mode: AtomicU64,
    running: AtomicBool,
    task_counter: AtomicU64,
    start_time: Instant,
    rng: Arc<std::sync::Mutex<ChaCha8Rng>>,
    
    // Scheduling statistics
    tasks_scheduled: AtomicU64,
    tasks_completed: AtomicU64,
    tasks_deadline_missed: AtomicU64,
    avg_wait_time_ns: AtomicU64,
}

#[pymethods]
impl KernelScheduler {
    #[new]
    fn new() -> Self {
        KernelScheduler {
            ready_queue: Arc::new(SegQueue::new()),
            priority_queue: Arc::new(DashMap::new()),
            uuid_tracker: Arc::new(UUIDTracker::new()),
            credit_budget: Arc::new(DashMap::new()),
            mode: AtomicU64::new(SchedulingMode::Normal as u64),
            running: AtomicBool::new(false),
            task_counter: AtomicU64::new(0),
            start_time: Instant::now(),
            rng: Arc::new(std::sync::Mutex::new(ChaCha8Rng::from_entropy())),
            tasks_scheduled: AtomicU64::new(0),
            tasks_completed: AtomicU64::new(0),
            tasks_deadline_missed: AtomicU64::new(0),
            avg_wait_time_ns: AtomicU64::new(0),
        }
    }

    /// Initialize credit budget for a resource domain
    #[pyo3(text_signature = "($self, domain, total_credits, replenishment_rate, replenish_interval_ms)")]
    fn init_budget(&self, domain: String, total_credits: u32, replenishment_rate: u32, replenish_interval_ms: u64) {
        let budget = CreditBudget {
            total_credits,
            available_credits: total_credits,
            replenishment_rate,
            last_replenish: 0,
            replenish_interval_ns: replenish_interval_ms as u128 * 1_000_000,
        };
        self.credit_budget.insert(domain, budget);
    }

    /// Submit a task to the scheduler
    #[pyo3(text_signature = "($self, task_type, deadline_ms, priority, credits_cost, domain, tags=None)")]
    fn submit_task(&self, task_type: String, deadline_ms: u64, priority: u8, credits_cost: u32, domain: String, tags: Option<Vec<String>>) -> Option<String> {
        // Check credit budget
        if let Some(mut budget) = self.credit_budget.get_mut(&domain) {
            if budget.available_credits < credits_cost {
                // Try to replenish first
                self.replenish_budget(&domain);
                
                // Check again after replenishment
                if let Some(budget) = self.credit_budget.get(&domain) {
                    if budget.available_credits < credits_cost {
                        // Reject task due to insufficient credits
                        return None;
                    }
                }
            }
            
            // Deduct credits
            budget.available_credits -= credits_cost;
        } else {
            // Initialize default budget if domain doesn't exist
            self.init_budget(domain.clone(), 1000, 100, 100);
        }
        
        // Create UUID for the task
        let uuid = self.uuid_tracker.create_task(task_type.clone(), priority, credits_cost, tags);
        let creation_order = self.task_counter.fetch_add(1, AtomicOrdering::SeqCst);
        
        let scheduled_task = ScheduledTask {
            uuid: uuid.clone(),
            deadline_ns: self.start_time.elapsed().as_nanos() + (deadline_ms as u128 * 1_000_000),
            priority,
            credits_cost,
            task_type: task_type.clone(),
            creation_order,
        };
        
        // Add to priority queue
        self.priority_queue
            .entry(priority)
            .or_insert_with(BinaryHeap::new)
            .push(scheduled_task);
        
        self.tasks_scheduled.fetch_add(1, AtomicOrdering::Relaxed);
        
        Some(uuid)
    }

    /// Schedule next task using EDF with priority boosting
    #[pyo3(text_signature = "($self)")]
    fn schedule_next(&self) -> Option<Py<PyAny>> {
        Python::with_gil(|py| {
            let now = self.start_time.elapsed().as_nanos();
            let mode = self.get_mode();
            
            // Collect all tasks from priority queues
            let mut all_tasks: Vec<ScheduledTask> = Vec::new();
            for mut entry in self.priority_queue.iter_mut() {
                while let Some(task) = entry.value_mut().pop() {
                    all_tasks.push(task);
                }
            }
            
            if all_tasks.is_empty() {
                return None;
            }
            
            // Sort by deadline (EDF)
            all_tasks.sort_by_key(|t| t.deadline_ns);
            
            // Apply mode-specific adjustments
            match mode {
                SchedulingMode::RealTime => {
                    // Boost priority for tasks close to deadline
                    for task in &mut all_tasks {
                        let time_to_deadline = task.deadline_ns.saturating_sub(now);
                        if time_to_deadline < 1_000_000 { // Less than 1ms
                            task.priority = task.priority.saturating_add(50);
                        }
                    }
                    all_tasks.sort_by(|a, b| b.priority.cmp(&a.priority));
                }
                SchedulingMode::Eco => {
                    // Batch tasks with same type
                    all_tasks.sort_by(|a, b| {
                        a.task_type.cmp(&b.task_type)
                            .then_with(|| a.deadline_ns.cmp(&b.deadline_ns))
                    });
                }
                SchedulingMode::Performance => {
                    // Prefer tasks with higher priority
                    all_tasks.sort_by(|a, b| b.priority.cmp(&a.priority));
                }
                _ => {} // Normal mode uses EDF
            }
            
            // Select the best task
            let selected_task = all_tasks.first()?.clone();
            
            // Put remaining tasks back
            for task in all_tasks.into_iter().skip(1) {
                self.priority_queue
                    .entry(task.priority)
                    .or_insert_with(BinaryHeap::new)
                    .push(task);
            }
            
            // Start the task
            self.uuid_tracker.start_task(selected_task.uuid.clone());
            
            // Return task info
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("uuid", selected_task.uuid).ok();
            dict.set_item("task_type", selected_task.task_type).ok();
            dict.set_item("priority", selected_task.priority).ok();
            dict.set_item("deadline_ns", selected_task.deadline_ns).ok();
            dict.set_item("credits_cost", selected_task.credits_cost).ok();
            Some(dict.into())
        })
    }

    /// Complete a task
    fn complete_task(&self, uuid: String) -> bool {
        let result = self.uuid_tracker.complete_task(uuid);
        if result.is_some() {
            self.tasks_completed.fetch_add(1, AtomicOrdering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Get scheduling statistics
    #[pyo3(text_signature = "($self)")]
    fn get_stats(&self) -> Py<PyAny> {
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("tasks_scheduled", self.tasks_scheduled.load(AtomicOrdering::Relaxed)).ok();
            dict.set_item("tasks_completed", self.tasks_completed.load(AtomicOrdering::Relaxed)).ok();
            dict.set_item("tasks_deadline_missed", self.tasks_deadline_missed.load(AtomicOrdering::Relaxed)).ok();
            dict.set_item("avg_wait_time_ns", self.avg_wait_time_ns.load(AtomicOrdering::Relaxed)).ok();
            dict.set_item("active_tasks", self.uuid_tracker.active_count()).ok();
            dict.set_item("completed_tasks", self.uuid_tracker.completed_count()).ok();
            
            // Credit budget stats
            let budgets: Vec<_> = self.credit_budget.iter().map(|r| {
                let budget_dict = pyo3::types::PyDict::new(py);
                budget_dict.set_item("domain", r.key().clone()).ok();
                budget_dict.set_item("available", r.value().available_credits).ok();
                budget_dict.set_item("total", r.value().total_credits).ok();
                budget_dict.into()
            }).collect();
            dict.set_item("budgets", budgets).ok();
            
            dict.into()
        })
    }

    /// Set scheduling mode
    fn set_mode(&self, mode: String) {
        let new_mode = match mode.to_lowercase().as_str() {
            "normal" => SchedulingMode::Normal,
            "realtime" | "real_time" => SchedulingMode::RealTime,
            "eco" => SchedulingMode::Eco,
            "performance" => SchedulingMode::Performance,
            _ => SchedulingMode::Normal,
        };
        self.mode.store(new_mode as u64, AtomicOrdering::SeqCst);
    }

    /// Get current mode
    fn get_mode(&self) -> String {
        match self.mode.load(AtomicOrdering::SeqCst) {
            0 => "normal",
            1 => "realtime",
            2 => "eco",
            3 => "performance",
            _ => "unknown",
        }.to_string()
    }

    /// Get latency statistics from UUID tracker
    fn get_latency_stats(&self) -> Py<PyAny> {
        self.uuid_tracker.get_latency_stats()
    }

    /// Stochastic scheduling - select task with probability based on priority
    #[pyo3(text_signature = "($self)")]
    fn schedule_stochastic(&self) -> Option<Py<PyAny>> {
        Python::with_gil(|py| {
            let mut all_tasks: Vec<ScheduledTask> = Vec::new();
            for mut entry in self.priority_queue.iter_mut() {
                while let Some(task) = entry.value_mut().pop() {
                    all_tasks.push(task);
                }
            }
            
            if all_tasks.is_empty() {
                return None;
            }
            
            // Weighted random selection based on priority
            let total_weight: u32 = all_tasks.iter().map(|t| t.priority as u32 + 1).sum();
            let mut rng = self.rng.lock().unwrap();
            let mut random = rng.gen_range(0..total_weight);
            
            let mut selected_idx = 0;
            for (idx, task) in all_tasks.iter().enumerate() {
                if random < (task.priority as u32 + 1) {
                    selected_idx = idx;
                    break;
                }
                random -= task.priority as u32 + 1;
            }
            
            let selected_task = all_tasks.remove(selected_idx);
            
            // Put remaining tasks back
            for task in all_tasks {
                self.priority_queue
                    .entry(task.priority)
                    .or_insert_with(BinaryHeap::new)
                    .push(task);
            }
            
            self.uuid_tracker.start_task(selected_task.uuid.clone());
            
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("uuid", selected_task.uuid).ok();
            dict.set_item("task_type", selected_task.task_type).ok();
            dict.set_item("priority", selected_task.priority).ok();
            Some(dict.into())
        })
    }

    /// Check for deadline violations
    fn check_deadlines(&self) -> Vec<String> {
        let now = self.start_time.elapsed().as_nanos();
        let mut violations = Vec::new();
        
        for entry in self.priority_queue.iter() {
            for task in entry.value().iter() {
                if task.deadline_ns < now {
                    violations.push(task.uuid.clone());
                }
            }
        }
        
        if !violations.is_empty() {
            self.tasks_deadline_missed.fetch_add(violations.len() as u64, AtomicOrdering::Relaxed);
        }
        
        violations
    }

    /// Replenish credit budget
    fn replenish_budget(&self, domain: &str) {
        if let Some(mut budget) = self.credit_budget.get_mut(domain) {
            let now = self.start_time.elapsed().as_nanos();
            let elapsed = now - budget.last_replenish;
            
            if elapsed >= budget.replenish_interval_ns {
                let replenish_amount = (elapsed / budget.replenish_interval_ns) as u32 * budget.replenishment_rate;
                budget.available_credits = (budget.available_credits + replenish_amount)
                    .min(budget.total_credits);
                budget.last_replenish = now;
            }
        }
    }

    /// Export scheduler state to JSON
    fn export_state(&self) -> String {
        let stats = self.get_stats();
        Python::with_gil(|py| {
            let stats_dict = stats.downcast::<pyo3::types::PyDict>(py).unwrap();
            serde_json::to_string_pretty(&stats_dict).unwrap_or_default()
        })
    }
}

/// Economic governor for resource allocation
#[pyclass]
pub struct EconomicGovernor {
    budgets: Arc<DashMap<String, CreditBudget>>,
    start_time: Instant,
}

#[pymethods]
impl EconomicGovernor {
    #[new]
    fn new() -> Self {
        EconomicGovernor {
            budgets: Arc::new(DashMap::new()),
            start_time: Instant::now(),
        }
    }

    #[pyo3(text_signature = "($self, domain, total_credits, replenishment_rate, replenish_interval_ms)")]
    fn allocate_budget(&self, domain: String, total_credits: u32, replenishment_rate: u32, replenish_interval_ms: u64) {
        let budget = CreditBudget {
            total_credits,
            available_credits: total_credits,
            replenishment_rate,
            last_replenish: 0,
            replenish_interval_ns: replenish_interval_ms as u128 * 1_000_000,
        };
        self.budgets.insert(domain, budget);
    }

    #[pyo3(text_signature = "($self, domain, cost)")]
    fn request_credits(&self, domain: String, cost: u32) -> bool {
        if let Some(mut budget) = self.budgets.get_mut(&domain) {
            // Auto-replenish if needed
            let now = self.start_time.elapsed().as_nanos();
            if now - budget.last_replenish >= budget.replenish_interval_ns {
                budget.available_credits = (budget.available_credits + budget.replenishment_rate)
                    .min(budget.total_credits);
                budget.last_replenish = now;
            }
            
            if budget.available_credits >= cost {
                budget.available_credits -= cost;
                return true;
            }
        }
        false
    }

    #[pyo3(text_signature = "($self, domain)")]
    fn get_budget_status(&self, domain: String) -> Option<Py<PyAny>> {
        Python::with_gil(|py| {
            if let Some(budget) = self.budgets.get(&domain) {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("domain", domain).ok();
                dict.set_item("available", budget.available_credits).ok();
                dict.set_item("total", budget.total_credits).ok();
                dict.set_item("replenishment_rate", budget.replenishment_rate).ok();
                Some(dict.into())
            } else {
                None
            }
        })
    }

    #[pyo3(text_signature = "($self)")]
    fn get_all_budgets(&self) -> Vec<Py<PyAny>> {
        Python::with_gil(|py| {
            self.budgets.iter().map(|r| {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("domain", r.key().clone()).ok();
                dict.set_item("available", r.value().available_credits).ok();
                dict.set_item("total", r.value().total_credits).ok();
                dict.set_item("replenishment_rate", r.value().replenishment_rate).ok();
                dict.into()
            }).collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_basic() {
        let scheduler = KernelScheduler::new();
        scheduler.init_budget("gpu".to_string(), 1000, 100, 100);
        
        let uuid = scheduler.submit_task(
            "inference".to_string(),
            50, // 50ms deadline
            150, // high priority
            10,  // credits cost
            "gpu".to_string(),
            Some(vec!["test".to_string()])
        );
        
        assert!(uuid.is_some());
        
        let task = scheduler.schedule_next();
        assert!(task.is_some());
        
        let uuid_str = uuid.unwrap();
        assert!(scheduler.complete_task(uuid_str.clone()));
    }

    #[test]
    fn test_economic_governor() {
        let governor = EconomicGovernor::new();
        governor.allocate_budget("cpu".to_string(), 100, 50, 1000);
        
        assert!(governor.request_credits("cpu".to_string(), 50));
        assert!(!governor.request_credits("cpu".to_string(), 60)); // Should fail
    }
}
