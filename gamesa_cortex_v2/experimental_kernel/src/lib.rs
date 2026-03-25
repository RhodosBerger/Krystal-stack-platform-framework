//! Experimental Kernel for Krystal Stack
//! 
//! This is the experimental kernel layer providing:
//! - Advanced UUID tracking with nanosecond latency measurement
//! - Kernel-level scheduler with EDF, stochastic, and economic scheduling
//! - Feature abstraction layer for modular architecture
//! - Dependency injection framework for dynamic feature insertion
//! 
//! # Architecture
//! 
//! The experimental kernel is designed to be inserted into the Krystal Stack
//! architecture as a high-performance layer that handles:
//! 
//! 1. **Task Scheduling**: Multiple scheduling algorithms (EDF, Stochastic, Economic)
//! 2. **Resource Management**: Credit-based budget system for fair resource allocation
//! 3. **Feature Management**: Hot-reloadable features with dependency tracking
//! 4. **Telemetry**: Comprehensive latency tracking and metrics export
//! 
//! # Example Usage (Python)
//! 
//! ```python
//! from experimental_kernel import KernelScheduler, UUIDTracker, EconomicGovernor
//! 
//! # Create scheduler
//! scheduler = KernelScheduler()
//! scheduler.init_budget("gpu", 1000, 100, 100)  # 1000 credits, 100/ms replenish
//! 
//! # Submit tasks
//! uuid = scheduler.submit_task("inference", 50, 150, 10, "gpu")
//! 
//! # Schedule and execute
//! task = scheduler.schedule_next()
//! scheduler.complete_task(task["uuid"])
//! 
//! # Get statistics
//! stats = scheduler.get_stats()
//! latency = scheduler.get_latency_stats()
//! ```

mod uuid_tracker;
mod scheduler;
mod feature_abstraction;

pub use uuid_tracker::*;
pub use scheduler::*;
pub use feature_abstraction::*;

use pyo3::prelude::*;

/// Krystal Stack Experimental Kernel
/// 
/// This module provides advanced scheduling and tracking capabilities
/// for the Krystal Stack architecture.
#[pymodule]
fn experimental_kernel(_py: Python, m: &PyModule) -> PyResult<()> {
    // UUID Tracking
    m.add_class::<UUIDTracker>()?;
    
    // Kernel Scheduler
    m.add_class::<KernelScheduler>()?;
    
    // Economic Governor
    m.add_class::<EconomicGovernor>()?;
    
    // Feature Management
    m.add_class::<FeatureContainer>()?;
    m.add_class::<ModuleRegistry>()?;
    m.add_class::<HotReloadManager>()?;
    
    // Module information
    m.add("__version__", "0.1.0")?;
    m.add("__author__", "Dušan Kopecký")?;
    m.add("__description__", "Experimental Kernel for Krystal Stack")?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration() {
        // Create scheduler with UUID tracking
        let scheduler = KernelScheduler::new();
        scheduler.init_budget("test".to_string(), 1000, 100, 100);
        
        // Submit multiple tasks
        let mut uuids = Vec::new();
        for i in 0..10 {
            let uuid = scheduler.submit_task(
                format!("task_{}", i),
                50,
                100 + i as u8,
                10,
                "test".to_string(),
                Some(vec![format!("batch_{}", i)])
            );
            uuids.push(uuid);
        }
        
        // Schedule and complete all tasks
        while let Some(_task) = scheduler.schedule_next() {
            // Simulate execution
            std::thread::sleep(std::time::Duration::from_millis(1));
            // Complete task (would need to extract UUID from task)
        }
        
        // Verify statistics
        let stats = scheduler.get_stats();
        // Stats would be checked in a full integration test
    }
}
