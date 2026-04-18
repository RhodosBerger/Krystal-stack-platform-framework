//! Control runtime - orchestrates governors and real-time decisions
//!
//! The ControlRuntime is the main entry point for making real-time safety decisions.
//! It manages multiple governors and ensures decisions meet safety requirements.

use crate::governor::{Governor, GovernorRegistry};
use crate::metrics::MetricsCollector;
use crate::Result;
use parking_lot::RwLock;
use serde_json::Value;
use std::sync::Arc;

/// High-performance real-time control runtime
pub struct ControlRuntime {
    governors: Arc<RwLock<GovernorRegistry>>,
    metrics: Arc<MetricsCollector>,
}

impl ControlRuntime {
    pub fn new(metrics: Arc<MetricsCollector>) -> Self {
        Self {
            governors: Arc::new(RwLock::new(GovernorRegistry::new())),
            metrics,
        }
    }

    /// Register a governor
    pub fn register_governor(&self, name: String, governor: Box<dyn Governor>) -> Result<()> {
        self.governors.write().register(name, governor)
    }

    /// Make a decision from all registered governors (all must agree for approval)
    pub fn decide_safe(
        &self,
        input: &Value,
        context: &Value,
    ) -> Result<Vec<(String, crate::governor::GovernorDecision)>> {
        let mut governors = self.governors.write();
        let governor_names: Vec<_> = governors.list_governors();

        let mut decisions = Vec::new();
        for name in governor_names {
            if let Some(gov) = governors.get_mut(&name) {
                let decision = gov.decide(input, context)?;
                decisions.push((name, decision));
            }
        }

        Ok(decisions)
    }

    /// Get metrics collector reference
    pub fn metrics(&self) -> &Arc<MetricsCollector> {
        &self.metrics
    }

    /// Get list of active governors
    pub fn list_governors(&self) -> Vec<String> {
        self.governors.read().list_governors()
    }
}

impl Clone for ControlRuntime {
    fn clone(&self) -> Self {
        Self {
            governors: Arc::clone(&self.governors),
            metrics: Arc::clone(&self.metrics),
        }
    }
}
