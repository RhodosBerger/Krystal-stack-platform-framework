//! Governor system - trait-based safety control mechanisms with statistical backing
//!
//! Governors are responsible for making real-time safety decisions based on:
//! - Current system state (metrics)
//! - Historical patterns (statistical models)
//! - Confidence intervals and uncertainty quantification
//!
//! Unlike deterministic governors (hard limits), statistical governors adapt safely
//! based on observed variance and confidence in the operating conditions.

mod config;
mod state;
mod traits;

pub use config::GovernorConfig;
pub use state::GovernorState;
pub use traits::Governor;

use crate::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Statistical governor decision with confidence and reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernorDecision {
    /// Unique decision ID for tracing
    pub id: String,
    /// Whether the action is allowed
    pub allowed: bool,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Recommended safe limit
    pub safe_limit: Option<f64>,
    /// Reason for the decision
    pub reason: String,
    /// Timestamp of decision
    pub timestamp: chrono::DateTime<Utc>,
}

impl GovernorDecision {
    pub fn new(allowed: bool, confidence: f64, reason: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            allowed,
            confidence,
            safe_limit: None,
            reason: reason.into(),
            timestamp: Utc::now(),
        }
    }

    pub fn with_limit(mut self, limit: f64) -> Self {
        self.safe_limit = Some(limit);
        self
    }
}

/// Registry for multiple governors working in concert
pub struct GovernorRegistry {
    governors: HashMap<String, Box<dyn Governor>>,
}

impl GovernorRegistry {
    pub fn new() -> Self {
        Self {
            governors: HashMap::new(),
        }
    }

    pub fn register(&mut self, name: String, governor: Box<dyn Governor>) -> Result<()> {
        self.governors.insert(name, governor);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&dyn Governor> {
        self.governors.get(name).map(|b| &**b)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut Box<dyn Governor>> {
        self.governors.get_mut(name)
    }

    pub fn list_governors(&self) -> Vec<String> {
        self.governors.keys().cloned().collect()
    }
}

impl Default for GovernorRegistry {
    fn default() -> Self {
        Self::new()
    }
}
