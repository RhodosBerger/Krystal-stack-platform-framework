//! Governor configuration

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Configuration for a governor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernorConfig {
    /// Governor name/type
    pub name: String,
    /// Minimum confidence threshold for allowing actions
    pub confidence_threshold: f64,
    /// Enable adaptive learning from observations
    pub adaptive: bool,
    /// Custom parameters (governor-specific)
    pub params: Value,
    /// Enable safety-first mode (stricter decisions)
    pub safety_first: bool,
    /// Update frequency in Hz
    pub update_frequency: f64,
}

impl GovernorConfig {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            confidence_threshold: 0.9,
            adaptive: true,
            params: Value::Object(Default::default()),
            safety_first: true,
            update_frequency: 1000.0, // 1000 Hz by default
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn with_safety_first(mut self, safety_first: bool) -> Self {
        self.safety_first = safety_first;
        self
    }

    pub fn with_params(mut self, params: Value) -> Self {
        self.params = params;
        self
    }

    pub fn with_adaptive(mut self, adaptive: bool) -> Self {
        self.adaptive = adaptive;
        self
    }
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self::new("default_governor")
    }
}
