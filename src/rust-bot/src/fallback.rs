//! Fallback Manager - Safety and Trust Validation
//!
//! Merges Python and Rust telemetry with validation.
//! Low trust forces conservative actions.

use crate::{TelemetrySnapshot, Action};
use crate::trust::TrustedDecision;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustLevel {
    High,
    Medium,
    Low,
}

pub struct FallbackManager {
    divergence_threshold: f64,
    last_trust_level: TrustLevel,
    consecutive_low_trust: u32,
}

impl FallbackManager {
    pub fn new() -> Self {
        Self {
            divergence_threshold: 0.2,  // 20% divergence = low trust
            last_trust_level: TrustLevel::High,
            consecutive_low_trust: 0,
        }
    }

    /// Validate telemetry by comparing sources
    pub fn validate_telemetry(&mut self, telemetry: &TelemetrySnapshot) -> TrustLevel {
        // Would compare Python telemetry vs Rust-collected telemetry
        // For now, simulate based on thermal headroom
        let trust = if telemetry.thermal_headroom() < 0.1 {
            TrustLevel::Low  // Thermal emergency = low trust
        } else if telemetry.cpu_util > 0.99 || telemetry.gpu_util > 0.99 {
            TrustLevel::Medium  // Saturated = medium trust
        } else {
            TrustLevel::High
        };

        if trust == TrustLevel::Low {
            self.consecutive_low_trust += 1;
        } else {
            self.consecutive_low_trust = 0;
        }

        self.last_trust_level = trust;
        trust
    }

    /// Parallel validation between Python and Rust sources
    pub fn parallel_validate_sources(
        &self,
        python_cpu: f64,
        python_gpu: f64,
        rust_cpu: f64,
        rust_gpu: f64,
    ) -> (f64, TrustLevel) {
        let cpu_divergence = (python_cpu - rust_cpu).abs();
        let gpu_divergence = (python_gpu - rust_gpu).abs();
        let max_divergence = cpu_divergence.max(gpu_divergence);

        let trust = if max_divergence > self.divergence_threshold {
            TrustLevel::Low
        } else if max_divergence > self.divergence_threshold / 2.0 {
            TrustLevel::Medium
        } else {
            TrustLevel::High
        };

        (max_divergence, trust)
    }

    /// Get fallback telemetry (Rust-only values)
    pub fn get_fallback_telemetry(&self, original: &TelemetrySnapshot) -> TelemetrySnapshot {
        // Return conservative estimate
        let mut fallback = original.clone();
        fallback.cpu_util = fallback.cpu_util.min(0.95);
        fallback.gpu_util = fallback.gpu_util.min(0.95);
        fallback
    }

    /// Force cooldown action when trust is low
    pub fn force_cooldown(&self, decision: &TrustedDecision) -> TrustedDecision {
        let mut safe_decision = decision.clone();
        safe_decision.action = Action::Throttle {
            zone_id: 0,
            target: 0.5,
        };
        safe_decision.confidence *= 0.5;
        safe_decision
    }

    pub fn get_consecutive_low_trust(&self) -> u32 {
        self.consecutive_low_trust
    }
}

impl Default for FallbackManager {
    fn default() -> Self { Self::new() }
}
