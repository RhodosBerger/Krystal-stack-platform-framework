//! Trust Engine - Validated, Signed Decision Responses

use crate::{Config, TelemetrySnapshot, Action, Domain};
use crate::policy::PolicyResult;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Trusted decision with validation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustedDecision {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub action: Action,
    pub domain: Domain,
    pub confidence: f64,
    pub signature: Option<String>,
    pub validation: ValidationMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetadata {
    pub thermal_safe: bool,
    pub power_safe: bool,
    pub policy_valid: bool,
    pub inference_used: bool,
    pub checks_passed: Vec<String>,
    pub checks_failed: Vec<String>,
}

pub struct TrustEngine {
    thermal_limits: (f64, f64),  // (cpu, gpu)
    power_limit: f64,
    signing_enabled: bool,
}

impl TrustEngine {
    pub fn new(config: &Config) -> Self {
        Self {
            thermal_limits: (config.thermal_limits.cpu_critical, config.thermal_limits.gpu_critical),
            power_limit: config.policy_config.power_budget_w,
            signing_enabled: config.trust_key_path.is_some(),
        }
    }

    pub fn create_decision(
        &self,
        policy_result: PolicyResult,
        inference_hint: Option<Action>,
        telemetry: &TelemetrySnapshot,
    ) -> TrustedDecision {
        let mut checks_passed = Vec::new();
        let mut checks_failed = Vec::new();

        // Thermal validation
        let thermal_safe = telemetry.temp_cpu < self.thermal_limits.0
            && telemetry.temp_gpu < self.thermal_limits.1;
        if thermal_safe {
            checks_passed.push("thermal_limits".into());
        } else {
            checks_failed.push("thermal_limits".into());
        }

        // Power validation
        let power_safe = telemetry.power_draw < self.power_limit;
        if power_safe {
            checks_passed.push("power_budget".into());
        } else {
            checks_failed.push("power_budget".into());
        }

        // Determine final action
        let action = if !thermal_safe {
            Action::Throttle {
                zone_id: telemetry.zone_id,
                target: 0.5,
            }
        } else if let Some(hint) = inference_hint {
            hint
        } else {
            policy_result.action
        };

        // Calculate confidence
        let confidence = if thermal_safe && power_safe {
            policy_result.confidence
        } else {
            policy_result.confidence * 0.5
        };

        TrustedDecision {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            action,
            domain: policy_result.domain,
            confidence,
            signature: if self.signing_enabled {
                Some(self.sign(&policy_result))
            } else {
                None
            },
            validation: ValidationMetadata {
                thermal_safe,
                power_safe,
                policy_valid: policy_result.valid,
                inference_used: inference_hint.is_some(),
                checks_passed,
                checks_failed,
            },
        }
    }

    fn sign(&self, _result: &PolicyResult) -> String {
        // Placeholder - would use actual signing key
        format!("sig_{}", Uuid::new_v4())
    }
}
