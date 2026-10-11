use crate::types::*;
use crate::economic_engine::{EconomicEngine, ScoredAction};
use std::collections::HashSet;

/// ActionGate: validates and filters actions based on safety guardrails
pub struct ActionGate {
    config: SafetyConfig,
    quarantined_rules: HashSet<String>,
    emergency_cooldown_active: bool,
}

#[derive(Debug, Clone)]
pub enum ValidationResult {
    Approved(Action),
    Rejected(RejectionReason),
    RequiresReview(Action, Vec<String>),
}

#[derive(Debug, Clone)]
pub enum RejectionReason {
    ThermalLimitExceeded { current: u32, max: u32 },
    MemoryLimitExceeded { available_mb: u32, required_mb: u32 },
    RestrictedPath(String),
    ProcessInjectionBlocked,
    RuleQuarantined(String),
    EmergencyCooldownActive,
    SafetyTierViolation { required: u8, provided: u8 },
    InvalidSyntax(String),
}

#[derive(Debug, Clone)]
pub struct GateDecision {
    pub action: Option<Action>,
    pub approved: bool,
    pub reason: Option<RejectionReason>,
    pub warnings: Vec<String>,
}

impl ActionGate {
    pub fn new(config: SafetyConfig) -> Self {
        Self {
            config,
            quarantined_rules: HashSet::new(),
            emergency_cooldown_active: false,
        }
    }

    /// Validate a single action against safety guardrails
    pub fn validate_action(
        &self,
        action: &Action,
        telemetry: &TelemetrySnapshot,
        rule: Option<&MicroInferenceRule>,
    ) -> ValidationResult {
        // Check emergency cooldown
        if self.emergency_cooldown_active {
            return ValidationResult::Rejected(RejectionReason::EmergencyCooldownActive);
        }

        // Check if rule is quarantined
        if let Some(r) = rule {
            if self.quarantined_rules.contains(&r.rule_id) {
                return ValidationResult::Rejected(RejectionReason::RuleQuarantined(
                    r.rule_id.clone(),
                ));
            }
        }

        // Thermal checks
        if telemetry.temp_cpu >= self.config.max_cpu_temp_c {
            return ValidationResult::Rejected(RejectionReason::ThermalLimitExceeded {
                current: telemetry.temp_cpu,
                max: self.config.max_cpu_temp_c,
            });
        }

        if telemetry.temp_gpu >= self.config.max_gpu_temp_c {
            return ValidationResult::Rejected(RejectionReason::ThermalLimitExceeded {
                current: telemetry.temp_gpu,
                max: self.config.max_gpu_temp_c,
            });
        }

        // Action-specific validation
        let mut warnings = Vec::new();

        match &action.action_type {
            ActionType::SetCpuAffinity => {
                // Validate affinity params
                if telemetry.temp_cpu > self.config.emergency_cooldown_threshold_c - 5 {
                    warnings.push("CPU temperature approaching emergency threshold".into());
                }
            }
            ActionType::SetGovernor => {
                if let Some(gov) = action.params.get("governor") {
                    if gov == "performance" && telemetry.temp_cpu > 80 {
                        warnings.push("Performance governor requested with elevated temps".into());
                    }
                }
            }
            ActionType::SetGpuPowerLimit => {
                // Check GPU thermal headroom
                if telemetry.temp_gpu > self.config.max_gpu_temp_c - 10 {
                    warnings.push("GPU thermal headroom limited".into());
                }
            }
            _ => {}
        }

        // Safety tier validation
        if let Some(r) = rule {
            if r.safety_tier < 2 {
                return ValidationResult::RequiresReview(action.clone(), warnings);
            }
        }

        if warnings.is_empty() {
            ValidationResult::Approved(action.clone())
        } else {
            ValidationResult::RequiresReview(action.clone(), warnings)
        }
    }

    /// Select best action from scored candidates
    pub fn select_action(
        &self,
        scored_actions: &[ScoredAction],
        telemetry: &TelemetrySnapshot,
        min_utility_threshold: f64,
    ) -> GateDecision {
        for scored in scored_actions {
            if scored.utility_score < min_utility_threshold {
                continue;
            }

            match self.validate_action(&scored.action, telemetry, None) {
                ValidationResult::Approved(action) => {
                    return GateDecision {
                        action: Some(action),
                        approved: true,
                        reason: None,
                        warnings: vec![],
                    };
                }
                ValidationResult::RequiresReview(action, warnings) => {
                    // Accept with warnings if utility is high enough
                    if scored.utility_score > min_utility_threshold + 0.2 {
                        return GateDecision {
                            action: Some(action),
                            approved: true,
                            reason: None,
                            warnings,
                        };
                    }
                }
                ValidationResult::Rejected(reason) => {
                    // Try next action
                    continue;
                }
            }
        }

        GateDecision {
            action: None,
            approved: false,
            reason: Some(RejectionReason::SafetyTierViolation {
                required: 2,
                provided: 0,
            }),
            warnings: vec!["No suitable action found".into()],
        }
    }

    /// Trigger emergency cooldown mode
    pub fn trigger_emergency_cooldown(&mut self) {
        self.emergency_cooldown_active = true;
    }

    /// Clear emergency cooldown
    pub fn clear_emergency_cooldown(&mut self) {
        self.emergency_cooldown_active = false;
    }

    /// Quarantine a problematic rule
    pub fn quarantine_rule(&mut self, rule_id: &str) {
        self.quarantined_rules.insert(rule_id.to_string());
    }

    /// Remove rule from quarantine
    pub fn unquarantine_rule(&mut self, rule_id: &str) {
        self.quarantined_rules.remove(rule_id);
    }

    /// Check thermal state and trigger cooldown if needed
    pub fn check_thermal_state(&mut self, telemetry: &TelemetrySnapshot) -> bool {
        if telemetry.temp_cpu >= self.config.emergency_cooldown_threshold_c
            || telemetry.temp_gpu >= self.config.emergency_cooldown_threshold_c
        {
            self.trigger_emergency_cooldown();
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_telemetry() -> TelemetrySnapshot {
        TelemetrySnapshot {
            timestamp: "2025-01-21T14:30:00Z".into(),
            cpu_util: 0.75,
            gpu_util: 0.70,
            frametime_ms: 14.2,
            temp_cpu: 72,
            temp_gpu: 68,
            active_process_category: "gaming".into(),
        }
    }

    #[test]
    fn test_approve_safe_action() {
        let gate = ActionGate::new(SafetyConfig::default());
        let action = Action {
            action_type: ActionType::SetGovernor,
            params: [("governor".into(), serde_json::json!("balanced"))]
                .into_iter()
                .collect(),
        };

        match gate.validate_action(&action, &sample_telemetry(), None) {
            ValidationResult::Approved(_) => {}
            _ => panic!("Expected approval"),
        }
    }

    #[test]
    fn test_reject_on_thermal_limit() {
        let gate = ActionGate::new(SafetyConfig::default());
        let mut telemetry = sample_telemetry();
        telemetry.temp_cpu = 96; // Exceeds default 95C limit

        let action = Action {
            action_type: ActionType::SetGovernor,
            params: [].into_iter().collect(),
        };

        match gate.validate_action(&action, &telemetry, None) {
            ValidationResult::Rejected(RejectionReason::ThermalLimitExceeded { .. }) => {}
            _ => panic!("Expected thermal rejection"),
        }
    }
}
