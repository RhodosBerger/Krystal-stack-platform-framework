//! Effect and Capability System
//!
//! Every component declares its effects and allowed capabilities,
//! enabling safe composition, auditability, and hot-reload.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use serde::{Deserialize, Serialize};

/// Effect types that components can declare
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Effect {
    // I/O Effects
    ReadTelemetry,
    WriteTelemetry,
    ReadConfig,
    WriteConfig,
    FileRead,
    FileWrite,
    NetworkRead,
    NetworkWrite,

    // System Effects
    CpuControl,
    GpuControl,
    MemoryControl,
    ThermalControl,
    ProcessControl,

    // State Effects
    StateRead,
    StateWrite,
    LogWrite,
    MetricsEmit,

    // Scheduling Effects
    PriorityChange,
    AffinityChange,
    BoostChange,

    // Learning Effects
    ExperienceRead,
    ExperienceWrite,
    PolicyUpdate,
}

/// Capability token granting permission for effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub id: String,
    pub effects: HashSet<Effect>,
    pub constraints: CapabilityConstraints,
    pub issued_at: u64,
    pub expires_at: Option<u64>,
    pub revocable: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapabilityConstraints {
    pub max_frequency_hz: Option<f64>,
    pub max_magnitude: Option<f64>,
    pub allowed_targets: Option<Vec<String>>,
    pub requires_approval: bool,
}

/// Component effect declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectDeclaration {
    pub component_id: String,
    pub required_effects: HashSet<Effect>,
    pub optional_effects: HashSet<Effect>,
    pub provided_effects: HashSet<Effect>,
    pub invariants: Vec<Invariant>,
}

/// Runtime invariant that must hold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invariant {
    pub id: String,
    pub description: String,
    pub check_fn: String,  // Expression to evaluate
    pub severity: InvariantSeverity,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum InvariantSeverity {
    Warning,
    Error,
    Critical,
}

/// Effect checker for safe composition
pub struct EffectChecker {
    declarations: HashMap<String, EffectDeclaration>,
    capabilities: HashMap<String, Vec<Capability>>,
    active_effects: HashSet<(String, Effect)>,
}

impl EffectChecker {
    pub fn new() -> Self {
        Self {
            declarations: HashMap::new(),
            capabilities: HashMap::new(),
            active_effects: HashSet::new(),
        }
    }

    /// Register a component's effect declaration
    pub fn register(&mut self, decl: EffectDeclaration) {
        self.declarations.insert(decl.component_id.clone(), decl);
    }

    /// Grant capability to a component
    pub fn grant_capability(&mut self, component_id: &str, cap: Capability) {
        self.capabilities
            .entry(component_id.to_string())
            .or_default()
            .push(cap);
    }

    /// Check if component can perform effect
    pub fn can_perform(&self, component_id: &str, effect: Effect) -> bool {
        if let Some(caps) = self.capabilities.get(component_id) {
            caps.iter().any(|c| c.effects.contains(&effect) && !self.is_expired(c))
        } else {
            false
        }
    }

    /// Validate composition of components
    pub fn validate_composition(&self, component_ids: &[&str]) -> CompositionResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut all_effects = HashSet::new();

        for &id in component_ids {
            if let Some(decl) = self.declarations.get(id) {
                // Check required effects are granted
                for effect in &decl.required_effects {
                    if !self.can_perform(id, *effect) {
                        errors.push(format!(
                            "Component '{}' requires {:?} but lacks capability",
                            id, effect
                        ));
                    }
                }

                // Track all effects for conflict detection
                for effect in &decl.required_effects {
                    all_effects.insert((id.to_string(), *effect));
                }
            } else {
                warnings.push(format!("Component '{}' not registered", id));
            }
        }

        // Check for conflicting effects
        let write_effects: Vec<_> = all_effects
            .iter()
            .filter(|(_, e)| matches!(e,
                Effect::WriteConfig | Effect::StateWrite |
                Effect::PolicyUpdate | Effect::WriteTelemetry))
            .collect();

        if write_effects.len() > 1 {
            let writers: Vec<_> = write_effects.iter().map(|(id, e)| format!("{}:{:?}", id, e)).collect();
            warnings.push(format!("Multiple writers detected: {:?}", writers));
        }

        CompositionResult { errors, warnings, valid: errors.is_empty() }
    }

    fn is_expired(&self, cap: &Capability) -> bool {
        if let Some(exp) = cap.expires_at {
            exp < current_timestamp()
        } else {
            false
        }
    }

    /// Record effect execution for audit
    pub fn record_effect(&mut self, component_id: &str, effect: Effect) {
        self.active_effects.insert((component_id.to_string(), effect));
    }

    /// Get audit trail
    pub fn get_audit_trail(&self) -> Vec<(String, Effect)> {
        self.active_effects.iter().cloned().collect()
    }
}

impl Default for EffectChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct CompositionResult {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub valid: bool,
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_grant_and_check() {
        let mut checker = EffectChecker::new();

        let cap = Capability {
            id: "cap-1".into(),
            effects: [Effect::ReadTelemetry, Effect::CpuControl].into_iter().collect(),
            constraints: Default::default(),
            issued_at: current_timestamp(),
            expires_at: None,
            revocable: true,
        };

        checker.grant_capability("guardian", cap);

        assert!(checker.can_perform("guardian", Effect::ReadTelemetry));
        assert!(checker.can_perform("guardian", Effect::CpuControl));
        assert!(!checker.can_perform("guardian", Effect::GpuControl));
    }

    #[test]
    fn test_composition_validation() {
        let mut checker = EffectChecker::new();

        checker.register(EffectDeclaration {
            component_id: "telemetry".into(),
            required_effects: [Effect::ReadTelemetry].into_iter().collect(),
            optional_effects: HashSet::new(),
            provided_effects: [Effect::WriteTelemetry].into_iter().collect(),
            invariants: vec![],
        });

        let cap = Capability {
            id: "cap-tel".into(),
            effects: [Effect::ReadTelemetry].into_iter().collect(),
            constraints: Default::default(),
            issued_at: current_timestamp(),
            expires_at: None,
            revocable: false,
        };
        checker.grant_capability("telemetry", cap);

        let result = checker.validate_composition(&["telemetry"]);
        assert!(result.valid);
    }
}
