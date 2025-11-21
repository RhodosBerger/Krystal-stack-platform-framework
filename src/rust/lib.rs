//! GAMESA/KrystalStack - Deterministic Stream
//!
//! This crate implements the deterministic (Rust) components of the system:
//! - Economic Engine: Resource budgeting and action scoring
//! - ActionGate: Safety guardrails and action validation
//! - RuleEvaluator: MicroInferenceRule evaluation

pub mod types;
pub mod economic_engine;
pub mod action_gate;
pub mod rule_evaluator;
pub mod feature_engine;
pub mod runtime;
pub mod allocation;
pub mod effects;
pub mod contracts;
pub mod signals;

pub use types::*;
pub use economic_engine::*;
pub use action_gate::*;
pub use rule_evaluator::*;
pub use feature_engine::*;
pub use runtime::*;
pub use allocation::*;
pub use effects::*;
pub use contracts::*;
pub use signals::*;

/// Main orchestrator combining all components
pub struct Guardian {
    pub economic_engine: EconomicEngine,
    pub action_gate: ActionGate,
    pub rule_evaluator: RuleEvaluator,
}

impl Guardian {
    pub fn new(profile: OperatorProfile, safety_config: SafetyConfig) -> Self {
        Self {
            economic_engine: EconomicEngine::new(profile),
            action_gate: ActionGate::new(safety_config),
            rule_evaluator: RuleEvaluator::new(),
        }
    }

    /// Main decision loop: evaluate rules, score actions, select best
    pub fn decide(
        &mut self,
        telemetry: &TelemetrySnapshot,
        budgets: &ResourceBudgets,
    ) -> Option<Action> {
        // Check thermal state first
        if self.action_gate.check_thermal_state(telemetry) {
            return Some(Action {
                action_type: ActionType::TriggerCooldown,
                params: Default::default(),
            });
        }

        // Evaluate all rules
        let results = self.rule_evaluator.evaluate_all(telemetry);

        // Collect triggered actions
        let candidate_actions: Vec<_> = results
            .into_iter()
            .filter(|r| r.matched && !r.shadow_mode)
            .flat_map(|r| r.triggered_actions)
            .collect();

        if candidate_actions.is_empty() {
            return None;
        }

        // Create economic profiles (simplified - in production, lookup from rule metadata)
        let action_profiles: Vec<_> = candidate_actions
            .iter()
            .map(|action| ActionEconomicProfile {
                action: action.clone(),
                estimated_power_cost_mw: 3000,
                estimated_thermal_impact_c: 2,
                expected_latency_benefit_ms: 1.0,
                stability_risk: 0.15,
                confidence: 0.80,
            })
            .collect();

        // Score and rank
        let scored = self.economic_engine.rank_actions(&action_profiles, budgets);

        // Select through gate
        let decision = self.action_gate.select_action(&scored, telemetry, 0.3);

        decision.action
    }
}
