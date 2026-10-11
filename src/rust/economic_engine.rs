use crate::types::*;
use std::collections::HashMap;

/// Economic Engine: scores candidate actions based on cost/benefit/risk
pub struct EconomicEngine {
    weights: ScoringWeights,
    profile: OperatorProfile,
}

#[derive(Debug, Clone)]
pub struct ScoringWeights {
    pub power_weight: f64,
    pub thermal_weight: f64,
    pub latency_weight: f64,
    pub stability_weight: f64,
    pub risk_penalty: f64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            power_weight: 0.15,
            thermal_weight: 0.25,
            latency_weight: 0.35,
            stability_weight: 0.20,
            risk_penalty: 0.30,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActionEconomicProfile {
    pub action: Action,
    pub estimated_power_cost_mw: i32,
    pub estimated_thermal_impact_c: i32,
    pub expected_latency_benefit_ms: f64,
    pub stability_risk: f64, // 0.0 - 1.0
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct ScoredAction {
    pub action: Action,
    pub utility_score: f64,
    pub breakdown: ScoreBreakdown,
}

#[derive(Debug, Clone)]
pub struct ScoreBreakdown {
    pub power_score: f64,
    pub thermal_score: f64,
    pub latency_score: f64,
    pub stability_score: f64,
    pub risk_penalty: f64,
}

impl EconomicEngine {
    pub fn new(profile: OperatorProfile) -> Self {
        let weights = match profile {
            OperatorProfile::Gaming => ScoringWeights {
                power_weight: 0.05,
                thermal_weight: 0.20,
                latency_weight: 0.50,
                stability_weight: 0.20,
                risk_penalty: 0.20,
            },
            OperatorProfile::Production => ScoringWeights {
                power_weight: 0.10,
                thermal_weight: 0.30,
                latency_weight: 0.20,
                stability_weight: 0.35,
                risk_penalty: 0.40,
            },
            OperatorProfile::PowerSaver => ScoringWeights {
                power_weight: 0.40,
                thermal_weight: 0.30,
                latency_weight: 0.10,
                stability_weight: 0.15,
                risk_penalty: 0.25,
            },
            OperatorProfile::Balanced => ScoringWeights::default(),
        };
        Self { weights, profile }
    }

    pub fn with_custom_weights(profile: OperatorProfile, weights: ScoringWeights) -> Self {
        Self { weights, profile }
    }

    /// Score a candidate action against current resource budgets
    pub fn score_action(
        &self,
        action_profile: &ActionEconomicProfile,
        budgets: &ResourceBudgets,
    ) -> ScoredAction {
        // Power score: penalize if exceeding budget
        let power_headroom = budgets.cpu_budget.headroom_pct / 100.0;
        let power_impact = action_profile.estimated_power_cost_mw as f64
            / budgets.cpu_budget.max_mw as f64;
        let power_score = (power_headroom - power_impact).max(0.0).min(1.0);

        // Thermal score: critical near limits
        let thermal_headroom = budgets.thermal_headroom.headroom_c as f64
            / (budgets.thermal_headroom.cpu_max_c - budgets.thermal_headroom.cpu_current_c) as f64;
        let thermal_impact = action_profile.estimated_thermal_impact_c as f64 / 20.0;
        let thermal_score = (thermal_headroom - thermal_impact).max(0.0).min(1.0);

        // Latency score: reward improvements
        let latency_score = if budgets.latency_budget.headroom_ms > 0.0 {
            (action_profile.expected_latency_benefit_ms / budgets.latency_budget.target_frametime_ms)
                .min(1.0)
                .max(0.0)
        } else {
            0.5 // neutral if no headroom data
        };

        // Stability score: inverse of risk
        let stability_score = 1.0 - action_profile.stability_risk;

        // Risk penalty based on confidence
        let risk_penalty = (1.0 - action_profile.confidence) * self.weights.risk_penalty;

        // Calculate weighted utility
        let utility_score = (self.weights.power_weight * power_score)
            + (self.weights.thermal_weight * thermal_score)
            + (self.weights.latency_weight * latency_score)
            + (self.weights.stability_weight * stability_score)
            - risk_penalty;

        ScoredAction {
            action: action_profile.action.clone(),
            utility_score: utility_score.max(0.0),
            breakdown: ScoreBreakdown {
                power_score,
                thermal_score,
                latency_score,
                stability_score,
                risk_penalty,
            },
        }
    }

    /// Score multiple actions and rank them
    pub fn rank_actions(
        &self,
        action_profiles: &[ActionEconomicProfile],
        budgets: &ResourceBudgets,
    ) -> Vec<ScoredAction> {
        let mut scored: Vec<ScoredAction> = action_profiles
            .iter()
            .map(|ap| self.score_action(ap, budgets))
            .collect();
        scored.sort_by(|a, b| b.utility_score.partial_cmp(&a.utility_score).unwrap());
        scored
    }

    /// Update weights based on LLM feedback
    pub fn update_weights(&mut self, new_weights: ScoringWeights) {
        self.weights = new_weights;
    }

    pub fn get_profile(&self) -> &OperatorProfile {
        &self.profile
    }

    pub fn get_weights(&self) -> &ScoringWeights {
        &self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaming_profile_favors_latency() {
        let engine = EconomicEngine::new(OperatorProfile::Gaming);
        assert!(engine.weights.latency_weight > engine.weights.power_weight);
    }

    #[test]
    fn test_score_action() {
        let engine = EconomicEngine::new(OperatorProfile::Balanced);
        let budgets = ResourceBudgets {
            cpu_budget: PowerBudget {
                current_mw: 45000,
                max_mw: 65000,
                headroom_pct: 30.8,
            },
            thermal_headroom: ThermalBudget {
                cpu_current_c: 72,
                cpu_max_c: 95,
                gpu_current_c: 68,
                gpu_max_c: 90,
                headroom_c: 23,
            },
            latency_budget: LatencyBudget {
                target_frametime_ms: 16.67,
                current_frametime_ms: 14.2,
                headroom_ms: 2.47,
            },
            ..Default::default()
        };

        let action_profile = ActionEconomicProfile {
            action: Action {
                action_type: ActionType::SetGovernor,
                params: HashMap::new(),
            },
            estimated_power_cost_mw: 5000,
            estimated_thermal_impact_c: 3,
            expected_latency_benefit_ms: 1.5,
            stability_risk: 0.1,
            confidence: 0.85,
        };

        let scored = engine.score_action(&action_profile, &budgets);
        assert!(scored.utility_score > 0.0);
    }
}
