use crate::types::*;
use std::collections::HashMap;

/// Evaluates MicroInferenceRules against current telemetry
pub struct RuleEvaluator {
    rules: HashMap<String, MicroInferenceRule>,
    shadow_results: Vec<ShadowEvaluation>,
}

#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub rule_id: String,
    pub matched: bool,
    pub triggered_actions: Vec<Action>,
    pub shadow_mode: bool,
}

#[derive(Debug, Clone)]
pub struct ShadowEvaluation {
    pub rule_id: String,
    pub timestamp: String,
    pub would_have_matched: bool,
    pub hypothetical_actions: Vec<Action>,
}

impl RuleEvaluator {
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            shadow_results: Vec::new(),
        }
    }

    /// Register a rule for evaluation
    pub fn register_rule(&mut self, rule: MicroInferenceRule) {
        self.rules.insert(rule.rule_id.clone(), rule);
    }

    /// Remove a rule
    pub fn unregister_rule(&mut self, rule_id: &str) -> Option<MicroInferenceRule> {
        self.rules.remove(rule_id)
    }

    /// Evaluate a single condition
    fn evaluate_condition(&self, condition: &Condition, telemetry: &TelemetrySnapshot) -> bool {
        let metric_value = self.get_metric_value(&condition.metric, telemetry);

        match metric_value {
            Some(MetricValue::Float(val)) => {
                let threshold = condition.value.as_f64().unwrap_or(0.0);
                match condition.operator {
                    ConditionOperator::Eq => (val - threshold).abs() < 0.001,
                    ConditionOperator::Ne => (val - threshold).abs() >= 0.001,
                    ConditionOperator::Gt => val > threshold,
                    ConditionOperator::Ge => val >= threshold,
                    ConditionOperator::Lt => val < threshold,
                    ConditionOperator::Le => val <= threshold,
                    ConditionOperator::Contains => false,
                }
            }
            Some(MetricValue::String(val)) => {
                let target = condition.value.as_str().unwrap_or("");
                match condition.operator {
                    ConditionOperator::Eq => val == target,
                    ConditionOperator::Ne => val != target,
                    ConditionOperator::Contains => val.contains(target),
                    _ => false,
                }
            }
            Some(MetricValue::Int(val)) => {
                let threshold = condition.value.as_i64().unwrap_or(0);
                match condition.operator {
                    ConditionOperator::Eq => val == threshold,
                    ConditionOperator::Ne => val != threshold,
                    ConditionOperator::Gt => val > threshold,
                    ConditionOperator::Ge => val >= threshold,
                    ConditionOperator::Lt => val < threshold,
                    ConditionOperator::Le => val <= threshold,
                    ConditionOperator::Contains => false,
                }
            }
            None => false,
        }
    }

    /// Get metric value from telemetry
    fn get_metric_value(&self, metric: &str, telemetry: &TelemetrySnapshot) -> Option<MetricValue> {
        match metric {
            "cpu_utilization" | "cpu_util" => Some(MetricValue::Float(telemetry.cpu_util)),
            "gpu_utilization" | "gpu_util" => Some(MetricValue::Float(telemetry.gpu_util)),
            "frametime_ms" => Some(MetricValue::Float(telemetry.frametime_ms)),
            "temp_cpu" | "cpu_temp" => Some(MetricValue::Int(telemetry.temp_cpu as i64)),
            "temp_gpu" | "gpu_temp" => Some(MetricValue::Int(telemetry.temp_gpu as i64)),
            "active_process_category" => {
                Some(MetricValue::String(telemetry.active_process_category.clone()))
            }
            "thermal_headroom_c" => {
                // Approximate thermal headroom
                let headroom = 95i64 - telemetry.temp_cpu as i64;
                Some(MetricValue::Int(headroom.max(0)))
            }
            _ => None,
        }
    }

    /// Evaluate all conditions for a rule
    fn evaluate_conditions(&self, rule: &MicroInferenceRule, telemetry: &TelemetrySnapshot) -> bool {
        if rule.conditions.is_empty() {
            return false;
        }

        let mut result = self.evaluate_condition(&rule.conditions[0], telemetry);

        for condition in rule.conditions.iter().skip(1) {
            let cond_result = self.evaluate_condition(condition, telemetry);

            match rule.conditions.iter().find(|c| c.metric == condition.metric) {
                Some(c) => match c.logical_op {
                    Some(LogicalOp::And) => result = result && cond_result,
                    Some(LogicalOp::Or) => result = result || cond_result,
                    None => result = result && cond_result,
                },
                None => result = result && cond_result,
            }
        }

        result
    }

    /// Evaluate a single rule
    pub fn evaluate_rule(
        &mut self,
        rule_id: &str,
        telemetry: &TelemetrySnapshot,
    ) -> Option<EvaluationResult> {
        let rule = self.rules.get(rule_id)?.clone();
        let matched = self.evaluate_conditions(&rule, telemetry);

        if rule.shadow_mode {
            self.shadow_results.push(ShadowEvaluation {
                rule_id: rule_id.to_string(),
                timestamp: telemetry.timestamp.clone(),
                would_have_matched: matched,
                hypothetical_actions: if matched {
                    rule.actions.clone()
                } else {
                    vec![]
                },
            });
        }

        Some(EvaluationResult {
            rule_id: rule_id.to_string(),
            matched,
            triggered_actions: if matched && !rule.shadow_mode {
                rule.actions.clone()
            } else {
                vec![]
            },
            shadow_mode: rule.shadow_mode,
        })
    }

    /// Evaluate all registered rules
    pub fn evaluate_all(&mut self, telemetry: &TelemetrySnapshot) -> Vec<EvaluationResult> {
        let rule_ids: Vec<String> = self.rules.keys().cloned().collect();
        rule_ids
            .iter()
            .filter_map(|id| self.evaluate_rule(id, telemetry))
            .collect()
    }

    /// Get shadow evaluation history for metacognitive analysis
    pub fn get_shadow_history(&self) -> &[ShadowEvaluation] {
        &self.shadow_results
    }

    /// Clear shadow history
    pub fn clear_shadow_history(&mut self) {
        self.shadow_results.clear();
    }

    /// Get rule by ID
    pub fn get_rule(&self, rule_id: &str) -> Option<&MicroInferenceRule> {
        self.rules.get(rule_id)
    }

    /// List all rule IDs
    pub fn list_rules(&self) -> Vec<&String> {
        self.rules.keys().collect()
    }
}

#[derive(Debug)]
enum MetricValue {
    Float(f64),
    Int(i64),
    String(String),
}

impl Default for RuleEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rule() -> MicroInferenceRule {
        MicroInferenceRule {
            rule_id: "test-rule-001".into(),
            version: "1.0.0".into(),
            source: RuleSource::LlmGenerated,
            safety_tier: 2,
            shadow_mode: false,
            conditions: vec![
                Condition {
                    metric: "cpu_utilization".into(),
                    operator: ConditionOperator::Gt,
                    value: serde_json::json!(0.70),
                    logical_op: Some(LogicalOp::And),
                },
                Condition {
                    metric: "active_process_category".into(),
                    operator: ConditionOperator::Eq,
                    value: serde_json::json!("gaming"),
                    logical_op: None,
                },
            ],
            actions: vec![Action {
                action_type: ActionType::SetGovernor,
                params: [("governor".into(), serde_json::json!("performance"))]
                    .into_iter()
                    .collect(),
            }],
            justification: "Boost performance for gaming workloads".into(),
        }
    }

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
    fn test_rule_matches() {
        let mut evaluator = RuleEvaluator::new();
        evaluator.register_rule(sample_rule());

        let result = evaluator.evaluate_rule("test-rule-001", &sample_telemetry());
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.matched);
        assert_eq!(result.triggered_actions.len(), 1);
    }

    #[test]
    fn test_rule_no_match() {
        let mut evaluator = RuleEvaluator::new();
        evaluator.register_rule(sample_rule());

        let mut telemetry = sample_telemetry();
        telemetry.cpu_util = 0.50; // Below threshold

        let result = evaluator.evaluate_rule("test-rule-001", &telemetry);
        assert!(result.is_some());
        assert!(!result.unwrap().matched);
    }

    #[test]
    fn test_shadow_mode() {
        let mut evaluator = RuleEvaluator::new();
        let mut rule = sample_rule();
        rule.shadow_mode = true;
        evaluator.register_rule(rule);

        let result = evaluator.evaluate_rule("test-rule-001", &sample_telemetry());
        assert!(result.unwrap().triggered_actions.is_empty());
        assert_eq!(evaluator.get_shadow_history().len(), 1);
    }
}
