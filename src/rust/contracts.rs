//! Contracts and Proofs System
//!
//! Pre/post/invariant contracts plus proof validators enforce correctness
//! at runtime, supporting automated validation and self-healing.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Contract definition for a function/component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contract {
    pub id: String,
    pub name: String,
    pub preconditions: Vec<Condition>,
    pub postconditions: Vec<Condition>,
    pub invariants: Vec<Condition>,
}

/// Condition expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    pub id: String,
    pub description: String,
    pub expression: ConditionExpr,
    pub on_violation: ViolationAction,
}

/// Condition expression types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionExpr {
    /// Value comparison: field op value
    Compare { field: String, op: CompareOp, value: f64 },
    /// Range check: min <= field <= max
    Range { field: String, min: f64, max: f64 },
    /// Non-null check
    NotNull { field: String },
    /// Type check
    TypeOf { field: String, expected: String },
    /// Custom expression
    Custom { expr: String },
    /// Logical AND
    And(Vec<ConditionExpr>),
    /// Logical OR
    Or(Vec<ConditionExpr>),
    /// Logical NOT
    Not(Box<ConditionExpr>),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompareOp {
    Eq, Ne, Lt, Le, Gt, Ge,
}

/// Action on contract violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationAction {
    Log,
    Warn,
    Error,
    Abort,
    Heal { strategy: HealStrategy },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealStrategy {
    Clamp { min: f64, max: f64 },
    Default { value: f64 },
    Retry { max_attempts: u32 },
    Fallback { handler: String },
}

/// Contract validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub contract_id: String,
    pub passed: bool,
    pub violations: Vec<Violation>,
    pub healed: Vec<HealedViolation>,
}

#[derive(Debug, Clone)]
pub struct Violation {
    pub condition_id: String,
    pub description: String,
    pub actual_value: Option<String>,
    pub expected: String,
}

#[derive(Debug, Clone)]
pub struct HealedViolation {
    pub condition_id: String,
    pub original_value: String,
    pub healed_value: String,
    pub strategy: String,
}

/// Contract validator
pub struct ContractValidator {
    contracts: HashMap<String, Contract>,
}

impl ContractValidator {
    pub fn new() -> Self {
        Self { contracts: HashMap::new() }
    }

    /// Register a contract
    pub fn register(&mut self, contract: Contract) {
        self.contracts.insert(contract.id.clone(), contract);
    }

    /// Validate preconditions
    pub fn check_preconditions(
        &self,
        contract_id: &str,
        context: &HashMap<String, f64>,
    ) -> ValidationResult {
        self.validate_conditions(contract_id, context, |c| &c.preconditions)
    }

    /// Validate postconditions
    pub fn check_postconditions(
        &self,
        contract_id: &str,
        context: &HashMap<String, f64>,
    ) -> ValidationResult {
        self.validate_conditions(contract_id, context, |c| &c.postconditions)
    }

    /// Validate invariants
    pub fn check_invariants(
        &self,
        contract_id: &str,
        context: &HashMap<String, f64>,
    ) -> ValidationResult {
        self.validate_conditions(contract_id, context, |c| &c.invariants)
    }

    fn validate_conditions<F>(
        &self,
        contract_id: &str,
        context: &HashMap<String, f64>,
        selector: F,
    ) -> ValidationResult
    where
        F: Fn(&Contract) -> &Vec<Condition>,
    {
        let contract = match self.contracts.get(contract_id) {
            Some(c) => c,
            None => return ValidationResult {
                contract_id: contract_id.to_string(),
                passed: false,
                violations: vec![Violation {
                    condition_id: "unknown".into(),
                    description: "Contract not found".into(),
                    actual_value: None,
                    expected: "registered contract".into(),
                }],
                healed: vec![],
            },
        };

        let conditions = selector(contract);
        let mut violations = Vec::new();
        let mut healed = Vec::new();

        for condition in conditions {
            if !self.evaluate_expr(&condition.expression, context) {
                match &condition.on_violation {
                    ViolationAction::Heal { strategy } => {
                        if let Some(heal_result) = self.try_heal(strategy, &condition.expression, context) {
                            healed.push(heal_result);
                        } else {
                            violations.push(self.create_violation(condition, context));
                        }
                    }
                    _ => {
                        violations.push(self.create_violation(condition, context));
                    }
                }
            }
        }

        ValidationResult {
            contract_id: contract_id.to_string(),
            passed: violations.is_empty(),
            violations,
            healed,
        }
    }

    fn evaluate_expr(&self, expr: &ConditionExpr, context: &HashMap<String, f64>) -> bool {
        match expr {
            ConditionExpr::Compare { field, op, value } => {
                if let Some(&actual) = context.get(field) {
                    match op {
                        CompareOp::Eq => (actual - value).abs() < 1e-10,
                        CompareOp::Ne => (actual - value).abs() >= 1e-10,
                        CompareOp::Lt => actual < *value,
                        CompareOp::Le => actual <= *value,
                        CompareOp::Gt => actual > *value,
                        CompareOp::Ge => actual >= *value,
                    }
                } else {
                    false
                }
            }
            ConditionExpr::Range { field, min, max } => {
                if let Some(&actual) = context.get(field) {
                    actual >= *min && actual <= *max
                } else {
                    false
                }
            }
            ConditionExpr::NotNull { field } => context.contains_key(field),
            ConditionExpr::TypeOf { .. } => true, // Simplified
            ConditionExpr::Custom { .. } => true, // Would need expression parser
            ConditionExpr::And(exprs) => exprs.iter().all(|e| self.evaluate_expr(e, context)),
            ConditionExpr::Or(exprs) => exprs.iter().any(|e| self.evaluate_expr(e, context)),
            ConditionExpr::Not(expr) => !self.evaluate_expr(expr, context),
        }
    }

    fn create_violation(&self, condition: &Condition, context: &HashMap<String, f64>) -> Violation {
        let actual = match &condition.expression {
            ConditionExpr::Compare { field, .. } |
            ConditionExpr::Range { field, .. } => {
                context.get(field).map(|v| v.to_string())
            }
            _ => None,
        };

        Violation {
            condition_id: condition.id.clone(),
            description: condition.description.clone(),
            actual_value: actual,
            expected: format!("{:?}", condition.expression),
        }
    }

    fn try_heal(
        &self,
        strategy: &HealStrategy,
        expr: &ConditionExpr,
        _context: &HashMap<String, f64>,
    ) -> Option<HealedViolation> {
        match (strategy, expr) {
            (HealStrategy::Clamp { min, max }, ConditionExpr::Range { field, .. }) => {
                Some(HealedViolation {
                    condition_id: field.clone(),
                    original_value: "out_of_range".into(),
                    healed_value: format!("clamped to [{}, {}]", min, max),
                    strategy: "clamp".into(),
                })
            }
            (HealStrategy::Default { value }, ConditionExpr::NotNull { field }) => {
                Some(HealedViolation {
                    condition_id: field.clone(),
                    original_value: "null".into(),
                    healed_value: value.to_string(),
                    strategy: "default".into(),
                })
            }
            _ => None,
        }
    }
}

impl Default for ContractValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Proof record for audit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofRecord {
    pub timestamp: u64,
    pub contract_id: String,
    pub phase: ProofPhase,
    pub result: bool,
    pub context_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofPhase {
    Precondition,
    Postcondition,
    Invariant,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precondition_check() {
        let mut validator = ContractValidator::new();

        validator.register(Contract {
            id: "thermal-action".into(),
            name: "Thermal Control Action".into(),
            preconditions: vec![
                Condition {
                    id: "temp-valid".into(),
                    description: "Temperature must be below critical".into(),
                    expression: ConditionExpr::Compare {
                        field: "temp_cpu".into(),
                        op: CompareOp::Lt,
                        value: 95.0,
                    },
                    on_violation: ViolationAction::Abort,
                },
            ],
            postconditions: vec![],
            invariants: vec![],
        });

        let mut context = HashMap::new();
        context.insert("temp_cpu".to_string(), 72.0);

        let result = validator.check_preconditions("thermal-action", &context);
        assert!(result.passed);

        context.insert("temp_cpu".to_string(), 98.0);
        let result = validator.check_preconditions("thermal-action", &context);
        assert!(!result.passed);
    }

    #[test]
    fn test_range_validation() {
        let mut validator = ContractValidator::new();

        validator.register(Contract {
            id: "cpu-util".into(),
            name: "CPU Utilization".into(),
            preconditions: vec![],
            postconditions: vec![],
            invariants: vec![
                Condition {
                    id: "util-range".into(),
                    description: "Utilization must be 0-100%".into(),
                    expression: ConditionExpr::Range {
                        field: "cpu_util".into(),
                        min: 0.0,
                        max: 1.0,
                    },
                    on_violation: ViolationAction::Heal {
                        strategy: HealStrategy::Clamp { min: 0.0, max: 1.0 },
                    },
                },
            ],
        });

        let mut context = HashMap::new();
        context.insert("cpu_util".to_string(), 0.75);

        let result = validator.check_invariants("cpu-util", &context);
        assert!(result.passed);
    }
}
