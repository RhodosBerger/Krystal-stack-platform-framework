use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Policy proposal from the Cognitive Stream (LLM)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyProposal {
    pub proposal_id: String,
    pub proposal_type: ProposalType,
    pub target: String,
    pub suggested_value: serde_json::Value,
    pub justification: String,
    pub confidence: f64,
    pub introspective_comment: Option<String>,
    pub related_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalType {
    ParameterAdjustment,
    RuleCreation,
    RuleModification,
    RuleDeactivation,
    ScoringWeightUpdate,
}

/// Resource budgets for the Economic Engine
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceBudgets {
    pub cpu_budget: PowerBudget,
    pub gpu_budget: PowerBudget,
    pub memory_tier_budget: MemoryBudget,
    pub thermal_headroom: ThermalBudget,
    pub latency_budget: LatencyBudget,
    pub time_budget: TimeBudget,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PowerBudget {
    pub current_mw: u32,
    pub max_mw: u32,
    pub headroom_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryBudget {
    pub bandwidth_used_gbps: f64,
    pub bandwidth_max_gbps: f64,
    pub hot_slots_used: u32,
    pub hot_slots_max: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThermalBudget {
    pub cpu_current_c: u32,
    pub cpu_max_c: u32,
    pub gpu_current_c: u32,
    pub gpu_max_c: u32,
    pub headroom_c: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LatencyBudget {
    pub target_frametime_ms: f64,
    pub current_frametime_ms: f64,
    pub headroom_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimeBudget {
    pub decision_cycle_ms: u32,
    pub used_ms: u32,
    pub remaining_ms: u32,
}

/// MicroInferenceRule - declarative rules from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroInferenceRule {
    pub rule_id: String,
    pub version: String,
    pub source: RuleSource,
    pub safety_tier: u8,
    pub shadow_mode: bool,
    pub conditions: Vec<Condition>,
    pub actions: Vec<Action>,
    pub justification: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleSource {
    LlmGenerated,
    UserDefined,
    SystemDefault,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    pub metric: String,
    pub operator: ConditionOperator,
    pub value: serde_json::Value,
    pub logical_op: Option<LogicalOp>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionOperator {
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
    Contains,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogicalOp {
    And,
    Or,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub action_type: ActionType,
    pub params: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionType {
    SetCpuAffinity,
    SetGovernor,
    SetGpuPowerLimit,
    SetMemoryTier,
    TriggerCooldown,
    LogEvent,
}

/// Telemetry snapshot
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TelemetrySnapshot {
    pub timestamp: String,
    pub cpu_util: f64,
    pub gpu_util: f64,
    pub frametime_ms: f64,
    pub temp_cpu: u32,
    pub temp_gpu: u32,
    pub active_process_category: String,
}

/// Experience tuple for training (S, A, R, S')
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub id: String,
    pub timestamp: String,
    pub state: TelemetrySnapshot,
    pub action: ActionRecord,
    pub reward: f64,
    pub next_state: TelemetrySnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRecord {
    pub rule_id: String,
    pub action_type: ActionType,
    pub params: HashMap<String, serde_json::Value>,
}

/// Operator profile for economic engine tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorProfile {
    Gaming,
    Production,
    Balanced,
    PowerSaver,
}

/// Safety guardrail configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConfig {
    pub max_cpu_temp_c: u32,
    pub max_gpu_temp_c: u32,
    pub min_free_ram_mb: u32,
    pub restricted_sysfs_paths: Vec<String>,
    pub allow_process_injection: bool,
    pub emergency_cooldown_threshold_c: u32,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            max_cpu_temp_c: 95,
            max_gpu_temp_c: 90,
            min_free_ram_mb: 1024,
            restricted_sysfs_paths: vec![
                "/sys/kernel/security".into(),
                "/sys/firmware".into(),
            ],
            allow_process_injection: false,
            emergency_cooldown_threshold_c: 90,
        }
    }
}
