use serde::{Deserialize, Serialize};
use crate::types::{PolicyVersion, StateVersion, VITAL_MAX_HP};
use crate::envelope::{ExpectedDelta, ResourceVector};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReversibilityClass {
    Reversible,
    Compensable,
    Irreversible,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SafetyTier {
    MemorySafety,
    ResourceSafety,
    PhysicalSafety,
    SystemInvariants,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SafetyVerdictType {
    Pass,
    Veto,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EvidenceVerdictType {
    Sufficient,
    Insufficient,
    Contradictory,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AuthorityVerdictType {
    CommitApproved,
    DeferInsufficientEvidence,
    QuarantineContradictoryEvidence,
    VetoSafetyBreach,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateAction {
    pub action_id: String,
    pub source_state_version: StateVersion,
    pub generated_at_ns: u64,
    pub valid_until_ns: u64,
    pub action_type: String,
    pub target_tenant_id: String,
    pub expected_delta: ExpectedDelta,
    pub required_resources: ResourceVector,
    pub evidence_refs: Vec<String>,
    pub reversibility: ReversibilityClass,
    #[serde(default = "default_vital_hp")]
    pub vital_hp: u32,
}

fn default_vital_hp() -> u32 {
    VITAL_MAX_HP
}

impl CandidateAction {
    pub fn is_valid_at(&self, current_version: &StateVersion, now_ns: u64) -> Result<(), String> {
        if !self.source_state_version.is_compatible_with(current_version) {
            return Err("StateVersion mismatch".to_string());
        }
        if now_ns > self.valid_until_ns {
            return Err("Action validity window expired".to_string());
        }
        if self.vital_hp != VITAL_MAX_HP {
            return Err(format!("VITAL_MAX_HP invariant breach: {}", self.vital_hp));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetyProof {
    pub verdict: SafetyVerdictType,
    pub memory_safety_check: serde_json::Value,
    pub resource_safety_check: serde_json::Value,
    pub physical_safety_check: serde_json::Value,
    pub system_invariants_check: serde_json::Value,
    pub policy_version: PolicyVersion,
    pub evaluated_at_ns: u64,
    #[serde(default)]
    pub violations: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceProof {
    pub verdict: EvidenceVerdictType,
    pub freshness_eval: serde_json::Value,
    pub provenance_eval: serde_json::Value,
    pub cross_sensor_consistency: serde_json::Value,
    pub sample_count_eval: serde_json::Value,
    pub calibration_error_eval: serde_json::Value,
    pub domain_eval: serde_json::Value,
    pub state_version_eval: serde_json::Value,
    pub clock_validity_eval: serde_json::Value,
    pub evaluated_at_ns: u64,
    #[serde(default)]
    pub contradictions: Vec<serde_json::Value>,
    #[serde(default)]
    pub gaps: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuthorityProofBundle {
    pub verdict: AuthorityVerdictType,
    pub safety_proof: SafetyProof,
    pub evidence_proof: EvidenceProof,
    pub policy_version: PolicyVersion,
    pub state_version: StateVersion,
    pub evaluated_at_ns: u64,
    pub evidence_refs: Vec<String>,
    pub action_id: String,
    pub lifecycle_action: String,
    #[serde(default)]
    pub audit_hash: String,
}

pub fn arbitrate_authority(
    safety: &SafetyVerdictType,
    evidence: &EvidenceVerdictType,
) -> (AuthorityVerdictType, &'static str) {
    match safety {
        SafetyVerdictType::Veto => (
            AuthorityVerdictType::VetoSafetyBreach,
            "REJECT_NOOP",
        ),
        SafetyVerdictType::Pass => match evidence {
            EvidenceVerdictType::Contradictory => (
                AuthorityVerdictType::QuarantineContradictoryEvidence,
                "QUARANTINE_AND_RECALIBRATE",
            ),
            EvidenceVerdictType::Insufficient => (
                AuthorityVerdictType::DeferInsufficientEvidence,
                "TRIGGER_ACTIVE_MICRO_PROBE",
            ),
            EvidenceVerdictType::Sufficient => (
                AuthorityVerdictType::CommitApproved,
                "WOULD_COMMIT",
            ),
        },
    }
}
