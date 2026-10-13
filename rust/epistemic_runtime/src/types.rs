use serde::{Deserialize, Serialize};

pub const VITAL_MAX_HP: u32 = 6;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ValidityState {
    Unknown,
    Observed,
    Derived,
    Predicted,
    Calibrated,
    Validated,
    Stale,
    Expired,
    Contradictory,
    Quarantined,
    Fresh, // Backward-compatible alias
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EvidenceKind {
    HardwareMeasured,
    RuntimeMeasured,
    IntegrationVerified,
    Derived,
    Inferred,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateVersion {
    pub major: u32,
    pub minor: u32,
    pub sequence: u64,
    pub wall_timestamp_utc: f64,
    #[serde(default = "default_clock_domain")]
    pub clock_domain_id: String,
    pub causal_parent: Option<String>,
}

fn default_clock_domain() -> String {
    "host_clock_0".to_string()
}

impl StateVersion {
    pub fn is_compatible_with(&self, other: &StateVersion) -> bool {
        self.major == other.major
            && self.minor == other.minor
            && self.sequence == other.sequence
            && self.clock_domain_id == other.clock_domain_id
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyVersion {
    pub policy_id: String,
    pub version: String,
    pub provenance: String,
    pub validity_domain: String,
    pub calibration_source: String,
    pub created_at_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpistemicValue<T> {
    pub value: T,
    pub provenance: String,
    pub evidence_kind: EvidenceKind,
    pub uncertainty: f64,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    pub valid_from_ns: u64,
    pub valid_to_ns: u64,
    pub domain: String,
    pub state_version: StateVersion,
    pub source_adapter: String,
    #[serde(default = "default_clock_domain")]
    pub clock_domain_id: String,
    #[serde(default)]
    pub calibration_error: f64,
    pub policy_version: Option<String>,
    pub causal_parent: Option<String>,
}

fn default_confidence() -> f64 {
    1.0
}

impl<T> EpistemicValue<T> {
    pub fn get_validity_state(&self, now_ns: u64, max_stale_ns: Option<u64>) -> ValidityState {
        if now_ns < self.valid_from_ns {
            return ValidityState::Unknown;
        }
        if now_ns > self.valid_to_ns {
            return ValidityState::Expired;
        }
        let stale_limit = max_stale_ns.unwrap_or(((self.valid_to_ns - self.valid_from_ns) as f64 * 0.7) as u64);
        if (now_ns - self.valid_from_ns) > stale_limit {
            return ValidityState::Stale;
        }
        ValidityState::Fresh
    }

    pub fn is_usable_for_decision(&self, now_ns: u64, max_stale_ns: Option<u64>) -> bool {
        self.get_validity_state(now_ns, max_stale_ns) == ValidityState::Fresh
    }
}
