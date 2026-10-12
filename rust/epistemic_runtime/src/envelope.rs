use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::types::{EpistemicValue, StateVersion};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceVector {
    #[serde(default)]
    pub cpu_normalized_share: f64,
    #[serde(default)]
    pub memory_bytes: u64,
    #[serde(default)]
    pub io_bandwidth_bytes_sec: f64,
    #[serde(default)]
    pub network_bandwidth_bytes_sec: f64,
    #[serde(default)]
    pub gpu_compute_units: f64,
    #[serde(default)]
    pub thermal_celsius: f64,
    #[serde(default)]
    pub power_watts: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpectedDelta {
    pub metric_deltas: HashMap<String, f64>,
    pub confidence: f64,
    pub validity_window_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceEvidenceEnvelope {
    pub schema_version: String,
    pub envelope_id: String,
    pub state_version: StateVersion,
    pub provenance: String,
    pub host_fingerprint: String,
    pub hypervisor_type: String,
    pub guest_id: String,
    pub clock_domain_id: String,
    pub host_monotonic_ns: u64,
    pub wall_timestamp_utc: f64,
    #[serde(default = "default_seq")]
    pub sequence_number: u64,
    pub causal_parent: Option<String>,
    pub canonical_normalized: HashMap<String, EpistemicValue<serde_json::Value>>,
    pub adapter_native: HashMap<String, serde_json::Value>,
    #[serde(default = "default_tag")]
    pub epistemic_tag: String,
}

fn default_seq() -> u64 {
    1
}

fn default_tag() -> String {
    "[CANONICAL_INVARIANT]".to_string()
}

impl ResourceEvidenceEnvelope {
    pub fn is_fresh(&self, now_ns: u64, max_age_ns: u64) -> bool {
        (now_ns - self.host_monotonic_ns) <= max_age_ns
    }
}
