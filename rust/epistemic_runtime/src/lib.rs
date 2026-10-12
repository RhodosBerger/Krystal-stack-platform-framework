pub mod types;
pub mod envelope;
pub mod authority;

pub use types::*;
pub use envelope::*;
pub use authority::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_authority_arbitration_mapping() {
        // PASS + SUFFICIENT -> COMMIT_APPROVED (WOULD_COMMIT)
        let (v, act) = arbitrate_authority(&SafetyVerdictType::Pass, &EvidenceVerdictType::Sufficient);
        assert_eq!(v, AuthorityVerdictType::CommitApproved);
        assert_eq!(act, "WOULD_COMMIT");

        // PASS + INSUFFICIENT -> DEFER_INSUFFICIENT_EVIDENCE
        let (v, act) = arbitrate_authority(&SafetyVerdictType::Pass, &EvidenceVerdictType::Insufficient);
        assert_eq!(v, AuthorityVerdictType::DeferInsufficientEvidence);
        assert_eq!(act, "TRIGGER_ACTIVE_MICRO_PROBE");

        // PASS + CONTRADICTORY -> QUARANTINE_CONTRADICTORY_EVIDENCE
        let (v, act) = arbitrate_authority(&SafetyVerdictType::Pass, &EvidenceVerdictType::Contradictory);
        assert_eq!(v, AuthorityVerdictType::QuarantineContradictoryEvidence);
        assert_eq!(act, "QUARANTINE_AND_RECALIBRATE");

        // VETO + ANY -> VETO_SAFETY_BREACH
        let (v, act) = arbitrate_authority(&SafetyVerdictType::Veto, &EvidenceVerdictType::Sufficient);
        assert_eq!(v, AuthorityVerdictType::VetoSafetyBreach);
        assert_eq!(act, "REJECT_NOOP");
    }

    #[test]
    fn test_candidate_action_expiration() {
        let sv = StateVersion {
            major: 1,
            minor: 0,
            sequence: 5,
            wall_timestamp_utc: 1000.0,
            clock_domain_id: "host_clock_0".to_string(),
            causal_parent: None,
        };

        let candidate = CandidateAction {
            action_id: "ACT-TEST-001".to_string(),
            source_state_version: sv.clone(),
            generated_at_ns: 1000,
            valid_until_ns: 2000,
            action_type: "PIN_VCPU".to_string(),
            target_tenant_id: "vm-1".to_string(),
            expected_delta: ExpectedDelta {
                metric_deltas: std::collections::HashMap::new(),
                confidence: 0.9,
                validity_window_ns: 1000,
            },
            required_resources: ResourceVector::default(),
            evidence_refs: vec!["ENV-1".to_string()],
            reversibility: ReversibilityClass::Reversible,
            vital_hp: 6,
        };

        // Valid within window and matching state version
        assert!(candidate.is_valid_at(&sv, 1500).is_ok());

        // Invalid if now > valid_until_ns
        assert!(candidate.is_valid_at(&sv, 2500).is_err());

        // Invalid if state version diverged
        let diverged_sv = StateVersion {
            sequence: 6,
            ..sv
        };
        assert!(candidate.is_valid_at(&diverged_sv, 1500).is_err());
    }

    impl Default for ResourceVector {
        fn default() -> Self {
            Self {
                cpu_normalized_share: 0.0,
                memory_bytes: 0,
                io_bandwidth_bytes_sec: 0.0,
                network_bandwidth_bytes_sec: 0.0,
                gpu_compute_units: 0.0,
                thermal_celsius: 0.0,
                power_watts: 0.0,
            }
        }
    }
}
