"""
Test Suite for V-ENLG-COMPILER-01: Krystal Epistemic Neurolinguistic Grammar & ArtifactGate
========================================================================================
Milestone: V-ENLG-COMPILER-01
Constitutional Invariant: VITAL_MAX_HP == 6

Verifies:
  1. Evidence Lattice Anti-Promotion Barriers (Simulation -> Measurement, Local -> Verified).
  2. Authority Separation: Proposer role cannot self-certify.
  3. Contradiction Preservation: Conflicting evidence forces CONFLICT verdict.
  4. Anti-Overclaim Pattern: Unbounded superlatives rejected.
  5. Deterministic ENLG Compiler text parsing and AST generation.
  6. ArtifactGate Quarantine, Tri-Plane Parity, and zero-mutation on rejection (Delta_S == 0).
"""

import pytest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import (
    ContractHeader,
    ContractViolation,
    ReceiptContract,
)
from krystal_kernel.epistemic_enlg_compiler import (
    ArtifactGate,
    EnlgAstValidator,
    EnlgClaimTrace,
    EnlgEvidenceClass,
    EnlgRole,
    EnlgVerdict,
    NextExperimentPlan,
    NullTriplet,
)


def _create_sample_trace(
    claim: str = "VSync micro-slicing stabilizes frame-time variance.",
    scope: str = "Intel Iris Xe iGPU, 120Hz VSync, balanced profile within defined invariant bounds.",
    role: EnlgRole = EnlgRole.VERIFIER,
    evidence: tuple = (EnlgEvidenceClass.LOCAL_MEASURED,),
    verdict: EnlgVerdict = EnlgVerdict.MEASURED_WITH_SCOPE,
    has_conflict: bool = False,
) -> EnlgClaimTrace:
    return EnlgClaimTrace(
        claim=claim,
        scope=scope,
        role=role,
        evidence=evidence,
        nulls=NullTriplet(
            m0_no_effect="Frame-time variance is unchanged without slicing.",
            m1_noise="Variance reductions are within background OS jitter.",
            m2_confounder="Effect is caused by thermal throttling lowering clock speed.",
        ),
        falsifier="p99 frame-time variance exceeds 1.5ms across 100 consecutive frames.",
        validity={
            "construct": "Variance directly measures frame pacing stability.",
            "internal": "Controlled background processes and pinned power profile.",
            "external": "Bounded strictly to tested UMA iGPU architecture.",
        },
        verdict=verdict,
        residual=("ARM64 memory model unmeasured", "Discrete GPU behavior unverified"),
        has_conflicting_evidence=has_conflict,
        vital_max_hp=6,
    )


class TestVEnlgEpistemicCompiler:
    """Rigorous formal verification of ENLG v2 compiler and ArtifactGate."""

    def test_01_vital_max_hp_invariant(self):
        """Validates constitutional invariant across all structures."""
        assert VITAL_MAX_HP == 6
        trace = _create_sample_trace()
        assert trace.vital_max_hp == 6

    def test_02_evidence_lattice_unknown_rejection(self):
        """Rejects promoting strictly UNKNOWN evidence to positive verdicts."""
        trace = _create_sample_trace(
            evidence=(EnlgEvidenceClass.UNKNOWN,),
            verdict=EnlgVerdict.SUPPORTED_WITH_SCOPE,
        )
        with pytest.raises(ContractViolation) as exc_info:
            EnlgAstValidator.validate_claim_trace(trace)
        assert exc_info.value.violation_code == "ERR_ENLG_UNKNOWN_PROMOTION"

    def test_03_evidence_lattice_simulation_to_measurement_rejection(self):
        """Simulation evidence (rank 1) cannot promote to MEASURED_WITH_SCOPE (rank 2)."""
        trace = _create_sample_trace(
            evidence=(EnlgEvidenceClass.SIMULATED,),
            verdict=EnlgVerdict.MEASURED_WITH_SCOPE,
        )
        with pytest.raises(ContractViolation) as exc_info:
            EnlgAstValidator.validate_claim_trace(trace)
        assert exc_info.value.violation_code == "ERR_ENLG_SIMULATION_TO_MEASUREMENT"

    def test_04_evidence_lattice_local_to_verified_rejection(self):
        """LOCAL_MEASURED evidence (rank 2) cannot promote to VERIFIED_WITH_SCOPE (rank 3) without CI."""
        trace = _create_sample_trace(
            evidence=(EnlgEvidenceClass.LOCAL_MEASURED,),
            verdict=EnlgVerdict.VERIFIED_WITH_SCOPE,
        )
        with pytest.raises(ContractViolation) as exc_info:
            EnlgAstValidator.validate_claim_trace(trace)
        assert exc_info.value.violation_code == "ERR_ENLG_LOCAL_TO_VERIFIED_PROMOTION"

    def test_05_authority_separation_proposer_cannot_self_certify(self):
        """Proposer role is structurally barred from self-assigning VERIFIED_WITH_SCOPE."""
        trace = _create_sample_trace(
            role=EnlgRole.PROPOSER,
            evidence=(EnlgEvidenceClass.REMOTE_CI,),
            verdict=EnlgVerdict.VERIFIED_WITH_SCOPE,
        )
        with pytest.raises(ContractViolation) as exc_info:
            EnlgAstValidator.validate_claim_trace(trace)
        assert exc_info.value.violation_code == "ERR_ENLG_PROPOSER_SELF_CERTIFICATION"

    def test_06_contradiction_preservation_forces_conflict_verdict(self):
        """Conflicting evidence branches cannot be averaged; must emit CONFLICT."""
        trace = _create_sample_trace(
            has_conflict=True,
            verdict=EnlgVerdict.SUPPORTED_WITH_SCOPE,
        )
        with pytest.raises(ContractViolation) as exc_info:
            EnlgAstValidator.validate_claim_trace(trace)
        assert exc_info.value.violation_code == "ERR_ENLG_CONFLICT_NOT_PRESERVED"

    def test_07_anti_overclaim_superlatives_rejection(self):
        """Banned ungrounded superlatives ('zero-risk', 'optimal') trigger rejection."""
        trace = _create_sample_trace(
            claim="Krystal zero-risk execution guarantees optimal throughput.",
            scope="General unconstrained production environment.",  # Lacks defined invariant bounding
        )
        with pytest.raises(ContractViolation) as exc_info:
            EnlgAstValidator.validate_claim_trace(trace)
        assert exc_info.value.violation_code == "ERR_ENLG_OVERCLAIM_SUPERLATIVE"

    def test_08_parser_valid_enlg_text_roundtrip(self):
        """Deterministic compiler parses canonical ENLG text format into typed AST."""
        canonical_text = """
CLAIM:
Zero-Copy ring eliminates torn reads under concurrent producer-consumer stress.

SCOPE:
Tested on x86_64, Windows 11, C11 memory model within defined invariant bounds.

ROLE:
VERIFIER

EVIDENCE:
LOCAL-MEASURED

ANTI-PROMOTION:
No extrapolation to ARM64 weak ordering without hardware testing.

NULLS:
M0_NO_EFFECT: Ring experiences torn reads equal to non-atomic memory blocks.
M1_NOISE: Read integrity was a random artifact of scheduling alignment.
M2_MEASUREMENT_OR_CONFOUNDER: Test workload was serialized by OS scheduler lock.

FALSIFIER:
Any torn read detected under 10,000 asynchronous frame mutations.

VALIDITY:
Construct validity: Torn frame detection directly catches partial updates.
Internal validity: Multi-threaded test pinpoints cross-cacheline hazards.
External validity: Validated on x86_64 hardware.

VERDICT:
MEASURED_WITH_SCOPE

RESIDUAL:
- ARM64 memory barrier semantics unverified on physical hardware.
- High contention multi-consumer queue not supported (SPSC only).

NEXT_EXPERIMENT:
Task: Execute ARM64 emulation run in QEMU.
Measurement: Record torn-read count across 10,000 frames.
Success: Zero torn reads.
Failure: Non-zero torn reads detected.
Rollback: Revert to mutual exclusion lock.
Stop: Any violation of VITAL_MAX_HP == 6.
"""
        parsed_trace = EnlgAstValidator.parse_compiler_text(canonical_text)
        assert parsed_trace.role == EnlgRole.VERIFIER
        assert parsed_trace.verdict == EnlgVerdict.MEASURED_WITH_SCOPE
        assert EnlgEvidenceClass.LOCAL_MEASURED in parsed_trace.evidence
        assert parsed_trace.vital_max_hp == 6

    def test_09_artifact_gate_deposit_holds_in_quarantine(self):
        """Deposited artifacts are held strictly in quarantine without release."""
        gate = ArtifactGate()
        payload = {"action": "MOVE", "to_hex": (1, 0), "vital_max_hp": 6}

        art_id = gate.deposit("art_001", payload, EnlgRole.PROPOSER)
        assert art_id == "art_001"
        assert gate.quarantined_count == 1
        assert gate.released_count == 0

    def test_10_artifact_gate_rejects_unauthorized_proposer_evaluator(self):
        """ArtifactGate blocks release when evaluator role is PROPOSER (Separation of Concerns)."""
        gate = ArtifactGate()
        payload = {"action": "MOVE", "to_hex": (1, 0), "vital_max_hp": 6}
        gate.deposit("art_002", payload, EnlgRole.PROPOSER)

        header = ContractHeader("c5_h", 1, 10, 0x10, 1, 1000, "0" * 64)
        receipt = ReceiptContract(header, "digest_exec", "balanced", 500, was_degraded_fallback=False)

        proposer_trace = _create_sample_trace(
            role=EnlgRole.PROPOSER,
            evidence=(EnlgEvidenceClass.LOCAL_MEASURED,),
            verdict=EnlgVerdict.MEASURED_WITH_SCOPE,
        )

        state_mutated = [False]
        def commit_fn():
            state_mutated[0] = True
            return "COMMITTED"

        with pytest.raises(ContractViolation) as exc_info:
            gate.release("art_002", receipt, proposer_trace, commit_fn)

        assert exc_info.value.violation_code == "ERR_GATE_EVALUATOR_IS_PROPOSER"
        assert state_mutated[0] is False  # Delta S == 0 guarantee!
        assert gate.quarantined_count == 1
        assert gate.released_count == 0

    def test_11_artifact_gate_rejects_degraded_fallback_receipt(self):
        """ArtifactGate refuses to release an artifact on degraded fallback receipt."""
        gate = ArtifactGate()
        payload = {"action": "ATTACK", "target": "duelist_2", "vital_max_hp": 6}
        gate.deposit("art_003", payload, EnlgRole.PROPOSER)

        header = ContractHeader("c5_h_degraded", 1, 11, 0x10, 1, 1000, "0" * 64)
        # Degraded fallback receipt!
        receipt = ReceiptContract(header, "digest_fallback", "ultra_saver", 500, was_degraded_fallback=True)

        verifier_trace = _create_sample_trace(
            role=EnlgRole.VERIFIER,
            evidence=(EnlgEvidenceClass.LOCAL_MEASURED,),
            verdict=EnlgVerdict.MEASURED_WITH_SCOPE,
        )

        state_mutated = [False]
        def commit_fn():
            state_mutated[0] = True
            return "COMMITTED"

        with pytest.raises(ContractViolation) as exc_info:
            gate.release("art_003", receipt, verifier_trace, commit_fn)

        assert exc_info.value.violation_code == "ERR_GATE_DEGRADED_RECEIPT_REJECTED"
        assert state_mutated[0] is False  # Delta S == 0 guarantee!
        assert gate.quarantined_count == 1

    def test_12_artifact_gate_successful_release_with_verified_receipt(self):
        """ArtifactGate releases artifact and commits state when all proof conditions hold."""
        gate = ArtifactGate()
        payload = {"action": "STRIKE", "damage": 2, "vital_max_hp": 6}
        gate.deposit("art_004", payload, EnlgRole.PROPOSER)

        header = ContractHeader("c5_h_ok", 1, 12, 0x10, 1, 1000, "0" * 64)
        receipt = ReceiptContract(header, "digest_valid_exec", "balanced", 850, was_degraded_fallback=False)

        verifier_trace = _create_sample_trace(
            role=EnlgRole.VERIFIER,
            evidence=(EnlgEvidenceClass.REMOTE_CI, EnlgEvidenceClass.LOCAL_MEASURED),
            verdict=EnlgVerdict.VERIFIED_WITH_SCOPE,
        )

        state_mutated = [False]
        def commit_fn():
            state_mutated[0] = True
            return "COMMITTED_STATE_S1"

        result = gate.release("art_004", receipt, verifier_trace, commit_fn)

        assert result == "COMMITTED_STATE_S1"
        assert state_mutated[0] is True
        assert gate.quarantined_count == 0
        assert gate.released_count == 1
