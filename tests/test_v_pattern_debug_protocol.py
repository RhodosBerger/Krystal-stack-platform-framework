"""
tests/test_v_pattern_debug_protocol.py
Test suite for Antigravity & Gemini 4 Argon Epistemic Debug Protocol and Pattern Revision.
Milestone: V-DEBUG-PROTOCOL-01 (SPEC-EPISTEMIC-DEBUG-01).
Constitutional Invariant: VITAL_MAX_HP == 6.
"""

import pytest

from krystal_kernel.pattern_debug_protocol import (
    EPISTEMIC_MANTRA,
    VITAL_MAX_HP,
    EpistemicContractViolation,
    EpistemicErrorClassifier,
    LockedAnchors,
    PatternCandidate,
    PatternCharacteristic,
    PatternStatus,
    PlanLockInGuard,
    ToolResultEnvelope,
    VersionedClaim,
    VersionedClaimStore,
    compute_epistemic_residual,
    verify_mantra_adherence,
)


def test_constitutional_invariant_hp():
    """Verify that VITAL_MAX_HP == 6 is enforced across all structures."""
    assert VITAL_MAX_HP == 6

    # Test LockedAnchors invariant check
    anchors = LockedAnchors(
        primary_goal="Verify deterministic zero-copy IPC",
        scope="krystal_kernel/zero_copy_ring.py",
        stop_conditions=["Zero packet drop", "Latency < 20us"],
    )
    assert anchors.invariants["VITAL_MAX_HP"] == 6

    # Tampering with invariant must raise EpistemicContractViolation
    with pytest.raises(EpistemicContractViolation):
        LockedAnchors(
            primary_goal="Compromised goal",
            scope="root",
            stop_conditions=[],
            invariants={"VITAL_MAX_HP": 10},
        )


def test_locked_anchors_context_drift():
    """Verify LockedAnchors detects drift in primary goal and scope."""
    anchors = LockedAnchors(
        primary_goal="Maintain VSync slice latency under 2.5ms",
        scope="krystal_kernel/krystal_nexus_accelerator.py",
        stop_conditions=["Frame drop == 0"],
    )
    has_drift, msg = anchors.check_context_drift("Maintain VSync slice latency", "krystal_kernel/krystal_nexus_accelerator.py")
    assert not has_drift

    has_drift_empty, msg_empty = anchors.check_context_drift("", "some_scope")
    assert has_drift_empty
    assert "Missing goal" in msg_empty


def test_pattern_combination_starts_as_candidate():
    """
    Combination is NOT proof.
    Combining two individually verified characteristics MUST initialize in PATTERN_CANDIDATE status.
    """
    p1 = PatternCharacteristic(
        name="VSyncBlankingMicroSlicing",
        characteristic="Fills VBlank bubbles with LPDDR5 matrix weight streaming",
        source="krystal_nexus_accelerator.py",
        scope="UMA display scanout",
        evidential_status="VERIFIED_EMPIRICAL",
        validity_bounds="120Hz panels with DWM blanking interval >= 1.2ms",
        dependencies=("dxgi_flip_queue",),
    )
    p2 = PatternCharacteristic(
        name="ZeroCopyRingTransfer",
        characteristic="Transfers tensors across processes without page duplication",
        source="zero_copy_ring.py",
        scope="IPC buffer arbitration",
        evidential_status="VERIFIED_EMPIRICAL",
        validity_bounds="Shared memory ring buffer <= 64MB",
        dependencies=("win32_shared_memory",),
    )

    candidate = PatternCandidate.combine(
        p1=p1,
        p2=p2,
        rationale="Combine micro-slicing with zero-copy IPC ring to minimize LLC pressure",
        falsifier="Ring buffer stalls > 50us or DWM dropped frame count > 0",
        proposer_id="gemini_4_argon_proposer",
    )

    assert candidate.status == PatternStatus.PATTERN_CANDIDATE
    assert len(candidate.components) == 2
    assert candidate.vital_max_hp == 6


def test_anti_self_certification_promotion():
    """
    Verify that a PatternCandidate cannot be promoted by the same entity that proposed it.
    Anti-Self-Certification: proposer_id != evaluator_id.
    """
    p = PatternCharacteristic(
        name="AdaptiveThermalGovernor",
        characteristic="Scales clock frequency inversely to socket temperature",
        source="cnc_adaptive_thermal_governor.py",
        scope="Core scheduling",
        evidential_status="HYPOTHESIS",
        validity_bounds="Junction temp <= 95C",
    )
    candidate = PatternCandidate(
        candidate_id="cand_therm_01",
        components=[p],
        falsifier="Temperature exceeds 95C under sustained matrix workload",
        proposer_id="agent_proposer",
    )

    # Self-certification attempt must fail
    with pytest.raises(EpistemicContractViolation, match="Anti-Self-Certification Violation"):
        candidate.promote_to_verified(
            evaluator_id="agent_proposer",
            test_trace="PASS: tests/test_thermal.py",
        )

    # Promotion without passing test trace must fail
    with pytest.raises(EpistemicContractViolation, match="valid passing test trace"):
        candidate.promote_to_verified(
            evaluator_id="independent_evaluator",
            test_trace="FAIL: Timeout exceeded",
        )

    # Valid external promotion succeeds
    candidate.promote_to_verified(
        evaluator_id="independent_auditor_01",
        test_trace="PASS: Hardware telemetry confirmed 82C max peak across 1000 runs",
    )
    assert candidate.status == PatternStatus.EMPIRICALLY_VERIFIED
    assert candidate.evaluator_id == "independent_auditor_01"


def test_versioned_claim_store_no_silent_overwrite():
    """
    Verify VersionedClaimStore prevents silent overwriting.
    Contradictions require explicit residual calculation and new versioning.
    """
    store = VersionedClaimStore()
    c1 = store.register_claim(
        claim_id="claim_ipc_throughput",
        statement="Zero-copy IPC sustains 48 GB/s bandwidth on Iris Xe UMA",
        evidence_status="EMPIRICAL_BENCHMARK",
    )
    assert c1.version == 1

    # Attempting to re-register the same claim must fail (anti-silent overwrite)
    with pytest.raises(EpistemicContractViolation, match="already exists"):
        store.register_claim(
            claim_id="claim_ipc_throughput",
            statement="Silent overwrite attempt",
            evidence_status="UNVERIFIED",
        )

    # Update with residual creates version 2
    residual = compute_epistemic_residual(48.0, 41.2)
    c2 = store.update_with_residual(
        claim_id="claim_ipc_throughput",
        new_statement="Zero-copy IPC sustains 41.2 GB/s bandwidth under concurrent 4K display scanout",
        new_evidence_status="UPDATED_EMPIRICAL",
        residual_error=residual,
        reason="Display engine memory arbitration steals 6.8 GB/s during active scanout",
    )
    assert c2.version == 2
    assert pytest.approx(c2.residual_error, 0.01) == 6.8
    assert store.get_latest("claim_ipc_throughput").version == 2
    assert len(store.get_history("claim_ipc_throughput")) == 2


def test_tool_result_envelope_separation():
    """
    Verify ToolResultEnvelope:
    1. Previous model output is NOT proof.
    2. Internet data is a hypothesis source, not local truth.
    3. Calibrated local telemetry qualifies as proof.
    """
    # 1. Previous model output
    env_model = ToolResultEnvelope(
        raw_tool_output="I believe the GPU driver is crashing due to timeout.",
        model_interpretation="Agent previously hypothesized TDR failure.",
        is_previous_model_output=True,
    )
    assert not env_model.is_valid_local_proof
    rec_model = env_model.format_epistemic_record()
    assert rec_model["status"] == "MODEL_PRIOR_STATE"
    assert not rec_model["is_valid_proof"]

    # 2. Internet source
    env_web = ToolResultEnvelope(
        raw_tool_output="StackOverflow post claiming Windows 11 disables DXGI flip queue by default.",
        model_interpretation="Web opinion suggesting registry key tweak.",
        is_external_internet=True,
    )
    assert not env_web.is_valid_local_proof
    rec_web = env_web.format_epistemic_record()
    assert rec_web["status"] == "HYPOTHESIS_EXTERNAL"

    # 3. Local test run
    env_local = ToolResultEnvelope(
        raw_tool_output={"exit_code": 0, "vsync_jitter_us": 14.2, "dropped_frames": 0},
        model_interpretation="Measured hardware jitter conforms to real-time budget.",
        is_external_internet=False,
        is_previous_model_output=False,
    )
    assert env_local.is_valid_local_proof
    rec_local = env_local.format_epistemic_record()
    assert rec_local["status"] == "LOCAL_TELEMETRY"
    assert rec_local["is_valid_proof"]


def test_plan_lock_in_guard_and_rational_irritation():
    """
    Verify that PlanLockInGuard enforces STOP -> Frame Audit -> Falsifier
    when Rational Irritation occurs.
    """
    guard = PlanLockInGuard(failure_threshold=2)

    # First step passes
    step1 = guard.record_step(passed=True, residual=0.01)
    assert step1["action"] == "PROCEED"
    assert guard.can_dispatch_next_action()

    # Step with anomaly triggers Rational Irritation immediately
    step2 = guard.record_step(passed=False, residual=0.85, unexpected_anomaly=True)
    assert step2["action"] == "STOP_AND_AUDIT_FRAME"
    assert not guard.can_dispatch_next_action()

    # Trying to execute frame audit with missing fields must fail
    with pytest.raises(EpistemicContractViolation):
        guard.execute_frame_audit("", "falsifier")

    with pytest.raises(EpistemicContractViolation):
        guard.execute_frame_audit("Audited summary", "")

    # Valid Frame Audit unlocks execution
    guard.execute_frame_audit(
        audited_model_summary="Identified Framing Error: Display scanout was not locked to DWM refresh rate.",
        formulated_falsifier="Jitter > 30us after applying VSync lock.",
    )
    assert guard.can_dispatch_next_action()


def test_epistemic_error_classifier_asymmetry():
    """
    Verify Epistemic Error Asymmetry:
    1. Error != proof of falsehood (syntax/import/runtime error is artifact error).
    2. Non-execution != falsification.
    """
    # 1. Non execution
    res_non_exec = EpistemicErrorClassifier.classify(
        error_type="None", execution_attempted=False, execution_succeeded=False
    )
    assert res_non_exec["category"] == "NON_EXECUTION"
    assert not res_non_exec["is_falsification"]
    assert res_non_exec["status"] == "UNTESTED"

    # 2. Syntax / import error
    res_syntax = EpistemicErrorClassifier.classify(
        error_type="SyntaxError: invalid syntax in test runner",
        execution_attempted=True,
        execution_succeeded=False,
    )
    assert res_syntax["category"] == "ARTIFACT_OR_ENV_ERROR"
    assert not res_syntax["is_falsification"]
    assert res_syntax["status"] == "EVIDENCE_INSUFFICIENT"

    # 3. Successful run
    res_success = EpistemicErrorClassifier.classify(
        error_type="", execution_attempted=True, execution_succeeded=True
    )
    assert res_success["category"] == "EXECUTION_SUCCESS"
    assert res_success["status"] == "VALID_EXECUTION"


def test_mantra_adherence():
    """Verify that verify_mantra_adherence detects adherence to the core mantra."""
    assert verify_mantra_adherence(EPISTEMIC_MANTRA)
    assert verify_mantra_adherence("Musíme držať cieľ stabilný, plán môžeš meniť a otváraj rámec pri rozpore.")
    assert not verify_mantra_adherence("Dokonči plán za každú cenu a prepíš staré výsledky.")
