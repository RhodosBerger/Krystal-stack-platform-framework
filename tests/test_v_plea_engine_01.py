"""
Krystal Stack Platform Framework - Phase-Locked Epistemic Action (PLEA) Test Suite
==================================================================================
Milestone: V-PLEA-ENGINE-01 (Verification of the 5 MTAP Axioms & PLEA Engine)
Constitutional Invariant: VITAL_MAX_HP == 6
"""

import time
import pytest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import ContractHeader, ContractViolation, ReceiptContract
from krystal_kernel.zero_copy_contracts import (
    BinaryContractType,
    KRYSTAL_MAGIC,
    KRYSTAL_PROTOCOL_VERSION,
    ZeroCopyContractFrame,
)
from krystal_kernel.epistemic_enlg_compiler import (
    EnlgRole,
    EnlgEvidenceClass,
    EnlgVerdict,
)
from krystal_kernel.plea_engine import (
    VBLANK_WINDOW_MS,
    PL1_BASE_POWER_WATTS,
    FLAG_PLEA_CONFLICT_BIFURCATION,
    EpistemicCalorimeter,
    HypothesisThermodynamics,
    BifurcatedConflictRing,
    BifurcatedBranch,
    VBlankExperimentInjector,
    CausalInertiaDamper,
    PleaEngine,
    PleaTrajectoryVerdict,
)


def _make_dummy_frame(seq: int = 1, vital_hp: int = 6) -> ZeroCopyContractFrame:
    """Helper to create a valid 128-byte ZeroCopyContractFrame."""
    f = ZeroCopyContractFrame(
        magic=KRYSTAL_MAGIC,
        version=KRYSTAL_PROTOCOL_VERSION,
        vital_hp=vital_hp,
        contract_type=BinaryContractType.STATE,
        payload_size=32,
        flags=0,
        sequence_id=seq,
        generation_id=0,
        timestamp_ns=time.time_ns(),
        capability_scope=1,
        nonce=1000 + seq,
        predecessor_hash_raw=b"\x11" * 32,
        profile_id=1,
        verdict_id=int(EnlgVerdict.MEASURED_WITH_SCOPE.value == "MEASURED_WITH_SCOPE"),
        approved_authority_mask=0xFF,
        metric1=120.0,
        metric2=1.5,
        signature_or_digest_raw=b"\x00" * 32,
    )
    sig = f.compute_binary_digest()
    return ZeroCopyContractFrame(
        magic=f.magic,
        version=f.version,
        vital_hp=f.vital_hp,
        contract_type=f.contract_type,
        payload_size=f.payload_size,
        flags=f.flags,
        sequence_id=f.sequence_id,
        generation_id=f.generation_id,
        timestamp_ns=f.timestamp_ns,
        capability_scope=f.capability_scope,
        nonce=f.nonce,
        predecessor_hash_raw=f.predecessor_hash_raw,
        profile_id=f.profile_id,
        verdict_id=f.verdict_id,
        approved_authority_mask=f.approved_authority_mask,
        metric1=f.metric1,
        metric2=f.metric2,
        signature_or_digest_raw=sig,
    )


class TestVPleaEngine01:
    """Verification suite for Milestone V-PLEA-ENGINE-01."""

    def test_01_vital_max_hp_invariant(self):
        """Invariant: VITAL_MAX_HP == 6 must hold across all PLEA subsystems."""
        assert VITAL_MAX_HP == 6
        cal = EpistemicCalorimeter()
        ring = BifurcatedConflictRing()
        inj = VBlankExperimentInjector()
        damper = CausalInertiaDamper()
        engine = PleaEngine(cal, ring, inj, damper)
        assert engine is not None

        # Negative test: tampering with HP in dataclass
        with pytest.raises(ContractViolation):
            HypothesisThermodynamics(hypothesis_id="h1", start_timestamp_ns=0, vital_max_hp=5)

    def test_02_calorimeter_thermodynamic_work_metering(self):
        """T1: Calorimeter tracks electrical energy (Joules) and UMA memory bytes."""
        cal = EpistemicCalorimeter(power_watts=28.0)
        entry = cal.register_hypothesis("hyp_opt_01")
        assert entry.cumulative_energy_joules == 0.0
        assert entry.cumulative_uma_bytes == 0

        # Run 100 ms of work with 1 MB transferred
        entry = cal.record_work("hyp_opt_01", duration_ms=100.0, uma_bytes_transferred=1024 * 1024)
        expected_energy = 28.0 * (100.0 / 1000.0)  # 2.8 Joules
        assert abs(entry.cumulative_energy_joules - expected_energy) < 1e-4
        assert entry.cumulative_uma_bytes == 1024 * 1024
        assert not entry.is_parasitic

    def test_03_calorimeter_parasitic_hypothesis_decoupling(self):
        """T1: Hypotheses exceeding energy budget without entropy reduction are decoupled."""
        cal = EpistemicCalorimeter(max_energy_joules=3.0, power_watts=28.0)
        cal.register_hypothesis("hyp_parasite")

        # Exceed 3 Joules (e.g. 150 ms * 28 W = 4.2 J) without reducing entropy
        entry = cal.record_work("hyp_parasite", duration_ms=150.0, uma_bytes_transferred=512, entropy_reduction=0.0)
        assert entry.is_parasitic
        assert cal.is_parasitic("hyp_parasite")

    def test_04_conflict_ring_homotopy_prohibits_convex_averaging(self):
        """T2: Convex averaging of conflicting evidence is strictly forbidden."""
        ring = BifurcatedConflictRing()
        with pytest.raises(ContractViolation) as exc_info:
            ring.attempt_convex_averaging(100.0, 200.0)
        assert "ERR_HOMOTOPY_CONVEX_AVERAGING_FORBIDDEN" in str(exc_info.value.violation_code)

    def test_05_conflict_ring_bifurcation_fork(self):
        """T2 + T4: Forks conflict into parallel Alpha and Beta branches sharing predecessor_hash."""
        ring = BifurcatedConflictRing()
        parent = _make_dummy_frame(seq=10)
        parent_digest = parent.compute_binary_digest()

        alpha, beta = ring.fork_conflict(
            conflict_id="conf_01",
            parent_frame=parent,
            claim_a="Throughput = 120 FPS",
            evidence_a=EnlgEvidenceClass.LOCAL_MEASURED,
            metric_a=120.0,
            claim_b="Throughput = 45 FPS",
            evidence_b=EnlgEvidenceClass.PHYSICAL_MEASURED,
            metric_b=45.0,
        )

        assert alpha.generation_id == 1
        assert beta.generation_id == 2
        assert alpha.fork_predecessor_hash == parent_digest
        assert beta.fork_predecessor_hash == parent_digest
        assert (alpha.frame.flags & FLAG_PLEA_CONFLICT_BIFURCATION) != 0
        assert (beta.frame.flags & FLAG_PLEA_CONFLICT_BIFURCATION) != 0
        assert alpha.frame.metric1 == 120.0
        assert beta.frame.metric1 == 45.0

    def test_06_conflict_ring_collapse_with_cdt_receipt(self):
        """T2 + T4: Collapses bifurcated superposition once CDT receipt is provided."""
        ring = BifurcatedConflictRing()
        parent = _make_dummy_frame(seq=5)
        ring.fork_conflict(
            conflict_id="conf_02",
            parent_frame=parent,
            claim_a="A",
            evidence_a=EnlgEvidenceClass.SIMULATED,
            metric_a=10.0,
            claim_b="B",
            evidence_b=EnlgEvidenceClass.PHYSICAL_MEASURED,
            metric_b=20.0,
        )

        header = ContractHeader("rcpt_cdt_01", 1, 1, 0x10, 1, 1000, "0" * 64)
        receipt = ReceiptContract(
            header=header,
            execution_digest="exec_cdt_01",
            applied_profile="balanced",
            actuation_duration_ns=500_000,
            was_degraded_fallback=False,
        )

        # Collapse to Beta (winning_branch_generation = 2)
        collapsed = ring.collapse_branch("conf_02", winning_branch_generation=2, cdt_receipt=receipt)
        assert collapsed.generation_id == 0  # Re-unified trunk
        assert collapsed.vital_hp == 6
        assert (collapsed.flags & FLAG_PLEA_CONFLICT_BIFURCATION) == 0
        assert collapsed.metric1 == 20.0

    def test_07_vblank_injector_shadow_phase_execution(self):
        """T3: Executes sub-millisecond experiments cleanly within the 2.2 ms shadow window."""
        injector = VBlankExperimentInjector(vblank_budget_ms=2.2)

        def mock_micro_experiment():
            time.sleep(0.0005)  # 0.5 ms
            return 42

        # 0.5 ms fits in 2.2 ms window with 0.0 ms elapsed
        success, result, duration_ms = injector.execute_in_shadow(
            experiment_fn=mock_micro_experiment,
            estimated_duration_ms=0.5,
            elapsed_vblank_ms=0.0,
        )
        assert success is True
        assert result == 42
        assert duration_ms >= 0.4

    def test_08_vblank_injector_overrun_protection_yields(self):
        """T3: Yields experiment execution before scanout if budget is insufficient."""
        injector = VBlankExperimentInjector(vblank_budget_ms=2.2)

        def mock_heavy_experiment():
            return 999

        # 1.5 ms estimated does NOT fit when 1.0 ms already elapsed (remaining = 1.2 ms)
        success, reason, dur = injector.execute_in_shadow(
            experiment_fn=mock_heavy_experiment,
            estimated_duration_ms=1.5,
            elapsed_vblank_ms=1.0,
        )
        assert success is False
        assert "ERR_VBLANK_BUDGET_EXCEEDED" in reason
        assert dur == 0.0

    def test_09_causal_inertia_damper_momentum_decay(self):
        """T4: Computes exponential causal momentum across verified evidence chain."""
        damper = CausalInertiaDamper(gamma=0.85)

        # Single physical measured = 1.0
        m1 = damper.compute_momentum([EnlgEvidenceClass.PHYSICAL_MEASURED])
        assert abs(m1 - 1.0) < 1e-4

        # Chain of 3 physical measured: 1.0 + 0.85 + 0.85^2 = 2.5725
        m3 = damper.compute_momentum([
            EnlgEvidenceClass.PHYSICAL_MEASURED,
            EnlgEvidenceClass.PHYSICAL_MEASURED,
            EnlgEvidenceClass.PHYSICAL_MEASURED,
        ])
        expected = 1.0 + 0.85 + (0.85 ** 2)
        assert abs(m3 - expected) < 1e-4

    def test_10_causal_inertia_damper_dampens_weak_speculative_proposals(self):
        """T4: Dampens unverified speculative proposals against high-momentum trunk."""
        damper = CausalInertiaDamper(gamma=0.85)
        trunk = [
            EnlgEvidenceClass.PHYSICAL_MEASURED,
            EnlgEvidenceClass.PHYSICAL_MEASURED,
            EnlgEvidenceClass.PHYSICAL_MEASURED,
        ]  # Trunk momentum ~ 2.57 > 1.5 threshold

        # Unverified proposal from PROPOSER
        admissible, reason = damper.evaluate_proposal_admissibility(
            trunk_evidence=trunk,
            proposal_evidence=EnlgEvidenceClass.SIMULATED,
            proposal_role=EnlgRole.PROPOSER,
        )
        assert admissible is False
        assert "PROPOSAL_DAMPENED" in reason

        # Verified evidence from VERIFIER is admissible
        admissible_v, reason_v = damper.evaluate_proposal_admissibility(
            trunk_evidence=trunk,
            proposal_evidence=EnlgEvidenceClass.PHYSICAL_MEASURED,
            proposal_role=EnlgRole.VERIFIER,
        )
        assert admissible_v is True
        assert reason_v == "PROPOSAL_ADMISSIBLE"

    def test_11_plea_engine_epistemic_action_integral(self):
        """Synthesized Action Integral S[pi] = alpha*Cost + beta*Conflict - gamma*Inertia."""
        engine = PleaEngine()

        # Lower cost + higher inertia -> negative (stable) action
        s_stable = engine.evaluate_action_integral(uma_cost=0.1, has_topological_conflict=False, causal_inertia=3.0)
        # 1.0*0.1 + 0.0 - 0.5*3.0 = 0.1 - 1.5 = -1.4
        assert abs(s_stable - (-1.4)) < 1e-4

        # High cost + conflict + zero inertia -> positive (destabilizing) action
        s_unstable = engine.evaluate_action_integral(uma_cost=2.0, has_topological_conflict=True, causal_inertia=0.0)
        # 1.0*2.0 + 2.0*1.0 - 0.0 = 4.0
        assert abs(s_unstable - 4.0) < 1e-4

    def test_12_plea_engine_full_trajectory_lifecycle(self):
        """End-to-end processing of proposals across the PLEA engine."""
        engine = PleaEngine()
        trunk = [EnlgEvidenceClass.PHYSICAL_MEASURED, EnlgEvidenceClass.LOCAL_MEASURED]

        # 1. Valid micro-proposal in VBlank
        verdict = engine.process_proposal(
            hypothesis_id="hyp_valid_01",
            proposal_role=EnlgRole.VERIFIER,
            proposal_evidence=EnlgEvidenceClass.LOCAL_MEASURED,
            trunk_history=trunk,
            estimated_vblank_ms=0.5,
            elapsed_vblank_ms=0.0,
            uma_bytes=4096,
        )
        assert verdict.admissible is True
        assert verdict.vital_max_hp == 6
        assert verdict.action_integral_value < 100.0

        # 2. VBlank budget overrun rejection
        verdict_overrun = engine.process_proposal(
            hypothesis_id="hyp_overrun_01",
            proposal_role=EnlgRole.VERIFIER,
            proposal_evidence=EnlgEvidenceClass.LOCAL_MEASURED,
            trunk_history=trunk,
            estimated_vblank_ms=3.0,  # Exceeds 2.2 ms VBlank
            elapsed_vblank_ms=0.0,
        )
        assert verdict_overrun.admissible is False
        assert verdict_overrun.status_code == "YIELDED_VBLANK_WINDOW_EXCEEDED"

    def test_13_noether_krystal_invariant_breach_rejection(self):
        """T5: Any state mutation or proposal violating VITAL_MAX_HP == 6 is rejected unconditionally."""
        engine = PleaEngine()
        trunk = [EnlgEvidenceClass.PHYSICAL_MEASURED]
        with pytest.raises(ContractViolation) as exc_info:
            engine.inertia_damper.evaluate_proposal_admissibility(
                trunk_evidence=trunk,
                proposal_evidence=EnlgEvidenceClass.LOCAL_MEASURED,
                proposal_role=EnlgRole.VERIFIER,
                is_hp_valid=False,
            )
        assert "Noether-Krystal invariant breach" in str(exc_info.value)

    def test_14_causal_inertia_damper_o1_recurrent_constant_time(self):
        """T4: Verifies O(1) recurrent Merkle-Momentum computation matches batch geometric sum exactly."""
        damper = CausalInertiaDamper(gamma=0.85)
        damper.reset_recurrent_state()

        stream = [
            EnlgEvidenceClass.SIMULATED,
            EnlgEvidenceClass.LOCAL_MEASURED,
            EnlgEvidenceClass.PHYSICAL_MEASURED,
            EnlgEvidenceClass.EXTERNAL_REPLICATED,
        ]

        # Step recurrently in O(1)
        for ev in stream:
            damper.step_recurrent(ev)

        recurrent_momentum = damper.current_momentum
        batch_momentum = damper.compute_momentum(stream)

        # Exact algebraic match
        assert abs(recurrent_momentum - batch_momentum) < 1e-6

        # Fast execution test: 1,000 steps should take under 5 milliseconds
        t0 = time.perf_counter()
        for _ in range(1000):
            damper.step_recurrent(EnlgEvidenceClass.PHYSICAL_MEASURED)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 5.0  # < 5 microseconds per step in pure Python!

    def test_15_causal_inertia_damper_cryptographic_chain_validation(self):
        """T4 + T5: Verifies cryptographic continuity of Merkle-Momentum frames."""
        damper = CausalInertiaDamper(gamma=0.85)
        damper.reset_recurrent_state()

        f1 = _make_dummy_frame(seq=1)
        damper.step_recurrent(EnlgEvidenceClass.PHYSICAL_MEASURED, frame=f1)

        # Frame 2 correctly chains predecessor_hash to f1.signature_or_digest_raw
        f2 = ZeroCopyContractFrame(
            magic=KRYSTAL_MAGIC,
            version=KRYSTAL_PROTOCOL_VERSION,
            vital_hp=6,
            contract_type=BinaryContractType.STATE,
            payload_size=32,
            flags=0,
            sequence_id=2,
            generation_id=0,
            timestamp_ns=time.time_ns(),
            capability_scope=1,
            nonce=1002,
            predecessor_hash_raw=f1.signature_or_digest_raw,
            profile_id=1,
            verdict_id=int(EnlgVerdict.MEASURED_WITH_SCOPE.value == "MEASURED_WITH_SCOPE"),
            approved_authority_mask=0xFF,
            metric1=120.0,
            metric2=1.5,
            signature_or_digest_raw=b"\x00" * 32,
        )
        sig2 = f2.compute_binary_digest()
        f2 = ZeroCopyContractFrame(
            magic=f2.magic,
            version=f2.version,
            vital_hp=f2.vital_hp,
            contract_type=f2.contract_type,
            payload_size=f2.payload_size,
            flags=f2.flags,
            sequence_id=f2.sequence_id,
            generation_id=f2.generation_id,
            timestamp_ns=f2.timestamp_ns,
            capability_scope=f2.capability_scope,
            nonce=f2.nonce,
            predecessor_hash_raw=f2.predecessor_hash_raw,
            profile_id=f2.profile_id,
            verdict_id=f2.verdict_id,
            approved_authority_mask=f2.approved_authority_mask,
            metric1=f2.metric1,
            metric2=f2.metric2,
            signature_or_digest_raw=sig2,
        )
        damper.step_recurrent(EnlgEvidenceClass.PHYSICAL_MEASURED, frame=f2)
        assert damper.current_momentum > 1.0

        # Frame 3 with corrupted predecessor_hash raises ERR_PLEA_MOMENTUM_CHAIN_BROKEN
        f3_corrupted = ZeroCopyContractFrame(
            magic=KRYSTAL_MAGIC,
            version=KRYSTAL_PROTOCOL_VERSION,
            vital_hp=6,
            contract_type=BinaryContractType.STATE,
            payload_size=32,
            flags=0,
            sequence_id=3,
            generation_id=0,
            timestamp_ns=time.time_ns(),
            capability_scope=1,
            nonce=1003,
            predecessor_hash_raw=b"\x99" * 32,  # Corrupted digest
            profile_id=1,
            verdict_id=int(EnlgVerdict.MEASURED_WITH_SCOPE.value == "MEASURED_WITH_SCOPE"),
            approved_authority_mask=0xFF,
            metric1=120.0,
            metric2=1.5,
            signature_or_digest_raw=b"\x00" * 32,
        )
        sig3 = f3_corrupted.compute_binary_digest()
        f3_corrupted = ZeroCopyContractFrame(
            magic=f3_corrupted.magic,
            version=f3_corrupted.version,
            vital_hp=f3_corrupted.vital_hp,
            contract_type=f3_corrupted.contract_type,
            payload_size=f3_corrupted.payload_size,
            flags=f3_corrupted.flags,
            sequence_id=f3_corrupted.sequence_id,
            generation_id=f3_corrupted.generation_id,
            timestamp_ns=f3_corrupted.timestamp_ns,
            capability_scope=f3_corrupted.capability_scope,
            nonce=f3_corrupted.nonce,
            predecessor_hash_raw=f3_corrupted.predecessor_hash_raw,
            profile_id=f3_corrupted.profile_id,
            verdict_id=f3_corrupted.verdict_id,
            approved_authority_mask=f3_corrupted.approved_authority_mask,
            metric1=f3_corrupted.metric1,
            metric2=f3_corrupted.metric2,
            signature_or_digest_raw=sig3,
        )

        with pytest.raises(ContractViolation) as exc_info:
            damper.step_recurrent(EnlgEvidenceClass.PHYSICAL_MEASURED, frame=f3_corrupted)
        assert "ERR_PLEA_MOMENTUM_CHAIN_BROKEN" in str(exc_info.value.violation_code)
