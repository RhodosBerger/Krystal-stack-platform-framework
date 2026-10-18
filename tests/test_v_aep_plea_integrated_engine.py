"""
TEST SUITE: V-AEP-PLEA-INTEGRATED-ENGINE-01
=============================================================================
Milestone: V-AEP-PLEA-01
Verification of Sovereign Phase-Locked Epistemic Execution Engine (AEP-PLEA):
  1. Constitutional Invariant: VITAL_MAX_HP == 6 across all state transitions
  2. Axiom 1 (T1/T3): Epistemic Calorimeter (thermal parasite decoupling & work accounting)
  3. Axiom 2 (T2): Homotopic Bifurcated Conflict preservation (non-averaging of branches)
  4. Axiom 3 (T3): Chronometric Shadow Slicing (VBlank shadow execution & overrun discard)
  5. Axiom 4 (T4/T5): Causal Inertia Damper (Merkle-Momentum proposal gating)
  6. Axiom 5 (P10): Zero-Dirty Mutation (Delta S == 0 on any rejection)
"""

import time
import pytest

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.contracts import ContractHeader, ReceiptContract
from krystal_kernel.zero_copy_contracts import (
    BinaryContractType,
    KRYSTAL_MAGIC,
    KRYSTAL_PROTOCOL_VERSION,
    ZeroCopyContractFrame,
)
from krystal_kernel.aep_engine import (
    AepVerdict,
    ExecutionLane,
    VitalHpInvariantRule,
    WorkloadType,
)
from krystal_kernel.plea_engine import (
    VBLANK_WINDOW_MS,
    EpistemicCalorimeter,
    BifurcatedConflictRing,
    VBlankExperimentInjector,
    CausalInertiaDamper,
    FLAG_PLEA_CONFLICT_BIFURCATION,
)
from krystal_kernel.epistemic_enlg_compiler import (
    EnlgRole,
    EnlgEvidenceClass,
    EnlgVerdict,
)
from krystal_kernel.aep_plea_integrated_engine import (
    AepPleaIntegratedEngine,
    PleaExecutionEnvelope,
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


class TestAepPleaIntegratedEngine:
    """Verification suite for Milestone V-AEP-PLEA-01."""

    def test_01_vital_max_hp_invariant_preserved(self):
        """Invariant: VITAL_MAX_HP == 6 must hold in engine initialization and execution."""
        assert VITAL_MAX_HP == 6
        engine = AepPleaIntegratedEngine(invariants=[VitalHpInvariantRule()])
        assert engine is not None

        # Negative test: invalid HP in envelope
        with pytest.raises(ContractViolation):
            PleaExecutionEnvelope(vital_max_hp=5)

    def test_02_successful_committed_transition_emits_plea_receipt(self):
        """Verifies standard valid transaction commits candidate state with PLEA telemetry."""
        engine = AepPleaIntegratedEngine(invariants=[VitalHpInvariantRule()])
        initial_state = {"actor_id": "hero_01", "vital_hp": 6, "gold": 100}

        def transition_earn_gold(s, payload):
            st = dict(s)
            st["gold"] += payload.get("amount", 0)
            return st

        envelope = PleaExecutionEnvelope(
            proposal_evidence=EnlgEvidenceClass.PHYSICAL_MEASURED,
            proposal_role=EnlgRole.EXECUTOR,
            expected_entropy_reduction=2.0,
            uma_bytes_estimated=4096,
        )

        res = engine.execute_plea_transition(
            runtime_id="rt-game-001",
            workload=WorkloadType.GAME,
            action="QUEST_REWARD",
            current_state=initial_state,
            transition_fn=transition_earn_gold,
            action_payload={"amount": 50},
            envelope=envelope,
        )

        assert res.committed is True
        assert res.state["gold"] == 150
        assert res.state["vital_hp"] == 6
        assert res.receipt.verdict == AepVerdict.COMMITTED
        assert res.receipt.plea_telemetry is not None
        assert res.receipt.plea_telemetry["vital_max_hp"] == 6
        assert "action_integral" in res.receipt.plea_telemetry
        assert res.receipt.plea_telemetry["causal_momentum"] > 0.0

    def test_03_zero_dirty_mutation_on_invariant_violation(self):
        """Constitutional: S' violating VITAL_MAX_HP is discarded, state remains Delta S == 0."""
        engine = AepPleaIntegratedEngine(invariants=[VitalHpInvariantRule()])
        initial_state = {"actor_id": "hero_01", "vital_hp": 6, "mana": 50}

        def invalid_transition(s, payload):
            st = dict(s)
            st["vital_hp"] = 999  # Invariant violation: HP > 6
            return st

        res = engine.execute_plea_transition(
            runtime_id="rt-game-002",
            workload=WorkloadType.GAME,
            action="ILLEGAL_BUFF",
            current_state=initial_state,
            transition_fn=invalid_transition,
            action_payload={},
        )

        assert res.committed is False
        assert res.state == initial_state  # Delta S == 0!
        assert res.receipt.verdict == AepVerdict.DISCARDED_INVARIANT_VIOLATION
        assert res.receipt.state_before == res.receipt.state_after

    def test_04_vblank_shadow_phase_execution_and_overrun_discard(self):
        """Chronometrics: VBlank shadow execution succeeds within budget, discards on overrun."""
        engine = AepPleaIntegratedEngine()
        initial_state = {"tick": 0, "vital_hp": 6}

        # Sub-test A: Quick transition within VBlank (0.1 ms < 2.20 ms)
        def fast_transition(s, payload):
            st = dict(s)
            st["tick"] += 1
            return st

        env_fast = PleaExecutionEnvelope(require_vblank_shadow=True, vblank_budget_ms=2.20)
        res_fast = engine.execute_plea_transition(
            runtime_id="rt-render-001",
            workload=WorkloadType.RENDER,
            action="FRAME_SYNC",
            current_state=initial_state,
            transition_fn=fast_transition,
            action_payload={},
            envelope=env_fast,
        )
        assert res_fast.committed is True
        assert res_fast.receipt.verdict == AepVerdict.COMMITTED
        assert res_fast.receipt.plea_telemetry["vblank_locked"] is True

        # Sub-test B: Slow transition violating VBlank budget (mocked through custom injector)
        class MockOverrunInjector(VBlankExperimentInjector):
            def execute_in_shadow(self, *args, **kwargs):
                return False, None, 999.0  # Overrun!

        engine_overrun = AepPleaIntegratedEngine(vblank_injector=MockOverrunInjector())
        res_slow = engine_overrun.execute_plea_transition(
            runtime_id="rt-render-002",
            workload=WorkloadType.RENDER,
            action="SLOW_FRAME",
            current_state=initial_state,
            transition_fn=fast_transition,
            action_payload={},
            envelope=env_fast,
        )
        assert res_slow.committed is False
        assert res_slow.state == initial_state  # Delta S == 0
        assert res_slow.receipt.verdict == AepVerdict.DISCARDED_VBLANK_OVERRUN

    def test_05_calorimeter_decouples_thermal_parasite(self):
        """Thermodynamics: Parasitic actions consuming energy without entropy reduction are decoupled."""
        # Setup calorimeter with a strict 0.05 Joule ceiling
        calorimeter = EpistemicCalorimeter(max_energy_joules=0.05)
        engine = AepPleaIntegratedEngine(calorimeter=calorimeter)
        initial_state = {"val": 10, "vital_hp": 6}

        def parasite_transition(s, payload):
            # Consumes time and power
            time.sleep(0.005)  # 5 ms at 28W = 0.14 Joules > 0.05 Joules
            return dict(s)

        env = PleaExecutionEnvelope(
            expected_entropy_reduction=0.0,  # Zero entropy reduction!
            uma_bytes_estimated=1024,
        )

        res = engine.execute_plea_transition(
            runtime_id="rt-parasite-001",
            workload=WorkloadType.SYSTEM,
            action="SPIN_LOCK_POLL",
            current_state=initial_state,
            transition_fn=parasite_transition,
            action_payload={},
            envelope=env,
        )

        assert res.committed is False
        assert res.state == initial_state  # Delta S == 0
        assert res.receipt.verdict == AepVerdict.DISCARDED_THERMAL_PARASITE

    def test_06_causal_inertia_damper_rejects_weak_speculative_action(self):
        """Inertia: High trunk momentum rejects weak speculative (UNKNOWN) proposals."""
        damper = CausalInertiaDamper()
        # Seed high causal trunk momentum with physical verified evidence
        damper.compute_momentum([
            EnlgEvidenceClass.PHYSICAL_MEASURED,
            EnlgEvidenceClass.PHYSICAL_MEASURED,
            EnlgEvidenceClass.EXTERNAL_REPLICATED,
        ])
        assert damper.current_momentum > 1.5

        engine = AepPleaIntegratedEngine(inertia_damper=damper)
        initial_state = {"status": "SETTLED", "vital_hp": 6}

        def speculative_transition(s, payload):
            st = dict(s)
            st["status"] = "SPECULATIVE_OVERRIDE"
            return st

        env = PleaExecutionEnvelope(
            proposal_evidence=EnlgEvidenceClass.UNKNOWN,  # Weak speculative evidence
            proposal_role=EnlgRole.PROPOSER,
        )

        res = engine.execute_plea_transition(
            runtime_id="rt-spec-001",
            workload=WorkloadType.SYSTEM,
            action="SPECULATIVE_FLIP",
            current_state=initial_state,
            transition_fn=speculative_transition,
            action_payload={},
            envelope=env,
        )

        assert res.committed is False
        assert res.state == initial_state  # Delta S == 0
        assert res.receipt.verdict == AepVerdict.DISCARDED_CAUSAL_DAMPENED
        assert "PROPOSAL_DAMPENED" in res.receipt.plea_telemetry["rejection_reason"]

    def test_07_homotopic_bifurcated_branch_preservation_and_collapse(self):
        """Homotopy: Conflicting candidate states are preserved in superposed branches without averaging."""
        engine = AepPleaIntegratedEngine()
        parent = _make_dummy_frame(seq=42)
        parent_digest = parent.compute_binary_digest()

        # Fork into superposed branches
        branch_a, branch_b = engine.fork_bifurcated_candidates(
            conflict_id="conf_aep_01",
            parent_frame=parent,
            claim_a="Throughput = 120 FPS",
            evidence_a=EnlgEvidenceClass.LOCAL_MEASURED,
            metric_a=120.0,
            claim_b="Throughput = 45 FPS",
            evidence_b=EnlgEvidenceClass.PHYSICAL_MEASURED,
            metric_b=45.0,
        )

        assert branch_a.generation_id == 1
        assert branch_b.generation_id == 2
        assert branch_a.fork_predecessor_hash == parent_digest
        assert branch_b.fork_predecessor_hash == parent_digest
        assert (branch_a.frame.flags & FLAG_PLEA_CONFLICT_BIFURCATION) != 0
        assert (branch_b.frame.flags & FLAG_PLEA_CONFLICT_BIFURCATION) != 0

        # Collapse with physical CDT receipt
        header = ContractHeader("rcpt_cdt_01", 1, 1, 0x10, 1, 1000, "0" * 64)
        receipt = ReceiptContract(
            header=header,
            execution_digest="exec_cdt_01",
            applied_profile="balanced",
            actuation_duration_ns=500_000,
            was_degraded_fallback=False,
        )
        winner_frame = engine.collapse_bifurcated_conflict(
            conflict_id="conf_aep_01",
            winning_branch_generation=1,
            cdt_receipt=receipt,
        )
        assert winner_frame.vital_hp == 6
        assert (winner_frame.flags & FLAG_PLEA_CONFLICT_BIFURCATION) == 0

    def test_08_bus_pressure_backpressure_rejection(self):
        """L7: Normalized bus pressure exceeding threshold triggers rate-limiting backpressure."""
        engine = AepPleaIntegratedEngine(max_pressure_threshold=0.80)
        initial_state = {"data": "ready", "vital_hp": 6}

        res = engine.execute_plea_transition(
            runtime_id="rt-bus-001",
            workload=WorkloadType.INFERENCE,
            action="BURST_INFERENCE",
            current_state=initial_state,
            transition_fn=lambda s, p: dict(s),
            action_payload={},
            current_bus_pressure=0.88,  # Exceeds 0.80 threshold
        )

        assert res.committed is False
        assert res.state == initial_state  # Delta S == 0
        assert res.receipt.verdict == AepVerdict.DISCARDED_BACKPRESSURE_REJECTED
