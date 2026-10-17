"""
Krystal-Stack Sovereign Principle Pack (KSPP v1) Blueprint Specification Test Suite
===================================================================================
Milestone: V-KSPP-BLUEPRINT-SPEC-01
References:
  - krystal_kernel/constants.py
  - krystal_kernel/zero_copy_contracts.py
  - krystal_kernel/zero_copy_ring.py
  - krystal_kernel/epistemic_enlg_compiler.py
  - krystal_kernel/plea_engine.py
  - krystal_kernel/causal_research_orchestrator.py
  - include/krystal_nexus_api.h

Validates the exact Blueprint Specification across all 6 core modules:
  0. Global constitutional constants and types.
  1. 128B Memory Contract (CL0/CL1 layout, 96B HMAC-SHA256, canonicalize_float32).
  2. Lock-free SPSC Ring Buffer (Acquire/Release publication, false-sharing elimination).
  3. Epistemic Compiler & ArtifactGate (EnlgCompiledBlock, Anti-Promotion Barriers).
  4. PLEA Engine (Calorimeter parasite decoupling, Homotopic bifurcation, VBlank shadow).
  5. Causal Research Orchestrator (Physical-first actuation state machine, IEEE-754 hex).
"""

import math
import struct
import hmac
import hashlib
import dataclasses
import pytest

from krystal_kernel.constants import (
    VITAL_MAX_HP,
    FRAME_MAGIC,
    FRAME_VERSION,
    FRAME_SIZE_BYTES,
    CACHE_LINE_BYTES,
    AUTHENTICATED_PREFIX_BYTES,
    UMA_SATURATION_THRESHOLD,
    HYSTERESIS_DWELL_FRAMES,
    HYSTERESIS_DOWNSCALE_RATIO,
    HYSTERESIS_UPSCALE_RATIO,
    ARRHENIUS_MAX_TEMP_C,
    CapabilityScope,
    FrameFlags,
    ContractViolation,
)
from krystal_kernel.zero_copy_contracts import (
    FRAME_STRUCT_FORMAT,
    canonicalize_float32,
    ZeroCopyContractFrame,
    BinaryContractType,
)
from krystal_kernel.zero_copy_ring import (
    RING_HEADER_FORMAT,
    ZeroCopyContractRing,
)
from krystal_kernel.epistemic_enlg_compiler import (
    EnlgRole,
    EnlgEvidenceClass,
    EnlgVerdict,
    EnlgCompiledBlock,
    EnlgEpistemicCompiler,
    ArtifactGate,
)
from krystal_kernel.plea_engine import (
    HypothesisMeter,
    EpistemicCalorimeter,
    BifurcatedConflictRing,
    VBlankExperimentInjector,
    CausalInertiaDamper,
    PleaEngine,
)
from krystal_kernel.causal_research_orchestrator import (
    ExperimentState,
    MetricObjective,
    canonical_float_repr,
    is_objective_satisfied,
    CausalTransactionReceipt,
    ClosedLoopCausalOrchestrator,
)


class TestVKsppV1BlueprintSpecs:
    """Rigorous verification suite for KSPP v1 Blueprint Specifications."""

    # -------------------------------------------------------------------------
    # 0. Global Constitutional Constants and Types
    # -------------------------------------------------------------------------
    def test_00_constitutional_constants_and_bitflags(self):
        """Validates all global constants, flags, and exception types."""
        assert VITAL_MAX_HP == 6
        assert FRAME_MAGIC == 0x4B525953
        assert FRAME_VERSION == 2
        assert FRAME_SIZE_BYTES == 128
        assert CACHE_LINE_BYTES == 64
        assert AUTHENTICATED_PREFIX_BYTES == 96

        assert UMA_SATURATION_THRESHOLD == 0.78
        assert HYSTERESIS_DWELL_FRAMES == 12
        assert HYSTERESIS_DOWNSCALE_RATIO == 0.90
        assert HYSTERESIS_UPSCALE_RATIO == 0.70
        assert ARRHENIUS_MAX_TEMP_C == 75.0

        # Bitflag invariants
        assert CapabilityScope.CAN_OBSERVE == 1 << 0
        assert CapabilityScope.CAN_PROPOSE == 1 << 1
        assert CapabilityScope.CAN_VERIFY == 1 << 2
        assert CapabilityScope.CAN_AUTHORIZE == 1 << 3
        assert CapabilityScope.CAN_EXECUTE == 1 << 4

        assert FrameFlags.VBLANK_ACTIVE == 1 << 0
        assert FrameFlags.UMA_THROTTLE_ACTIVE == 1 << 1
        assert FrameFlags.HYSTERESIS_HELD == 1 << 2
        assert FrameFlags.BIFURCATED_SUPERPOSITION == 1 << 3
        assert FrameFlags.PARASITIC_DECOUPLED == 1 << 4
        assert FrameFlags.FULL_HMAC_VERIFIED == 1 << 5

        # ContractViolation dual-inheritance
        cv = ContractViolation("Test violation", "ERR_TEST")
        assert isinstance(cv, RuntimeError)
        assert isinstance(cv, ValueError)

    # -------------------------------------------------------------------------
    # 1. 128-Byte Memory Contract
    # -------------------------------------------------------------------------
    def test_01_zero_copy_frame_blueprint_packing_and_hmac(self):
        """Validates 128B binary layout, canonical float32, and HMAC-SHA256 authentication."""
        assert struct.calcsize(FRAME_STRUCT_FORMAT) == 128
        session_key = b"session_secret_key_128b_krystal"

        # Canonicalize float32: reject NaN/Inf, normalize -0.0
        assert canonicalize_float32(-0.0) == 0.0
        assert canonicalize_float32(42.5) == 42.5
        with pytest.raises(ContractViolation):
            canonicalize_float32(float("nan"))
        with pytest.raises(ContractViolation):
            canonicalize_float32(float("inf"))

        # Construct frame per blueprint
        frame = ZeroCopyContractFrame(
            magic=FRAME_MAGIC,
            version=FRAME_VERSION,
            vital_hp=VITAL_MAX_HP,
            contract_type=1,
            payload_size=16,
            flags=int(FrameFlags.VBLANK_ACTIVE),
            sequence=10,
            generation=1,
            cap_scope=int(CapabilityScope.CAN_PROPOSE),
            timestamp_ns=1_000_000_000,
            nonce=777,
            predecessor_hash=b"\x05" * 32,
            profile_id=1,
            verdict_id=1,
            mask=0xFF,
            gpu_render_ms=6.5,
            drs_scale_factor=1.0,
        )

        # 1. Pack authenticated: computes 96B prefix HMAC into bytes 96..127
        raw_128 = frame.pack_authenticated(session_key)
        assert len(raw_128) == 128
        assert len(raw_128) % 64 == 0

        # Verify HMAC match
        prefix_96 = raw_128[:96]
        expected_hmac = hmac.new(session_key, prefix_96, hashlib.sha256).digest()
        assert raw_128[96:] == expected_hmac

        # 2. Unpack and verify
        unpacked = ZeroCopyContractFrame.unpack_and_verify(raw_128, session_key)
        assert unpacked.vital_hp == 6
        assert unpacked.sequence == 10
        assert unpacked.gpu_render_ms == 6.5
        assert unpacked.flags == int(FrameFlags.VBLANK_ACTIVE)

        # 3. Torn frame / tampering rejection
        tampered = bytearray(raw_128)
        tampered[16] ^= 0xFF  # Flip bit in flags
        with pytest.raises(ContractViolation, match="HMAC mismatch or CL0/CL1 torn frame"):
            ZeroCopyContractFrame.unpack_and_verify(bytes(tampered), session_key)

        # 4. Vital HP breach rejection
        tampered_hp = bytearray(raw_128)
        tampered_hp[8] = 5  # Corrupt vital_hp to 5
        # Recompute HMAC to test unpack_and_verify invariant check
        tampered_hp[96:] = hmac.new(session_key, bytes(tampered_hp[:96]), hashlib.sha256).digest()
        with pytest.raises(ContractViolation, match="VITAL_MAX_HP == 6"):
            ZeroCopyContractFrame.unpack_and_verify(bytes(tampered_hp), session_key)

    # -------------------------------------------------------------------------
    # 2. Lock-Free SPSC Ring Buffer
    # -------------------------------------------------------------------------
    def test_02_spsc_ring_buffer_blueprint_publish_consume(self):
        """Validates SPSC ring buffer with CL0 producer, CL1 consumer, and Acquire/Release."""
        assert struct.calcsize(RING_HEADER_FORMAT) == 128
        session_key = b"spsc_key_32_bytes_long_secret_!"

        ring = ZeroCopyContractRing(capacity_power_of_two=16, session_key=session_key)
        assert ring.capacity == 16
        assert ring.read_head_acquire() == 0
        assert ring.read_tail_acquire() == 0

        # Slot alignment
        assert ring.slot_offset(0) == 128
        assert ring.slot_offset(1) == 256
        assert ring.slot_offset(0) % 64 == 0
        assert ring.slot_offset(0) % 128 == 0

        # Build valid frame
        frame = ZeroCopyContractFrame(
            magic=FRAME_MAGIC,
            version=FRAME_VERSION,
            vital_hp=VITAL_MAX_HP,
            contract_type=1,
            sequence=1,
            generation=0,
            gpu_render_ms=5.0,
        )

        # Publish frame
        success = ring.publish(frame)
        assert success is True
        assert ring.read_head_acquire() == 1
        assert ring.read_tail_acquire() == 0

        # Consume frame
        consumed = ring.consume()
        assert consumed is not None
        assert consumed.vital_hp == 6
        assert consumed.sequence == 1
        assert consumed.gpu_render_ms == 5.0
        assert ring.read_head_acquire() == 1
        assert ring.read_tail_acquire() == 1

        # Buffer now empty
        assert ring.consume() is None

        # Fill buffer to capacity (16 frames)
        for i in range(16):
            f = ZeroCopyContractFrame(sequence=100 + i, vital_hp=6)
            assert ring.publish(f) is True

        # 17th publish must fail with drop count increment
        overflow_frame = ZeroCopyContractFrame(sequence=999, vital_hp=6)
        assert ring.publish(overflow_frame) is False
        assert ring.drop_count() == 1

    # -------------------------------------------------------------------------
    # 3. Epistemic Compiler and ArtifactGate
    # -------------------------------------------------------------------------
    def test_03_epistemic_compiler_and_artifact_gate_blueprint(self):
        """Validates EnlgCompiledBlock, EnlgEpistemicCompiler validation rules, and ArtifactGate release."""
        # Valid compiled block
        block = EnlgCompiledBlock(
            claim="LPDDR5 memory sustains 48 GB/s bandwidth under 120Hz VSync",
            scope="Hardware memory subsystem within defined invariant bounds",
            role=EnlgRole.VERIFIER,
            evidence={EnlgEvidenceClass.LOCAL_MEASURED, EnlgEvidenceClass.PHYSICAL_MEASURED},
            falsifier="Bandwidth drops below 40 GB/s across 100 frames",
            null_m0_no_effect="Bandwidth unchanged",
            null_m1_noise="Variance within noise",
            null_m2_confounder="Thermal clock throttling",
            verdict=EnlgVerdict.MEASURED_WITH_SCOPE,
            has_contradiction=False,
            vital_hp=VITAL_MAX_HP,
        )
        assert EnlgEpistemicCompiler.validate_block(block) is True

        # Rule [10]: Contradiction detected without CONFLICT verdict
        contradictory_block = dataclasses.replace(
            block,
            has_contradiction=True,
            verdict=EnlgVerdict.MEASURED_WITH_SCOPE,
        )
        with pytest.raises(ContractViolation, match="Contradiction detected: verdict must be CONFLICT"):
            EnlgEpistemicCompiler.validate_block(contradictory_block)

        # Rule [3]: Proposer self-certification of VERIFIED
        proposer_block = dataclasses.replace(
            block,
            role=EnlgRole.PROPOSER,
            verdict=EnlgVerdict.VERIFIED_WITH_SCOPE,
        )
        with pytest.raises(ContractViolation, match="Authority Separation: PROPOSER cannot self-certify VERIFIED"):
            EnlgEpistemicCompiler.validate_block(proposer_block)

        # Rule [1] & [4]: Simulation cannot promote to Measurement
        sim_block = dataclasses.replace(
            block,
            evidence={EnlgEvidenceClass.SIMULATED},
            verdict=EnlgVerdict.MEASURED_WITH_SCOPE,
        )
        with pytest.raises(ContractViolation, match="Simulation -/-> Measurement"):
            EnlgEpistemicCompiler.validate_block(sim_block)

        # ArtifactGate blueprint operations
        gate = ArtifactGate()
        gate.deposit("art_block_01", block)
        assert gate.quarantined_count >= 1

        # Proposer cannot authorize release (Delta S = 0)
        with pytest.raises(ContractViolation, match="PROPOSER cannot authorize ArtifactGate release"):
            gate.release("art_block_01", evaluator_role=EnlgRole.PROPOSER, receipt_verified=True)

        # Unverified receipt leaves artifact in quarantine
        released_unverified = gate.release("art_block_01", evaluator_role=EnlgRole.VERIFIER, receipt_verified=False)
        assert released_unverified is False
        assert "art_block_01" in gate.quarantine_store

        # Verified receipt releases artifact
        released_ok = gate.release("art_block_01", evaluator_role=EnlgRole.VERIFIER, receipt_verified=True)
        assert released_ok is True
        assert "art_block_01" in gate.released_store

    # -------------------------------------------------------------------------
    # 4. PLEA Engine
    # -------------------------------------------------------------------------
    def test_04_plea_engine_blueprint_components(self):
        """Validates EpistemicCalorimeter, BifurcatedConflictRing, VBlankInjector, CausalDamper, and PleaEngine."""
        # 1. EpistemicCalorimeter: parasite decoupling
        cal = EpistemicCalorimeter(max_joules=5.0, max_uma_mb=100.0)
        m1 = cal.record_work("hyp_prod", delta_joules=3.0, delta_mb=50.0, promoted=True)
        assert m1.is_decoupled_parasite is False
        assert m1.entropy_reduced is True

        m2 = cal.record_work("hyp_waste", delta_joules=3.0, delta_mb=50.0, promoted=False)
        assert m2.is_decoupled_parasite is False
        # Burning more energy without promotion -> decouples as thermal parasite!
        m2_waste = cal.record_work("hyp_waste", delta_joules=3.0, delta_mb=60.0, promoted=False)
        assert m2_waste.is_decoupled_parasite is True

        with pytest.raises(ContractViolation, match="already decoupled as thermal parasite"):
            cal.record_work("hyp_waste", delta_joules=1.0, delta_mb=1.0, promoted=False)

        # 2. BifurcatedConflictRing: Homotopic bifurcation & collapse
        ring = BifurcatedConflictRing()
        with pytest.raises(ContractViolation, match="Homotopic Conflict Law"):
            ring.attempt_convex_averaging(10.0, 20.0)

        f_alpha = ZeroCopyContractFrame(sequence=1, vital_hp=6)
        f_beta = ZeroCopyContractFrame(sequence=1, vital_hp=6)
        parent_hash = b"\x22" * 32

        b_alpha, b_beta = ring.fork_superposition(parent_hash, f_alpha, f_beta)
        assert b_alpha.generation == 1
        assert b_beta.generation == 2
        assert (b_alpha.flags & int(FrameFlags.BIFURCATED_SUPERPOSITION)) != 0
        assert (b_beta.flags & int(FrameFlags.BIFURCATED_SUPERPOSITION)) != 0

        winner = ring.collapse_with_cdt_receipt(winning_generation=2, cdt_receipt_valid=True)
        assert winner.generation == 2
        assert (winner.flags & int(FrameFlags.BIFURCATED_SUPERPOSITION)) == 0

        # 3. VBlankExperimentInjector: shadow phase gating
        inj = VBlankExperimentInjector(vblank_budget_ms=2.20, jitter_guard_ms=0.05)
        task_executed = [False]
        def dummy_task():
            task_executed[0] = True
            return True

        # Scanout active -> deferred
        ok1, reason1 = inj.execute_in_shadow(scanout_active=True, estimated_duration_ms=1.0, task=dummy_task)
        assert ok1 is False
        assert reason1 == "DEFERRED_SCANOUT_ACTIVE"

        # Overrun budget -> yielded
        ok2, reason2 = inj.execute_in_shadow(scanout_active=False, estimated_duration_ms=2.5, task=dummy_task)
        assert ok2 is False
        assert reason2 == "YIELDED_BEFORE_DEADLINE"

        # Safe execution
        ok3, reason3 = inj.execute_in_shadow(scanout_active=False, estimated_duration_ms=1.5, task=dummy_task)
        assert ok3 is True
        assert reason3 == "EXECUTED_IN_VBLANK_SHADOW"
        assert task_executed[0] is True

        # 4. CausalInertiaDamper: O(1) Merkle-Momentum
        damper = CausalInertiaDamper(gamma=0.85)
        f_ver = ZeroCopyContractFrame(predecessor_hash=b"\x00" * 32, vital_hp=6)
        f_ver.frame_hmac = b"\x33" * 32

        mom1 = damper.push_verified_frame(f_ver, EnlgEvidenceClass.PHYSICAL_MEASURED)
        assert mom1 == 2.5
        assert damper.can_override_trunk(proposal_weight=2.0) is True
        assert damper.can_override_trunk(proposal_weight=0.5) is False

        # 5. PleaEngine: Action Integral
        engine = PleaEngine(alpha=1.0, beta=2.0, gamma_weight=0.5)
        s_val = engine.evaluate_action_integral(uma_cost=10.0, topo_conflict=0.0, causal_inertia=2.0)
        # S = 1.0 * 10.0 + 2.0 * 0.0 - 0.5 * 2.0 = 9.0
        assert abs(s_val - 9.0) < 1e-6

    # -------------------------------------------------------------------------
    # 5. Causal Research Orchestrator
    # -------------------------------------------------------------------------
    def test_05_causal_research_orchestrator_blueprint(self):
        """Validates 2-phase physical actuation state machine, IEEE-754 hex, and rollback."""
        secret_key = b"causal_orchestrator_key_9999"
        orchestrator = ClosedLoopCausalOrchestrator(secret_key)

        # Canonical IEEE-754 hex bijection
        h1 = canonical_float_repr(0.0)
        h2 = canonical_float_repr(-0.0)
        assert h1 == h2 == float(0.0).hex()
        assert canonical_float_repr(12.5) == float(12.5).hex()

        # Objective satisfaction
        assert is_objective_satisfied(10.0, 8.0, MetricObjective.MINIMIZE) is True
        assert is_objective_satisfied(10.0, 12.0, MetricObjective.MINIMIZE) is False
        assert is_objective_satisfied(10.0, 12.0, MetricObjective.MAXIMIZE) is True

        # Scenario 1: Successful optimization (latency reduced from 14.0 ms to 11.0 ms)
        system_val = [14.0]
        applied = [False]
        rolled_back = [False]

        def sample():
            return system_val[0]

        def actuate():
            applied[0] = True
            system_val[0] = 11.0

        def rollback():
            rolled_back[0] = True
            system_val[0] = 14.0

        rcpt_win = orchestrator.execute_grounded_transaction(
            exp_id="exp_boost_pl1",
            objective=MetricObjective.MINIMIZE,
            sampler=sample,
            actuate_fn=actuate,
            rollback_fn=rollback,
        )

        assert rcpt_win.committed is True
        assert rcpt_win.state == ExperimentState.KNOWLEDGE_ELIGIBLE
        assert rcpt_win.vital_hp == 6
        assert applied[0] is True
        assert rolled_back[0] is False
        assert rcpt_win.verify_signature(secret_key) is True

        # Scenario 2: Regressive optimization (latency increases from 11.0 ms to 16.0 ms)
        applied_fail = [False]
        rolled_back_fail = [False]

        def actuate_bad():
            applied_fail[0] = True
            system_val[0] = 16.0

        def rollback_bad():
            rolled_back_fail[0] = True
            system_val[0] = 11.0

        rcpt_fail = orchestrator.execute_grounded_transaction(
            exp_id="exp_bad_boost",
            objective=MetricObjective.MINIMIZE,
            sampler=sample,
            actuate_fn=actuate_bad,
            rollback_fn=rollback_bad,
        )

        assert rcpt_fail.committed is False
        assert rcpt_fail.state == ExperimentState.RECEIPT_FINALIZED
        assert applied_fail[0] is True
        assert rolled_back_fail[0] is True
        # System state rolled back to baseline (Delta S == 0)
        assert system_val[0] == 11.0
        assert rcpt_fail.verify_signature(secret_key) is True
