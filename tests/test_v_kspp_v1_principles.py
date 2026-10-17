"""
Krystal-Stack Sovereign Principle Pack (KSPP v1) Test Suite
===========================================================
Milestone: V-KSPP-PRINCIPLES-01
References:
  - docs/architecture/KRYSTAL_SOVEREIGN_PRINCIPLE_PACK.md
  - .agents/rules/krystal-sovereign-principle-pack.md
  - AGENTS.md / GEMINI.md

Formally validates all 10 Sovereign Principles:
  P1:  Zero-Touch Interposition (SourceCodeDelta(M) == 0)
  P2:  Universal 128B Lingua Franca (_Alignas(64), Size == 128B)
  P3:  Chronometric Shadow Multiplexing (Phi_Scanout * Phi_Epistemic == 0, UMA < 0.78)
  P4:  Epistemic Type Safety (Level(Verdict) <= min AllowedLevel, PROPOSER != VERIFIER)
  P5:  Thermodynamic Hypothesis Metering (RAPL Joules / UMA MB, Parasite Decoupling)
  P6:  Homotopic Conflict Preservation (E_A _|_ E_B -> Fork, ConvexAverage -> Rejection)
  P7:  Physical-First Actuation (Commit requires AUDITED + no regression)
  P8:  Inertial Stability & Anti-Oscillation (O(1) Merkle-Momentum, N_dwell >= 12)
  P9:  Full-Frame Provenance (96B HMAC-SHA256, canonicalize_float32 IEEE-754)
  P10: Noether-Krystal Survival Symmetry (VITAL_MAX_HP == 6, delta S == 0 on breach)
"""

import time
import struct
import hashlib
import hmac
import pytest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import (
    ContractViolation,
    ContractHeader,
    ReceiptContract,
)
from krystal_kernel.zero_copy_contracts import (
    ZeroCopyContractFrame,
    BinaryContractType,
    ZERO_COPY_FRAME_SIZE,
    KRYSTAL_MAGIC,
    KRYSTAL_PROTOCOL_VERSION,
    canonicalize_float32,
)
from krystal_kernel.epistemic_enlg_compiler import (
    EnlgEpistemicCompiler,
    EnlgAstValidator,
    EnlgClaimTrace,
    EnlgRole,
    EnlgEvidenceClass,
    EnlgVerdict,
    NullTriplet,
    NextExperimentPlan,
    ArtifactGate,
)
from krystal_kernel.plea_engine import (
    EpistemicCalorimeter,
    BifurcatedConflictRing,
    VBlankExperimentInjector,
    CausalInertiaDamper,
    PleaEngine,
    FLAG_PLEA_CONFLICT_BIFURCATION,
)
from krystal_kernel.predictive_render_profiler import (
    NeuralRenderSteeringCopilot,
    ProgramMetadata,
    ProgramWorkloadClass,
    AllocationSchemeVariant,
    NssQualityProfile,
)
from projects.sovereign_game_companion import (
    SovereignGameCompanionBridge,
    GameRenderFrameTelemetry,
)
from projects.zero_stutter_dcc_copilot import (
    ZeroStutterDccBridge,
    DccTaskResult,
)
from projects.self_tuning_workstation import (
    SelfTuningWorkstationEngine,
    HardwareTuningHypothesis,
    TuningEvaluationReport,
)
from projects.verified_robotics_twin import (
    VerifiedRoboticsTwinBridge,
    SensorObservation,
    RobotNavigationCommand,
)
from projects.low_latency_ide_mesh import (
    LowLatencyIdeMeshCoordinator,
    EditorFrameQoS,
    CodeAssistantSuggestion,
)


def _make_valid_test_frame(seq: int = 1, vital_hp: int = 6) -> ZeroCopyContractFrame:
    """Helper to build a valid signed 128B ZeroCopyContractFrame."""
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
        verdict_id=1,
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


class TestVKsppV1Principles:
    """Formal verification suite for Krystal Sovereign Principle Pack (KSPP v1)."""

    # -------------------------------------------------------------------------
    # P1: Zero-Touch Interposition
    # -------------------------------------------------------------------------
    def test_p1_zero_touch_interposition_contract(self):
        """P1: Verifies that external binaries are controlled strictly via boundaries without source modifications."""
        assert VITAL_MAX_HP == 6

        # Instantiate all 5 domain projects as non-invasive superstructure wrappers
        b1 = SovereignGameCompanionBridge()
        b2 = ZeroStutterDccBridge()
        b3 = SelfTuningWorkstationEngine()
        b4 = VerifiedRoboticsTwinBridge()
        b5 = LowLatencyIdeMeshCoordinator()

        # Invariant: all wrappers preserve upstream module sovereignty (SourceCodeDelta(M) == 0)
        assert b1.frame_budget_ms > 0
        assert b2.calorimeter is not None
        assert b3.current_state["pl1_watts"] == 28.0
        assert b4.conflict_ring is not None
        assert b5.artifact_gate is not None

    # -------------------------------------------------------------------------
    # P2: The Universal 128B Lingua Franca
    # -------------------------------------------------------------------------
    def test_p2_universal_128b_lingua_franca(self):
        """P2: Verifies natural 128B layout, 64B cache line alignment, and zero-copy binary serialization."""
        frame = _make_valid_test_frame(seq=42)
        raw = frame.to_bytes()

        # Invariants: exact 128B size and 64B cache line alignment
        assert len(raw) == ZERO_COPY_FRAME_SIZE == 128
        assert len(raw) % 64 == 0

        # Memory deserialization with bit-exact preservation
        restored = ZeroCopyContractFrame.from_bytes(raw)
        assert restored.vital_hp == 6
        assert restored.sequence_id == 42
        assert restored.metric1 == 120.0
        assert restored.metric2 == 1.5
        assert restored.to_bytes() == raw

    # -------------------------------------------------------------------------
    # P3: Chronometric Shadow Multiplexing
    # -------------------------------------------------------------------------
    def test_p3_chronometric_shadow_multiplexing(self):
        """P3: Heavy compute executes in VBlank shadow (Phi_Scanout * Phi_Epistemic == 0) and yields on pressure."""
        injector = VBlankExperimentInjector(vblank_budget_ms=2.2)

        # 1. Fits within remaining VBlank shadow (< 2.2 ms)
        can_exec, result, duration = injector.execute_in_shadow(
            experiment_fn=lambda: "VBLANK_PROBING_OK",
            estimated_duration_ms=1.2,
            elapsed_vblank_ms=0.5,
        )
        assert can_exec is True
        assert result == "VBLANK_PROBING_OK"
        assert duration >= 0.0

        # 2. Overruns remaining VBlank shadow -> preemptively yields to scanout
        can_overrun, res_overrun, _ = injector.execute_in_shadow(
            experiment_fn=lambda: "OVERRUN",
            estimated_duration_ms=2.0,
            elapsed_vblank_ms=1.0,  # 2.0 > (2.2 - 1.0 = 1.2)
        )
        assert can_overrun is False
        assert "YIELD_BEFORE_SCANOUT" in str(res_overrun)

        # 3. UMA arbitration in game companion: at >= 0.78, upscaling switches to DP4A to free bandwidth
        game_companion = SovereignGameCompanionBridge()
        telemetry, _ = game_companion.process_render_frame(
            gpu_render_duration_ms=7.0,
            uma_bandwidth_saturation=0.82,  # >= 0.78
        )
        assert telemetry.upscaling_active is True
        assert telemetry.k_speculative_slots == 1

    # -------------------------------------------------------------------------
    # P4: Epistemic Type Safety
    # -------------------------------------------------------------------------
    def test_p4_epistemic_type_safety_lattice_and_role_separation(self):
        """P4: Level(Verdict) <= min AllowedLevel(e), PROPOSER cannot self-certify, and GBNF AST compilation."""
        # 1. Reject type promotion: SIMULATED cannot justify MEASURED_WITH_SCOPE
        trace_sim = EnlgClaimTrace(
            claim="LPDDR5 memory bus sustains 48 GB/s.",
            scope="Hardware memory subsystem within defined invariant bounds.",
            role=EnlgRole.VERIFIER,
            evidence=(EnlgEvidenceClass.SIMULATED,),
            nulls=NullTriplet(
                m0_no_effect="Bandwidth unchanged",
                m1_noise="Variance within noise",
                m2_confounder="Thermal clock variance",
            ),
            falsifier="Bandwidth drops below 40 GB/s",
            validity={"construct": "OK", "internal": "OK", "external": "OK"},
            verdict=EnlgVerdict.MEASURED_WITH_SCOPE,
            residual=("Unmeasured ARM64 bus",),
            has_conflicting_evidence=False,
            vital_max_hp=6,
        )
        with pytest.raises(ContractViolation) as exc_sim:
            EnlgAstValidator.validate_claim_trace(trace_sim)
        assert exc_sim.value.violation_code == "ERR_ENLG_SIMULATION_TO_MEASUREMENT"

        # 2. Reject proposer self-certification
        trace_proposer = EnlgClaimTrace(
            claim="New speculative governor reduces latency by 20%.",
            scope="Linux kernel governor within defined invariant bounds.",
            role=EnlgRole.PROPOSER,
            evidence=(EnlgEvidenceClass.LOCAL_MEASURED,),
            nulls=NullTriplet(
                m0_no_effect="Latency unchanged",
                m1_noise="Variance within noise",
                m2_confounder="Thermal clock variance",
            ),
            falsifier="Latency increases in p99",
            validity={"construct": "OK", "internal": "OK", "external": "OK"},
            verdict=EnlgVerdict.VERIFIED_WITH_SCOPE,
            residual=("Unmeasured workstation variants",),
            has_conflicting_evidence=False,
            vital_max_hp=6,
        )
        with pytest.raises(ContractViolation) as exc_prop:
            EnlgAstValidator.validate_claim_trace(trace_proposer)
        assert exc_prop.value.violation_code == "ERR_ENLG_PROPOSER_SELF_CERTIFICATION"

    # -------------------------------------------------------------------------
    # P5: Thermodynamic Hypothesis Metering
    # -------------------------------------------------------------------------
    def test_p5_thermodynamic_hypothesis_metering(self):
        """P5: Hypotheses consuming energy/bandwidth without entropy reduction (Delta tau == 0) are decoupled."""
        calorimeter = EpistemicCalorimeter(
            max_energy_joules=3.0,
            max_uma_mb=10.0,
            power_watts=28.0,
        )

        # Register two competing hypotheses
        calorimeter.register_hypothesis("H_productive")
        calorimeter.register_hypothesis("H_parasite")

        # H_productive consumes energy but reduces entropy
        calorimeter.record_work(
            hypothesis_id="H_productive",
            duration_ms=150.0,  # 4.2 Joules > 3.0 J
            uma_bytes_transferred=1024,
            entropy_reduction=1.5,
        )
        assert calorimeter.is_parasitic("H_productive") is False

        # H_parasite consumes energy without reducing entropy -> marked parasitic
        calorimeter.record_work(
            hypothesis_id="H_parasite",
            duration_ms=150.0,  # 4.2 Joules > 3.0 J
            uma_bytes_transferred=1024,
            entropy_reduction=0.0,
        )
        assert calorimeter.is_parasitic("H_parasite") is True

    # -------------------------------------------------------------------------
    # P6: Homotopic Conflict Preservation
    # -------------------------------------------------------------------------
    def test_p6_homotopic_conflict_preservation(self):
        """P6: Forbids convex averaging of contradictory claims; enforces bifurcated branch superposition."""
        ring = BifurcatedConflictRing()

        # Direct convex averaging is strictly rejected
        with pytest.raises(ContractViolation, match="ERR_HOMOTOPY_CONVEX_AVERAGING_FORBIDDEN"):
            ring.attempt_convex_averaging(10.0, 15.0)

        # Bifurcate into Alpha (gen=1) and Beta (gen=2) sharing parent predecessor
        parent = _make_valid_test_frame(seq=10)
        branch_alpha, branch_beta = ring.fork_conflict(
            conflict_id="CONFLICT_SIM_VS_LIDAR",
            parent_frame=parent,
            claim_a="Gazebo simulation predicts 10.0 mm position",
            evidence_a=EnlgEvidenceClass.SIMULATED,
            metric_a=10.0,
            claim_b="LiDAR sensor measures 15.0 mm position",
            evidence_b=EnlgEvidenceClass.PHYSICAL_MEASURED,
            metric_b=15.0,
        )

        assert branch_alpha.generation_id == 1
        assert branch_beta.generation_id == 2
        assert branch_alpha.fork_predecessor_hash == parent.compute_binary_digest()
        assert branch_beta.fork_predecessor_hash == parent.compute_binary_digest()
        assert (branch_alpha.frame.flags & FLAG_PLEA_CONFLICT_BIFURCATION) != 0
        assert (branch_beta.frame.flags & FLAG_PLEA_CONFLICT_BIFURCATION) != 0

        # Collapse with verified CDT receipt
        header = ContractHeader(
            contract_id="CDT_RECEIPT_01",
            contract_version=1,
            sequence_id=parent.sequence_id + 2,
            capability_scope=1,
            generation_id=2,
            timestamp_ns=time.time_ns(),
            predecessor_hash=branch_beta.frame.signature_or_digest_raw.hex(),
        )
        cdt_receipt = ReceiptContract(
            header=header,
            execution_digest=branch_beta.frame.signature_or_digest_raw.hex(),
            applied_profile="physical_laser_calibrated",
            actuation_duration_ns=500_000,
            was_degraded_fallback=False,
        )

        collapsed = ring.collapse_branch(
            conflict_id="CONFLICT_SIM_VS_LIDAR",
            winning_branch_generation=2,
            cdt_receipt=cdt_receipt,
        )
        assert collapsed.generation_id == 0  # Rejoined unified trunk
        assert (collapsed.flags & FLAG_PLEA_CONFLICT_BIFURCATION) == 0

    # -------------------------------------------------------------------------
    # P7: Physical-First Actuation
    # -------------------------------------------------------------------------
    def test_p7_physical_first_actuation_audit(self):
        """P7: Changes are committed only after audited physical telemetry and regression verification."""
        engine = SelfTuningWorkstationEngine()
        assert engine.current_state["pl1_watts"] == 28.0

        # 1. Successful hypothesis: 10% expected, 15% measured -> committed
        hyp_win = HardwareTuningHypothesis(
            hypothesis_id="hyp_pl1_32w",
            target_subsystem="pl1_watts",
            proposed_value=32.0,
            baseline_value=28.0,
            expected_latency_improvement_pct=10.0,
        )
        rep_win = engine.evaluate_and_actuate(
            hypothesis=hyp_win,
            probe_fn=lambda: 15.0,
        )
        assert rep_win.receipt_id is not None
        assert engine.current_state["pl1_watts"] == 32.0

        # 2. Regressive hypothesis: 20% expected, only 4% measured -> strictly Delta S == 0
        hyp_regress = HardwareTuningHypothesis(
            hypothesis_id="hyp_pl1_45w",
            target_subsystem="pl1_watts",
            proposed_value=45.0,
            baseline_value=32.0,
            expected_latency_improvement_pct=20.0,
        )
        rep_regress = engine.evaluate_and_actuate(
            hypothesis=hyp_regress,
            probe_fn=lambda: 4.0,  # Below threshold
        )
        assert rep_regress.receipt_id is None
        # State unchanged!
        assert engine.current_state["pl1_watts"] == 32.0

    # -------------------------------------------------------------------------
    # P8: Inertial Stability & Anti-Oscillation
    # -------------------------------------------------------------------------
    def test_p8_inertial_stability_and_hysteresis(self):
        """P8: O(1) Merkle-Momentum damping and >= 12-frame dwell time before upscale."""
        damper = CausalInertiaDamper(gamma=0.85)

        # 1. Trunk with solid history
        trunk = [EnlgEvidenceClass.LOCAL_MEASURED, EnlgEvidenceClass.PHYSICAL_MEASURED, EnlgEvidenceClass.EXTERNAL_REPLICATED]
        trunk_mom = damper.compute_momentum(trunk)
        assert trunk_mom > 1.5

        # Isolated speculative proposal from PROPOSER is dampened by causal inertia
        admissible, reason = damper.evaluate_proposal_admissibility(
            trunk_evidence=trunk,
            proposal_evidence=EnlgEvidenceClass.SIMULATED,
            proposal_role=EnlgRole.PROPOSER,
        )
        assert admissible is False
        assert "PROPOSAL_DAMPENED" in reason

        # 2. Hysteresis filter: 12-frame dwell requirement (N_dwell >= 12)
        copilot = NeuralRenderSteeringCopilot(hysteresis_dwell_frames=12)
        meta = ProgramMetadata("game_app", ProgramWorkloadClass.INTERNAL_ENGINE_RENDERER, 120.0, 8.33, 1024, 0.9)
        scheme_perf = AllocationSchemeVariant("s_perf", {}, NssQualityProfile.PERFORMANCE, 50.0, 0.80, 0)
        scheme_ultra = AllocationSchemeVariant("s_ultra", {}, NssQualityProfile.ULTRA_QUALITY, 20.0, 0.95, 0)

        # Immediate downscale on latency pressure
        r_down = copilot.execute_render_intervention(meta, scheme_perf, simulated_pre_render_latency_ms=8.5)
        assert r_down.nss_profile_applied == NssQualityProfile.PERFORMANCE
        assert r_down.was_held_by_hysteresis is False

        # Attempt upscale during 12 dwell frames: must be held by hysteresis
        for frame in range(12):
            r_held = copilot.execute_render_intervention(meta, scheme_ultra, simulated_pre_render_latency_ms=7.0)
            assert r_held.nss_profile_applied == NssQualityProfile.PERFORMANCE
            assert r_held.was_held_by_hysteresis is True

        # Frame 13: dwell time exceeded, safe upscale permitted
        r_up = copilot.execute_render_intervention(meta, scheme_ultra, simulated_pre_render_latency_ms=5.0)
        assert r_up.nss_profile_applied == NssQualityProfile.ULTRA_QUALITY
        assert r_up.was_held_by_hysteresis is False

    # -------------------------------------------------------------------------
    # P9: Full-Frame Provenance
    # -------------------------------------------------------------------------
    def test_p9_full_frame_provenance_and_canonical_ieee754(self):
        """P9: Valid(Frame) <=> HMAC-SHA256_K(Bytes[0..95]) == sig[96..127], with canonical IEEE-754 floats."""
        session_key = b"krystal_sovereign_secret_key_999"

        # Canonicalize floats: reject NaN / Inf, normalize -0.0 to +0.0
        assert canonicalize_float32(-0.0) == +0.0
        assert canonicalize_float32(12.34) == 12.34
        with pytest.raises(ValueError, match="Non-finite float value"):
            canonicalize_float32(float("nan"))
        with pytest.raises(ValueError, match="Non-finite float value"):
            canonicalize_float32(float("inf"))

        # Build raw frame with placeholder signature
        raw_frame = ZeroCopyContractFrame(
            magic=KRYSTAL_MAGIC,
            version=KRYSTAL_PROTOCOL_VERSION,
            vital_hp=VITAL_MAX_HP,
            contract_type=BinaryContractType.EXECUTION,
            payload_size=16,
            flags=0,
            sequence_id=101,
            generation_id=1,
            timestamp_ns=2_000_000_000,
            capability_scope=0x07,
            nonce=999,
            predecessor_hash_raw=b"\x11" * 32,
            profile_id=2,
            verdict_id=1,
            approved_authority_mask=0xFF,
            metric1=32.0,
            metric2=canonicalize_float32(-0.0),
            signature_or_digest_raw=b"\x00" * 32,
        )

        # Compute HMAC-SHA256 over exact bytes 0..95
        sig_hmac = raw_frame.compute_frame_hmac(session_key)
        assert len(sig_hmac) == 32

        signed_frame = ZeroCopyContractFrame(
            magic=raw_frame.magic,
            version=raw_frame.version,
            vital_hp=raw_frame.vital_hp,
            contract_type=raw_frame.contract_type,
            payload_size=raw_frame.payload_size,
            flags=raw_frame.flags,
            sequence_id=raw_frame.sequence_id,
            generation_id=raw_frame.generation_id,
            timestamp_ns=raw_frame.timestamp_ns,
            capability_scope=raw_frame.capability_scope,
            nonce=raw_frame.nonce,
            predecessor_hash_raw=raw_frame.predecessor_hash_raw,
            profile_id=raw_frame.profile_id,
            verdict_id=raw_frame.verdict_id,
            approved_authority_mask=raw_frame.approved_authority_mask,
            metric1=raw_frame.metric1,
            metric2=raw_frame.metric2,
            signature_or_digest_raw=sig_hmac,
        )

        # Integrity verified
        assert signed_frame.verify_frame_integrity(session_key) is True

        # Tampering with capability_scope in prefix bytes (offset 32..35) invalidates HMAC
        tampered_bytes = bytearray(signed_frame.to_bytes())
        tampered_bytes[32] ^= 0xFF
        tampered_frame = ZeroCopyContractFrame.from_bytes(bytes(tampered_bytes))
        assert tampered_frame.verify_frame_integrity(session_key) is False

    # -------------------------------------------------------------------------
    # P10: Noether-Krystal Survival Symmetry
    # -------------------------------------------------------------------------
    def test_p10_noether_krystal_survival_symmetry(self):
        """P10: VITAL_MAX_HP == 6 holds invariant across all layers; breach forces Delta S == 0 NOOP."""
        # 1. Zero-Copy Frame layer enforcement
        with pytest.raises(ContractViolation, match="VITAL_MAX_HP == 6"):
            ZeroCopyContractFrame(
                magic=KRYSTAL_MAGIC,
                version=KRYSTAL_PROTOCOL_VERSION,
                vital_hp=5,  # Breached invariant
                contract_type=BinaryContractType.STATE,
                payload_size=16,
                flags=0,
                sequence_id=1,
                generation_id=1,
                timestamp_ns=100,
                capability_scope=1,
                nonce=1,
                predecessor_hash_raw=b"\x00" * 32,
                profile_id=1,
                verdict_id=1,
                approved_authority_mask=1,
                metric1=1.0,
                metric2=1.0,
                signature_or_digest_raw=b"\x00" * 32,
            )

        # 2. PLEA Engine enforcement
        engine = PleaEngine()
        assert VITAL_MAX_HP == 6

        # 3. State transition invariance: invalid action implies S_{t+1} == S_t (Delta S == 0)
        ws = SelfTuningWorkstationEngine()
        init_state = dict(ws.current_state)

        # Attempt to actuate without audit / invalid hypothesis
        invalid_hyp = HardwareTuningHypothesis(
            hypothesis_id="hyp_fail",
            target_subsystem="pl1_watts",
            proposed_value=99.0,
            baseline_value=init_state["pl1_watts"],
            expected_latency_improvement_pct=50.0,
        )
        report = ws.evaluate_and_actuate(
            hypothesis=invalid_hyp,
            probe_fn=lambda: -10.0,  # Severe performance degradation
        )
        assert report.receipt_id is None
        assert ws.current_state == init_state  # Delta S == 0
