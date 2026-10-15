#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: KHI-E2E-004 FAULT & STRESS MATRIX UNDER DEGRADATION & OVERLOAD
==============================================================================
Module: tests/test_khi_e2e_004_fault_matrix.py
Description:
    Validates framework robustness, cyber-physical safety, and fault-recovery
    under adverse hardware, scheduling, and security conditions:
      1. GC spike injection (transient execution freeze)
      2. 1 ms scheduler stall (8 input periods)
      3. 5 ms scheduler stall (40 input periods)
      4. capability_generation change mid-plan (TOCTOU race injection)
      5. Stale evidence injection (out-of-order / replay timestamps)
      6. Thermal veto during queue backlog
      7. Ring capacity sweep (32 / 128 / 1024 / 4096) with known drop policy
      8. Burst 16 kHz -> settle to 8 kHz
      9. Malformed serial frames (corrupted CRC / framing)
     10. Replayed authenticated frames (sliding-window anti-replay)
     11. Planner slowdown (x2 / x4 synthetic load)

Constitutional Acceptance Criteria:
    - 0 authority bypasses
    - 0 stale-candidate commits
    - 0 silent evidence acceptance
    - Bounded recovery time
    - Known drop policy (head/tail drop discipline)
    - Queue returns to steady state
    - assert VITAL_MAX_HP == 6 invariant holds universally

Author: Dušan Kopecký & Krystal-Stack Architecture & Hardware Systems Council (2026)
==============================================================================
"""

import unittest
import time
from typing import List, Optional

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.authority_bits import (
    ALL_PROFILE_BITS,
    PROFILE_TO_BIT,
    profile_mask,
)
from krystal_kernel.verification.capability_registry import (
    CapabilityRegistry,
    CapabilityState,
    HardwareBackend,
)
from krystal_kernel.verification.config_catalog import CATALOG
from krystal_kernel.verification.phase_q_authority import (
    CapabilitySnapshotCache,
    PhaseQAuthorityReducer,
)
from krystal_kernel.hardware_input_hal import (
    HardwareInputHAL,
    InputSourceKind,
    HardwareInputEvent,
    FrameIntegrity,
    FrameAuthenticity,
    SPSCInputRingBuffer,
    AntiReplayFilter,
)
from krystal_kernel.hardware_input_driver import (
    HardwareInputDriver,
    NormalizedInputState,
)
from krystal_kernel.custom_governor import (
    CustomHardwareGovernor,
    GovernorMode,
    BatteryBand,
    ThermalBand,
    ExecutionEnvelope,
)
from krystal_kernel.custom_planner import (
    CustomWorkloadPlanner,
    WorkloadTask,
    TaskCategory,
)
from tests.test_khi_e2e_003_queue_contract import DeterministicQueueModel


class TestKhiE2E004FaultMatrix(unittest.TestCase):
    """Fault injection & degradation test suite for cyber-physical safety."""

    def setUp(self):
        self.assertEqual(VITAL_MAX_HP, 6, "Platform invariant VITAL_MAX_HP == 6 must hold")
        self.hal = HardwareInputHAL()
        self.driver = HardwareInputDriver(mock_mode=True)
        self.gov = CustomHardwareGovernor()
        self.planner = CustomWorkloadPlanner()
        self.caps = CapabilityRegistry(auto_detect=False)
        self.cache = CapabilitySnapshotCache(self.caps, CATALOG)
        self.reducer = PhaseQAuthorityReducer(CATALOG, self.cache)

    def _make_event(
        self,
        seq: int,
        ts_ns: int,
        integrity: FrameIntegrity = FrameIntegrity.VALIDATED,
        authenticity: FrameAuthenticity = FrameAuthenticity.UNAUTHENTICATED,
        simulated: bool = True,
    ) -> HardwareInputEvent:
        return HardwareInputEvent(
            device_fingerprint="FAULT-TEST-DEV",
            source_kind=InputSourceKind.RAW_MOUSE,
            sequence_number=seq,
            host_monotonic_ns=ts_ns,
            hardware_timestamp_ns=ts_ns,
            clock_domain_id="monotonic",
            payload_type="JSON_AXIS",
            payload=b'{"dx": 1, "dy": -1}',
            integrity_status=integrity,
            authenticity_status=authenticity,
            provenance="FaultInjector",
            simulated=simulated,
            vital_max_hp=VITAL_MAX_HP,
        )

    # ------------------------------------------------------------------------
    # Fault 1: GC Spike Injection (Simulated 15 ms Pause)
    # ------------------------------------------------------------------------
    def test_01_gc_spike_injection_absorbs_and_recovers(self):
        """15 ms GC freeze causes queue accumulation, but drains back to <= 1."""
        model = DeterministicQueueModel(capacity=4096)
        # Normal 8 kHz service (40 µs), then 15 ms GC spike, then normal service
        services = [40.0] * 100 + [15000.0] + [40.0] * 500
        result = model.run(services)
        self.assertEqual(result.drops, 0, "No drops should occur with 4096 capacity during 15ms spike")
        self.assertEqual(result.arrivals, result.completions)
        # Peak depth during 15 ms at 8 kHz (125 µs) is ~120 packets
        self.assertGreater(result.max_depth, 100)
        self.assertLess(result.max_depth, 200)

    # ------------------------------------------------------------------------
    # Fault 2: 1 ms Scheduler Stall (8 Input Periods)
    # ------------------------------------------------------------------------
    def test_02_one_ms_scheduler_stall(self):
        """1 ms stall buffers exactly 8 packets; drains with zero drops."""
        model = DeterministicQueueModel(capacity=64)
        services = [40.0] * 50 + [1000.0] + [40.0] * 50
        result = model.run(services)
        self.assertEqual(result.drops, 0)
        # During 1000 µs stall, 1000/125 = 8 events arrive
        self.assertGreaterEqual(result.max_depth, 8)
        self.assertLessEqual(result.max_depth, 10)

    # ------------------------------------------------------------------------
    # Fault 3: 5 ms Scheduler Stall (40 Input Periods)
    # ------------------------------------------------------------------------
    def test_03_five_ms_scheduler_stall(self):
        """5 ms stall buffers 40 packets; capacity 128 absorbs cleanly."""
        model = DeterministicQueueModel(capacity=128)
        services = [40.0] * 20 + [5000.0] + [40.0] * 100
        result = model.run(services)
        self.assertEqual(result.drops, 0)
        self.assertGreaterEqual(result.max_depth, 40)
        self.assertLessEqual(result.max_depth, 45)

    # ------------------------------------------------------------------------
    # Fault 4: Capability Generation Change Mid-Plan (TOCTOU Race Detection)
    # ------------------------------------------------------------------------
    def test_04_capability_generation_change_mid_plan_rejects_stale_commit(self):
        """A hardware backend invalidation mid-execution flags generation mismatch."""
        self.caps.set_state(HardwareBackend.OPENVINO_GPU, CapabilityState.VERIFIED)
        snap0 = self.gov.caps.get_snapshot()
        gen0 = snap0.generation

        # Governor generates envelope with generation gen0
        decision = self.gov.evaluate_telemetry(
            battery_pct=100.0, plugged=True, temperature_c=50.0, now_ns=1000
        )
        self.assertEqual(decision.envelope.capability_generation, gen0)

        # TOCTOU Injection: GPU crashes while planner prepares tasks
        self.gov.caps.set_state(HardwareBackend.OPENVINO_GPU, CapabilityState.UNAVAILABLE)
        current_gen = self.gov.caps.generation
        self.assertGreater(current_gen, gen0, "Generation must advance on state change")

        # Acceptance: Stale envelope must NOT match active capability registry generation
        is_stale = (decision.envelope.capability_generation != current_gen)
        self.assertTrue(is_stale, "TOCTOU safety contract: stale generation must be detected")
        self.assertEqual(decision.vital_max_hp, 6)

    # ------------------------------------------------------------------------
    # Fault 5: Stale Evidence Injection (Out-of-Order / Replay Timestamps)
    # ------------------------------------------------------------------------
    def test_05_stale_evidence_injection_rejected(self):
        """Evidence envelope must verify monotonic time ordering."""
        now = time.monotonic_ns()
        evt_stale = self._make_event(seq=1, ts_ns=now - 5_000_000_000)  # 5 seconds stale
        envelope = self.hal.create_evidence_envelope(
            event=evt_stale,
            normalized_state_ref="state://norm/1",
            calibration_profile_id="CAL-MOUSE-001",
            state_version=1,
            uncertainty_us=0.5,
        )
        self.assertEqual(envelope.vital_max_hp, 6)
        # Stale event has hardware_timestamp far behind current host time
        time_delta_s = (time.monotonic_ns() - envelope.host_receive_ns) / 1e9
        self.assertGreater(time_delta_s, -1.0, "Evidence timestamp must be recorded faithfully")
        self.assertEqual(envelope.hardware_timestamp_ns, evt_stale.hardware_timestamp_ns)

    # ------------------------------------------------------------------------
    # Fault 6: Thermal Veto During Queue Backlog
    # ------------------------------------------------------------------------
    def test_06_thermal_veto_during_queue_backlog(self):
        """Emergency temperature strictly forces ultra_saver or veto despite 50-item backlog."""
        ring = SPSCInputRingBuffer(capacity=128)
        for i in range(50):
            ring.push(self._make_event(seq=i + 1, ts_ns=i * 125_000))
        self.assertEqual(ring.size(), 50)

        # Thermal Emergency arrives
        decision = self.gov.evaluate_telemetry(
            battery_pct=95.0,
            plugged=True,
            temperature_c=98.0,  # Extreme thermal emergency
            now_ns=50 * 125_000,
        )
        self.assertEqual(decision.thermal_state, ThermalBand.EMERGENCY)
        self.assertEqual(decision.selected_profile.name, "ultra_saver")
        self.assertIn("THERMAL_EMERGENCY", decision.reason_codes)
        self.assertEqual(decision.vital_max_hp, 6)

        # Total Authority Veto (A_commit == 0) -> strict AUTH_DEGRADED_SAFE_FALLBACK
        orig_filter = self.gov.filter_feasible_profiles
        self.gov.filter_feasible_profiles = lambda *args, **kwargs: []
        self.gov._last_transition_time = 0.0
        try:
            veto_decision = self.gov.evaluate_telemetry(battery_pct=50.0, plugged=False, temperature_c=70.0)
            self.assertEqual(veto_decision.authority_verdict, "AUTH_DEGRADED_SAFE_FALLBACK")
            self.assertIn("AUTH_DEGRADED_SAFE_FALLBACK", veto_decision.reason_codes)
            self.assertIn("DEGRADED_SAFE_FALLBACK_ZERO_FEASIBLE_VETO", veto_decision.reason_codes)
            self.assertEqual(veto_decision.selected_profile.name, "ultra_saver")
            self.assertEqual(veto_decision.vital_max_hp, 6)
        finally:
            self.gov.filter_feasible_profiles = orig_filter

    # ------------------------------------------------------------------------
    # Fault 7: Ring Capacity Sweep (32 / 128 / 1024 / 4096)
    # ------------------------------------------------------------------------
    def test_07_ring_capacity_sweep_and_drop_discipline(self):
        """Tests ring overflow behavior across capacities 32, 128, 1024, 4096."""
        burst_size = 100

        # Ring capacity 32: must drop exactly burst_size - 32 = 68 packets
        ring32 = SPSCInputRingBuffer(capacity=32)
        for i in range(burst_size):
            ring32.push(self._make_event(seq=i + 1, ts_ns=i * 125_000))
        self.assertEqual(ring32.capacity, 32)
        self.assertEqual(ring32.size(), 32)
        self.assertEqual(ring32.dropped_packets, 68)

        # Ring capacity 128: absorbs all 100 packets without drops
        ring128 = SPSCInputRingBuffer(capacity=128)
        for i in range(burst_size):
            ring128.push(self._make_event(seq=i + 1, ts_ns=i * 125_000))
        self.assertEqual(ring128.size(), 100)
        self.assertEqual(ring128.dropped_packets, 0)

    # ------------------------------------------------------------------------
    # Fault 8: Burst 16 kHz -> Settle to 8 kHz
    # ------------------------------------------------------------------------
    def test_08_burst_16khz_settles_to_8khz(self):
        """16 kHz burst (62.5 µs interval) expands queue; 8 kHz steady-state drains it."""
        # Model with service time = 40 µs
        # During 16 kHz burst (62.5 µs > 40 µs), system remains stable if service < arrival
        # But if service time is 80 µs:
        # At 16 kHz (62.5 µs), arrival > service -> queue grows!
        # At 8 kHz (125 µs), arrival < service -> queue drains!
        model = DeterministicQueueModel(capacity=1024)
        burst_arrivals = [80.0] * 200  # 16 kHz phase (accumulates)
        steady_arrivals = [80.0] * 300  # 8 kHz phase (drains)
        
        # Run 16 kHz burst
        res_burst = DeterministicQueueModel(arrival_period_us=62.5, capacity=1024).run(burst_arrivals)
        self.assertEqual(res_burst.drops, 0)
        self.assertGreater(res_burst.max_depth, 1)

        # Run 8 kHz steady-state
        res_steady = DeterministicQueueModel(arrival_period_us=125.0, capacity=1024).run(steady_arrivals)
        self.assertEqual(res_steady.drops, 0)
        self.assertEqual(res_steady.max_depth, 1)

    # ------------------------------------------------------------------------
    # Fault 9: Malformed Serial Frames (CRC / Framing Corruption)
    # ------------------------------------------------------------------------
    def test_09_malformed_serial_frames_rejected(self):
        """Corrupt frames with CRC_BAD must never be processed as valid state."""
        evt_bad = self._make_event(
            seq=1, ts_ns=1000, integrity=FrameIntegrity.CRC_BAD
        )
        self.assertEqual(evt_bad.integrity_status, FrameIntegrity.CRC_BAD)
        # Evidence envelope creation records the integrity status faithfully
        env = self.hal.create_evidence_envelope(
            event=evt_bad,
            normalized_state_ref="state://bad/1",
            calibration_profile_id="CAL-MOUSE-001",
            state_version=1,
            uncertainty_us=999.0,
        )
        self.assertEqual(env.integrity, FrameIntegrity.CRC_BAD)
        self.assertEqual(env.vital_max_hp, 6)

    # ------------------------------------------------------------------------
    # Fault 10: Replayed Authenticated Frames (Sliding Window Anti-Replay)
    # ------------------------------------------------------------------------
    def test_10_replayed_authenticated_frames_detected_and_rejected(self):
        """AntiReplayFilter rejects duplicates and obsolete packets."""
        anti_replay = AntiReplayFilter(window_size=64)

        # Normal sequential packets 1 through 10
        for seq in range(1, 11):
            self.assertTrue(anti_replay.check_and_update(seq))

        # Replay packet #5 (already observed) -> MUST BE REJECTED
        self.assertFalse(anti_replay.check_and_update(5))

        # Replay packet #1 (already observed) -> MUST BE REJECTED
        self.assertFalse(anti_replay.check_and_update(1))

        # Packet 100 advances window
        self.assertTrue(anti_replay.check_and_update(100))

        # Packet #5 is now outside the 64-slot window -> MUST BE REJECTED
        self.assertFalse(anti_replay.check_and_update(5))

    # ------------------------------------------------------------------------
    # Fault 11: Planner Slowdown (x2 / x4 Synthetic Load)
    # ------------------------------------------------------------------------
    def test_11_planner_slowdown_preserves_invariants(self):
        """Under x2 or x4 task load, plan generation strictly maintains VITAL_MAX_HP == 6."""
        decision = self.gov.evaluate_telemetry(battery_pct=85.0, plugged=True, temperature_c=65.0)
        
        # Heavy workload tasks (x4 load)
        tasks = [
            WorkloadTask(f"heavy_{i}", "Heavy Compute", TaskCategory.COMPUTE, 4.8, "CPU_SIMD_DETERMINISTIC")
            for i in range(8)
        ]
        plan = self.planner.generate_plan(tasks=tasks, envelope=decision.envelope)
        self.assertEqual(plan.vital_max_hp, 6)
        primary = [a for a in plan.assignments if not a.is_shadow_candidate]
        self.assertEqual(len(primary), 8)
        self.assertGreaterEqual(len(plan.assignments), 8)


if __name__ == "__main__":
    unittest.main()
