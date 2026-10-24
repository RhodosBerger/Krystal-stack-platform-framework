#!/usr/bin/env python3
"""
==============================================================================
UNIT TEST SUITE: UMA INDUCTIVE IMPEDANCE TERMINALS & VOLTAGE GOVERNOR
==============================================================================
Module: tests/test_uma_inductive_voltage_governor.py
Constitutional Invariant: VITAL_MAX_HP == 6
Milestone: V-UMA-INDUCTIVE-VOLT-VERIFY-01
Normative Reference: artifacts/benchmarks/H07_SILICON_TELEMETRY_AND_OPENVINO_LEXICON_SPEC.md

Verifies:
  1. Sacred constitutional invariant VITAL_MAX_HP == 6 across all inductive structures.
  2. InductiveChokeTerminalBlock transient slew rate (di/dt), back-EMF voltage,
     AC impedance magnitude, reflection coefficient, and return loss in dB.
  3. Multi-channel UMA terminal block evaluation (CPU, GPU, Vulkan USM, LLM KV).
  4. Dynamic Voltage Limit Planning (DVFS compensation, droop guard, 1.05V ceiling).
  5. H-07 Hex Lattice event code emission (0x20, 0x21, 0x30, 0x31, 0x10, 0xFF).
  6. 128-byte cache-line aligned ZeroCopyContractFrame generation and authentication.
  7. Falsification & anti-tampering guards (ContractViolation on HP != 6).

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.silicon_telemetry_lexicon import PulseSequenceLogger, TelemetryEventCode
from krystal_kernel.zero_copy_contracts import ZeroCopyContractFrame
from krystal_kernel.uma_inductive_voltage_governor import (
    TerminalChannel,
    TerminalChokeState,
    UmaVoltagePlan,
    InductiveChokeTerminalBlock,
    UmaInductiveVoltageGovernor,
    GLOBAL_UMA_INDUCTIVE_VOLTAGE_GOVERNOR,
    ABSOLUTE_MAX_VOLTAGE_V,
    MINIMUM_STABLE_VOLTAGE_V,
)


class TestUmaInductiveVoltageGovernor(unittest.TestCase):
    """Test suite verifying Inductive Terminal Blocks and UMA Voltage Limit Governor."""

    def setUp(self):
        self.logger = PulseSequenceLogger(capacity=1024)
        self.governor = UmaInductiveVoltageGovernor(
            logger=self.logger,
            vital_max_hp=6
        )

    def test_01_vital_max_hp_invariants(self):
        """Invariant VITAL_MAX_HP == 6 must strictly hold across all inductive components."""
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(self.governor.vital_max_hp, 6)

        # Falsification check: any mutation of vital_max_hp must raise ContractViolation
        with self.assertRaises(ContractViolation):
            UmaInductiveVoltageGovernor(vital_max_hp=5)

        with self.assertRaises(ContractViolation):
            InductiveChokeTerminalBlock(
                channel=TerminalChannel.CPU_AVX512_UMA_LANE,
                vital_max_hp=7
            )

    def test_02_single_inductive_choke_terminal_behavior(self):
        """Validates Faraday back-EMF, AC impedance, and stored magnetic energy in choke block."""
        terminal = InductiveChokeTerminalBlock(
            channel=TerminalChannel.GPU_IRIS_XE_UMA_LANE,
            inductance_uh=0.50,
            esr_resistance_m_ohm=3.0,
            vital_max_hp=6
        )
        self.assertEqual(terminal.channel, TerminalChannel.GPU_IRIS_XE_UMA_LANE)

        # Evaluate transient under a sudden 15A step demand
        state = terminal.evaluate_transient(current_demand_a=15.0, clock_frequency_mhz=2400.0)
        self.assertIsInstance(state, TerminalChokeState)
        self.assertEqual(state.vital_max_hp, 6)
        self.assertGreater(state.current_instantaneous_a, 4.0)
        self.assertNotEqual(state.di_dt_amps_per_us, 0.0)
        self.assertLess(state.back_emf_volts, 0.0)  # Opposes positive current surge (Faraday/Lenz)
        self.assertGreater(state.reactance_x_l_ohms, 0.0)
        self.assertGreater(state.total_impedance_magnitude_ohms, 0.0)
        self.assertGreater(state.return_loss_db, 0.0)
        self.assertGreater(state.stored_magnetic_energy_uj, 0.0)

    def test_03_multi_channel_terminal_states(self):
        """Validates evaluation across all 4 UMA architectural lanes."""
        states = self.governor.evaluate_terminal_states()
        self.assertEqual(len(states), 4)
        for channel in TerminalChannel:
            self.assertIn(channel, states)
            st = states[channel]
            self.assertEqual(st.vital_max_hp, 6)
            self.assertEqual(st.channel, channel)

    def test_04_voltage_limit_planning_nominal(self):
        """Validates voltage limit plan under nominal load and temperature."""
        plan = self.governor.plan_voltage_limits(
            requested_uma_bandwidth_gbps=28.0,
            junction_temp_c=65.0
        )

        self.assertIsInstance(plan, UmaVoltagePlan)
        self.assertEqual(plan.vital_max_hp, 6)
        self.assertLessEqual(plan.target_core_voltage_v, ABSOLUTE_MAX_VOLTAGE_V)
        self.assertGreaterEqual(plan.target_core_voltage_v, MINIMUM_STABLE_VOLTAGE_V)
        self.assertLessEqual(plan.target_gpu_voltage_v, plan.target_core_voltage_v)
        self.assertEqual(plan.bus_contention_status, "CONTENTION_LOW")
        self.assertEqual(plan.noise_floor_status, "NOMINAL")
        self.assertIn("0x20:UMA_CONTENTION_LOW", plan.hex_event_codes)
        self.assertIn("0x30:NOISE_FLOOR_NOMINAL_DB", plan.hex_event_codes)
        self.assertIn("0xFF:CONSTITUTIONAL_HEARTBEAT", plan.hex_event_codes)
        self.assertEqual(len(plan.provenance_hash), 64)

    def test_05_voltage_limit_planning_high_stress(self):
        """Validates voltage plan and droop compensation under high bandwidth & thermal stress."""
        # 48 GB/s UMA bandwidth (heavy LLM prompt prefill), 85 C temperature
        plan = self.governor.plan_voltage_limits(
            requested_uma_bandwidth_gbps=48.0,
            junction_temp_c=85.0,
            lane_demands_amperes={
                TerminalChannel.CPU_AVX512_UMA_LANE: 18.0,
                TerminalChannel.GPU_IRIS_XE_UMA_LANE: 24.0,
                TerminalChannel.VULKAN_USM_LANE: 10.0,
                TerminalChannel.LLM_KV_CACHE_LANE: 12.0,
            }
        )

        self.assertEqual(plan.bus_contention_status, "CONTENTION_HIGH")
        self.assertIn("0x21:UMA_CONTENTION_HIGH", plan.hex_event_codes)
        self.assertIn("0x12:SILICON_THERMAL_THROTTLE", plan.hex_event_codes)
        self.assertGreater(plan.voltage_compensation_droop_v, 0.0)  # Droop headroom added
        self.assertLessEqual(plan.target_core_voltage_v, ABSOLUTE_MAX_VOLTAGE_V)

        # Check telemetry events written to logger
        events = self.logger.read_all()
        self.assertGreaterEqual(len(events), 4)

    def test_06_zero_copy_contract_frame_emission(self):
        """Validates synthesis of 128B cache-line aligned contract frame from voltage plan."""
        plan = self.governor.plan_voltage_limits(32.0, 70.0)
        frame = self.governor.create_zero_copy_contract_frame(plan, sequence=101)

        self.assertEqual(frame.vital_hp, 6)
        self.assertEqual(frame.sequence, 101)
        self.assertEqual(int(frame.contract_type), 0x01)  # STATE

        session_key = b"krystal_uma_inductive_key_32bytes"
        packed = frame.pack_authenticated(session_key=session_key)
        self.assertEqual(len(packed), 128)

        # Unpack & verify
        unpacked = ZeroCopyContractFrame.unpack_and_verify(packed, session_key=session_key)
        self.assertEqual(unpacked.vital_hp, 6)
        self.assertEqual(unpacked.sequence, 101)
        self.assertAlmostEqual(unpacked.gpu_render_ms, plan.target_core_voltage_v, places=3)
        self.assertAlmostEqual(unpacked.drs_scale_factor, plan.planned_uma_bandwidth_gbps, places=1)


if __name__ == "__main__":
    unittest.main()
