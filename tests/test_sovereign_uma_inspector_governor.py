#!/usr/bin/env python3
"""
================================================================================
UNIT TESTS: UMA INSPECTOR, HEX GRID, TOKEN MONITOR & GOVERNOR ADVISOR
================================================================================
Module:     tests/test_sovereign_uma_inspector_governor.py
Author:     Antigravity Cognitive Invariants Group (2026)
Invariants: VITAL_MAX_HP == 6 | Proposer != Evaluator | Delta S == 0 on breach
================================================================================
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.sovereign_runtime import SubsystemNodeId
from krystal_kernel.sovereign_uma_inspector_governor import (
    HardwarePriorityLane,
    HexMemoryCellState,
    GovernorAdviceKind,
    MicroOpKind,
    HexGridCellDescriptor,
    GovernorAdvice,
    MicroOperation,
    PlannedMicroOpBatch,
    UmaInspectorHexGrid,
    HardwarePrioritizationTokenMonitor,
    SovereignGovernorAdvisor,
    MicroOperationAllocationBot,
    calculate_physical_clock_period_ns,
    GLOBAL_UMA_HEX_GRID,
    GLOBAL_HW_TOKEN_MONITOR,
    GLOBAL_GOVERNOR_ADVISOR,
    GLOBAL_MICRO_OP_BOT,
)


class TestUmaInspectorHexGrid(unittest.TestCase):
    """Verifies 16x16 hexadecimal grid mapping, Hex-HSL color mapping, and ASCII rendering."""

    def setUp(self) -> None:
        self.inspector = UmaInspectorHexGrid(vital_max_hp=VITAL_MAX_HP)

    def test_grid_initialization_256_cells(self) -> None:
        """Grid must contain exactly 256 cells from 0x00 to 0xFF."""
        self.assertEqual(len(self.inspector.cells), 256)
        self.assertEqual(self.inspector.cells[0].hex_addr_tag, "0x00")
        self.assertEqual(self.inspector.cells[255].hex_addr_tag, "0xFF")
        self.assertEqual(self.inspector.cells[0].state, HexMemoryCellState.FREE)

    def test_cell_allocation_and_hex_hsl_mapping(self) -> None:
        """Allocating a cell updates state, hue angle (120° -> 0°), and HSLA u32."""
        # Low pressure (0.1) -> Hue near 120° (Emerald)
        cell_cool = self.inspector.allocate_cell(
            cell_index=0x2A,
            state=HexMemoryCellState.ACTIVE_BUS_MASTER,
            subsystem=SubsystemNodeId.CHRONO_SOC,
            bandwidth_pressure=0.10
        )
        self.assertEqual(cell_cool.state, HexMemoryCellState.ACTIVE_BUS_MASTER)
        self.assertAlmostEqual(cell_cool.hue_angle_deg, 108.0, delta=2.0)

        # High pressure (0.8) -> Hue near 24° (Red Ember)
        cell_hot = self.inspector.allocate_cell(
            cell_index=0x2B,
            state=HexMemoryCellState.THROTTLED_COOLDOWN,
            subsystem=SubsystemNodeId.CHRONO_SOC,
            bandwidth_pressure=0.80
        )
        self.assertEqual(cell_hot.state, HexMemoryCellState.THROTTLED_COOLDOWN)
        self.assertAlmostEqual(cell_hot.hue_angle_deg, 24.0, delta=2.0)

    def test_mean_active_uma_pressure(self) -> None:
        """Calculates correct mean bandwidth saturation across active cells."""
        self.inspector.allocate_cell(0x01, HexMemoryCellState.ACTIVE_BUS_MASTER, SubsystemNodeId.CHRONO_SOC, 0.40)
        self.inspector.allocate_cell(0x02, HexMemoryCellState.DMA_RING_ALLOCATED, SubsystemNodeId.WSL_BRIDGE, 0.60)
        self.assertAlmostEqual(self.inspector.get_average_uma_pressure(), 0.50, places=2)

    def test_ascii_inspector_rendering(self) -> None:
        """generate_ascii_hex_inspector must output formatted 16x16 grid."""
        ascii_view = self.inspector.generate_ascii_hex_inspector()
        self.assertIn("UMA INSPECTOR: 16x16 HEXADECIMAL GRID ASSET", ascii_view)
        self.assertIn("0: |", ascii_view)
        self.assertIn("F: |", ascii_view)
        self.assertIn("Legend:", ascii_view)

    def test_out_of_bounds_cell_rejection(self) -> None:
        """Attempting to allocate cell >= 256 or < 0 must raise ContractViolation."""
        with self.assertRaises(ContractViolation):
            self.inspector.allocate_cell(256, HexMemoryCellState.FREE, SubsystemNodeId.CHRONO_SOC, 0.0)


class TestHardwarePrioritizationTokenMonitor(unittest.TestCase):
    """Verifies token tracking across priority lanes and starvation detection."""

    def setUp(self) -> None:
        self.monitor = HardwarePrioritizationTokenMonitor(vital_max_hp=VITAL_MAX_HP)

    def test_token_registration_and_lane_occupancy(self) -> None:
        """Tokens register into specific priority lanes."""
        self.monitor.register_token("TOK-L0-1", HardwarePriorityLane.LANE_0_MICRO_SILICON, SubsystemNodeId.CHRONO_SOC, 500)
        self.monitor.register_token("TOK-L1-1", HardwarePriorityLane.LANE_1_MESO_VBLANK, SubsystemNodeId.GODOT_HOLOBRIDGE, 3500)

        self.assertEqual(self.monitor.get_lane_occupancy(HardwarePriorityLane.LANE_0_MICRO_SILICON), 1)
        self.assertEqual(self.monitor.get_lane_occupancy(HardwarePriorityLane.LANE_1_MESO_VBLANK), 1)
        self.assertEqual(self.monitor.get_lane_occupancy(HardwarePriorityLane.LANE_2_MACRO_CADENCE), 0)

    def test_starvation_detection_on_excessive_dwell(self) -> None:
        """Tokens dwelling >= 12 cycles without release are flagged for starvation risk."""
        tok = self.monitor.register_token("TOK-SLOW", HardwarePriorityLane.LANE_2_MACRO_CADENCE, SubsystemNodeId.VPP_GRID_BROKER, 10000)

        # Advance 12 cycles
        for _ in range(12):
            tok = self.monitor.record_quantum_progress("TOK-SLOW", elapsed_ns=100)

        self.assertTrue(tok.is_starvation_risk)
        starving = self.monitor.get_starving_tokens()
        self.assertEqual(len(starving), 1)
        self.assertEqual(starving[0].token_id, "TOK-SLOW")

    def test_token_release(self) -> None:
        """Releasing token removes it from tracking."""
        self.monitor.register_token("TOK-DEL", HardwarePriorityLane.LANE_0_MICRO_SILICON, SubsystemNodeId.CHRONO_SOC, 500)
        self.assertTrue(self.monitor.release_token("TOK-DEL"))
        self.assertEqual(self.monitor.get_lane_occupancy(HardwarePriorityLane.LANE_0_MICRO_SILICON), 0)


class TestSovereignGovernorAdvisor(unittest.TestCase):
    """Verifies advisory recommendations under thermal and UMA stress."""

    def setUp(self) -> None:
        self.advisor = SovereignGovernorAdvisor(vital_max_hp=VITAL_MAX_HP)

    def test_critical_thermal_emergency(self) -> None:
        """Temp >= 75.0°C must trigger EMERGENCY_QUOTA_CUT (0.50 multiplier)."""
        advice = self.advisor.evaluate_and_advise(current_uma=0.30, current_temp_c=76.0)
        self.assertEqual(advice.advice_kind, GovernorAdviceKind.EMERGENCY_QUOTA_CUT)
        self.assertEqual(advice.suggested_quota_multiplier, 0.50)
        self.assertTrue(bool(advice.hmac_digest))

    def test_uma_saturation_downscale(self) -> None:
        """UMA pressure >= 0.78 must trigger THROTTLE_DOWNSCALE (0.90 multiplier)."""
        advice = self.advisor.evaluate_and_advise(current_uma=0.82, current_temp_c=50.0)
        self.assertEqual(advice.advice_kind, GovernorAdviceKind.THROTTLE_DOWNSCALE)
        self.assertEqual(advice.suggested_quota_multiplier, 0.90)

    def test_clean_vblank_burst_authorize(self) -> None:
        """Clean VBlank window with low UMA pressure allows BURST_AUTHORIZE (1.25 multiplier)."""
        advice = self.advisor.evaluate_and_advise(current_uma=0.25, current_temp_c=48.0, is_vblank_active=True)
        self.assertEqual(advice.advice_kind, GovernorAdviceKind.BURST_AUTHORIZE)
        self.assertEqual(advice.suggested_quota_multiplier, 1.25)

    def test_nominal_sustain(self) -> None:
        """Normal conditions yield SUSTAIN_NOMINAL (1.00 multiplier)."""
        advice = self.advisor.evaluate_and_advise(current_uma=0.45, current_temp_c=55.0, is_vblank_active=False)
        self.assertEqual(advice.advice_kind, GovernorAdviceKind.SUSTAIN_NOMINAL)
        self.assertEqual(advice.suggested_quota_multiplier, 1.00)


class TestMicroOperationAllocationBot(unittest.TestCase):
    """Verifies cycle quota measurement and greedy micro-operation knapsack packing."""

    def setUp(self) -> None:
        self.advisor = SovereignGovernorAdvisor(vital_max_hp=VITAL_MAX_HP)
        self.bot = MicroOperationAllocationBot(governor_advisor=self.advisor, vital_max_hp=VITAL_MAX_HP)

    def _create_sample_micro_ops(self) -> list[MicroOperation]:
        return [
            MicroOperation("UOP-01", MicroOpKind.SAMPLE_SENSOR_PULSE, SubsystemNodeId.CHRONO_SOC, 50, 0.1, 32, True),
            MicroOperation("UOP-02", MicroOpKind.PACK_128B_FRAME, SubsystemNodeId.CHRONO_SOC, 120, 0.3, 128, True),
            MicroOperation("UOP-03", MicroOpKind.DISPATCH_DMA_SLOT, SubsystemNodeId.WSL_BRIDGE, 250, 0.8, 256, False),
            MicroOperation("UOP-04", MicroOpKind.EXECUTE_ALU_KERNEL, SubsystemNodeId.CHRONO_SOC, 600, 2.5, 512, False),
            MicroOperation("UOP-05", MicroOpKind.RECONCILE_HMAC_RECEIPT, SubsystemNodeId.SLOVAK_LLM_CASCADE, 350, 1.2, 128, True),
        ]

    def test_quota_measurement(self) -> None:
        """Bot accurately measures available cycles from governor advice."""
        cycles, energy, uma = self.bot.measure_available_cycle_quota(
            base_window_ns=1_000_000,
            current_uma=0.30,
            current_temp_c=45.0,
            is_vblank=True
        )
        self.assertGreater(cycles, 0)
        self.assertGreater(energy, 0.0)
        self.assertGreater(uma, 0)

    def test_greedy_packing_never_exceeds_quota(self) -> None:
        """Packed batch must strictly respect cycle, energy, and UMA byte limits."""
        ops = self._create_sample_micro_ops()
        # Tight budget: 500 ns
        batch = self.bot.plan_behavioral_batch(
            candidate_ops=ops,
            allocated_cycles_ns=500,
            allocated_energy_uj=2.0,
            allocated_uma_bytes=600
        )
        self.assertTrue(batch.is_quota_respected)
        self.assertLessEqual(batch.total_planned_cycles_ns, 500)
        self.assertLessEqual(batch.total_planned_energy_uj, 2.0)
        self.assertLessEqual(batch.total_planned_uma_bytes, 600)
        self.assertGreater(len(batch.operations), 0)
        self.assertTrue(bool(batch.hmac_signature))

        # Check critical path priority: UOP-01 (critical) should be included
        op_ids = [op.micro_op_id for op in batch.operations]
        self.assertIn("UOP-01", op_ids)

    def test_constitutional_invariant_enforcement(self) -> None:
        """Initializing with VITAL_MAX_HP != 6 must raise ContractViolation."""
        with self.assertRaises(ContractViolation):
            MicroOperationAllocationBot(governor_advisor=self.advisor, vital_max_hp=5)

    def test_resource_packing_efficiency_and_zero_planned_overrun(self) -> None:
        """Planned batch must compute valid RPE metrics and guarantee planned_overrun_rate == 0.0."""
        ops = self._create_sample_micro_ops()
        batch = self.bot.plan_behavioral_batch(
            candidate_ops=ops,
            allocated_cycles_ns=1000,
            allocated_energy_uj=5.0,
            allocated_uma_bytes=2048
        )
        self.assertEqual(batch.planned_overrun_rate, 0.0)
        self.assertGreater(batch.rpe_time, 0.0)
        self.assertGreater(batch.rpe_energy, 0.0)
        self.assertGreater(batch.rpe_memory, 0.0)
        self.assertGreater(batch.aggregate_rpe, 0.0)
        self.assertEqual(batch.measured_overrun_rate, 0.0)


class TestPhysicalTimescaleAndContentionTopology(unittest.TestCase):
    """Verifies physical clock calculations and UMA contention topology inspection."""

    def test_physical_clock_period_calculation(self) -> None:
        """Calculates clock period T = 1 / f in nanoseconds."""
        self.assertAlmostEqual(calculate_physical_clock_period_ns(250.0), 4.0, places=3)
        self.assertAlmostEqual(calculate_physical_clock_period_ns(500.0), 2.0, places=3)
        self.assertAlmostEqual(calculate_physical_clock_period_ns(100.0), 10.0, places=3)

        lane0 = HardwarePriorityLane.LANE_0_MICRO_SILICON
        self.assertAlmostEqual(lane0.physical_clock_period_ns(250.0), 4.0, places=3)

    def test_contention_topology_inspection(self) -> None:
        """UMA Hex Grid produces complete contention topology report."""
        grid = UmaInspectorHexGrid()
        grid.allocate_cell(
            cell_index=0x10,
            state=HexMemoryCellState.ACTIVE_BUS_MASTER,
            subsystem=SubsystemNodeId.CHRONO_SOC,
            bandwidth_pressure=0.85,
            active_token_id="TOK-HOT-01",
            dwell_age_cycles=5
        )
        grid.allocate_cell(
            cell_index=0x20,
            state=HexMemoryCellState.DMA_RING_ALLOCATED,
            subsystem=SubsystemNodeId.WSL_BRIDGE,
            bandwidth_pressure=0.40,
            active_token_id="TOK-NORM-01",
            dwell_age_cycles=2
        )

        topo = grid.inspect_contention_topology()
        self.assertEqual(topo["total_cells"], 256)
        self.assertEqual(topo["total_storage_bytes"], 32768)
        self.assertEqual(topo["hotspot_count"], 1)
        self.assertEqual(topo["hotspots"][0]["hex_addr"], "0x10")
        self.assertEqual(topo["hotspots"][0]["token_id"], "TOK-HOT-01")
        self.assertIn("CHRONO_SOC", topo["active_subsystems"])
        self.assertIn("WSL_BRIDGE", topo["active_subsystems"])
        self.assertEqual(topo["active_token_count"], 2)


class TestResourceCapabilityTokenAndGovernorProfile(unittest.TestCase):
    """Verifies ResourceCapabilityToken tracking and ArchitectureProfile governor configuration."""

    def test_resource_capability_token_registration(self) -> None:
        """HardwarePrioritizationTokenMonitor tracks full capability tokens."""
        from krystal_kernel.sovereign_uma_inspector_governor import ResourceCapabilityToken
        from krystal_kernel.constants import CapabilityScope
        monitor = HardwarePrioritizationTokenMonitor()

        cap_token = ResourceCapabilityToken(
            token_id="CAP-TOK-001",
            subsystem=SubsystemNodeId.CHRONO_SOC,
            lane=HardwarePriorityLane.LANE_0_MICRO_SILICON,
            uma_region_hex_start=0x00,
            uma_region_hex_end=0x3F,
            allocated_quantum_ns=800,
            max_energy_budget_uj=2.5,
            max_memory_footprint_bytes=512,
            granted_capabilities=CapabilityScope.CAN_EXECUTE
        )

        entry = monitor.register_capability_token(cap_token)
        self.assertEqual(entry.token_id, "CAP-TOK-001")
        self.assertEqual(monitor.get_lane_occupancy(HardwarePriorityLane.LANE_0_MICRO_SILICON), 1)

        retrieved = monitor.get_capability_token("CAP-TOK-001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.uma_region_hex_start, 0x00)
        self.assertEqual(retrieved.uma_region_hex_end, 0x3F)

        # Release
        self.assertTrue(monitor.release_token("CAP-TOK-001"))
        self.assertIsNone(monitor.get_capability_token("CAP-TOK-001"))

    def test_governor_advisor_with_custom_architecture_profile(self) -> None:
        """Governor advisor adapts thresholds to custom ArchitectureProfile."""
        from krystal_kernel.sovereign_subsystem_turn_coordinator import ArchitectureProfile

        custom_profile = ArchitectureProfile(
            configured_thermal_limit_c=80.0,
            configured_uma_limit=0.85,
            emergency_uma_limit=0.95,
            nominal_uma_limit=0.65
        )
        advisor = SovereignGovernorAdvisor(profile=custom_profile)

        # Temp 76°C is below 80°C limit and below 80 - 5 = 75°C downscale threshold
        # With UMA 0.50, this is SUSTAIN_NOMINAL under custom profile!
        advice_nominal = advisor.evaluate_and_advise(current_uma=0.50, current_temp_c=74.0)
        self.assertEqual(advice_nominal.advice_kind, GovernorAdviceKind.SUSTAIN_NOMINAL)

        # UMA 0.86 triggers downscale under custom profile
        advice_stress = advisor.evaluate_and_advise(current_uma=0.86, current_temp_c=60.0)
        self.assertEqual(advice_stress.advice_kind, GovernorAdviceKind.THROTTLE_DOWNSCALE)


class TestFourModeBenchmarkEngine(unittest.TestCase):
    """Verifies four-mode comparative benchmark across S0, S1, S2, and K modes."""

    def test_four_mode_benchmark_execution(self) -> None:
        """Benchmark executes cleanly across all 4 modes and confirms Mode K superiority."""
        from krystal_kernel.sovereign_uma_inspector_governor import run_four_mode_uma_governor_benchmark

        result = run_four_mode_uma_governor_benchmark(rounds=30)
        self.assertEqual(result.vital_max_hp, VITAL_MAX_HP)

        # Mode K has 0.0 planned and measured overrun, 0.0 rollbacks, and 0.0 conflicts
        self.assertEqual(result.mode_k.measured_overrun_rate, 0.0)
        self.assertEqual(result.mode_k.rollback_rate, 0.0)
        self.assertEqual(result.mode_k.uma_conflict_rate, 0.0)

        # Baseline S0 experiences conflicts due to lack of UMA awareness
        self.assertGreater(result.mode_s0.uma_conflict_rate, 0.0)

        # Mode K achieves higher useful work per joule than baseline S0
        self.assertGreater(result.joule_efficiency_gain_k_vs_s0, 0.0)
        self.assertGreater(result.throughput_gain_k_vs_s0, 0.0)

        # Markdown summary table contains all modes
        self.assertIn("S0 (Baseline)", result.summary_markdown_table)
        self.assertIn("S1 (UMA-aware)", result.summary_markdown_table)
        self.assertIn("S2 (Adaptive Q)", result.summary_markdown_table)
        self.assertIn("K (Full Stack)", result.summary_markdown_table)


if __name__ == "__main__":
    unittest.main()

