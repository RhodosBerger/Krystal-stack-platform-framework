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
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation, CapabilityScope
from krystal_kernel.sovereign_runtime import SubsystemNodeId
from krystal_kernel.sovereign_uma_inspector_governor import (
    HardwarePriorityLane,
    HexMemoryCellState,
    GovernorAdviceKind,
    MemoryBottleneckCause,
    CauseAwareMemoryAction,
    MicroOpKind,
    HexGridCellDescriptor,
    GovernorAdvice,
    MemoryControllerPressureVector,
    MCServiceRequest,
    calculate_mc_service_debt,
    evaluate_t5_mc_subtransition,
    MicroOperation,
    PlannedMicroOpBatch,
    ResourceCapabilityToken,
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
        """Planned batch must compute valid RPE metrics, guarantee planned_overrun_rate == 0.0, and represent unmeasured state as None."""
        ops = self._create_sample_micro_ops()
        batch = self.bot.plan_behavioral_batch(
            candidate_ops=ops,
            allocated_cycles_ns=1000,
            allocated_energy_uj=5.0,
            allocated_uma_bytes=2048
        )
        self.assertEqual(batch.planned_overrun_rate, 0.0)
        self.assertEqual(batch.problem_formulation, "3D_MULTIDIMENSIONAL_KNAPSACK")
        self.assertEqual(batch.solver_kind, "GREEDY_APPROXIMATION_HEURISTIC")
        self.assertGreater(batch.rpe_time, 0.0)
        self.assertGreater(batch.rpe_energy, 0.0)
        self.assertGreater(batch.rpe_memory, 0.0)
        self.assertGreater(batch.aggregate_rpe, 0.0)
        self.assertGreater(batch.resource_packing_balance, 0.0)
        self.assertEqual(batch.rpe_vector, (batch.rpe_time, batch.rpe_energy, batch.rpe_memory))
        self.assertGreater(batch.rpe_dominant, 0.0)
        self.assertIn(batch.dominant_bottleneck_resource, ("TIME", "ENERGY", "MEMORY"))

        # Epistemically NOT_MEASURED prior to reconciliation
        self.assertIsNone(batch.measured_overrun_rate)
        self.assertFalse(batch.is_measured)

        # After reconciliation with actual measurements:
        reconciled = batch.reconcile_measurements(
            actual_time_ns=batch.total_planned_cycles_ns,
            actual_energy_uj=batch.total_planned_energy_uj,
            actual_uma_bytes=batch.total_planned_uma_bytes
        )
        self.assertEqual(reconciled.measured_overrun_rate, 0.0)
        self.assertTrue(reconciled.is_measured)

    def test_resource_capability_token_enforcement_gate(self) -> None:
        """ResourceCapabilityToken enforces Cap(op) <= Cap(T), Region(op) <= Region(T), Cost(op) <= Budget(T), and expiry."""
        token = ResourceCapabilityToken(
            token_id="TOK-AUTH-01",
            subsystem=SubsystemNodeId.CHRONO_SOC,
            lane=HardwarePriorityLane.LANE_0_MICRO_SILICON,
            uma_region_hex_start=0x00,
            uma_region_hex_end=0x3F,
            allocated_quantum_ns=500,
            max_energy_budget_uj=2.0,
            max_memory_footprint_bytes=1024,
            granted_capabilities=CapabilityScope.CAN_EXECUTE | CapabilityScope.CAN_OBSERVE,
            expires_at_ns=10_000_000_000
        )

        valid_op = MicroOperation(
            micro_op_id="UOP-VALID",
            kind=MicroOpKind.SAMPLE_SENSOR_PULSE,
            target_subsystem=SubsystemNodeId.CHRONO_SOC,
            cycle_cost_ns=200,
            energy_cost_uj=1.0,
            uma_footprint_bytes=256,
            required_capability=CapabilityScope.CAN_EXECUTE
        )
        is_exec, reason = token.authorizes_micro_op(valid_op, now_ns=1_000_000_000)
        self.assertTrue(is_exec)
        self.assertEqual(reason, "AUTHORIZED")
        self.assertTrue(token.is_executable(valid_op, now_ns=1_000_000_000))

        # 1. Subsystem mismatch
        alien_op = MicroOperation(
            micro_op_id="UOP-ALIEN",
            kind=MicroOpKind.SAMPLE_SENSOR_PULSE,
            target_subsystem=SubsystemNodeId.WSL_BRIDGE,
            cycle_cost_ns=200,
            energy_cost_uj=1.0,
            uma_footprint_bytes=256
        )
        is_exec, reason = token.authorizes_micro_op(alien_op, now_ns=1_000_000_000)
        self.assertFalse(is_exec)
        self.assertIn("Subsystem mismatch", reason)

        # 2. Capability breach
        admin_op = MicroOperation(
            micro_op_id="UOP-ADMIN",
            kind=MicroOpKind.SAMPLE_SENSOR_PULSE,
            target_subsystem=SubsystemNodeId.CHRONO_SOC,
            cycle_cost_ns=200,
            energy_cost_uj=1.0,
            uma_footprint_bytes=256,
            required_capability=CapabilityScope.CAN_AUTHORIZE
        )
        is_exec, reason = token.authorizes_micro_op(admin_op, now_ns=1_000_000_000)
        self.assertFalse(is_exec)
        self.assertIn("Capability breach", reason)

        # 3. Budget breach: time
        slow_op = MicroOperation(
            micro_op_id="UOP-SLOW",
            kind=MicroOpKind.SAMPLE_SENSOR_PULSE,
            target_subsystem=SubsystemNodeId.CHRONO_SOC,
            cycle_cost_ns=800,  # > 500
            energy_cost_uj=1.0,
            uma_footprint_bytes=256
        )
        is_exec, reason = token.authorizes_micro_op(slow_op, now_ns=1_000_000_000)
        self.assertFalse(is_exec)
        self.assertIn("Time budget breach", reason)

        # 4. Expired token
        is_exec, reason = token.authorizes_micro_op(valid_op, now_ns=12_000_000_000)
        self.assertFalse(is_exec)
        self.assertIn("Token expired", reason)

        # 5. Region bounds breach
        is_exec, reason = token.authorizes_micro_op(
            valid_op,
            now_ns=1_000_000_000,
            target_region_hex_start=0x40,
            target_region_hex_end=0x50
        )
        self.assertFalse(is_exec)
        self.assertIn("Region breach", reason)

    def test_bot_planning_with_resource_capability_token_filter(self) -> None:
        """MicroOperationAllocationBot enforces token limits and filters unauthorized micro-ops."""
        token = ResourceCapabilityToken(
            token_id="TOK-FILTER",
            subsystem=SubsystemNodeId.CHRONO_SOC,
            lane=HardwarePriorityLane.LANE_0_MICRO_SILICON,
            uma_region_hex_start=0x00,
            uma_region_hex_end=0x1F,
            allocated_quantum_ns=400,
            max_energy_budget_uj=2.0,
            max_memory_footprint_bytes=512,
            granted_capabilities=CapabilityScope.CAN_EXECUTE
        )

        ops = [
            MicroOperation("OP-OK", MicroOpKind.SAMPLE_SENSOR_PULSE, SubsystemNodeId.CHRONO_SOC, 150, 0.5, 128),
            MicroOperation("OP-WRONG-SUB", MicroOpKind.SAMPLE_SENSOR_PULSE, SubsystemNodeId.WSL_BRIDGE, 150, 0.5, 128),
            MicroOperation("OP-TOO-EXPENSIVE", MicroOpKind.SAMPLE_SENSOR_PULSE, SubsystemNodeId.CHRONO_SOC, 600, 3.0, 1024),
        ]

        batch = self.bot.plan_behavioral_batch(
            candidate_ops=ops,
            allocated_cycles_ns=1000,
            allocated_energy_uj=10.0,
            allocated_uma_bytes=4096,
            token=token
        )
        # Only OP-OK should be authorized and admitted
        self.assertEqual(len(batch.operations), 1)
        self.assertEqual(batch.operations[0].micro_op_id, "OP-OK")
        self.assertLessEqual(batch.total_planned_cycles_ns, token.allocated_quantum_ns)
        self.assertLessEqual(batch.total_planned_energy_uj, token.max_energy_budget_uj)
        self.assertLessEqual(batch.total_planned_uma_bytes, token.max_memory_footprint_bytes)


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


class TestResourceControlPlaneRefinements(unittest.TestCase):
    """Verifies timing decomposition, canonical 9-tuple cell, decoupled VBlank, and lane timescales."""

    def test_hardware_priority_lane_deadline_bounds(self) -> None:
        """Each priority lane has explicit physical timescale deadline upper bound."""
        self.assertEqual(HardwarePriorityLane.LANE_0_MICRO_SILICON.deadline_upper_bound_ns, 250)
        self.assertEqual(HardwarePriorityLane.LANE_1_MESO_VBLANK.deadline_upper_bound_ns, 16_666_667)
        self.assertEqual(HardwarePriorityLane.LANE_2_MACRO_CADENCE.deadline_upper_bound_ns, 500_000_000)
        self.assertEqual(HardwarePriorityLane.LANE_3_EPOCH_GOVERN.deadline_upper_bound_ns, 60_000_000_000)

    def test_canonical_nine_tuple_cell_descriptor(self) -> None:
        """HexGridCellDescriptor represents canonical 9-tuple Resource Topology Cell."""
        cell = HexGridCellDescriptor(
            cell_index=0x4A,
            hex_addr_tag="0x4A",
            storage_bytes=128,
            owner_subsystem=SubsystemNodeId.CHRONO_SOC,
            bandwidth_pressure=0.67,
            dwell_age_cycles=4,
            active_token_id="TOK-0x8F12",
            state=HexMemoryCellState.DMA_RING_ALLOCATED,
            lane=HardwarePriorityLane.LANE_1_MESO_VBLANK,
            read_write_mode="WRITE",
            confidence_score=0.98
        )
        self.assertEqual(cell.hex_addr_tag, "0x4A")
        self.assertEqual(cell.state, HexMemoryCellState.DMA_RING_ALLOCATED)
        self.assertEqual(cell.lane, HardwarePriorityLane.LANE_1_MESO_VBLANK)
        self.assertEqual(cell.read_write_mode, "WRITE")
        self.assertEqual(cell.confidence_score, 0.98)

    def test_decoupled_vblank_headroom_governor(self) -> None:
        """VBlank alone is not enough; insufficient measured bus headroom rejects BURST_AUTHORIZE."""
        advisor = SovereignGovernorAdvisor(vital_max_hp=VITAL_MAX_HP)

        # Case 1: VBlank active, but bus headroom is constrained (0.30 <= 0.60) -> No Burst
        adv_no_burst = advisor.evaluate_and_advise(
            current_uma=0.20,
            current_temp_c=45.0,
            is_vblank_active=True,
            measured_bus_headroom=0.30
        )
        self.assertNotEqual(adv_no_burst.advice_kind, GovernorAdviceKind.BURST_AUTHORIZE)

        # Case 2: VBlank active AND measured headroom is ample (0.80 > 0.60) -> Burst Authorized
        adv_burst = advisor.evaluate_and_advise(
            current_uma=0.20,
            current_temp_c=45.0,
            is_vblank_active=True,
            measured_bus_headroom=0.80
        )
        self.assertEqual(adv_burst.advice_kind, GovernorAdviceKind.BURST_AUTHORIZE)
        self.assertEqual(adv_burst.suggested_quota_multiplier, 1.25)

    def test_planned_batch_timing_decomposition(self) -> None:
        """PlannedMicroOpBatch correctly decomposes sequential vs pipelined total critical path."""
        op = MicroOperation("UOP-TEST", MicroOpKind.EXECUTE_ALU_KERNEL, SubsystemNodeId.CHRONO_SOC, 1450, 2.5, 512)
        batch = PlannedMicroOpBatch(
            batch_id="BATCH-TEST",
            allocated_cycle_quota_ns=3000,
            total_planned_cycles_ns=1450,
            allocated_energy_quota_uj=10.0,
            total_planned_energy_uj=2.5,
            allocated_uma_quota_bytes=1024,
            total_planned_uma_bytes=512,
            operations=[op],
            quota_utilization_ratio=1450 / 3000,
            is_quota_respected=True,
            hmac_signature="sig",
            t_observe_ns=150,
            t_decision_ns=350,
            t_transport_ns=400,
            t_validate_ns=250
        )
        # T_system = 150 + 350 + 400 + 250 = 1150 ns
        self.assertEqual(batch.t_system_overhead_ns, 1150)
        # T_sequential = 1150 + 1450 = 2600 ns
        self.assertEqual(batch.t_sequential_total_ns, 2600)
        # T_pipelined = max(1150, 1450) = 1450 ns
        self.assertEqual(batch.t_pipelined_total_ns, 1450)


class TestMemoryControllerPressureVectorAndT5Subtransition(unittest.TestCase):
    """
    Verifies Memory Controller Pressure Vector (MCP), Service Debt (D_MC),
    Subtransition T_{5, MC}, and HexGridCellDescriptor controller telemetry.
    """

    def test_mcp_vector_properties_and_mu_mc(self) -> None:
        """MCP vector computes correct mu_MC and identifies dominant cause."""
        mcp = MemoryControllerPressureVector(
            queue_occupancy=0.20,
            bank_conflict_rate=0.30,
            row_miss_pressure=0.65,
            refresh_blocking_pressure=0.10,
            service_debt_pressure=0.25,
            write_drain_pressure=0.15,
            bandwidth_headroom=0.80  # 1 - H = 0.20
        )
        self.assertEqual(mcp.mu_mc, 0.65)
        self.assertEqual(mcp.dominant_cause, MemoryBottleneckCause.ROW_MISS_THRASHING)
        self.assertEqual(mcp.as_tuple(), (0.20, 0.30, 0.65, 0.10, 0.25, 0.15, 0.80))

    def test_mcp_vital_hp_invariant(self) -> None:
        """Constitutional invariant VITAL_MAX_HP == 6 must hold in MCP."""
        with self.assertRaises(ContractViolation):
            MemoryControllerPressureVector(vital_max_hp=5)

    def test_mc_service_debt_calculation(self) -> None:
        """D_MC accurately computes weighted latency debt across requests."""
        # Empty list -> 0.0
        self.assertEqual(calculate_mc_service_debt([]), 0.0)

        # 3 requests:
        # req1: age=50, deadline=100 -> ratio=0.5, weight=1.0 -> 0.5
        # req2: age=120, deadline=100 -> ratio=1.2, weight=1.0 -> 1.2
        # req3: age=80, deadline=100 -> ratio=0.8, weight=2.0 -> 1.6
        # sum = 3.3, N=3 -> D_MC = 1.10 (Service debt breach!)
        reqs = [
            MCServiceRequest("R1", age_ns=50, deadline_ns=100, weight=1.0),
            MCServiceRequest("R2", age_ns=120, deadline_ns=100, weight=1.0),
            MCServiceRequest("R3", age_ns=80, deadline_ns=100, weight=2.0),
        ]
        debt = calculate_mc_service_debt(reqs)
        self.assertAlmostEqual(debt, 1.10, places=3)
        self.assertGreater(debt, 1.0)  # Service envelope breach

    def test_mc_service_request_validation(self) -> None:
        """MCServiceRequest rejects invalid deadlines and non-6 HP."""
        with self.assertRaises(ContractViolation):
            MCServiceRequest("R-BAD", age_ns=10, deadline_ns=0)
        with self.assertRaises(ContractViolation):
            MCServiceRequest("R-BAD", age_ns=10, deadline_ns=100, vital_max_hp=3)

    def test_t5_mc_case_a_high_volume_healthy_controller(self) -> None:
        """Case A: High volume (RPE_m = 0.90) but controller healthy (mu_MC = 0.35) -> NO throttling."""
        mcp = MemoryControllerPressureVector(
            queue_occupancy=0.30,
            bank_conflict_rate=0.20,
            row_miss_pressure=0.25,
            refresh_blocking_pressure=0.10,
            service_debt_pressure=0.35,
            write_drain_pressure=0.15,
            bandwidth_headroom=0.75  # 1 - H = 0.25
        )
        self.assertEqual(mcp.mu_mc, 0.35)
        cause, action, advice = evaluate_t5_mc_subtransition(
            rpe_m=0.90,
            mcp=mcp,
            mc_throttle_threshold=0.75,
            service_debt_limit=1.0,
            rpe_m_dominant=False
        )
        self.assertEqual(cause, MemoryBottleneckCause.NOMINAL)
        self.assertEqual(action, CauseAwareMemoryAction.SUSTAIN)
        self.assertEqual(advice, GovernorAdviceKind.SUSTAIN_NOMINAL)

    def test_t5_mc_case_b_low_volume_bank_conflict(self) -> None:
        """Case B: Low volume (RPE_m = 0.45) but severe bank conflicts (mu_MC = 0.91) -> Targeted action."""
        mcp = MemoryControllerPressureVector(
            queue_occupancy=0.30,
            bank_conflict_rate=0.91,
            row_miss_pressure=0.40,
            refresh_blocking_pressure=0.10,
            service_debt_pressure=0.20,
            write_drain_pressure=0.10,
            bandwidth_headroom=0.60
        )
        self.assertEqual(mcp.mu_mc, 0.91)
        self.assertEqual(mcp.dominant_cause, MemoryBottleneckCause.BANK_CONFLICT)

        cause, action, advice = evaluate_t5_mc_subtransition(
            rpe_m=0.45,
            mcp=mcp,
            mc_throttle_threshold=0.75,
            service_debt_limit=1.0
        )
        self.assertEqual(cause, MemoryBottleneckCause.BANK_CONFLICT)
        self.assertEqual(action, CauseAwareMemoryAction.REORDER_REMAP_DEFER)
        self.assertEqual(advice, GovernorAdviceKind.EMERGENCY_QUOTA_CUT)

    def test_t5_mc_service_debt_breach(self) -> None:
        """Service debt breach triggers PRIORITY_INHERITANCE_PROMOTION."""
        mcp = MemoryControllerPressureVector(
            queue_occupancy=0.40,
            bank_conflict_rate=0.20,
            row_miss_pressure=0.30,
            refresh_blocking_pressure=0.10,
            service_debt_pressure=1.20,  # > 1.0 limit
            write_drain_pressure=0.15,
            bandwidth_headroom=0.70
        )
        cause, action, advice = evaluate_t5_mc_subtransition(
            rpe_m=0.50,
            mcp=mcp,
            service_debt_limit=1.0
        )
        self.assertEqual(cause, MemoryBottleneckCause.SERVICE_DEBT_BREACH)
        self.assertEqual(action, CauseAwareMemoryAction.PRIORITY_INHERITANCE_PROMOTION)

    def test_t5_mc_refresh_blocking(self) -> None:
        """Refresh blocking triggers HOLD_BURST_FOR_REFRESH."""
        mcp = MemoryControllerPressureVector(
            refresh_blocking_pressure=0.80,
            bandwidth_headroom=0.80
        )
        cause, action, advice = evaluate_t5_mc_subtransition(
            rpe_m=0.50,
            mcp=mcp,
            mc_throttle_threshold=0.75
        )
        self.assertEqual(cause, MemoryBottleneckCause.REFRESH_BLOCKING)
        self.assertEqual(action, CauseAwareMemoryAction.HOLD_BURST_FOR_REFRESH)
        self.assertEqual(advice, GovernorAdviceKind.THROTTLE_DOWNSCALE)

    def test_t5_mc_queue_congestion(self) -> None:
        """Queue congestion triggers REDUCE_PRODUCERS_INCREASE_QUANTUM."""
        mcp = MemoryControllerPressureVector(
            queue_occupancy=0.85,
            bandwidth_headroom=0.80
        )
        cause, action, advice = evaluate_t5_mc_subtransition(
            rpe_m=0.50,
            mcp=mcp,
            mc_throttle_threshold=0.75
        )
        self.assertEqual(cause, MemoryBottleneckCause.QUEUE_CONGESTION)
        self.assertEqual(action, CauseAwareMemoryAction.REDUCE_PRODUCERS_INCREASE_QUANTUM)

    def test_governor_advisor_mcp_integration_case_a_and_b(self) -> None:
        """GovernorAdvisor distinguishes Case A from Case B when MCP is supplied."""
        advisor = SovereignGovernorAdvisor(vital_max_hp=VITAL_MAX_HP)

        # Case A: UMA is high (0.85 >= 0.78 throttle limit) but MCP is healthy (mu_MC = 0.30)
        mcp_healthy = MemoryControllerPressureVector(
            queue_occupancy=0.25,
            bank_conflict_rate=0.20,
            row_miss_pressure=0.25,
            refresh_blocking_pressure=0.10,
            service_debt_pressure=0.30,
            write_drain_pressure=0.20,
            bandwidth_headroom=0.80
        )
        adv_a = advisor.evaluate_and_advise(
            current_uma=0.85,
            current_temp_c=45.0,
            mcp=mcp_healthy
        )
        self.assertEqual(adv_a.advice_kind, GovernorAdviceKind.SUSTAIN_NOMINAL)
        self.assertEqual(adv_a.suggested_quota_multiplier, 1.0)
        self.assertIn("HIGH VOLUME STREAMING", adv_a.rationale)

        # Case B: UMA is low (0.45) but bank conflict is critical (B = 0.91)
        mcp_conflict = MemoryControllerPressureVector(
            bank_conflict_rate=0.91,
            bandwidth_headroom=0.60
        )
        adv_b = advisor.evaluate_and_advise(
            current_uma=0.45,
            current_temp_c=45.0,
            mcp=mcp_conflict
        )
        self.assertEqual(adv_b.advice_kind, GovernorAdviceKind.EMERGENCY_QUOTA_CUT)
        self.assertEqual(adv_b.mc_cause, MemoryBottleneckCause.BANK_CONFLICT)
        self.assertEqual(adv_b.mc_action, CauseAwareMemoryAction.REORDER_REMAP_DEFER)
        self.assertIn("CONTROLLER CONTENTION", adv_b.rationale)

    def test_hex_grid_cell_descriptor_mc_telemetry(self) -> None:
        """HexGridCellDescriptor accurately stores controller-local telemetry."""
        grid = UmaInspectorHexGrid(vital_max_hp=VITAL_MAX_HP)
        cell = grid.allocate_cell(
            cell_index=0x15,
            state=HexMemoryCellState.ACTIVE_BUS_MASTER,
            subsystem=SubsystemNodeId.CHRONO_SOC,
            bandwidth_pressure=0.50,
            controller_channel_id=1,
            bank_id=7,
            bank_conflict_score=0.42,
            request_age_cycles=18,
            refresh_blocked=True,
            mc_pressure=0.65
        )
        self.assertEqual(cell.controller_channel_id, 1)
        self.assertEqual(cell.bank_id, 7)
        self.assertAlmostEqual(cell.bank_conflict_score, 0.42, places=2)
        self.assertEqual(cell.request_age_cycles, 18)
        self.assertTrue(cell.refresh_blocked)
        self.assertAlmostEqual(cell.mc_pressure, 0.65, places=2)


if __name__ == "__main__":
    unittest.main()

