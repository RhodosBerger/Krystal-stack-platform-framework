#!/usr/bin/env python3
"""
==============================================================================
UNIT TEST SUITE: PARAVIRTUALIZED MATRIX HYPERVISOR & LAYER INJECTION SUBSYSTEM
==============================================================================
Test Target: krystal_kernel/matrix_hypervisor_subsystem.py
Framework: unittest
Invariant: VITAL_MAX_HP == 6 strictly preserved across all rings and hypercalls.
==============================================================================
"""

import os
import sys
import unittest
import threading

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.matrix_hypervisor_subsystem import (
    GLOBAL_MATRIX_HYPERVISOR,
    MatrixHypervisorSubsystem,
    CentralMachineUnit,
    ZeroCopySharedMemoryRing,
    DmaMemoryFrame,
    UnifiedMatrixDataCluster,
    HypervisorLayerInterceptor,
    ParavirtualizedAutonomousGenerator,
    HypercallOpcode,
    BackendLayerDomain,
    HypervisorPrivilegeRing
)
from krystal_kernel.multilock_matrix_autocompleter import (
    GLOBAL_MULTILOCK_AUTOCOMPLETER,
    MultilockMatrixAutocompleter
)


class TestMatrixHypervisorSubsystem(unittest.TestCase):
    """Rigorous unit tests for Matrix Hypervisor, DMA, Interceptor, and Generator."""

    def setUp(self):
        self.hypervisor = MatrixHypervisorSubsystem()

    def test_vital_max_hp_invariant(self):
        """Platform invariant VITAL_MAX_HP == 6 must hold universally across all hypervisor components."""
        self.assertEqual(VITAL_MAX_HP, 6)
        manifest = self.hypervisor.get_system_manifest()
        self.assertEqual(manifest["vital_max_hp"], 6)
        self.assertEqual(self.hypervisor.central_unit.vital_hp, 6)

    def test_central_machine_unit_telemetry_and_actuation(self):
        """Central Machine Unit must sample 5D vector and simulate thermal dynamics safely."""
        unit = self.hypervisor.central_unit
        vec = unit.sample_telemetry_vector()
        self.assertEqual(len(vec), 5)
        self.assertEqual(vec[0], 12.5)  # Current
        self.assertEqual(vec[1], 220.0) # Temp

        # Actuate spindle and coolant
        unit.apply_control_actuation(rpm_delta=2000.0, feed_delta=0.010, coolant_bar=35.0)
        self.assertEqual(unit.spindle_rpm, 14000.0)
        self.assertEqual(unit.coolant_pressure_bar, 35.0)
        self.assertGreater(unit.total_cutting_cycles, 0)

    def test_zero_copy_shared_memory_ring(self):
        """Shared memory ring must push/pop frames without loss and overwrite gracefully on overflow."""
        ring = ZeroCopySharedMemoryRing(capacity=4)
        self.assertEqual(ring.current_occupancy(), 0)

        for i in range(4):
            frame = DmaMemoryFrame(
                frame_id=i + 1,
                source_domain=BackendLayerDomain.CENTRAL_MACHINE_UNIT,
                target_domain=BackendLayerDomain.PROCEDURAL_SYNTHESIS,
                timestamp_ns=i * 1000,
                payload_type="TEST_FRAME",
                data_vector=[float(i), float(i * 2)],
                vital_max_hp=6
            )
            pushed = ring.push_frame(frame)
            self.assertTrue(pushed)

        self.assertEqual(ring.current_occupancy(), 4)
        latest = ring.peek_latest()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.frame_id, 4)

        # Pop one
        popped = ring.pop_frame()
        self.assertIsNotNone(popped)
        self.assertEqual(popped.frame_id, 1)
        self.assertEqual(ring.current_occupancy(), 3)

    def test_hypercall_dma_direct_transfer(self):
        """HYPERCALL_DMA_DIRECT_TRANSFER must move machine telemetry from Central Unit to ring."""
        res = self.hypervisor.execute_hypercall(
            opcode=HypercallOpcode.HYPERCALL_DMA_DIRECT_TRANSFER,
            calling_domain=BackendLayerDomain.VULKAN_MULTIMESH_STREAM
        )
        self.assertEqual(res["status"], "HYPERCALL_SUCCESS")
        self.assertEqual(res["opcode"], "HYPERCALL_DMA_DIRECT_TRANSFER")
        self.assertIn("frame_transferred", res)
        self.assertEqual(res["frame_transferred"]["source_domain"], "CENTRAL_MACHINE_UNIT")
        self.assertEqual(res["frame_transferred"]["target_domain"], "VULKAN_MULTIMESH_STREAM")

    def test_hypercall_cluster_unify_with_apex_projection(self):
        """HYPERCALL_CLUSTER_UNIFY must assemble a unified cluster with strain tensor sigma and apex."""
        res = self.hypervisor.execute_hypercall(
            opcode=HypercallOpcode.HYPERCALL_CLUSTER_UNIFY,
            calling_domain=BackendLayerDomain.COPILOT_LLM_COGNITION
        )
        self.assertEqual(res["status"], "HYPERCALL_SUCCESS")
        cluster = res["unified_cluster"]
        self.assertIn("strain_tensor_sigma", cluster)
        self.assertIn("apex_projection_x", cluster)
        self.assertEqual(cluster["vital_max_hp"], 6)
        self.assertAlmostEqual(cluster["golden_ratio_phi"], 1.6180339887, places=5)

    def test_hypercall_inject_between_layers(self):
        """Interceptor must capture and inject packets between designated layers."""
        res = self.hypervisor.execute_hypercall(
            opcode=HypercallOpcode.HYPERCALL_INJECT_BETWEEN_LAYERS,
            calling_domain=BackendLayerDomain.PROCEDURAL_SYNTHESIS,
            payload={
                "source_layer": "CENTRAL_MACHINE_UNIT",
                "target_layer": "PROCEDURAL_SYNTHESIS",
                "action": "THERMAL_COMPENSATION_OVERRIDE",
                "data": {"offset_rpm": -800.0, "coolant_boost": True},
                "priority": 1
            }
        )
        self.assertEqual(res["status"], "HYPERCALL_SUCCESS")
        self.assertIn("injected_packet", res)

        # Test routing through interceptor
        base_data = {"layer": "procedural", "lod": 1}
        routed = self.hypervisor.interceptor.process_and_route(
            BackendLayerDomain.CENTRAL_MACHINE_UNIT,
            BackendLayerDomain.PROCEDURAL_SYNTHESIS,
            base_data
        )
        self.assertIn("_injected_hypervisor_packet", routed)
        self.assertEqual(routed["_injected_hypervisor_packet"]["intercept_action"], "THERMAL_COMPENSATION_OVERRIDE")

    def test_autonomous_generator_modes(self):
        """Autonomous generator must step harmonic waves and 96-byte MultiMesh buffers."""
        # 1. Telemetry wave
        wave = self.hypervisor.generator.step_generate_telemetry_wave(base_temp=250.0)
        self.assertIn("simulated_rpm", wave)
        self.assertIn("simulated_temperature_c", wave)
        self.assertEqual(wave["vital_hp"], 6)

        # 2. MultiMesh buffer
        mesh_buf = self.hypervisor.generator.step_generate_multimesh_buffer(count=8)
        self.assertEqual(mesh_buf["instance_count"], 8)
        self.assertEqual(mesh_buf["byte_size_vulkan"], 8 * 96)
        self.assertEqual(len(mesh_buf["instances"]), 8)
        self.assertEqual(mesh_buf["vital_max_hp"], 6)

    def test_multilock_autocompleter_hypervisor_facade(self):
        """Global autocompleter must seamlessly integrate hypervisor operations."""
        autocomp = MultilockMatrixAutocompleter()
        
        # Test direct DMA trigger
        dma_res = autocomp.trigger_dma_direct_transfer(BackendLayerDomain.PROCEDURAL_SYNTHESIS)
        self.assertEqual(dma_res["status"], "HYPERCALL_SUCCESS")

        # Test predict autocompletions includes hypervisor manifest
        suggestions = autocomp.predict_autocompletions("vytvor mi krystal")
        self.assertIn("hypervisor_manifest", suggestions.multilock_status)
        self.assertGreater(suggestions.multilock_status["hypervisor_manifest"]["uptime_seconds"], -1.0)
        self.assertEqual(suggestions.vital_max_hp, 6)


if __name__ == "__main__":
    unittest.main()
