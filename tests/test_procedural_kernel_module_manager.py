#!/usr/bin/env python3
"""
Unit tests for Procedural Kernel Module Manager, Conduct Code Bridge,
and Interactive Processing Command Mindmap (Matrix-to-Statistical Model Bridge).
"""

import os
import sys
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.procedural_kernel_module_manager import (
    KernelModuleRecord,
    MindmapNode,
    MindmapEdge,
    ProcessingCommandMindmap,
    ProceduralKernelModuleManager,
    execute_agent_pipeline_with_mindmap,
    GLOBAL_KERNEL_MODULE_MANAGER
)


class TestProceduralKernelModuleManager(unittest.TestCase):
    """Test suite for Kernel Module Manager, Conduct Code Bridge & Mindmap."""

    def setUp(self):
        self.manager = ProceduralKernelModuleManager()

    def test_default_modules_and_vital_invariant(self):
        """Verify default modules exist and satisfy VITAL_MAX_HP == 6."""
        modules = self.manager.list_modules()
        self.assertGreaterEqual(len(modules), 8)

        # Check default IDs exist including Node 7 and Node 8
        mod_ids = [m["module_id"] for m in modules]
        self.assertIn("CNC_THERMAL_GOVERNOR", mod_ids)
        self.assertIn("TENSOR_MATH_OBSERVER", mod_ids)
        self.assertIn("LINGUISTIC_PREDICTOR", mod_ids)
        self.assertIn("IMAGE_RASTER_DIRECTOR", mod_ids)
        self.assertIn("MULTIMESH_VULKAN_STREAMER", mod_ids)
        self.assertIn("SENSE_CONFIRMATION_FLOW", mod_ids)
        self.assertIn("HARDWARE_SAFETY_INTERLOCK", mod_ids)
        self.assertIn("FLEET_TELEMETRY_AGGREGATOR", mod_ids)

        for m in modules:
            self.assertEqual(m["vital_hp"], VITAL_MAX_HP)
            self.assertTrue(m["vital_intact"])
            self.assertIn("conduct_flags", m)
            self.assertIn("conduct_badge", m)
            self.assertIn("throughput_tokens_per_sec", m)
            self.assertIn("context_window_size", m)
            # Verify badge contains all 5 Conduct Code characters
            badge = m["conduct_badge"]
            self.assertTrue("[ACT]" in badge or "[!ACT]" in badge)
            self.assertTrue("[LCK]" in badge or "[!LCK]" in badge)
            self.assertTrue("[CNF]" in badge or "[!CNF]" in badge)
            self.assertTrue("[HAZ]" in badge or "[!HAZ]" in badge)
            self.assertIn(f"[VIT {VITAL_MAX_HP}HP]", badge)

    def test_crud_operations(self):
        """Test full Create, Read, Update, Delete lifecycle of kernel modules."""
        # 1. CREATE
        new_mod = self.manager.create_module(
            module_id="TEST_QUANTUM_DISPATCHER",
            name="Quantum State Dispatcher",
            role="QUANTUM_MODULE",
            inputs={"max_throughput_tokens_sec": 512.0, "context_window": 4096},
            outputs={"output_dim": 16, "format": "TENSOR_F32"},
            matrix_coupling="C_unified",
            statistical_target="W_usage"
        )
        self.assertEqual(new_mod.module_id, "TEST_QUANTUM_DISPATCHER")
        self.assertEqual(new_mod.vital_hp, 6)
        self.assertTrue(new_mod.is_active)
        self.assertFalse(new_mod.is_locked)

        # 2. READ
        retrieved = self.manager.get_module("TEST_QUANTUM_DISPATCHER")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Quantum State Dispatcher")
        self.assertEqual(retrieved.throughput_tokens_per_sec, 512.0)

        # 3. UPDATE
        updated = self.manager.update_module(
            "TEST_QUANTUM_DISPATCHER",
            name="Quantum State Dispatcher (Overclocked)",
            is_locked=True
        )
        self.assertIsNotNone(updated)
        self.assertEqual(updated.name, "Quantum State Dispatcher (Overclocked)")
        self.assertTrue(updated.is_locked)

        # 4. DELETE
        deleted = self.manager.delete_module("TEST_QUANTUM_DISPATCHER")
        self.assertTrue(deleted)
        self.assertIsNone(self.manager.get_module("TEST_QUANTUM_DISPATCHER"))

    def test_boolean_conduct_flags_and_toggles(self):
        """Verify conduct code boolean character toggles."""
        mod_id = "CNC_THERMAL_GOVERNOR"
        initial = self.manager.get_module(mod_id)
        self.assertIsNotNone(initial)

        # Toggle is_locked
        prev_lock = initial.is_locked
        updated = self.manager.toggle_conduct_flag(mod_id, "is_locked")
        self.assertEqual(updated.is_locked, not prev_lock)

        # Toggle back
        restored = self.manager.toggle_conduct_flag(mod_id, "is_locked")
        self.assertEqual(restored.is_locked, prev_lock)

        # Toggle is_hazardous
        haz = self.manager.toggle_conduct_flag(mod_id, "is_hazardous")
        self.assertTrue(haz.is_hazardous)
        # Restore safe
        safe = self.manager.toggle_conduct_flag(mod_id, "is_hazardous")
        self.assertFalse(safe.is_hazardous)

    def test_system_telemetry(self):
        """Verify system telemetry calculations."""
        telemetry = self.manager.get_system_telemetry()
        self.assertEqual(telemetry["vital_max_hp"], VITAL_MAX_HP)
        self.assertGreaterEqual(telemetry["total_modules"], 8)
        self.assertGreater(telemetry["aggregate_throughput_tok_sec"], 0.0)
        self.assertIn("active_modules", telemetry)

    def test_agent_pipeline_execution_and_mindmap(self):
        """Verify agent pipeline execution synthesizing command mindmap bridging matrix & stats."""
        prompt = "vytvor proceduralny ostrov s gotickou vezou a riekou"
        result = execute_agent_pipeline_with_mindmap(prompt)

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["vital_max_hp"], VITAL_MAX_HP)
        self.assertIn("pipeline_id", result)
        self.assertIn("coherence_score", result)
        self.assertIn("frobenius_norm_ratio", result)
        self.assertIn("dominant_domain", result)
        self.assertGreater(result["engine_asset_count"], 0)
        self.assertLess(result["execution_latency_ms"], 25.0)  # Ultra-fast local execution

        # Verify mindmap graph structure (8 nodes and 10 edges)
        mindmap = result["mindmap"]
        self.assertEqual(mindmap["prompt"], prompt)
        nodes = mindmap["nodes"]
        edges = mindmap["edges"]

        self.assertEqual(len(nodes), 8)
        self.assertEqual(len(edges), 10)

        node_types = [n["type"] for n in nodes]
        expected_types = [
            "INPUT_COMMAND",
            "LINGUISTIC_TENSOR",
            "MULTILOCK_MATRICES",
            "STATISTICAL_PREDICTION",
            "HARDWARE_SAFETY_INTERLOCK",
            "PROCEDURAL_ASSETS",
            "VISUAL_RASTERIZER",
            "FLEET_TELEMETRY_AGGREGATOR"
        ]
        for exp in expected_types:
            self.assertIn(exp, node_types)

        # Verify each node maintains VITAL_MAX_HP == 6 and conduct flags
        for n in nodes:
            self.assertEqual(n["vital_hp"], VITAL_MAX_HP)
            self.assertIn("conduct_flags", n)
            self.assertIn("conduct_badge", n)
            self.assertTrue(n["conduct_flags"]["vital_intact"])

        # Verify Edge connections across the full 8-node DAG
        edge_labels = [e["label"] for e in edges]
        self.assertIn("COHERENCE_PROJECT", edge_labels)
        self.assertIn("MULTILOCK_ACQUIRE", edge_labels)
        self.assertIn("SOFTMAX_PREDICTION", edge_labels)
        self.assertIn("FAIL_SAFE_SUPERVISION", edge_labels)
        self.assertIn("INTERLOCK_AUTHORIZE", edge_labels)
        self.assertIn("ASSET_SYNTHESIS", edge_labels)
        self.assertIn("PIXEL_RASTERIZE", edge_labels)
        self.assertIn("RESONANCE_BRIDGE", edge_labels)
        self.assertIn("SAFETY_TELEMETRY_STREAM", edge_labels)
        self.assertIn("TAYLOR_ARRHENIUS_EXPORT", edge_labels)

        # Verify statistical and matrix couplings in summary
        stats = result["statistical_model_summary"]
        self.assertIn("prediction_distribution", stats)
        self.assertIn("dominant_cluster", stats)
        self.assertIn("temperature_scaling", stats)

        matrices = result["matrix_connections_summary"]
        self.assertIn("a_discrete_frobenius", matrices)
        self.assertIn("c_physical_frobenius", matrices)
        self.assertIn("spectral_radius_bound", matrices)
        self.assertIn("frobenius_coupling_ratio", matrices)

    def test_cnc_machining_pipeline_mindmap(self):
        """Verify pipeline with CNC machining prompt correctly maps to CNC domain."""
        prompt = "vygeneruj adaptivny CNC cyklus s chladenim pre kalenu ocel 1018"
        result = execute_agent_pipeline_with_mindmap(prompt)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["dominant_domain"], "CNC_MACHINING")
        self.assertGreater(result["coherence_score"], 0.70)
        self.assertEqual(result["vital_max_hp"], 6)

    def test_hardware_safety_interlock_fail_safe_trigger(self):
        """
        Verify Node 7 HARDWARE_SAFETY_INTERLOCK trips fail-safe upon fault:
        - Sets is_hazardous = True, is_active = False, is_locked = True
        - Triggers rescue cycle: Z-retract +4 mm, M07 coolant switch
        - Degrades HP (vital_hp = 4) while respecting VITAL_MAX_HP == 6 platform rule
        - Deactivates non-essential synthesis edges while keeping safety telemetry stream active
        """
        prompt = "vysokorychlostne frezovanie titanu Ti-6Al-4V"
        result = execute_agent_pipeline_with_mindmap(prompt, hardware_safety_fault=True)

        mindmap = result["mindmap"]
        nodes_by_id = {n["id"]: n for n in mindmap["nodes"]}
        edges_by_label = {e["label"]: e for e in mindmap["edges"]}

        # Check Hardware Safety Interlock node
        hw_node = nodes_by_id["node_hardware_interlock"]
        self.assertEqual(hw_node["status"], "BLOCKED")
        self.assertFalse(hw_node["conduct_flags"]["is_active"])
        self.assertTrue(hw_node["conduct_flags"]["is_locked"])
        self.assertFalse(hw_node["conduct_flags"]["is_confirmed"])
        self.assertTrue(hw_node["conduct_flags"]["is_hazardous"])
        self.assertFalse(hw_node["conduct_flags"]["vital_intact"])
        self.assertEqual(hw_node["vital_hp"], 4)
        self.assertEqual(hw_node["conduct_badge"], "[!ACT] [LCK] [!CNF] [HAZ] [VIT 4HP]")

        # Check safety hardware metrics payload
        hw_metrics = hw_node["metrics"]
        self.assertFalse(hw_metrics["limit_switches_cleared"])
        self.assertTrue(hw_metrics["e_stop_engaged"])
        self.assertEqual(hw_metrics["emulsion_pressure_bar"], 0.0)
        self.assertEqual(hw_metrics["rescue_action"], "Z_RETRACT_4MM_M07")
        self.assertTrue(hw_metrics["rescue_cycle_active"])
        self.assertEqual(hw_metrics["binary_protocol"], "ISO_13849_PLd")

        # Verify edge authorization gating
        self.assertFalse(edges_by_label["INTERLOCK_AUTHORIZE"]["flow_active"])
        self.assertFalse(edges_by_label["ASSET_SYNTHESIS"]["flow_active"])
        self.assertFalse(edges_by_label["PIXEL_RASTERIZE"]["flow_active"])
        self.assertFalse(edges_by_label["RESONANCE_BRIDGE"]["flow_active"])
        self.assertFalse(edges_by_label["TAYLOR_ARRHENIUS_EXPORT"]["flow_active"])

        # Safety telemetry must remain active to alert cloud fleet bridge
        self.assertTrue(edges_by_label["SAFETY_TELEMETRY_STREAM"]["flow_active"])

    def test_fleet_telemetry_aggregator_metrics_and_rul(self):
        """
        Verify Node 8 FLEET_TELEMETRY_AGGREGATOR:
        - B2B multi-spindle cloud tier ($1,200/mo)
        - Non-blocking async I/O on E-cores
        - Aggregated Frobenius norm across fleet
        - Taylor-Arrhenius tool RUL prediction
        """
        prompt = "optimalizuj rezne podmienky pre vreteno 3"
        result = execute_agent_pipeline_with_mindmap(prompt)

        mindmap = result["mindmap"]
        nodes_by_id = {n["id"]: n for n in mindmap["nodes"]}

        fleet_node = nodes_by_id["node_fleet_telemetry"]
        self.assertEqual(fleet_node["status"], "ACTIVE")
        self.assertTrue(fleet_node["conduct_flags"]["is_active"])
        self.assertFalse(fleet_node["conduct_flags"]["is_hazardous"])
        self.assertEqual(fleet_node["vital_hp"], VITAL_MAX_HP)
        self.assertEqual(fleet_node["conduct_badge"], f"[ACT] [!LCK] [CNF] [!HAZ] [VIT {VITAL_MAX_HP}HP]")

        metrics = fleet_node["metrics"]
        self.assertEqual(metrics["monetization_tier"], "$1,200/mo OEM Fleet Bridge")
        self.assertGreaterEqual(metrics["fleet_spindles_online"], 16)
        self.assertGreater(metrics["fleet_frobenius_norm"], 0.0)
        self.assertGreater(metrics["taylor_arrhenius_rul_hours"], 0.0)
        self.assertEqual(metrics["async_io_affinity"], "E_CORES_ISOLATED")

    def test_dag_acyclic_graph_validation_and_sub_5ms_latency(self):
        """
        Verify that the 8-node, 10-edge processing graph is strictly an Acyclic Directed Graph (DAG)
        via Kahn's topological sort algorithm, and validates synthesis execution latency under 5 ms.
        """
        import time

        result = execute_agent_pipeline_with_mindmap("analyzuj DAG stabilitu grafu")
        nodes = result["mindmap"]["nodes"]
        edges = result["mindmap"]["edges"]

        node_ids = set(n["id"] for n in nodes)
        self.assertEqual(len(node_ids), 8)

        # Build in-degree map and adjacency list for Kahn's algorithm
        in_degree = {nid: 0 for nid in node_ids}
        adj = {nid: [] for nid in node_ids}

        for edge in edges:
            src = edge["source"]
            tgt = edge["target"]
            self.assertIn(src, node_ids, f"Invalid source node {src}")
            self.assertIn(tgt, node_ids, f"Invalid target node {tgt}")
            adj[src].append(tgt)
            in_degree[tgt] += 1

        # Kahn's algorithm
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        visited_count = 0

        while queue:
            curr = queue.pop(0)
            visited_count += 1
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If visited_count == len(node_ids), graph is strictly acyclic (DAG)
        self.assertEqual(
            visited_count,
            len(node_ids),
            f"Cycle detected in DAG! Visited {visited_count} of {len(node_ids)} nodes."
        )

        # Benchmark 50 iterations for sub-5ms latency test
        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            execute_agent_pipeline_with_mindmap("vysokorychlostny test latencie")
            latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_latency = sum(latencies) / len(latencies)
        self.assertLess(avg_latency, 5.0, f"Average latency {avg_latency:.2f}ms exceeds 5ms threshold!")


if __name__ == "__main__":
    unittest.main()

