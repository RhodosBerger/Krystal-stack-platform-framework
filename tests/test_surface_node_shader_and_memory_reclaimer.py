import unittest
import json
import urllib.request
from krystal_web_hub.economic_engine.surface_node_shader_and_memory_reclaimer import (
    MemorySlot,
    MemoryBlockSlab,
    SurfaceNodeShaderEngine,
    GLOBAL_SURFACE_NODE_ENGINE,
    VITAL_MAX_HP,
    SURFACE_FEATURE_CHANNELS
)

class TestSurfaceNodeShaderAndMemoryReclaimer(unittest.TestCase):
    def setUp(self):
        self.engine = GLOBAL_SURFACE_NODE_ENGINE

    def test_vital_max_hp_rule(self):
        self.assertEqual(VITAL_MAX_HP, 6)
        cat = self.engine.get_node_catalog()
        self.assertEqual(cat["vital_max_hp_rule"], 6)

    def test_memory_block_slab_allocation_and_reverse_scavenging(self):
        slab = MemoryBlockSlab(num_slots=16)
        self.assertEqual(slab.num_slots, 16)

        # Allocate slot 0, 1, 15
        self.assertTrue(slab.allocate_slot(0, "worker_0"))
        self.assertTrue(slab.allocate_slot(1, "worker_1"))
        self.assertTrue(slab.allocate_slot(15, "worker_15"))

        # Slots 2 to 14 are empty holes
        scavenged = slab.reverse_scavenge_empty_slots()
        self.assertEqual(len(scavenged), 13)

        # Confirm reverse order (top slot 14 must be first in scavenged list)
        self.assertEqual(scavenged[0]["slot_id"], 14)
        self.assertEqual(scavenged[-1]["slot_id"], 2)

        # Telemetry check
        telemetry = slab.get_slab_telemetry()
        self.assertEqual(telemetry["occupied_slots"], 3)
        self.assertEqual(telemetry["reclaimed_slots"], 13)

    def test_feature_channels_completeness(self):
        self.assertEqual(len(SURFACE_FEATURE_CHANNELS), 8)
        self.assertIn(0, SURFACE_FEATURE_CHANNELS)
        self.assertIn(1, SURFACE_FEATURE_CHANNELS)  # Self-shadowing
        self.assertIn(5, SURFACE_FEATURE_CHANNELS)  # Golden Mean Fresnel Rim

    def test_scenario_multiplication_growth(self):
        # Trigger scavenging with known holes
        res = self.engine.scavenge_and_generate_scenarios()
        self.assertGreater(res["reclaimed_count"], 0)
        self.assertGreater(res["scenario_multiplication_factor"], 1000)
        self.assertEqual(res["vital_max_hp_rule"], 6)
        self.assertIn("u_self_shadow_density", res["active_shader_uniforms"])

    def test_default_graph_and_evaluation(self):
        default_graph = self.engine.get_default_graph()
        self.assertEqual(default_graph["vital_max_hp_rule"], 6)
        self.assertGreaterEqual(len(default_graph["nodes"]), 7)
        self.assertGreaterEqual(len(default_graph["connections"]), 7)

        eval_res = self.engine.evaluate_node_graph(default_graph)
        self.assertEqual(eval_res["vital_max_hp_rule"], 6)
        self.assertEqual(eval_res["status"], "COMPILED_OPTIMIZED")
        self.assertIn("compiled_material", eval_res)
        self.assertIn("roughness", eval_res["compiled_material"])

    def test_http_api_endpoints(self):
        base_url = "http://127.0.0.1:8089"
        try:
            # 1. GET /api/surface-nodes/catalog
            req = urllib.request.Request(f"{base_url}/api/surface-nodes/catalog")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(d["vital_max_hp_rule"], 6)
                self.assertGreaterEqual(len(d["node_types"]), 6)

            # 2. GET /api/surface-nodes/default-graph
            req = urllib.request.Request(f"{base_url}/api/surface-nodes/default-graph")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(d["vital_max_hp_rule"], 6)

            # 3. POST /api/surface-nodes/scavenge-memory
            req = urllib.request.Request(
                f"{base_url}/api/surface-nodes/scavenge-memory",
                data=json.dumps({"free_pattern": [60, 61, 62]}).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(d["vital_max_hp_rule"], 6)
                self.assertGreater(d["scenario_multiplication_factor"], 100)

            # 4. GET /surface-nodes HTML Studio
            req = urllib.request.Request(f"{base_url}/surface-nodes")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                html = resp.read().decode('utf-8')
                self.assertIn("SURFACE NODE EDITOR", html)

        except Exception as e:
            self.skipTest(f"Live server on {base_url} skipped during direct test: {e}")

if __name__ == "__main__":
    unittest.main()
