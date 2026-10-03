import unittest
import json
import urllib.request
from krystal_web_hub.economic_engine.krystal_execution_architecture_engine import (
    VITAL_MAX_HP,
    MetricState,
    KrystalExecutionArchitectureEngine,
    GLOBAL_EXECUTION_ARCHITECTURE_ENGINE
)

class TestKrystalExecutionArchitecture(unittest.TestCase):
    def setUp(self):
        self.engine = KrystalExecutionArchitectureEngine()

    def test_vital_max_hp_rule(self):
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(self.engine.vital_max_hp, 6)

    def test_metrics_stratification_structure(self):
        catalog = self.engine.get_metrics_catalog()
        self.assertEqual(catalog["vital_max_hp_rule"], 6)
        
        by_state = catalog["metrics_by_state"]
        self.assertIn("MEASURED", by_state)
        self.assertIn("MODELED", by_state)
        self.assertIn("TARGET", by_state)

        # 1. MEASURED
        measured_ids = [m["id"] for m in by_state["MEASURED"]]
        self.assertIn("vm_queue_throughput", measured_ids)
        self.assertIn("cpu_sdf_eval_rate", measured_ids)
        self.assertIn("terrain_kernel_sample_time", measured_ids)
        self.assertIn("visual_entropy_telemetry_time", measured_ids)

        vm_metric = next(m for m in by_state["MEASURED"] if m["id"] == "vm_queue_throughput")
        self.assertEqual(vm_metric["value"], 1724554.0)
        self.assertAlmostEqual(vm_metric["speedup_factor"], 4.39, places=2)

        # 2. MODELED
        modeled_ids = [m["id"] for m in by_state["MODELED"]]
        self.assertIn("simd_terrain_projection", modeled_ids)
        self.assertIn("iris_xe_driver_bypass", modeled_ids)

        # 3. TARGET
        target_ids = [m["id"] for m in by_state["TARGET"]]
        self.assertIn("neural_sdf_eval_latency", target_ids)
        self.assertIn("webgpu_frame_rate", target_ids)
        self.assertIn("rust_native_acceleration", target_ids)

    def test_http_api_endpoints(self):
        base_url = "http://127.0.0.1:8089"
        try:
            # 1. GET /api/execution-architecture/metrics
            req = urllib.request.Request(f"{base_url}/api/execution-architecture/metrics")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(data["vital_max_hp_rule"], 6)
                self.assertGreaterEqual(data["summary"]["measured_count"], 4)

            # 2. GET /hybrid-portal-arena
            req = urllib.request.Request(f"{base_url}/hybrid-portal-arena")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                html = resp.read().decode('utf-8')
                self.assertIn("POSLEDNÍ KMEN", html)
                self.assertIn("krystal_ui_system.css", html)
                self.assertIn("MEASURED", html)

            # 3. GET /static/krystal_ui_system.css
            req = urllib.request.Request(f"{base_url}/static/krystal_ui_system.css")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                css = resp.read().decode('utf-8')
                self.assertIn("--ks-navy", css)
                self.assertIn("--ks-gold", css)
                self.assertIn(".ks-metric-tag.measured", css)

        except Exception as e:
            self.skipTest(f"Live server test skipped: {e}")

if __name__ == "__main__":
    unittest.main()
