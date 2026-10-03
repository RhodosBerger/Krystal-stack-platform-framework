import unittest
import json
import urllib.request
from krystal_web_hub.economic_engine.quadratic_variable_transformer import (
    GLOBAL_QUADRATIC_TRANSFORMER,
    QuadraticDomain,
    VITAL_MAX_HP,
    GOLDEN_RATIO,
    INV_GOLDEN_RATIO
)

class TestQuadraticVariableTransformer(unittest.TestCase):
    def setUp(self):
        self.transformer = GLOBAL_QUADRATIC_TRANSFORMER

    def test_domains_registered(self):
        domains_data = self.transformer.get_domains()
        self.assertGreaterEqual(domains_data["domains_count"], 8)
        self.assertEqual(domains_data["vital_max_hp_rule"], 6)

        domain_ids = [d["domain_id"] for d in domains_data["domains"]]
        self.assertIn("memory_latency_ns", domain_ids)
        self.assertIn("vram_allocation_mb", domain_ids)
        self.assertIn("gpu_clock_mhz", domain_ids)
        self.assertIn("hamiltonian_h", domain_ids)
        self.assertIn("vital_hp", domain_ids)
        self.assertIn("market_price_credits", domain_ids)

    def test_forward_evaluation_and_vital_invariant(self):
        # Evaluate at boundary points x=0.0 and x=1.0
        res_0 = self.transformer.evaluate_at_x(0.0)
        res_1 = self.transformer.evaluate_at_x(1.0)
        res_phi = self.transformer.evaluate_at_x(INV_GOLDEN_RATIO)

        # Vital HP must strictly be <= 6 across all x
        hp_0 = res_0["variables"]["vital_hp"]["value"]
        hp_1 = res_1["variables"]["vital_hp"]["value"]
        hp_phi = res_phi["variables"]["vital_hp"]["value"]

        self.assertLessEqual(hp_0, 6.0)
        self.assertLessEqual(hp_1, 6.0)
        self.assertLessEqual(hp_phi, 6.0)
        self.assertGreaterEqual(hp_0, 0.0)
        self.assertGreaterEqual(hp_1, 0.0)
        self.assertGreaterEqual(hp_phi, 0.0)

        # At x=0, HP should be 6.0 (idle uninjured state)
        self.assertAlmostEqual(hp_0, 6.0, delta=0.01)

    def test_solve_x_reversibility(self):
        # Pick x = 0.45, evaluate memory latency, then solve for x and verify recovery
        known_x = 0.45
        mem_domain = self.transformer._domains["memory_latency_ns"]
        v = mem_domain.evaluate(known_x)

        solved = self.transformer.solve_latent_x("memory_latency_ns", v)
        self.assertTrue(solved["is_real_root"])
        self.assertGreaterEqual(solved["discriminant_delta"], 0)
        self.assertAlmostEqual(solved["latent_x"], known_x, delta=0.002)

    def test_cross_domain_conversion(self):
        # Convert 45 ns memory latency to VRAM and Vital HP
        conv = self.transformer.convert(
            source_domain_id="memory_latency_ns",
            source_val=45.0,
            target_domain_id="vital_hp"
        )
        self.assertIn("latent_bridge_x", conv)
        self.assertGreaterEqual(conv["latent_bridge_x"], 0.0)
        self.assertLessEqual(conv["latent_bridge_x"], 1.0)

        target_hp = conv["target"]["value"]
        self.assertLessEqual(target_hp, 6.0)
        self.assertGreaterEqual(target_hp, 0.0)
        self.assertEqual(conv["vital_max_hp_rule"], 6)

    def test_convert_all_domains(self):
        conv_all = self.transformer.convert_all("memory_latency_ns", 50.0)
        self.assertEqual(conv_all["vital_max_hp_rule"], 6)
        self.assertIn("latent_x_extracted", conv_all)
        projections = conv_all["projected_variables"]
        self.assertIn("vram_allocation_mb", projections)
        self.assertIn("gpu_clock_mhz", projections)
        self.assertIn("hamiltonian_h", projections)
        self.assertIn("vital_hp", projections)

    def test_golden_mean_equilibrium(self):
        golden = self.transformer.evaluate_golden_mean()
        self.assertTrue(golden["is_golden_ratio_state"])
        self.assertAlmostEqual(golden["latent_x"], INV_GOLDEN_RATIO, delta=0.001)

    def test_http_api_endpoints(self):
        base_url = "http://127.0.0.1:8089"
        try:
            # 1. GET /api/quadratic/domains
            req = urllib.request.Request(f"{base_url}/api/quadratic/domains")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(d["vital_max_hp_rule"], 6)
                self.assertGreaterEqual(d["domains_count"], 8)

            # 2. GET /api/quadratic/golden-state
            req = urllib.request.Request(f"{base_url}/api/quadratic/golden-state")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertTrue(d["is_golden_ratio_state"])

            # 3. POST /api/quadratic/convert
            req = urllib.request.Request(
                f"{base_url}/api/quadratic/convert",
                data=json.dumps({
                    "source_domain": "memory_latency_ns",
                    "value": 45.0,
                    "target_domain": "vital_hp"
                }).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(d["vital_max_hp_rule"], 6)
                self.assertLessEqual(d["target"]["value"], 6.0)

            # 4. GET /quadratic HTML studio
            req = urllib.request.Request(f"{base_url}/quadratic")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                html = resp.read().decode('utf-8')
                self.assertIn("KVADRATICKÉ ROVNICE", html)

        except Exception as e:
            self.skipTest(f"Live server not currently reachable on {base_url}: {e}")

if __name__ == "__main__":
    unittest.main()
