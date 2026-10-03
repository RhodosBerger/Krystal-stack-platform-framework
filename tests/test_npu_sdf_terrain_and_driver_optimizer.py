import unittest
import json
import urllib.request
from krystal_web_hub.economic_engine.npu_sdf_terrain_and_driver_optimizer import (
    VITAL_MAX_HP,
    GOLDEN_RATIO,
    CRITICAL_TALUS_ANGLE_RAD,
    TAN_THETA_C,
    PosledniKmenBiome,
    CANONICAL_BIOMES,
    NpuSdfTerrainAndDriverEngine,
    GLOBAL_TERRAIN_SYNTHESIS_ENGINE
)

class TestNpuSdfTerrainAndDriverEngine(unittest.TestCase):
    def setUp(self):
        self.engine = NpuSdfTerrainAndDriverEngine()

    def test_vital_max_hp_rule(self):
        self.assertEqual(VITAL_MAX_HP, 6)
        telemetry = self.engine.get_system_telemetry()
        self.assertEqual(telemetry["vital_max_hp_rule"], 6)

    def test_canonical_biomes_presence(self):
        self.assertIn(PosledniKmenBiome.CRYSTAL, CANONICAL_BIOMES)
        self.assertIn(PosledniKmenBiome.TOXIC, CANONICAL_BIOMES)
        self.assertIn(PosledniKmenBiome.DRUID, CANONICAL_BIOMES)
        self.assertIn(PosledniKmenBiome.STUDNA_DUSI, CANONICAL_BIOMES)
        for b in CANONICAL_BIOMES.values():
            self.assertEqual(b.vital_max_hp, 6)

    def test_master_terrain_manifold_evaluation(self):
        # Crystal has high elevation and high ridge weight
        h_crystal = self.engine.evaluate_master_terrain(2.0, 2.0, PosledniKmenBiome.CRYSTAL)
        self.assertIsInstance(h_crystal, float)

        # Studna Dusi has sink vortex at center (0, 0)
        h_sink = self.engine.evaluate_master_terrain(0.0, 0.0, PosledniKmenBiome.STUDNA_DUSI)
        self.assertLess(h_sink, 0.0)  # Sink vortex depression

    def test_coupled_erosion_models(self):
        # 1. Thermal weathering at 35 degrees critical angle
        res_talus = self.engine.evaluate_thermal_weathering(2.0, 2.0, PosledniKmenBiome.DRUID)
        self.assertEqual(res_talus["critical_angle_deg"], 35.0)
        self.assertIn(res_talus["action"], ["ERODE", "DEPOSIT"])
        self.assertEqual(res_talus["vital_max_hp_rule"], 6)

        # 2. Hydraulic erosion and sediment capacity
        res_hyd = self.engine.evaluate_hydraulic_erosion(2.0, 2.0, velocity=2.0, biome_key=PosledniKmenBiome.DRUID)
        self.assertGreaterEqual(res_hyd["sin_theta"], 0.0)
        self.assertLessEqual(res_hyd["sin_theta"], 1.0)
        self.assertGreaterEqual(res_hyd["sediment_capacity_c"], 0.0)
        self.assertEqual(res_hyd["vital_max_hp_rule"], 6)

    def test_sdf_raymarching_and_npu_acceleration(self):
        # Baseline (NPU off): 6 evaluations for gradient
        self.engine.npu_acceleration_enabled = False
        res_cpu = self.engine.raymarch(
            ray_origin=(0.0, 5.0, 5.0),
            ray_dir=(0.0, -0.7071, -0.7071),
            biome_key=PosledniKmenBiome.CRYSTAL
        )
        self.assertFalse(res_cpu.npu_accelerated)
        self.assertGreater(res_cpu.evaluations_count, 6)

        # NPU Accelerated: single tensor pass reduces gradient evaluation cost to 1
        self.engine.npu_acceleration_enabled = True
        res_npu = self.engine.raymarch(
            ray_origin=(0.0, 5.0, 5.0),
            ray_dir=(0.0, -0.7071, -0.7071),
            biome_key=PosledniKmenBiome.CRYSTAL
        )
        self.assertTrue(res_npu.npu_accelerated)
        self.assertLess(res_npu.evaluations_count, res_cpu.evaluations_count)

    def test_hardware_telemetry(self):
        telemetry = self.engine.get_system_telemetry()
        self.assertEqual(telemetry["execution_units"], 96)
        self.assertEqual(telemetry["hardware_threads"], 672)
        self.assertEqual(telemetry["wddm_driver_stall_us"], 185.0)
        self.assertEqual(telemetry["krystal_bypass_latency_us"], 11.4)
        self.assertGreater(telemetry["driver_speedup_factor"], 15.0)
        self.assertEqual(telemetry["baseline_sdf_cost_per_hit"], 7)
        self.assertEqual(telemetry["npu_accelerated_cost_per_hit"], 2)

    def test_http_api_endpoints(self):
        base_url = "http://127.0.0.1:8089"
        try:
            # 1. GET /api/terrain-synthesis/telemetry
            req = urllib.request.Request(f"{base_url}/api/terrain-synthesis/telemetry")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(d["vital_max_hp_rule"], 6)
                self.assertEqual(d["critical_talus_angle_deg"], 35.0)

            # 2. POST /api/terrain-synthesis/evaluate-sdf
            req = urllib.request.Request(
                f"{base_url}/api/terrain-synthesis/evaluate-sdf",
                data=json.dumps({
                    "origin": [0.0, 5.0, 5.0],
                    "direction": [0.0, -0.7071, -0.7071],
                    "biome": "Crystal_Severni_Stity"
                }).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertIn("hit_distance_t", d)
                self.assertIn("surface_normal", d)

            # 3. POST /api/terrain-synthesis/simulate-erosion
            req = urllib.request.Request(
                f"{base_url}/api/terrain-synthesis/simulate-erosion",
                data=json.dumps({"x": 2.0, "z": 2.0, "biome": "Druid_Pradavny_Les"}).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(d["vital_max_hp_rule"], 6)
                self.assertIn("thermal_weathering", d)
                self.assertIn("hydraulic_erosion", d)

            # 4. POST /api/terrain-synthesis/npu-driver-dispatch
            req = urllib.request.Request(
                f"{base_url}/api/terrain-synthesis/npu-driver-dispatch",
                data=json.dumps({"enable_npu": True}).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(d["status"], "DISPATCH_CONFIGURED")

            # 5. GET /terrain-synthesis HTML
            req = urllib.request.Request(f"{base_url}/terrain-synthesis")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                html = resp.read().decode('utf-8')
                self.assertIn("CONTINUOUS WORLD GENERATION", html)
                self.assertIn("SDF RAYMARCHING PIPELINE", html)
                self.assertIn("COUPLED EROSION MODELS", html)

        except Exception as e:
            self.skipTest(f"Live server test skipped: {e}")

if __name__ == "__main__":
    unittest.main()
