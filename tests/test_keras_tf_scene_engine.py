#!/usr/bin/env python3
"""
==============================================================================
UNIT TESTS: KERAS/TF SCENE ENGINE, MEMORY BENCHMARK & CORTEX DEDUCTION
==============================================================================
Tests:
  1. KerasSceneModelSpec: Keras 3 / TensorFlow Functional graph structure.
  2. KerasSceneDetector: Multi-channel tensor evaluation & biome classification.
  3. TensorSceneGenerator: Procedural 3D scene synthesis & 25-card deck invariants.
  4. CacheAlignedArenaAllocator: 64-byte alignment, canary integrity, and corruption catch.
  5. LowLevelMemoryBenchmarkSuite: High-throughput memory benchmark & zero corruption.
  6. CortexThreadSwitcher: Processor core affinity evaluation (P-core vs E-core).
  7. Janet S-Expression Specification: cortex_deductive_scheduler.janet structure.
  8. Web Hub REST Handlers: /api/neural/scene, /api/benchmark/memory, /api/cortex.
  9. System Invariant: VITAL_MAX_HP == 6.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import json
import unittest
from pathlib import Path
from io import BytesIO

from krystal_kernel.keras_tf_scene_engine import (
    KerasSceneModelSpec,
    KerasSceneDetector,
    TensorSceneGenerator,
    BIOME_CLASSES,
    VITAL_MAX_HP
)
from krystal_kernel.low_level_memory_benchmark import (
    CacheAlignedArenaAllocator,
    LowLevelMemoryBenchmarkSuite,
    MemoryCorruptionError,
    RingBufferTelemetryQueue
)
from krystal_kernel.cortex_thread_switcher import (
    CortexThreadSwitcher,
    STRATEGIES
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


class TestKerasTFSceneEngine(unittest.TestCase):

    def test_vital_max_hp_invariant(self):
        """Invariant: VITAL_MAX_HP must strictly be 6 across all systems."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_keras_model_specification(self):
        """Verify Keras Functional API model specification schema."""
        cfg = KerasSceneModelSpec.get_keras_model_config((16, 16, 4))
        self.assertEqual(cfg["class_name"], "Functional")
        self.assertEqual(cfg["backend"], "tensorflow")
        layers = cfg["config"]["layers"]
        layer_names = [l["name"] for l in layers]
        self.assertIn("scene_tensor_in", layer_names)
        self.assertIn("conv1", layer_names)
        self.assertIn("biome_head", layer_names)
        self.assertIn("metrics_head", layer_names)

    def test_keras_scene_detector(self):
        """Verify scene detector feature extraction and biome probabilities."""
        detector = KerasSceneDetector(use_arena=True)
        # Create an elevation-heavy tensor
        grid = [[[0.8, 0.1, 0.9, 0.1] for _ in range(16)] for _ in range(16)]
        res = detector.detect_scene_features(grid)

        self.assertEqual(res.vital_max_hp, 6)
        self.assertIn(res.detected_biome, BIOME_CLASSES)
        self.assertGreater(res.biome_confidence, 0.0)
        self.assertLessEqual(res.biome_confidence, 1.0)
        self.assertGreaterEqual(res.gameplay_consistency_score, 0.0)
        self.assertLessEqual(res.gameplay_consistency_score, 1.0)

    def test_tensor_scene_generator_biomes(self):
        """Verify procedural scene generator for biomes with 25 cards & 6 HP invariant."""
        generator = TensorSceneGenerator()
        for biome in ["CRYSTAL_GLACIER", "TOXIC_SILT", "ANCIENT_DRUID", "AETHER_CITADEL"]:
            scene = generator.generate_scene(biome=biome, seed=123, grid_size=16)
            self.assertEqual(scene.biome, biome)
            self.assertEqual(len(scene.heightmap_sfd), 16)
            self.assertEqual(len(scene.heightmap_sfd[0]), 16)
            self.assertGreater(len(scene.tactical_hexes), 0)
            self.assertEqual(scene.deck_preset["total_cards"], 25)
            self.assertEqual(scene.deck_preset["vital_max_hp"], 6)
            self.assertIn("overview", scene.camera_presets)

    def test_low_level_arena_cache_alignment_and_canaries(self):
        """Verify 64-byte cache alignment and front/rear guard canary integrity."""
        arena = CacheAlignedArenaAllocator(capacity_bytes=1024 * 1024)
        self.assertEqual(arena.alignment, 64)

        # Allocate 3 blocks
        blk1 = arena.allocate(128, tag="chunk_1")
        blk2 = arena.allocate(256, tag="chunk_2")
        arena.write_payload(blk1, b"\x11" * 128)
        arena.write_payload(blk2, b"\x22" * 256)

        # Verify reading data back
        self.assertEqual(arena.read_payload(blk1), b"\x11" * 128)
        self.assertEqual(arena.read_payload(blk2), b"\x22" * 256)
        passed, count = arena.verify_all_allocations()
        self.assertTrue(passed)
        self.assertEqual(count, 2)

        # Test intentional canary tampering
        arena._raw_storage[blk1["rear_canary_offset"]] = 0xFF
        with self.assertRaises(MemoryCorruptionError):
            arena.verify_block(blk1)

    def test_low_level_memory_benchmark_suite(self):
        """Verify memory benchmark execution with zero corruptions."""
        res = LowLevelMemoryBenchmarkSuite.run_benchmark(iterations=1000, chunk_size=128)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(res["canary_verification_passed"])
        self.assertEqual(res["corruptions_detected"], 0)
        self.assertEqual(res["memory_integrity_score"], 1.0)
        self.assertGreater(res["bandwidth_mb_s"], 0.0)
        self.assertEqual(res["vital_max_hp"], 6)

    def test_cortex_thread_switcher(self):
        """Verify Cortex thread switcher strategy evaluation and affinity masks."""
        switcher = CortexThreadSwitcher()
        dec = switcher.evaluate_optimal_strategy()
        self.assertEqual(dec.vital_max_hp, 6)
        self.assertIn(dec.current_strategy, STRATEGIES)

        # Override with P-Core Burst
        override_p = switcher.evaluate_optimal_strategy("STRATEGY_ISOLATED_P_CORE_BURST")
        self.assertEqual(override_p.current_strategy, "STRATEGY_ISOLATED_P_CORE_BURST")
        self.assertEqual(override_p.affinity_mask_hex, "0x000F")

        # Override with E-Core Offload
        override_e = switcher.evaluate_optimal_strategy("STRATEGY_COOPERATIVE_E_CORE_THROTTLE")
        self.assertEqual(override_e.current_strategy, "STRATEGY_COOPERATIVE_E_CORE_THROTTLE")
        self.assertEqual(override_e.affinity_mask_hex, "0x00F0")

    def test_janet_dsl_cortex_deductive_scheduler(self):
        """Verify existence and syntax elements of cortex_deductive_scheduler.janet."""
        janet_path = WORKSPACE_ROOT / "krystal_janet" / "cortex_deductive_scheduler.janet"
        self.assertTrue(janet_path.exists(), f"Missing Janet file: {janet_path}")
        content = janet_path.read_text(encoding="utf-8")
        self.assertIn("VITAL-MAX-HP 6", content)
        self.assertIn("kernel/deduce-memory-integrity", content)
        self.assertIn("kernel/deduce-scene-safety", content)
        self.assertIn("cortex/evaluate-telemetry", content)
        self.assertIn("STRATEGY-ISOLATED-P-CORE-BURST", content)
        self.assertIn("STRATEGY-COOPERATIVE-E-CORE-THROTTLE", content)
        self.assertIn("0xDEADBEEFCAFEBAFE", content)

    def test_server_neural_scene_and_memory_routes(self):
        """Verify GET & POST endpoints in KrystalHubHandler for neural scenes & benchmark."""
        from krystal_web_hub.server import KrystalHubHandler

        class DummyServer:
            pass

        class MockHandler(KrystalHubHandler):
            def __init__(self, path, body=b""):
                self.path = path
                self.wfile = BytesIO()
                self.headers = {"Content-Length": str(len(body))}
                self.rfile = BytesIO(body)
                self.server = DummyServer()
                self.client_address = ("127.0.0.1", 8080)
                self.requestline = f"GET {path} HTTP/1.1"
                self.command = "GET"
                self.close_connection = True
            def send_response(self, code, message=None):
                self._code = code
            def send_header(self, keyword, value):
                pass
            def end_headers(self):
                pass

        # 1. Test GET /api/neural/scene/generate
        h_gen = MockHandler("/api/neural/scene/generate?biome=CRYSTAL_GLACIER")
        h_gen.do_GET()
        res_gen = json.loads(h_gen.wfile.getvalue().decode("utf-8"))
        self.assertEqual(res_gen["status"], "OK")
        self.assertEqual(res_gen["vital_max_hp"], 6)
        self.assertEqual(res_gen["scene"]["biome"], "CRYSTAL_GLACIER")

        # 2. Test GET /api/benchmark/memory_safety
        h_mem = MockHandler("/api/benchmark/memory_safety?iterations=500")
        h_mem.do_GET()
        res_mem = json.loads(h_mem.wfile.getvalue().decode("utf-8"))
        self.assertEqual(res_mem["status"], "SUCCESS")
        self.assertEqual(res_mem["corruptions_detected"], 0)
        self.assertEqual(res_mem["vital_max_hp"], 6)

        # 3. Test GET /api/cortex/scheduling_strategy
        h_cor = MockHandler("/api/cortex/scheduling_strategy")
        h_cor.do_GET()
        res_cor = json.loads(h_cor.wfile.getvalue().decode("utf-8"))
        self.assertEqual(res_cor["status"], "OK")
        self.assertEqual(res_cor["vital_max_hp"], 6)
        self.assertIn("decision", res_cor)


if __name__ == "__main__":
    unittest.main()
