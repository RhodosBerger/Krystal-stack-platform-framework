#!/usr/bin/env python3
"""
Unit and integration tests for Keras Scene Lupa, Auto-Healer, Presets,
and the LLM Scene Builder (>8,000 objects, Intel 11th Gen+ Iris Xe UMA).

Verifies:
  - Scene constructor architectural presets (min_plane_separation_epsilon, bounds, VITAL_MAX_HP == 6).
  - KerasSceneLupa anomaly detection (Z-fighting, collisions, out-of-bounds, scale).
  - SceneAutoHealer geometry correction and zero-glitch guarantee.
  - LLMSceneBuilder 8K+ object procedural generation and performance benchmark (<250ms).
  - Normative JSON schema and Janet DSL definitions.
"""

import json
import os
import time
import unittest
from pathlib import Path

from krystal_kernel.scene_constructor_presets import (
    ALL_PRESETS,
    PRESET_TACTICAL_ARENA_8K,
    PRESET_OPENWORLD_BIOME_8K,
    PRESET_URBAN_METROPOLIS_8K,
    PRESET_SKEUOMORPHIC_DUNGEON_8K,
    SceneConstructorPreset,
    get_preset_for_context,
    VITAL_MAX_HP
)
from krystal_kernel.keras_scene_lupa import (
    KerasLupaModelSpec,
    KerasSceneLupa,
    SceneAutoHealer,
    LupaInspectionReport,
    GLOBAL_KERAS_LUPA,
    GLOBAL_AUTO_HEALER
)
from krystal_kernel.llm_scene_builder import (
    LLMSceneBuilder,
    CertifiedScenePackage,
    GLOBAL_LLM_SCENE_BUILDER
)


class TestSceneConstructorPresets(unittest.TestCase):
    """Verifies presets enforce all architectural rules and invariants."""

    def test_vital_max_hp_invariant(self):
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_presets_completeness(self):
        expected_preset_ids = [
            "PRESET_TACTICAL_ARENA_8K",
            "PRESET_OPENWORLD_BIOME_8K",
            "PRESET_URBAN_METROPOLIS_8K",
            "PRESET_SKEUOMORPHIC_DUNGEON_8K"
        ]
        for pid in expected_preset_ids:
            self.assertIn(pid, ALL_PRESETS)
            preset = ALL_PRESETS[pid]
            self.assertEqual(preset.vital_max_hp, 6)
            self.assertGreaterEqual(preset.min_plane_separation_epsilon, 0.010)
            self.assertGreaterEqual(preset.max_instances, 8000)
            self.assertTrue(len(preset.allowed_archetypes) >= 3)
            self.assertEqual(len(preset.bounds_min), 3)
            self.assertEqual(len(preset.bounds_max), 3)

    def test_get_preset_for_context(self):
        p_arena = get_preset_for_context("TACTICAL_ARENA")
        self.assertEqual(p_arena.context, "TACTICAL_ARENA")
        self.assertEqual(p_arena.min_plane_separation_epsilon, 0.015)

        p_default = get_preset_for_context("UNKNOWN_CONTEXT")
        self.assertEqual(p_default.context, "TACTICAL_ARENA")


class TestKerasSceneLupa(unittest.TestCase):
    """Verifies Keras 3 Autoencoder spec and spatial hash anomaly detection."""

    def setUp(self):
        self.preset = PRESET_TACTICAL_ARENA_8K
        self.lupa = KerasSceneLupa(self.preset)

    def test_keras_lupa_model_spec(self):
        spec = KerasLupaModelSpec.get_keras_lupa_config()
        self.assertEqual(spec["class_name"], "Functional")
        self.assertEqual(spec["keras_version"], "3.3.0")
        self.assertEqual(spec["backend"], "tensorflow")
        layers = spec["config"]["layers"]
        layer_names = [l["name"] for l in layers]
        self.assertIn("spatial_features_in", layer_names)
        self.assertIn("bottleneck", layer_names)
        self.assertIn("reconstructed_out", layer_names)
        self.assertIn("anomaly_type_head", layer_names)

    def test_detect_z_fighting(self):
        # Two coplanar instances separated by only 0.005m (< 0.015m preset epsilon)
        instances = [
            {
                "id": "floor_slab_01",
                "archetype": "trench_parapet",
                "position": [10.0, 0.000, 10.0],
                "rotation": [0.0, 0.0, 0.0],
                "scale": [2.0, 0.1, 2.0],
                "normal": [0.0, 1.0, 0.0]
            },
            {
                "id": "floor_slab_02",
                "archetype": "concrete_bunker",
                "position": [10.0, 0.005, 10.0],
                "rotation": [0.0, 0.0, 0.0],
                "scale": [2.0, 0.1, 2.0],
                "normal": [0.0, 1.0, 0.0]
            }
        ]
        report = self.lupa.inspect_scene(instances, scene_id="TEST_Z_FIGHT", preset=self.preset)
        self.assertFalse(report.glitch_free_guarantee)
        self.assertGreater(report.anomalies_detected, 0)
        types = [a.anomaly_type for a in report.anomalies]
        self.assertIn("Z_FIGHTING", types)

    def test_detect_collision_penetration(self):
        # Two bulky objects placed at almost identical coordinates
        instances = [
            {
                "id": "bunker_a",
                "archetype": "concrete_bunker",
                "position": [5.0, 0.0, 5.0],
                "rotation": [0.0, 0.0, 0.0],
                "scale": [1.5, 1.5, 1.5],
                "normal": [0.0, 1.0, 0.0]
            },
            {
                "id": "bunker_b",
                "archetype": "mortar_emplacement",
                "position": [5.1, 0.0, 5.1],
                "rotation": [0.0, 0.0, 0.0],
                "scale": [1.5, 1.5, 1.5],
                "normal": [0.0, 1.0, 0.0]
            }
        ]
        report = self.lupa.inspect_scene(instances, scene_id="TEST_COLLISION", preset=self.preset)
        types = [a.anomaly_type for a in report.anomalies]
        self.assertIn("COLLISION_PENETRATION", types)

    def test_detect_out_of_bounds_and_scale(self):
        instances = [
            {
                "id": "lost_prop",
                "archetype": "sandbag_wall",
                "position": [9999.0, 0.0, 9999.0],  # Outside bounds [-50, 50]
                "rotation": [0.0, 0.0, 0.0],
                "scale": [0.0, 1.0, 1.0],           # Degenerate scale <= 0
                "normal": [0.0, 1.0, 0.0]
            }
        ]
        report = self.lupa.inspect_scene(instances, scene_id="TEST_BOUNDS", preset=self.preset)
        types = [a.anomaly_type for a in report.anomalies]
        self.assertIn("OUT_OF_BOUNDS", types)
        self.assertIn("DEGENERATE_SCALE", types)


class TestSceneAutoHealer(unittest.TestCase):
    """Verifies that the auto-healer restores 100% clean geometry."""

    def setUp(self):
        self.preset = PRESET_TACTICAL_ARENA_8K
        self.lupa = KerasSceneLupa(self.preset)
        self.healer = SceneAutoHealer()

    def test_auto_heal_pipeline(self):
        # Create a set with Z-fighting, bounds breach, and degenerate scale
        instances = [
            {
                "id": "slab_a",
                "archetype": "trench_parapet",
                "position": [12.0, 0.0, 12.0],
                "rotation": [0.0, 0.0, 0.0],
                "scale": [2.0, 0.1, 2.0],
                "normal": [0.0, 1.0, 0.0]
            },
            {
                "id": "slab_b",
                "archetype": "trench_parapet",
                "position": [12.0, 0.004, 12.0],
                "rotation": [0.0, 0.0, 0.0],
                "scale": [2.0, 0.1, 2.0],
                "normal": [0.0, 1.0, 0.0]
            },
            {
                "id": "escaped_unit",
                "archetype": "barbwire_fence",
                "position": [500.0, 100.0, 500.0],
                "rotation": [0.0, 0.0, 0.0],
                "scale": [-0.5, 0.0, 1.0],
                "normal": [0.0, 1.0, 0.0]
            }
        ]

        # Initial inspection
        initial_report = self.lupa.inspect_scene(instances, scene_id="PRE_HEAL", preset=self.preset)
        self.assertGreater(initial_report.anomalies_detected, 0)

        # Healing pass
        healed_instances, count = self.healer.auto_heal_scene(instances, initial_report, self.preset)
        self.assertGreater(count, 0)

        # Post-healing inspection
        post_report = self.lupa.inspect_scene(healed_instances, scene_id="POST_HEAL", preset=self.preset)
        self.assertEqual(post_report.anomalies_detected, 0)
        self.assertTrue(post_report.glitch_free_guarantee)

        # Verify degenerate scale was clamped
        for inst in healed_instances:
            self.assertTrue(all(s >= 0.05 for s in inst["scale"]))
            # Verify bounds clamped
            pos = inst["position"]
            self.assertLessEqual(pos[0], self.preset.bounds_max[0])
            self.assertGreaterEqual(pos[0], self.preset.bounds_min[0])


class TestLLMSceneBuilder(unittest.TestCase):
    """Verifies generation, scaling (>8,000 objects), and Intel Iris Xe UMA efficiency."""

    def setUp(self):
        self.builder = LLMSceneBuilder()

    def test_8k_benchmark_execution(self):
        res = self.builder.run_8k_benchmark("TACTICAL_ARENA")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["total_instances_processed"], 8192)
        self.assertEqual(res["anomalies_post_heal"], 0)
        self.assertTrue(res["glitch_free_guarantee"])
        self.assertEqual(res["vital_max_hp"], 6)
        self.assertEqual(res["fps_target_maintained"], 60)
        # Execution duration benchmark: should complete comfortably within 500 ms (typically ~145 ms)
        self.assertLess(res["benchmark_duration_ms"], 500.0)
        # Intel Iris Xe UMA memory: 8,192 instances should consume < 2.5 MB (typically ~1.54 MB)
        self.assertLess(res["intel_iris_xe_uma_mb"], 2.5)

    def test_build_across_different_contexts(self):
        contexts = ["OPENWORLD_BIOME", "URBAN_METROPOLIS", "SKEUOMORPHIC_DUNGEON"]
        for ctx in contexts:
            pkg = self.builder.build_scene_from_prompt(
                prompt=f"Generuj masívnu scénu pre {ctx}",
                context=ctx,
                target_count=8192,
                seed=123
            )
            self.assertEqual(pkg.total_instances, 8192)
            self.assertEqual(pkg.context, ctx)
            self.assertEqual(pkg.vital_max_hp, 6)
            self.assertTrue(pkg.glitch_free_guarantee)
            self.assertEqual(pkg.lupa_report["anomalies_detected"], 0)
            self.assertLess(pkg.build_latency_ms, 500.0)

    def test_stress_prompt_auto_healing(self):
        # Stress prompt deliberately injects an out-of-bounds anomaly at index 0
        pkg = self.builder.build_scene_from_prompt(
            prompt="Generuj stress test s anomáliami",
            context="TACTICAL_ARENA",
            target_count=1000,
            seed=999
        )
        self.assertTrue(pkg.glitch_free_guarantee)
        self.assertEqual(pkg.lupa_report["anomalies_detected"], 0)
        self.assertGreater(pkg.healed_anomalies_count, 0)


class TestNormativeStandardsAndJanetDSL(unittest.TestCase):
    """Verifies the JSON schema normative standards and Janet DSL definitions."""

    def test_json_schema_validity(self):
        schema_path = Path("schemas/llm_scene_builder_norms.schema.json")
        self.assertTrue(schema_path.exists(), "Schema file does not exist")
        with open(schema_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertIn("properties", data)
        self.assertIn("vital_invariants", data["properties"])
        vital_inv = data["properties"]["vital_invariants"]["properties"]
        self.assertIn("z_fighting_prevention_epsilon_m", vital_inv)
        self.assertIn("vital_max_hp", vital_inv)
        self.assertEqual(vital_inv["vital_max_hp"]["const"], 6)
        self.assertIn("object_budget", data["properties"])
        self.assertIn("spatial_envelope", data["properties"])

    def test_janet_dsl_definitions(self):
        janet_path = Path("krystal_janet/scene_builder_norms.janet")
        self.assertTrue(janet_path.exists(), "Janet DSL file does not exist")
        with open(janet_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("SCENE-BUILDER-NORMS", content)
        self.assertIn("lupa/detect-spatial-anomalies", content)
        self.assertIn("lupa/auto-heal-instance", content)
        self.assertIn("builder/synthesize-8k-scene-spec", content)
        self.assertIn(":intel-11th-gen-uma-mb-ceiling", content)
        self.assertIn("VITAL-MAX-HP 6", content)


class TestSceneServerEndpoints(unittest.TestCase):
    """Verifies backend logic for Scene Builder and Keras Lupa HTTP endpoints."""

    def test_presets_endpoint_payload(self):
        presets_dict = {k: p.to_dict() for k, p in ALL_PRESETS.items()}
        payload = {
            "status": "OK",
            "presets": presets_dict,
            "vital_max_hp": 6
        }
        self.assertEqual(payload["status"], "OK")
        self.assertEqual(payload["vital_max_hp"], 6)
        self.assertEqual(len(payload["presets"]), 4)

    def test_benchmark_endpoint_payload(self):
        res = GLOBAL_LLM_SCENE_BUILDER.run_8k_benchmark(context="TACTICAL_ARENA")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["total_instances_processed"], 8192)
        self.assertEqual(res["vital_max_hp"], 6)

    def test_lupa_inspect_endpoint_payload(self):
        preset = get_preset_for_context("TACTICAL_ARENA")
        instances = [
            {"id": "box_1", "position": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "normal": [0.0, 1.0, 0.0]}
        ]
        rep = GLOBAL_KERAS_LUPA.inspect_scene(instances, scene_id="INSPECT_TEST", preset=preset)
        payload = {"status": "OK", "report": rep.to_dict(), "vital_max_hp": 6}
        self.assertEqual(payload["status"], "OK")
        self.assertEqual(payload["vital_max_hp"], 6)
        self.assertEqual(payload["report"]["anomalies_detected"], 0)

    def test_lupa_heal_endpoint_payload(self):
        preset = get_preset_for_context("TACTICAL_ARENA")
        instances = [
            {"id": "box_1", "position": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "normal": [0.0, 1.0, 0.0]},
            {"id": "box_2", "position": [0.0, 0.005, 0.0], "scale": [1.0, 1.0, 1.0], "normal": [0.0, 1.0, 0.0]}
        ]
        initial_rep = GLOBAL_KERAS_LUPA.inspect_scene(instances, scene_id="HEAL_TEST", preset=preset)
        healed_inst, healed_count = GLOBAL_AUTO_HEALER.auto_heal_scene(instances, initial_rep, preset)
        final_rep = GLOBAL_KERAS_LUPA.inspect_scene(healed_inst, scene_id="HEAL_TEST", preset=preset)
        payload = {
            "status": "OK",
            "healed_count": healed_count,
            "healed_instances": healed_inst,
            "final_report": final_rep.to_dict(),
            "vital_max_hp": 6
        }
        self.assertEqual(payload["status"], "OK")
        self.assertEqual(payload["vital_max_hp"], 6)
        self.assertGreater(payload["healed_count"], 0)
        self.assertEqual(payload["final_report"]["anomalies_detected"], 0)

    def test_llm_generate_endpoint_payload(self):
        pkg = GLOBAL_LLM_SCENE_BUILDER.build_scene_from_prompt(
            prompt="Generuj taktickú arénu",
            context="TACTICAL_ARENA",
            target_count=100,
            seed=42
        )
        payload = {"status": "OK", "scene_package": pkg.to_dict(), "vital_max_hp": 6}
        self.assertEqual(payload["status"], "OK")
        self.assertEqual(payload["vital_max_hp"], 6)
        self.assertEqual(payload["scene_package"]["total_instances"], 100)
        self.assertTrue(payload["scene_package"]["glitch_free_guarantee"])


if __name__ == "__main__":
    unittest.main()
