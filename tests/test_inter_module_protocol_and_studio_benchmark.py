#!/usr/bin/env python3
"""
Unit tests for Krystal Inter-Module Communication Protocol, Three-Tier Scope Bus,
Coarse Decision Scales, Reasoning Patterns, and Game Design Studio Benchmark.
"""

import os
import sys
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.inter_module_protocol import (
    ProtocolScopeTier,
    ProtocolType,
    QoSPriority,
    ProtocolPacket,
    ThreeTierMemoryBus,
    TacticalThreatScale,
    SpatialProximityScale,
    ResourceCapacityScale,
    CoarseDecisionQuantizer,
    ControlFlowInstruction,
    BranchExecutionEngine,
    EarlyTriageReturn,
    TemplateCompiler,
    DepthLayeringPattern,
    FocalPointAnchorPattern,
    ChromaticHarmonicBalancePattern,
    MortarBallisticsCoverPattern,
    CapacityTriageBalancingPattern,
    CompositeSceneAssembler,
    GLOBAL_SCENE_ASSEMBLER
)
from krystal_kernel.game_design_studio_benchmark import (
    GameDesignStudioBenchmarkSuite,
    GLOBAL_STUDIO_BENCHMARK
)


class TestInterModuleProtocolAndStudioBenchmark(unittest.TestCase):

    def test_packet_envelope_and_integrity(self):
        packet = ProtocolPacket.create(
            source="sim.mortar_battery",
            target="gpu.webgl_compositor",
            protocol_type=ProtocolType.COMMAND,
            payload={"elevation_deg": 65.0, "salvo": 2},
            scope_tier=ProtocolScopeTier.GLOBAL,
            qos=QoSPriority.CRITICAL
        )
        self.assertTrue(packet.verify_integrity())
        self.assertEqual(packet.header.qos_priority, QoSPriority.CRITICAL)
        self.assertEqual(packet.header.scope_tier, ProtocolScopeTier.GLOBAL)

        # Corrupt payload and verify integrity check fails
        corrupted = ProtocolPacket(
            header=packet.header,
            payload={"elevation_deg": 40.0, "salvo": 2}
        )
        self.assertFalse(corrupted.verify_integrity())

    def test_three_tier_memory_bus_scoping_and_cascading(self):
        bus = ThreeTierMemoryBus()
        # Set explicit tiers
        bus.set("@local:scratch_counter", 10)
        bus.set("@global:active_biome", "VOLCANIC_CRATER")
        bus.set("@superglobal:world_cosmological_seed", 4242)

        self.assertEqual(bus.get("@local:scratch_counter"), 10)
        self.assertEqual(bus.get("@global:active_biome"), "VOLCANIC_CRATER")
        self.assertEqual(bus.get("@superglobal:world_cosmological_seed"), 4242)

        # Test cascading resolution
        self.assertEqual(bus.get("scratch_counter"), 10)
        self.assertEqual(bus.get("active_biome"), "VOLCANIC_CRATER")
        self.assertEqual(bus.get("world_cosmological_seed"), 4242)

        # Local shadow overrides global in cascading
        bus.set("@local:active_biome", "LOCAL_SHADOWED_BIOME")
        self.assertEqual(bus.get("active_biome"), "LOCAL_SHADOWED_BIOME")
        self.assertEqual(bus.get("@global:active_biome"), "VOLCANIC_CRATER")

    def test_three_tier_memory_bus_vital_max_hp_invariant(self):
        bus = ThreeTierMemoryBus()
        self.assertEqual(bus.get("@superglobal:vital_max_hp"), VITAL_MAX_HP)

        # Attempt to set hp > 6
        bus.set("@global:player_hp", 15)
        self.assertEqual(bus.get("@global:player_hp"), VITAL_MAX_HP)

        # Attempt to override superglobal vital_max_hp
        bus.set("@superglobal:vital_max_hp", 99)
        self.assertEqual(bus.get("@superglobal:vital_max_hp"), VITAL_MAX_HP)

    def test_local_stack_frame_push_pop(self):
        bus = ThreeTierMemoryBus()
        bus.set("@local:var_frame0", "alpha")
        self.assertEqual(bus.get("@local:var_frame0"), "alpha")

        bus.push_local_frame({"var_frame1": "beta"})
        self.assertEqual(bus.get("@local:var_frame1"), "beta")
        self.assertIsNone(bus.get("@local:var_frame0"))

        popped = bus.pop_local_frame()
        self.assertEqual(popped.get("var_frame1"), "beta")
        self.assertEqual(bus.get("@local:var_frame0"), "alpha")

    def test_coarse_decision_scales_quantization(self):
        self.assertEqual(CoarseDecisionQuantizer.quantize_threat(0.1), TacticalThreatScale.NEGLIGIBLE)
        self.assertEqual(CoarseDecisionQuantizer.quantize_threat(0.4), TacticalThreatScale.ELEVATED)
        self.assertEqual(CoarseDecisionQuantizer.quantize_threat(0.7), TacticalThreatScale.CRITICAL)
        self.assertEqual(CoarseDecisionQuantizer.quantize_threat(0.95), TacticalThreatScale.LETHAL)

        # Mortar dead-zone check
        self.assertEqual(CoarseDecisionQuantizer.quantize_proximity(8.0, is_high_angle_mortar=True), SpatialProximityScale.DEAD_ZONE)
        self.assertEqual(CoarseDecisionQuantizer.quantize_proximity(8.0, is_high_angle_mortar=False), SpatialProximityScale.CLOSE_QUARTERS)
        self.assertEqual(CoarseDecisionQuantizer.quantize_proximity(25.0, is_high_angle_mortar=True), SpatialProximityScale.OPTIMAL_ZONE)

        # Capacity scale
        self.assertEqual(CoarseDecisionQuantizer.quantize_capacity(10.0), ResourceCapacityScale.DEPLETED)
        self.assertEqual(CoarseDecisionQuantizer.quantize_capacity(30.0), ResourceCapacityScale.CONSTRAINED)
        self.assertEqual(CoarseDecisionQuantizer.quantize_capacity(60.0), ResourceCapacityScale.BALANCED)
        self.assertEqual(CoarseDecisionQuantizer.quantize_capacity(90.0), ResourceCapacityScale.SURPLUS)

    def test_control_flow_if_then_else_and_early_triage_return(self):
        bus = ThreeTierMemoryBus()
        bus.set("@local:target_distance", 6.0) # Dead zone for mortar

        instructions = [
            ControlFlowInstruction(
                op="IF",
                condition={
                    "scale": "SPATIAL_PROXIMITY",
                    "target": "@local:target_distance",
                    "operator": "EQUALS",
                    "value": "DEAD_ZONE",
                    "is_mortar": True
                },
                then_branch=[
                    ControlFlowInstruction(
                        op="RETURN",
                        value={"status": "ABORT_DEAD_ZONE", "action": "SWITCH_TO_MELEE"}
                    )
                ],
                else_branch=[
                    ControlFlowInstruction(op="ASSIGN", target="@local:status", value="FIRED")
                ]
            )
        ]

        engine = BranchExecutionEngine(bus)
        with self.assertRaises(EarlyTriageReturn) as cm:
            engine.execute_block(instructions)

        self.assertEqual(cm.exception.payload["status"], "ABORT_DEAD_ZONE")
        self.assertEqual(cm.exception.payload["action"], "SWITCH_TO_MELEE")
        self.assertTrue(engine.early_exit_triggered)

    def test_control_flow_while_bounded_loop(self):
        bus = ThreeTierMemoryBus()
        bus.set("@local:ammo_count", 3)

        # Safe loop that assigns and exits
        instructions = [
            ControlFlowInstruction(
                op="WHILE",
                condition={
                    "scale": "",
                    "target": "@local:ammo_count",
                    "operator": "GREATER_THAN",
                    "value": 0
                },
                max_iterations=8,
                body=[
                    ControlFlowInstruction(op="EXECUTE_MICROTASK", value="MICROT_FIRE_ROUND"),
                    ControlFlowInstruction(op="ASSIGN", target="@local:ammo_count", value=0)
                ]
            )
        ]
        engine = BranchExecutionEngine(bus)
        engine.execute_block(instructions)
        self.assertEqual(bus.get("@local:ammo_count"), 0)
        self.assertIn("MICROT_FIRE_ROUND", engine.executed_microtasks)

    def test_template_compiler_microtask_reduction(self):
        bus = ThreeTierMemoryBus()
        tasks, stats = TemplateCompiler.compile_template("TACTICAL_SIEGE_MORTAR", bus)
        self.assertGreater(len(tasks), 3)
        self.assertGreater(stats["operational_reduction_pct"], 80.0)
        self.assertEqual(stats["template_id"], "TACTICAL_SIEGE_MORTAR")

    def test_five_reasoning_patterns(self):
        bus = ThreeTierMemoryBus()

        # 1. DepthLayeringPattern
        depth = DepthLayeringPattern.execute({"foreground_fog": 0.2, "background_haze": 0.6}, bus)
        self.assertGreaterEqual(depth["depth_contrast_ratio"], 3.5)
        self.assertTrue(depth["rule_depth_satisfied"])

        # 2. FocalPointAnchorPattern
        focal = FocalPointAnchorPattern.execute({"actors": ["MORTAR", "COMMANDER"]}, bus)
        self.assertEqual(focal["focal_anchor_count"], 2)
        self.assertTrue(focal["rule_of_thirds_met"])

        # 3. ChromaticHarmonicBalancePattern
        color = ChromaticHarmonicBalancePattern.execute({"palette_type": "SPLIT_COMPLEMENTARY"}, bus)
        self.assertGreaterEqual(color["harmony_cohesion_index"], 0.90)

        # 4. MortarBallisticsCoverPattern
        mortar_norm = MortarBallisticsCoverPattern.execute({"elevation_deg": 65.0, "distance_m": 30.0, "trench_cover": False}, bus)
        self.assertEqual(mortar_norm["verdict"], "VALID_FIRE_SOLUTION")
        self.assertTrue(mortar_norm["is_high_angle"])

        mortar_trench = MortarBallisticsCoverPattern.execute({"elevation_deg": 65.0, "distance_m": 30.0, "trench_cover": True}, bus)
        self.assertAlmostEqual(mortar_trench["effective_damage"], 2.4, places=2)

        mortar_dead = MortarBallisticsCoverPattern.execute({"elevation_deg": 65.0, "distance_m": 8.0}, bus)
        self.assertEqual(mortar_dead["verdict"], "TARGET_IN_DEAD_ZONE")

        # 5. CapacityTriageBalancingPattern
        cap = CapacityTriageBalancingPattern.execute({"spell_capacity": 60.0, "mechanics_capacity": 80.0, "salvo_count": 2}, bus)
        self.assertEqual(cap["vital_max_hp_invariant"], VITAL_MAX_HP)
        self.assertLessEqual(cap["clamped_hp"], VITAL_MAX_HP)

    def test_composite_scene_assembler(self):
        manifest = GLOBAL_SCENE_ASSEMBLER.assemble_scene_from_template(
            template_id="TACTICAL_SIEGE_MORTAR",
            custom_params={
                "target_distance": 25.0,
                "elevation_deg": 62.0,
                "spell_capacity": 70.0,
                "mechanics_capacity": 80.0,
                "player_hp": VITAL_MAX_HP
            }
        )
        self.assertGreaterEqual(manifest.composition_synergy_index, 0.85)
        self.assertEqual(manifest.vital_max_hp, VITAL_MAX_HP)
        self.assertEqual(manifest.reduction_stats["template_id"], "TACTICAL_SIEGE_MORTAR")
        self.assertIn("foreground_plane", manifest.depth_planes)
        self.assertFalse(manifest.early_triage_exit)

    def test_composite_scene_assembler_dead_zone_early_exit(self):
        manifest = GLOBAL_SCENE_ASSEMBLER.assemble_scene_from_template(
            template_id="TACTICAL_SIEGE_MORTAR",
            custom_params={
                "target_distance": 6.0, # Within dead-zone!
                "elevation_deg": 65.0
            }
        )
        self.assertTrue(manifest.early_triage_exit)
        self.assertEqual(manifest.cohesive_verdict, "EARLY_TRIAGE_ABORTED")

    def test_game_design_studio_benchmark(self):
        result = GLOBAL_STUDIO_BENCHMARK.run_benchmark(
            template_id="TACTICAL_SIEGE_MORTAR",
            iterations=50
        )
        self.assertGreater(result.operational_reduction_pct, 80.0)
        self.assertGreaterEqual(result.composition_synergy_index, 0.85)
        self.assertEqual(result.vital_max_hp, VITAL_MAX_HP)
        self.assertIn("authoring_time_saved_per_scene_hours", result.studio_roi_metrics)
        self.assertIn("local_stack", result.krystal_protocol_stack["scope_access_latency_us"])

    def test_server_protocol_endpoints(self):
        import json
        from io import BytesIO
        from krystal_web_hub.server import KrystalHubHandler

        class DummyServer:
            pass

        class MockHandler(KrystalHubHandler):
            def __init__(self, path: str, method: str = "GET", body: dict = None):
                self.path = path
                self.command = method
                self.requestline = f"{method} {path} HTTP/1.1"
                self.wfile = BytesIO()
                self.server = DummyServer()
                self.client_address = ("127.0.0.1", 8080)
                self.close_connection = True
                self._code = 200

                if body is not None:
                    body_bytes = json.dumps(body).encode("utf-8")
                    self.rfile = BytesIO(body_bytes)
                    self.headers = {"Content-Length": str(len(body_bytes)), "Content-Type": "application/json"}
                else:
                    self.rfile = BytesIO()
                    self.headers = {}

            def send_response(self, code, message=None):
                self._code = code

            def send_header(self, keyword, value):
                pass

            def end_headers(self):
                pass

        # 1. Test GET /api/protocol/standards
        h_get = MockHandler("/api/protocol/standards", "GET")
        h_get.do_GET()
        res_get = json.loads(h_get.wfile.getvalue().decode("utf-8"))
        self.assertEqual(res_get["status"], "OK")
        self.assertEqual(res_get["vital_max_hp"], 6)
        self.assertIn("LOCAL", res_get["scope_tiers"])
        self.assertIn("SUPERGLOBAL", res_get["scope_tiers"])

        # 2. Test POST /api/protocol/execute_pipeline
        h_post = MockHandler("/api/protocol/execute_pipeline", "POST", {
            "template_id": "TACTICAL_SIEGE_MORTAR",
            "target_distance": 28.0,
            "elevation_deg": 65.0,
            "target_in_trench": True
        })
        h_post.do_POST()
        res_post = json.loads(h_post.wfile.getvalue().decode("utf-8"))
        self.assertEqual(res_post["status"], "OK")
        self.assertEqual(res_post["manifest"]["vital_max_hp"], 6)

        # 3. Test POST /api/protocol/studio_benchmark
        h_bench = MockHandler("/api/protocol/studio_benchmark", "POST", {
            "template_id": "TACTICAL_SIEGE_MORTAR",
            "iterations": 20
        })
        h_bench.do_POST()
        res_bench = json.loads(h_bench.wfile.getvalue().decode("utf-8"))
        self.assertEqual(res_bench["status"], "OK")
        self.assertEqual(res_bench["benchmark"]["vital_max_hp"], 6)


if __name__ == "__main__":
    unittest.main()
