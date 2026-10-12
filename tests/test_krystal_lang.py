"""
Unit and Integration Tests for Krystal-Lang Topological Engine
==============================================================
Tests:
1. Lexer and Parser tokenization
2. Program AST construction
3. Queue partitioning and dependency graph
4. Topological bytecode generation
5. Bytecode-to-shape 3D SDF and ASCII rendering
6. Topological VM queue execution and packet throughput
"""

import unittest
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from krystal_lang.compiler import KrystalCompiler, KrystalLexer
from krystal_lang.bytecode_to_shape import BytecodeShapeTranspiler
from krystal_lang.virtual_machine import TopologicalVM

SAMPLE_CODE = """
module NeuralAsciiDispatcher

queue InboundStream { priority: 3, capacity: 512, spatial: [0.0, 0.0, -5.0] }
queue ProcessingCore { priority: 2, capacity: 1024, spatial: [0.0, 1.5, 0.0] }
queue OutboundRaster { priority: 1, capacity: 256, spatial: [0.0, 0.0, 5.0] }

shape CoreManifold {
    type: IFS_FRACTAL
    scale: 2.0
    csg: SMOOTH_MIN(0.2)
}

shape MemoryGyroid {
    type: GYROID
    frequency: 1.8
    thickness: 0.1
    csg: INTERSECTION
}

pipeline IngestToProcess {
    from: InboundStream
    to: ProcessingCore
    transform: MIRROR_DN(folds: 8)
    transform: TWIST(rate: 0.5)
    action: EVAL_SDF
}

pipeline ProcessToRaster {
    from: ProcessingCore
    to: OutboundRaster
    action: PASS
}

loop SymmetryHarmonics(iterations: 8, fold: 8) {
    transform: DISPLACE(amp: 0.1, freq: 2.0)
}
"""

class TestKrystalLang(unittest.TestCase):

    def setUp(self):
        self.compiler = KrystalCompiler()

    def test_01_lexer_and_parser(self):
        tokens = KrystalLexer.tokenize(SAMPLE_CODE)
        self.assertGreater(len(tokens), 30)

        res = self.compiler.compile(SAMPLE_CODE)
        self.assertEqual(res["module_name"], "NeuralAsciiDispatcher")
        self.assertEqual(res["queues_count"], 3)
        self.assertEqual(res["shapes_count"], 2)
        self.assertEqual(res["pipelines_count"], 2)

    def test_02_queue_partitioner(self):
        res = self.compiler.compile(SAMPLE_CODE)
        stages = res["queue_partitions"]
        self.assertGreaterEqual(len(stages), 3)

        # Stage 0 should be Ingestion
        self.assertEqual(stages[0]["type"], "INGESTION_PARALLEL")
        self.assertIn("InboundStream", stages[0]["queues"])

        # Final stage should be Output Sinks
        sink_stage = [s for s in stages if s["type"] == "OUTPUT_SINKS"]
        self.assertTrue(len(sink_stage) > 0)
        self.assertIn("OutboundRaster", sink_stage[0]["queues"])

    def test_03_bytecode_generation(self):
        res = self.compiler.compile(SAMPLE_CODE)
        bc = res["bytecode"]
        self.assertGreater(len(bc), 5)

        op_names = [instr["op"] for instr in bc]
        self.assertIn("OP_ALLOC_QUEUE", op_names)
        self.assertIn("OP_EMIT_SHAPE", op_names)
        self.assertIn("OP_GEOM_TRANSFORM", op_names)
        self.assertIn("OP_QUEUE_DISPATCH", op_names)
        self.assertIn("OP_SYMMETRY_FOLD", op_names)

    def test_04_bytecode_to_shape_rendering(self):
        res = self.compiler.compile(SAMPLE_CODE)
        transpiler = BytecodeShapeTranspiler(res)

        # Evaluate 3D SDF at origin
        d_val = transpiler.evaluate_program_sdf((0.0, 0.0, 0.0), t=0.5)
        self.assertIsInstance(d_val, float)

        # Render ASCII representation of the code manifold
        ascii_frame = transpiler.render_shape_ascii(width=48, height=12, t=0.5)
        self.assertEqual(len(ascii_frame.splitlines()), 12)
        # Check non-empty representation of code
        self.assertTrue(any(c in ascii_frame for c in ".:;+=*#%@"))

    def test_05_topological_vm_execution(self):
        res = self.compiler.compile(SAMPLE_CODE)
        vm = TopologicalVM(res)

        # Inject 10 packets into InboundStream
        for i in range(10):
            ok = vm.inject_input("InboundStream", {"id": i, "payload": f"data_{i}"})
            self.assertTrue(ok)

        # Step 1: Moves from InboundStream -> ProcessingCore
        m1 = vm.step_execution(cycles=1)
        self.assertGreater(m1["packets_processed"], 0)

        # Step 2: Moves from ProcessingCore -> OutboundRaster
        m2 = vm.step_execution(cycles=1)
        self.assertGreater(m2["packets_processed"], 0)
        self.assertGreater(m2["queue_depths"]["OutboundRaster"], 0)

    def test_06_fast_ring_buffer_vm_execution(self):
        res = self.compiler.compile(SAMPLE_CODE)
        vm = TopologicalVM(res, use_fast_queue=True)
        self.assertEqual(vm.metrics["engine_mode"], "LOCK_FREE_RING")

        # Inject 100 packets into InboundStream
        for i in range(100):
            ok = vm.inject_input("InboundStream", {"id": i, "payload": f"fast_data_{i}"})
            self.assertTrue(ok)

        # Execute multiple cycles
        m = vm.step_execution(cycles=20)
        self.assertGreater(m["packets_processed"], 50)
        self.assertGreater(m["average_throughput_pps"], 1000.0)

    def test_07_conditional_branching_if_elif_else(self):
        """Tests compiling and executing if / elif / else conditional ladders."""
        code = """
        module ThermalGovernorModule
        queue InboundTelemetry { priority: 3, capacity: 64 }
        queue CriticalThrottleQueue { priority: 3, capacity: 64 }
        queue WarningQueue { priority: 2, capacity: 64 }
        queue NominalQueue { priority: 1, capacity: 64 }

        if ThermalLadder on InboundTelemetry {
            when: temp_cpu > 90.0
            to: CriticalThrottleQueue
            action: THROTTLE
        }
        elif {
            when: temp_cpu > 75.0
            to: WarningQueue
            action: WARN
        }
        else {
            to: NominalQueue
            action: PASS
        }
        """
        res = self.compiler.compile(code)
        self.assertEqual(res["module_name"], "ThermalGovernorModule")
        ast = res["program_ast"]
        self.assertEqual(len(ast.conditionals), 1)
        ladder = ast.conditionals[0]
        self.assertEqual(ladder.name, "ThermalLadder")
        self.assertEqual(ladder.source_queue, "InboundTelemetry")
        self.assertEqual(ladder.if_branch.target_queue, "CriticalThrottleQueue")
        self.assertEqual(len(ladder.elif_branches), 1)
        self.assertEqual(ladder.elif_branches[0].target_queue, "WarningQueue")
        self.assertIsNotNone(ladder.else_branch)
        self.assertEqual(ladder.else_branch.target_queue, "NominalQueue")

        # Verify bytecode opcodes
        ops = [instr["op"] for instr in res["bytecode"]]
        self.assertIn("OP_BRANCH_IF", ops)
        self.assertIn("OP_BRANCH_ELIF", ops)
        self.assertIn("OP_BRANCH_ELSE", ops)

        # Run in TopologicalVM
        vm = TopologicalVM(res)
        # 1. Packet exceeding 90°C -> CriticalThrottleQueue
        vm.inject_input("InboundTelemetry", {"id": 1, "temp_cpu": 95.5})
        # 2. Packet between 75°C and 90°C -> WarningQueue
        vm.inject_input("InboundTelemetry", {"id": 2, "temp_cpu": 82.0})
        # 3. Packet below 75°C -> NominalQueue
        vm.inject_input("InboundTelemetry", {"id": 3, "temp_cpu": 60.0})

        m = vm.step_execution(cycles=5)
        self.assertGreaterEqual(m["packets_processed"], 3)
        self.assertEqual(vm.metrics["queue_depths"]["CriticalThrottleQueue"], 1)
        self.assertEqual(vm.metrics["queue_depths"]["WarningQueue"], 1)
        self.assertEqual(vm.metrics["queue_depths"]["NominalQueue"], 1)

    def test_08_what_if_speculative_counterfactual_execution(self):
        """Tests compiling and executing what_if counterfactual speculative blocks."""
        code = """
        module SpeculativeGovernorModule
        queue CandidateStream { priority: 3, capacity: 64 }
        queue TurboQueue { priority: 3, capacity: 64 }
        queue EcoQueue { priority: 2, capacity: 64 }
        queue SafeFallbackQueue { priority: 1, capacity: 64 }

        what_if PolicyEvaluation on CandidateStream {
            scenario OverclockHigh {
                hypothesis: "MAX_FPS"
                to: TurboQueue
                action: BOOST
                cost: 0.15
                utility: 0.95
            }
            scenario EcoLowPower {
                hypothesis: "LOW_HEAT"
                to: EcoQueue
                action: THROTTLE
                cost: 0.05
                utility: 0.70
            }
            fallback: SafeFallbackQueue
            criterion: MAX_UTILITY
        }
        """
        res = self.compiler.compile(code)
        ast = res["program_ast"]
        self.assertEqual(len(ast.what_ifs), 1)
        wib = ast.what_ifs[0]
        self.assertEqual(wib.name, "PolicyEvaluation")
        self.assertEqual(wib.source_queue, "CandidateStream")
        self.assertEqual(len(wib.scenarios), 2)
        self.assertEqual(wib.commit_criterion, "MAX_UTILITY")

        # Verify bytecode
        ops = [instr["op"] for instr in res["bytecode"]]
        self.assertIn("OP_WHAT_IF_SPECULATE", ops)

        # Run in TopologicalVM
        vm = TopologicalVM(res)
        vm.inject_input("CandidateStream", {"tick": 42, "workload": "raymarching"})
        m = vm.step_execution(cycles=1)

        self.assertGreater(m["packets_processed"], 0)
        self.assertEqual(m["what_if_resolutions"], 1)
        # Winner must be OverclockHigh (utility 0.95 - 0.15 = 0.80 vs 0.65)
        self.assertEqual(vm.metrics["queue_depths"]["TurboQueue"], 1)
        winner_pkt = vm.channels["TurboQueue"].get_nowait()
        self.assertEqual(winner_pkt["what_if_winner"], "OverclockHigh")
        self.assertAlmostEqual(winner_pkt["utility_score"], 0.80, places=2)

    def test_09_branching_and_what_if_shape_rendering(self):
        """Tests 3D SDF and ASCII manifold rendering of branching and what-if programs."""
        code = """
        module ManifoldBifurcation
        queue InboundStream { priority: 3, capacity: 64 }
        queue BranchA { priority: 2, capacity: 64 }
        queue BranchB { priority: 2, capacity: 64 }

        if Bifurcation on InboundStream {
            when: energy > 50.0
            to: BranchA
            action: PASS
        }
        else {
            to: BranchB
            action: PASS
        }
        """
        res = self.compiler.compile(code)
        transpiler = BytecodeShapeTranspiler(res)
        d_val = transpiler.evaluate_program_sdf((0.0, 0.0, 0.0), t=1.0)
        self.assertIsInstance(d_val, float)

        ascii_art = transpiler.render_shape_ascii(width=40, height=10, t=1.0)
        self.assertEqual(len(ascii_art.splitlines()), 10)


if __name__ == "__main__":
    unittest.main()

