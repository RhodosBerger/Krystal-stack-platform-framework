#!/usr/bin/env python3
"""
==============================================================================
UNIT TEST SUITE: SOVEREIGN JANET-OPENVINO ENGINE & BYTECODE DISPATCHER
==============================================================================
Module: tests/test_janet_openvino_core.py
Description: Validates 64-bit binary bytecode encoding (KSYN/KSOV), S-expression
             parsing, SIMD fallback execution (AVX2/SSE4.2), and dispatch speedup.

Platform Invariant: VITAL_MAX_HP = 6
Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from krystal_kernel.janet_openvino_core import (
    TensorOpcode,
    TargetSimdTier,
    CompiledTensorInstruction,
    JanetOpenVinoBenchmarkReport,
    JanetTensorBytecodeComposer,
    LegacyIntelSimdFallbackEngine,
    SovereignJanetOpenVinoEngine,
    GLOBAL_JANET_OPENVINO_ENGINE,
    VITAL_MAX_HP
)


class TestJanetOpenVinoCore(unittest.TestCase):
    """Test suite for the sovereign Janet-OpenVINO engine."""

    def test_01_invariant_constant(self):
        """Ensures the sacred system invariant is strictly 6."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_02_bytecode_instruction_compilation(self):
        """Tests that instruction compiles to exactly 8 bytes (64 bits)."""
        ins = JanetTensorBytecodeComposer.compile_instruction(
            opcode=TensorOpcode.OP_MATMUL_DP4A_I8,
            dest=4,
            src_a=2,
            src_b=3,
            prec=1,
            payload=0x1234
        )
        self.assertIsInstance(ins, CompiledTensorInstruction)
        self.assertEqual(len(ins.raw_word_bytes), 8)
        self.assertEqual(ins.opcode, TensorOpcode.OP_MATMUL_DP4A_I8)
        self.assertEqual(ins.reg_dest, 4)

    def test_03_sexpr_to_bytecode_compilation(self):
        """Tests S-expression translation to binary stream."""
        sexpr = """
        (ov/pipeline
          (ov/assert-vital-hp)
          (ov/quant-i8 :src 1 :dest 2)
          (ov/matmul :a 2 :b 3 :dest 4)
          (ov/rmsnorm :src 4 :dest 5)
          (ov/softmax :src 5 :dest 6))
        """
        bytecode = JanetTensorBytecodeComposer.compile_janet_sexpr_to_bytecode(sexpr)
        self.assertTrue(bytecode.startswith(b"KSOV"))
        # Header (4B) + 6 instructions * 8B = 52 bytes
        self.assertEqual(len(bytecode), 4 + 6 * 8)

    def test_04_fallback_quantization_and_operators(self):
        """Tests symmetric INT8 quantization and math operators."""
        data = [-1.0, 0.0, 0.5, 1.0]
        i8_data, scale = LegacyIntelSimdFallbackEngine.quantize_fp32_to_int8(data, scale=127.0)
        self.assertEqual(len(i8_data), 4)
        self.assertEqual(i8_data[0], -127)
        self.assertEqual(i8_data[3], 127)

        # Softmax test
        probs = LegacyIntelSimdFallbackEngine.execute_softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(probs), 1.0, places=4)
        self.assertTrue(probs[2] > probs[1] > probs[0])

    def test_05_engine_execution_stream(self):
        """Tests end-to-end execution of a compiled tensor bytecode stream."""
        engine = GLOBAL_JANET_OPENVINO_ENGINE
        sexpr = "(ov/pipeline (ov/assert-vital-hp) (ov/quant-i8) (ov/rmsnorm))"
        bytecode = JanetTensorBytecodeComposer.compile_janet_sexpr_to_bytecode(sexpr)

        result = engine.execute_bytecode_stream(bytecode)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertTrue(result["vital_hp_verified"])
        self.assertGreater(result["executed_instructions"], 0)
        self.assertLess(result["dispatch_latency_us"], 5000.0)

    def test_06_benchmark_dispatch_speedup(self):
        """Tests that Janet bytecode dispatch demonstrates speedup over OpenAPI/JSON."""
        engine = GLOBAL_JANET_OPENVINO_ENGINE
        rep = engine.benchmark_dispatch_speedup(iterations=20)
        self.assertIsInstance(rep, JanetOpenVinoBenchmarkReport)
        self.assertGreater(rep.dispatch_speedup_factor, 10.0)
        self.assertLess(rep.janet_bytecode_latency_us, rep.json_openapi_latency_us)
        self.assertEqual(rep.vital_max_hp, 6)


if __name__ == "__main__":
    unittest.main()
