#!/usr/bin/env python3
"""
Unit tests for OpenCL & OpenGL Alternative Ruleset Noise/Cosine Decoder Engine.
"""

import unittest
from krystal_kernel.opencl_opengl_noise_cosine_decoder import (
    OpenCLOpenGLNoiseCosineDecoderEngine,
    GLOBAL_OPENCL_OPENGL_DECODER,
    AccelerationBackend,
    NoiseDecoderMode,
    OpenCLDeviceTier,
    AlternativeRulesetConfig,
    VITAL_MAX_HP
)


class TestOpenCLOpenGLNoiseCosineDecoder(unittest.TestCase):
    def setUp(self):
        self.engine = OpenCLOpenGLNoiseCosineDecoderEngine()

    def test_ruleset_validation_and_contracts(self):
        cfg = AlternativeRulesetConfig()
        self.assertEqual(len(cfg.validate_rules()), 0)
        self.assertEqual(cfg.vital_max_hp, 6)
        
        # Test violation on bad workgroup size
        bad_cfg = AlternativeRulesetConfig(workgroup_size_x=64, workgroup_size_y=64) # 4096 > 1024
        violations = bad_cfg.validate_rules()
        self.assertGreater(len(violations), 0)
        self.assertIn("Rule 2", violations[0])

    def test_opencl_c_kernel_synthesis(self):
        # IDCT Kernel
        idct_src = self.engine.generate_opencl_c_kernel(NoiseDecoderMode.DISCRETE_COSINE_IDCT_2D)
        self.assertIn("__kernel void idct_2d_cosine_decoder", idct_src)
        self.assertIn("__local float local_block", idct_src)
        self.assertIn("barrier(CLK_LOCAL_MEM_FENCE)", idct_src)
        
        # Cosine FBM Kernel
        fbm_src = self.engine.generate_opencl_c_kernel(NoiseDecoderMode.COSINE_HARMONIC_FBM)
        self.assertIn("__kernel void cosine_fbm_noise_decoder", fbm_src)
        self.assertIn("cosine_interpolate", fbm_src)

    def test_opengl_glsl_compute_shader_synthesis(self):
        glsl_src = self.engine.generate_opengl_glsl_compute_shader(NoiseDecoderMode.COSINE_HARMONIC_FBM)
        self.assertIn("#version 460 core", glsl_src)
        self.assertIn("local_size_x = 16, local_size_y = 16", glsl_src)
        self.assertIn("layout (std430, binding = 0)", glsl_src)
        self.assertIn("layout (std430, binding = 1)", glsl_src)
        self.assertIn("cosine_smooth", glsl_src)

    def test_decode_all_modes(self):
        for mode in NoiseDecoderMode:
            res = self.engine.decode_field(
                decoder_mode=mode,
                width=32,
                height=32,
                frequency=0.1,
                octaves=3
            )
            self.assertEqual(res.width, 32)
            self.assertEqual(res.height, 32)
            self.assertEqual(len(res.preview_grid_8x8), 8)
            self.assertEqual(len(res.preview_grid_8x8[0]), 8)
            self.assertGreaterEqual(res.max_value, res.min_value)
            self.assertEqual(res.vital_max_hp, VITAL_MAX_HP)
            self.assertEqual(res.vital_max_hp, 6)

    def test_engine_ruleset_summary_and_status(self):
        summary = self.engine.get_ruleset_summary()
        self.assertEqual(len(summary["rule_contracts"]), 5)
        self.assertIn("Rule 1: Backend Arbitration", summary["rule_contracts"][0]["rule"])
        self.assertEqual(summary["vital_max_hp"], 6)

        status = GLOBAL_OPENCL_OPENGL_DECODER.get_status()
        self.assertGreaterEqual(status["cached_fields_count"], 4)
        self.assertEqual(status["vital_max_hp"], 6)


if __name__ == '__main__':
    unittest.main()
