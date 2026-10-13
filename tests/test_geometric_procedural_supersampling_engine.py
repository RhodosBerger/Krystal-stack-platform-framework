#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TESTS FOR GEOMETRIC PROCEDURAL SUPERSAMPLING ENGINE
==============================================================================
File: tests/test_geometric_procedural_supersampling_engine.py
Covers:
  - Differential geometry & harmonic cosine topology sampling
  - Signed distance field evaluation (crystal spire, polynomial smooth min)
  - All 6 learned procedural scenes (crystal spire, Riemannian fold, etc.)
  - Intermediate frame generation & motion-vector interpolation (medzisnímky)
  - NPU multi-frame accumulation & perceptual color bit-depth refinement (10/12-bit)
  - Blue-noise spatial-temporal dithering & reconstruction filters
  - Scientific image evaluation suite (PSNR, SSIM, Shannon entropy, Laplacian edge energy)
  - End-to-end scene stream pipeline & valid PNG lossless export
  - Constitutional Invariant: VITAL_MAX_HP == 6
==============================================================================
"""

import unittest
import base64
import math

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.geometric_procedural_supersampling_engine import (
    ProceduralSceneKind,
    NpuReconstructionFilter,
    ColorBitDepth,
    NpuSupersamplingSoftwareProfile,
    CANONICAL_NPU_PROFILES,
    ProceduralFrame,
    GeometricManifoldSynthesizer,
    IntermediateFrameInterpolator,
    NpuMultiFrameSupersampler,
    ImageQualityEvaluator,
    ImageQualityEvaluationReport,
    SupersampledSceneBundle,
    GeometricProceduralSupersamplingEngine,
    GLOBAL_GEOMETRIC_PROCEDURAL_SUPERSAMPLING_ENGINE,
    GOLDEN_RATIO,
)


class TestGeometricProceduralSupersamplingEngine(unittest.TestCase):
    """Validation test suite for the geometric procedural supersampling engine."""

    def setUp(self):
        self.engine = GeometricProceduralSupersamplingEngine()

    def test_constitutional_invariant(self):
        """Verify the constitutional invariant VITAL_MAX_HP == 6 holds strictly."""
        self.assertEqual(VITAL_MAX_HP, 6)
        report = ImageQualityEvaluator.evaluate([[(128, 128, 128)]])
        self.assertTrue(report.passed_invariant_check)
        self.assertEqual(report.vital_max_hp, 6)

    def test_canonical_profiles_loaded(self):
        """Verify all canonical NPU software parameter profiles are present."""
        self.assertIn("NPU_ULTRA_QUALITY_12BIT", CANONICAL_NPU_PROFILES)
        self.assertIn("NPU_BALANCED_10BIT", CANONICAL_NPU_PROFILES)
        self.assertIn("NPU_PERFORMANCE_8BIT", CANONICAL_NPU_PROFILES)
        self.assertIn("NPU_CINEMATIC_120FPS", CANONICAL_NPU_PROFILES)

        ultra = CANONICAL_NPU_PROFILES["NPU_ULTRA_QUALITY_12BIT"]
        self.assertEqual(ultra.target_bit_depth, ColorBitDepth.DEEP_12BIT)
        self.assertEqual(ultra.accumulation_frames, 16)
        self.assertEqual(ultra.vital_max_hp, 6)

        profile_dict = ultra.to_dict()
        self.assertEqual(profile_dict["profile_id"], "NPU_ULTRA_QUALITY_12BIT")
        self.assertEqual(profile_dict["vital_max_hp"], 6)

    def test_cosine_topology_sampling(self):
        """Verify harmonic Fourier/cosine radius derivative and steepness ratio ξ(θ)."""
        theta = math.pi / 4.0
        r_val, dr_val, xi = GeometricManifoldSynthesizer.sample_cosine_topology(theta, harmonics=4, seed=42)
        self.assertIsInstance(r_val, float)
        self.assertIsInstance(dr_val, float)
        self.assertIsInstance(xi, float)
        self.assertGreater(r_val, 0.0)
        self.assertGreaterEqual(xi, 0.0)
        # Ratio should match formula: |dr/dθ| / r
        expected_xi = abs(dr_val) / max(0.01, r_val)
        self.assertAlmostEqual(xi, expected_xi, places=6)

    def test_crystal_spire_sdf_evaluation(self):
        """Verify crystal spire polynomial smooth min SDF returns continuous values."""
        sdf_center = GeometricManifoldSynthesizer.sample_crystal_spire_sdf(0.0, 0.0, 0.0)
        sdf_outer = GeometricManifoldSynthesizer.sample_crystal_spire_sdf(2.0, 2.0, 0.0)
        self.assertIsInstance(sdf_center, float)
        self.assertIsInstance(sdf_outer, float)
        self.assertLess(sdf_center, sdf_outer)

    def test_all_learned_scenes_sampling(self):
        """Verify each of the 6 procedural scenes generates valid potential and RGB."""
        for scene_kind in ProceduralSceneKind:
            pot, (r, g, b) = GeometricManifoldSynthesizer.sample_scene_field(
                scene=scene_kind,
                u=0.2,
                v=-0.3,
                time_phase=1.0,
                seed=101
            )
            self.assertGreaterEqual(pot, 0.0, f"Potential out of bounds for {scene_kind}")
            self.assertLessEqual(pot, 1.0, f"Potential out of bounds for {scene_kind}")
            self.assertTrue(0 <= r <= 255, f"Red out of range for {scene_kind}")
            self.assertTrue(0 <= g <= 255, f"Green out of range for {scene_kind}")
            self.assertTrue(0 <= b <= 255, f"Blue out of range for {scene_kind}")

    def test_intermediate_frame_interpolation(self):
        """Verify bidirectional intermediate frame synthesis (medzisnímky)."""
        w, h = 8, 8
        frame_a = ProceduralFrame(
            width=w,
            height=h,
            pixels=[[(0.0, 0.0, 0.0) for _ in range(w)] for _ in range(h)],
            potentials=[[0.0 for _ in range(w)] for _ in range(h)],
            timestamp_phase=0.0,
            frame_index=0
        )
        frame_b = ProceduralFrame(
            width=w,
            height=h,
            pixels=[[(1.0, 1.0, 1.0) for _ in range(w)] for _ in range(h)],
            potentials=[[1.0 for _ in range(w)] for _ in range(h)],
            timestamp_phase=1.0,
            frame_index=1
        )

        intermediates = IntermediateFrameInterpolator.generate_intermediate_frames(frame_a, frame_b, count=3)
        self.assertEqual(len(intermediates), 3)

        for idx, inter in enumerate(intermediates):
            self.assertTrue(inter.is_intermediate)
            self.assertEqual(inter.width, w)
            self.assertEqual(inter.height, h)
            expected_alpha = (idx + 1) / 4.0
            self.assertAlmostEqual(inter.timestamp_phase, expected_alpha, places=4)
            # Center pixel check
            self.assertAlmostEqual(inter.pixels[4][4][0], expected_alpha, places=4)
            self.assertAlmostEqual(inter.potentials[4][4], expected_alpha, places=4)

    def test_halton_sequence_determinism(self):
        """Verify Halton sub-pixel jitter sequence outputs values in [-0.5, 0.5]."""
        for i in range(1, 20):
            h2 = NpuMultiFrameSupersampler.halton_sequence(i, 2)
            h3 = NpuMultiFrameSupersampler.halton_sequence(i, 3)
            self.assertGreaterEqual(h2, -0.5)
            self.assertLessEqual(h2, 0.5)
            self.assertGreaterEqual(h3, -0.5)
            self.assertLessEqual(h3, 0.5)

    def test_npu_multi_frame_accumulation_and_refinement(self):
        """Verify temporal accumulation, blue-noise dithering, and 12-bit quantization."""
        w, h = 24, 24
        frames = [
            ProceduralFrame(
                width=w,
                height=h,
                pixels=[[(0.5, 0.6, 0.7) for _ in range(w)] for _ in range(h)],
                potentials=[[0.5 for _ in range(w)] for _ in range(h)],
                timestamp_phase=i * 0.1,
                frame_index=i
            )
            for i in range(8)
        ]
        profile = CANONICAL_NPU_PROFILES["NPU_ULTRA_QUALITY_12BIT"]
        refined, telemetry = NpuMultiFrameSupersampler.accumulate_and_refine(frames, profile)

        self.assertEqual(len(refined), h)
        self.assertEqual(len(refined[0]), w)
        self.assertEqual(telemetry["effective_bit_depth"], "DEEP_12BIT")
        self.assertEqual(telemetry["quantization_levels"], 4096)
        self.assertEqual(telemetry["vital_max_hp"], 6)

        # Invariant HUD pips must be present in the bottom right corner
        hp_pip = refined[h - 3][w - 14]
        self.assertEqual(hp_pip, (255, 0, 110))

    def test_image_quality_evaluator_metrics(self):
        """Verify multi-definition image quality evaluation suite (PSNR, SSIM, entropy)."""
        w, h = 16, 16
        pixels_a = [[(100, 150, 200) for _ in range(w)] for _ in range(h)]
        pixels_b = [[(102, 148, 198) for _ in range(w)] for _ in range(h)]

        report = ImageQualityEvaluator.evaluate(
            supersampled_pixels=pixels_a,
            baseline_pixels=pixels_b,
            bit_depth_target="EXTENDED_10BIT"
        )

        self.assertIsInstance(report, ImageQualityEvaluationReport)
        self.assertGreater(report.psnr_db, 20.0)
        self.assertGreater(report.ssim_score, 0.8)
        self.assertGreaterEqual(report.color_entropy_bits, 0.0)
        self.assertGreaterEqual(report.laplacian_edge_energy, 0.0)
        self.assertTrue(report.passed_invariant_check)
        self.assertEqual(report.vital_max_hp, 6)

        report_dict = report.to_dict()
        self.assertIn("psnr_db", report_dict)
        self.assertIn("ssim_score", report_dict)
        self.assertIn("color_entropy_bits", report_dict)
        self.assertEqual(report_dict["vital_max_hp"], 6)

    def test_master_pipeline_generate_scene_stream(self):
        """Test full end-to-end execution of GeometricProceduralSupersamplingEngine."""
        bundle = self.engine.generate_scene_stream(
            scene=ProceduralSceneKind.EPICENTER_CRYSTAL_SPIRE,
            profile_id="NPU_BALANCED_10BIT",
            width=24,
            height=24,
            keyframes=3,
            seed=42
        )

        self.assertIsInstance(bundle, SupersampledSceneBundle)
        self.assertEqual(bundle.scene_kind, "EPICENTER_CRYSTAL_SPIRE")
        self.assertEqual(bundle.profile_used, "NPU_BALANCED_10BIT")
        self.assertEqual(bundle.width, 24)
        self.assertEqual(bundle.height, 24)
        self.assertEqual(bundle.keyframe_count, 3)
        # With 2 intermediates per keyframe: 3 keys + 3 * 2 intermediates = 9 total
        self.assertEqual(bundle.total_frames_generated, 9)
        self.assertEqual(bundle.intermediate_frame_count, 6)
        self.assertGreater(bundle.generation_throughput_fps, 0.0)

        # PNG Base64 Check
        self.assertTrue(bundle.png_base64.startswith("data:image/png;base64,"))
        raw_b64 = bundle.png_base64.split(",", 1)[1]
        decoded_png = base64.b64decode(raw_b64)
        self.assertEqual(decoded_png[:8], b"\x89PNG\r\n\x1a\n")

        # Telemetry & Quality
        self.assertIn("psnr_db", bundle.quality_evaluation)
        self.assertEqual(bundle.quality_evaluation["vital_max_hp"], 6)
        self.assertEqual(bundle.telemetry["effective_bit_depth"], "EXTENDED_10BIT")
        self.assertEqual(bundle.vital_max_hp, 6)

        # Serialization to dict
        b_dict = bundle.to_dict()
        self.assertEqual(b_dict["vital_max_hp"], 6)
        self.assertEqual(b_dict["total_frames_generated"], 9)

    def test_global_singleton(self):
        """Verify GLOBAL_GEOMETRIC_PROCEDURAL_SUPERSAMPLING_ENGINE is operational."""
        self.assertIsNotNone(GLOBAL_GEOMETRIC_PROCEDURAL_SUPERSAMPLING_ENGINE)
        scenes = GLOBAL_GEOMETRIC_PROCEDURAL_SUPERSAMPLING_ENGINE.scenes
        self.assertEqual(len(scenes), 6)


if __name__ == "__main__":
    unittest.main()
