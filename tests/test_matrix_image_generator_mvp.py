#!/usr/bin/env python3
"""
Unit tests for Matrix-Convolutional Procedural & Latent Image Generator MVP.
Verifies all 6 architectural layers, zero-dependency PNG/SVG rasterizers,
and the constitutional invariant assert VITAL_MAX_HP == 6.
"""

import unittest
import base64
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.matrix_image_generator_mvp import (
    GLOBAL_MATRIX_IMAGE_GENERATOR_ENGINE,
    ImageGenerationConfig,
    GeneratedImageArtifact,
    LatentSeedMode,
    ImageColorPalette,
    LatentTopologyLayer,
    ConvolutionalDiffusionLayer,
    HarmonicColorTensorLayer,
    EpistemicHudInlayLayer,
    MatrixImageRasterExporter
)


class TestMatrixImageGeneratorMvp(unittest.TestCase):

    def setUp(self):
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_layer1_latent_topology_seeding(self):
        for mode in LatentSeedMode:
            cfg = ImageGenerationConfig(width=16, height=16, seed=42, latent_mode=mode)
            mat = LatentTopologyLayer.generate_seed_matrix(cfg)
            self.assertEqual(mat.rows, 16)
            self.assertEqual(mat.cols, 16)
            self.assertGreaterEqual(mat.active_bit_count, 0)

    def test_layer2_convolutional_diffusion_pipeline(self):
        cfg = ImageGenerationConfig(width=16, height=16, seed=123)
        mat = LatentTopologyLayer.generate_seed_matrix(cfg)
        steps, telemetry = ConvolutionalDiffusionLayer.execute_pipeline(
            mat, ["diffuse_3x3", "edge_detect_3x3", "majority_3x3", "invariant_6hp_3x3"]
        )
        self.assertEqual(len(steps), 5)  # 1 initial + 4 convolved
        self.assertEqual(len(telemetry), 5)

    def test_layer3_harmonic_color_tensor_layer(self):
        cfg = ImageGenerationConfig(width=16, height=16, seed=77, palette=ImageColorPalette.KRYSTAL_CYBERPUNK)
        mat = LatentTopologyLayer.generate_seed_matrix(cfg)
        steps, _ = ConvolutionalDiffusionLayer.execute_pipeline(mat, ["diffuse_3x3", "majority_3x3"])
        pixels = HarmonicColorTensorLayer.colorize(steps, cfg)
        self.assertEqual(len(pixels), 16)
        self.assertEqual(len(pixels[0]), 16)
        # Verify 24-bit RGB ranges
        for row in pixels:
            for r, g, b in row:
                self.assertTrue(0 <= r <= 255)
                self.assertTrue(0 <= g <= 255)
                self.assertTrue(0 <= b <= 255)

    def test_layer4_epistemic_hud_inlay_invariant(self):
        cfg = ImageGenerationConfig(width=24, height=24, seed=99, enable_hud_overlay=True, vital_hp=6)
        mat = LatentTopologyLayer.generate_seed_matrix(cfg)
        steps, _ = ConvolutionalDiffusionLayer.execute_pipeline(mat, ["diffuse_3x3"])
        pixels = HarmonicColorTensorLayer.colorize(steps, cfg)
        overlaid = EpistemicHudInlayLayer.apply_hud_overlay(pixels, cfg)
        self.assertEqual(len(overlaid), 24)

    def test_layer5_lossless_exporters(self):
        pixels = [[(10, 20, 30) for _ in range(8)] for _ in range(8)]
        png_bytes = MatrixImageRasterExporter.to_png_bytes(pixels, scale=2)
        # Check PNG magic header
        self.assertEqual(png_bytes[:8], b"\x89PNG\r\n\x1a\n")

        b64_png = MatrixImageRasterExporter.to_base64_png(pixels, scale=2)
        decoded = base64.b64decode(b64_png)
        self.assertEqual(decoded[:8], b"\x89PNG\r\n\x1a\n")

        svg_xml = MatrixImageRasterExporter.to_svg(pixels, scale=4)
        self.assertTrue(svg_xml.startswith("<svg"))
        self.assertTrue(svg_xml.endswith("</svg>"))

        ascii_art = MatrixImageRasterExporter.to_ascii(pixels)
        self.assertEqual(len(ascii_art.split("\n")), 8)

    def test_full_engine_generate_artifact(self):
        cfg = ImageGenerationConfig(
            width=16,
            height=16,
            seed=42,
            latent_mode=LatentSeedMode.EPICENTER_CRYSTAL,
            palette=ImageColorPalette.ALCHEMICAL_GOLD,
            scale_factor=4
        )
        artifact = GLOBAL_MATRIX_IMAGE_GENERATOR_ENGINE.generate_image(cfg)
        self.assertIsInstance(artifact, GeneratedImageArtifact)
        self.assertEqual(artifact.width, 16)
        self.assertEqual(artifact.height, 16)
        self.assertEqual(artifact.scale, 4)
        self.assertEqual(artifact.vital_max_hp, 6)
        self.assertGreater(len(artifact.png_base64), 100)
        self.assertGreater(len(artifact.svg_xml), 100)
        self.assertEqual(artifact.epistemic_status, "SAFE_VERIFIED_CONVOLVED")

        # Telemetry dict
        d = artifact.to_dict()
        self.assertEqual(d["vital_max_hp"], 6)
        self.assertTrue(d["png_data_uri"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
