#!/usr/bin/env python3
"""
Unit tests for Polygraphic DTP 3D Scene Generator & 2B Parameter Model Engine.
"""

import unittest
from krystal_kernel.polygraphic_dtp_generator import (
    PolygraphicDtpGeneratorEngine,
    GLOBAL_POLYGRAPHIC_DTP_ENGINE,
    DtpProductType,
    PaperStandard,
    ColorSeparationChannel,
    VulkanPipelineFormat,
    VulkanMsaaSamples,
    QuantizationPrecision,
    TwoBillionParameterModelSpec,
    DtpGeometrySpecs,
    VulkanRuntimeRenderParameters,
    CANONICAL_2B_PARAMETERS,
    VITAL_MAX_HP
)


class TestPolygraphicDtpGenerator(unittest.TestCase):
    def setUp(self):
        self.engine = PolygraphicDtpGeneratorEngine()

    def test_canonical_model_spec(self):
        spec = TwoBillionParameterModelSpec()
        self.assertEqual(spec.total_parameters, CANONICAL_2B_PARAMETERS)
        self.assertEqual(spec.total_parameters, 2_048_000_000)
        self.assertEqual(spec.num_layers, 24)
        self.assertEqual(spec.hidden_dimension, 2048)
        self.assertEqual(spec.num_attention_heads, 16)
        self.assertEqual(spec.num_kv_heads, 8)
        self.assertEqual(spec.quantization, QuantizationPrecision.INT4_AWQ)
        self.assertLess(spec.vram_footprint_mb, 1200.0)

    def test_geometry_and_prepress_boxes(self):
        geo = DtpGeometrySpecs(
            width_mm=210.0,
            height_mm=297.0,
            depth_mm=10.0,
            bleed_mm=3.0,
            margin_safe_mm=8.0
        )
        self.assertEqual(geo.trim_box, (210.0, 297.0))
        self.assertEqual(geo.bleed_box, (216.0, 303.0))
        self.assertEqual(geo.safe_box, (194.0, 281.0))
        self.assertEqual(geo.dpi, 300)

    def test_scene_synthesis_from_prompt(self):
        prompt = "Prémiová skladaná brožúra A4 trifold s matným laminom a striebornou razbou."
        scene = self.engine.generate_scene_from_prompt(prompt, product_type=DtpProductType.TRIFOLD_BROCHURE)
        self.assertEqual(scene.product_type, DtpProductType.TRIFOLD_BROCHURE)
        self.assertIn("TRIFOLD", scene.title)
        self.assertEqual(len(scene.geometry.fold_angles_deg), 2)
        self.assertEqual(scene.vital_max_hp, VITAL_MAX_HP)
        self.assertEqual(scene.vital_max_hp, 6)
        self.assertIn("Cyan", scene.cmyk_densities)
        self.assertIn("Magenta", scene.cmyk_densities)

    def test_direct_blender_editor_script(self):
        scene = self.engine.generate_scene_from_prompt("Luxusná kniha s pevnou väzbou a zlatou razbou.")
        script = scene.blender_python_script
        self.assertIn("import bpy", script)
        self.assertIn("import bmesh", script)
        self.assertIn("Krystal_DTP_3D_Scene", script)
        self.assertIn("ShaderNodeBsdfPrincipled", script)
        self.assertIn("CAM_DTP_Studio_Macro", script)
        self.assertIn("LIGHT_DTP_KeySoftbox", script)
        self.assertIn("krystal_dtp.generate_scene", script)
        self.assertIn("VITAL_MAX_HP = 6", script)

    def test_vulkan_pipeline_descriptor(self):
        scene = self.engine.generate_scene_from_prompt("Kozmetická škatuľka s lakom.")
        vk_desc = self.engine.generate_vulkan_pipeline_descriptor(scene)
        self.assertEqual(vk_desc["vulkan_version"], "1.3.268")
        self.assertEqual(len(vk_desc["render_pass"]["attachments"]), 6)
        self.assertEqual(len(vk_desc["render_pass"]["subpasses"]), 2)
        self.assertTrue(vk_desc["ray_tracing_khr"]["enabled"])
        self.assertEqual(vk_desc["push_constants"]["size_bytes"], 128)
        self.assertEqual(vk_desc["vital_max_hp"], 6)

    def test_vram_and_throughput_estimation(self):
        scene = self.engine.generate_scene_from_prompt("Vizitky")
        budget = self.engine.estimate_vram_and_throughput(scene)
        self.assertIn("model_weights_vram_mb", budget)
        self.assertIn("vulkan_framebuffers_vram_mb", budget)
        self.assertIn("total_estimated_vram_mb", budget)
        self.assertGreater(budget["total_estimated_vram_mb"], 3000.0)
        self.assertLess(budget["vulkan_predicted_frametime_ms"], 16.6)
        self.assertEqual(budget["vital_max_hp_rule"], 6)

    def test_engine_status_and_global_singleton(self):
        status = GLOBAL_POLYGRAPHIC_DTP_ENGINE.get_status()
        self.assertGreaterEqual(status["scenes_catalog_count"], 3)
        self.assertEqual(status["model_specification"]["total_parameters"], 2_048_000_000)
        self.assertIn("HARDCOVER_BOOK", status["supported_product_types"])
        self.assertIn("VK_FORMAT_R16G16B16A16_SFLOAT", status["supported_vulkan_formats"])
        self.assertEqual(status["vital_max_hp"], 6)


if __name__ == '__main__':
    unittest.main()
