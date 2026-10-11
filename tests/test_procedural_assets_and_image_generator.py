#!/usr/bin/env python3
"""
==============================================================================
UNIT TEST SUITE: PROCEDURAL ASSET SYNTHESIZER & LOCAL LLM IMAGE GENERATOR
==============================================================================
Test Targets:
  1. krystal_kernel/procedural_asset_synthesizer.py
  2. krystal_kernel/local_llm_image_generator.py
Framework: unittest
Invariant: VITAL_MAX_HP == 6 strictly preserved across all assets and renders.
==============================================================================
"""

import os
import sys
import unittest
import base64

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.procedural_asset_synthesizer import (
    ProceduralAssetSynthesizer,
    ProceduralAsset,
    ProceduralAssetType,
    HarmonicColorTensor,
    GLOBAL_ASSET_SYNTHESIZER
)
from krystal_kernel.local_llm_image_generator import (
    LocalLLMImageGenerator,
    ImageGenerationResult,
    GLOBAL_LOCAL_IMAGE_GENERATOR
)


class TestProceduralAssetSynthesizer(unittest.TestCase):
    """Rigorous tests for multi-domain procedural asset synthesis."""

    def setUp(self):
        self.synth = ProceduralAssetSynthesizer()

    def test_vital_max_hp_invariant(self):
        """Platform invariant VITAL_MAX_HP == 6 must hold universally."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_harmonic_color_tensor(self):
        """Harmonic color tensor must produce valid RGB and Hex strings."""
        tensor = HarmonicColorTensor.from_seed(seed=42)
        self.assertGreaterEqual(tensor.hue_deg, 0.0)
        self.assertLess(tensor.hue_deg, 360.0)
        self.assertTrue(tensor.rgb_hex.startswith("#"))
        self.assertEqual(len(tensor.rgb_hex), 7)
        for c in tensor.rgb_tuple:
            self.assertGreaterEqual(c, 0)
            self.assertLessEqual(c, 255)

    def test_cnc_smart_tool_synthesis_and_thermal_dependency(self):
        """Tool asset must reflect cutting physics and thermal governor coupling."""
        # Cool nominal state (220 C)
        cool_tool = self.synth.synthesize_cnc_tool(seed=101, live_temp_c=220.0)
        self.assertEqual(cool_tool.asset_type, ProceduralAssetType.CNC_SMART_TOOL)
        self.assertIn("flank_wear_vb_mm", cool_tool.properties)
        self.assertEqual(cool_tool.vital_max_hp, 6)
        self.assertEqual(cool_tool.vital_hp, 6)
        self.assertFalse(cool_tool.visual_descriptor["heat_bluing_active"])

        # High thermal load state (650 C)
        hot_tool = self.synth.synthesize_cnc_tool(seed=101, live_temp_c=650.0)
        self.assertLessEqual(hot_tool.vital_hp, 4)
        self.assertTrue(hot_tool.visual_descriptor["heat_bluing_active"])
        self.assertGreater(hot_tool.properties["flank_wear_vb_mm"], cool_tool.properties["flank_wear_vb_mm"])

    def test_material_substrate_synthesis(self):
        """Material asset must incorporate thermophysical constants from MATERIAL_CATALOG."""
        mat_asset = self.synth.synthesize_material_substrate(seed=202)
        self.assertEqual(mat_asset.asset_type, ProceduralAssetType.MATERIAL_SUBSTRATE)
        self.assertIn("specific_cutting_force_kc_mpa", mat_asset.properties)
        self.assertIn("thermal_diffusivity_alpha", mat_asset.properties)
        self.assertGreater(mat_asset.properties["machinability_index"], 0.0)
        self.assertEqual(mat_asset.vital_hp, 6)
        self.assertEqual(mat_asset.vital_max_hp, 6)

    def test_skeuomorphic_artifact_synthesis(self):
        """Skeuomorphic artifact must enforce durability pips <= 6."""
        artifact = self.synth.synthesize_skeuomorphic_artifact(seed=303)
        self.assertEqual(artifact.asset_type, ProceduralAssetType.SKEUOMORPHIC_ARTIFACT)
        self.assertIn("resonance_frequency_hz", artifact.properties)
        self.assertGreaterEqual(artifact.vital_hp, 1)
        self.assertLessEqual(artifact.vital_hp, 6)
        self.assertEqual(artifact.vital_max_hp, 6)

    def test_computational_kernel_node_synthesis(self):
        """Compute node must couple with microarchitectural IPC and L2 cache hit %."""
        node = self.synth.synthesize_kernel_node(seed=404, ipc=1.85, cache_hit=98.2)
        self.assertEqual(node.asset_type, ProceduralAssetType.COMPUTATIONAL_KERNEL_NODE)
        self.assertEqual(node.properties["runtime_ipc"], 1.85)
        self.assertEqual(node.properties["l2_cache_hit_rate_pct"], 98.2)
        self.assertEqual(node.vital_hp, 6)
        self.assertEqual(node.vital_max_hp, 6)

    def test_procedural_biome_construct_synthesis(self):
        """Biome construct must follow Fermat/Gielis geometry and preserve 6 Max HP."""
        biome = self.synth.synthesize_biome_construct(seed=505)
        self.assertEqual(biome.asset_type, ProceduralAssetType.PROCEDURAL_BIOME_CONSTRUCT)
        self.assertIn("golden_spiral_constant", biome.properties)
        self.assertEqual(biome.vital_max_hp, 6)

    def test_random_asset_selection_diversity_and_determinism(self):
        """Selection must yield requested count across diverse types and be deterministic."""
        sel1 = self.synth.generate_random_asset_selection(count=5, seed=777)
        sel2 = self.synth.generate_random_asset_selection(count=5, seed=777)
        self.assertEqual(len(sel1), 5)
        self.assertEqual(len(sel2), 5)
        for a1, a2 in zip(sel1, sel2):
            self.assertEqual(a1.asset_id, a2.asset_id)
            self.assertEqual(a1.name, a2.name)
            self.assertEqual(a1.vital_hp, a2.vital_hp)
            self.assertEqual(a1.vital_max_hp, 6)


class TestLocalLLMImageGenerator(unittest.TestCase):
    """Tests for Local LLM Image Generator and Neuro-Symbolic Raster Engine."""

    def setUp(self):
        self.generator = LocalLLMImageGenerator()

    def test_neuro_symbolic_semantic_tokenizer_inference(self):
        """Tokenizer must infer palette, hero object, and vitality from prompt text."""
        spec_cyber = self.generator.neuro_symbolic_semantic_tokenizer(
            "Cybernetic neon carbide endmill spinning at 14000 RPM in dark laboratory",
            seed=42
        )
        self.assertEqual(spec_cyber["palette"], "cyberpunk")
        self.assertEqual(spec_cyber["center_object"], "cutting_tool")
        self.assertEqual(spec_cyber["vital_hp"], 6)
        self.assertEqual(spec_cyber["vital_max_hp"], 6)

        spec_hot = self.generator.neuro_symbolic_semantic_tokenizer(
            "Thermal red hot fiery tool tip damaged under severe cutting friction",
            seed=42
        )
        self.assertEqual(spec_hot["palette"], "thermal_flame")
        self.assertLessEqual(spec_hot["vital_hp"], 3)  # Damaged prompt triggers lower HP

    def test_canvas_rendering_and_formats(self):
        """Raster engine must produce valid PNG bytes, data URI, SVG, and ASCII."""
        res = self.generator.generate_image_from_prompt(
            prompt="Alchemical golden talisman on basalt altar with reflection",
            seed=123,
            width=32,
            height=32
        )
        self.assertEqual(res.status, "SUCCESS")
        self.assertIn("data:image/png;base64,", res.png_data_uri)
        
        # Verify valid PNG header in decoded base64 bytes: 0x89 'P' 'N' 'G'
        raw_png = base64.b64decode(res.png_base64)
        self.assertTrue(raw_png.startswith(b"\x89PNG\r\n\x1a\n"))

        # Verify SVG XML
        self.assertIn("<svg", res.svg_xml)
        self.assertIn("</svg>", res.svg_xml)

        # Verify ASCII
        self.assertGreater(len(res.ascii_art.splitlines()), 10)
        self.assertEqual(res.vital_max_hp, 6)

    def test_direct_asset_to_image_generation(self):
        """Direct asset portrait generation must render smoothly with accurate telemetry."""
        synth = ProceduralAssetSynthesizer()
        tool_asset = synth.synthesize_cnc_tool(seed=999, live_temp_c=250.0)
        
        res = self.generator.generate_image_from_asset(tool_asset, width=32, height=32)
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.generator_backend, "DIRECT_PROCEDURAL_ASSET_PAINTER")
        self.assertEqual(res.vital_hp, tool_asset.vital_hp)
        self.assertEqual(res.vital_max_hp, 6)
        self.assertGreater(len(res.png_base64), 500)


if __name__ == "__main__":
    unittest.main()
