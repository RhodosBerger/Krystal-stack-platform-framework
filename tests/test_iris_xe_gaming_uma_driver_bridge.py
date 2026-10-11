#!/usr/bin/env python3
"""
==============================================================================
UNIT TEST SUITE: IRIS XE GAMING UMA DRIVER BRIDGE (DX12 / VULKAN & LOCAL LLM)
==============================================================================
Module: tests/test_iris_xe_gaming_uma_driver_bridge.py
Description: Validates Ubisoft Snowdrop (The Division) and FromSoftware (Elden Ring)
             gaming profiles, Asahi UMA slab allocation, local LLM VSync interleaving,
             and drop-in community configuration generation.

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

from krystal_kernel.iris_xe_gaming_uma_driver_bridge import (
    TargetGameEngine,
    UmaAllocationStrategy,
    GraphicsApiBackend,
    EnginePerformanceTarget,
    AsahiMemorySlabReport,
    LlmGamingCoexistenceStatus,
    AsahiUmaMemoryManager,
    IrisXeGamingOptimizationEngine,
    LocalLlmGamingCoexistenceGovernor,
    CommunityMvpArtifactGenerator,
    GLOBAL_ASAHI_UMA_MANAGER,
    GLOBAL_GAMING_OPTIMIZATION_ENGINE,
    GLOBAL_COEXISTENCE_GOVERNOR,
    VITAL_MAX_HP
)


class TestIrisXeGamingUmaDriverBridge(unittest.TestCase):
    """Test suite for the Gaming UMA driver bridge and local LLM coexistence."""

    def test_01_invariant_constant(self):
        """Ensures the sacred system invariant is strictly 6."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_02_ubisoft_snowdrop_profile(self):
        """Tests Ubisoft Snowdrop Engine profile for The Division at 1080p 30 FPS Medium."""
        profile = IrisXeGamingOptimizationEngine.get_profile_ubisoft_snowdrop()
        self.assertEqual(profile.engine_name, TargetGameEngine.UBISOFT_SNOWDROP)
        self.assertEqual(profile.target_framerate_fps, 30.0)
        self.assertGreaterEqual(profile.framerate_1pct_low_target, 28.0)
        self.assertIn("1920x1080", profile.target_resolution)
        self.assertTrue(profile.dx12_async_compute_active)
        self.assertLess(profile.optimized_bandwidth_gb_s, profile.memory_bandwidth_demand_gb_s)
        self.assertEqual(profile.vital_max_hp, 6)

    def test_03_elden_ring_profile(self):
        """Tests FromSoftware Elden Ring profile at 30 FPS Medium."""
        profile = IrisXeGamingOptimizationEngine.get_profile_elden_ring()
        self.assertEqual(profile.engine_name, TargetGameEngine.FROMSOFTWARE_ELDEN)
        self.assertEqual(profile.target_framerate_fps, 30.0)
        self.assertGreaterEqual(profile.framerate_1pct_low_target, 28.0)
        self.assertTrue(profile.shader_cache_prewarmed)
        self.assertEqual(profile.vital_max_hp, 6)

    def test_04_asahi_uma_slab_partitioning(self):
        """Tests Asahi Linux-style UMA memory slab partitioning."""
        mgr = AsahiUmaMemoryManager(system_ram_gb=16.0)
        report = mgr.allocate_coexistence_pools(TargetGameEngine.FROMSOFTWARE_ELDEN, llm_model_size_params_b=7.0)

        self.assertIsInstance(report, AsahiMemorySlabReport)
        self.assertEqual(report.total_physical_ram_mb, 16384)
        self.assertGreaterEqual(report.game_vram_placed_heap_mb, 3500)
        self.assertGreaterEqual(report.local_llm_weights_quota_mb, 3500)
        self.assertTrue(report.zero_copy_active)
        self.assertTrue(report.wddm_128mb_clamp_bypassed)
        self.assertGreater(report.transient_alias_savings_mb, 500)
        self.assertEqual(report.vital_max_hp, 6)

    def test_05_local_llm_gaming_coexistence_governor(self):
        """Tests VSync blanking window token micro-slicing."""
        status = LocalLlmGamingCoexistenceGovernor.evaluate_coexistence_cycle(
            game_render_time_ms=20.5,
            target_fps=30.0,
            game_profile=IrisXeGamingOptimizationEngine.get_profile_elden_ring()
        )
        self.assertIsInstance(status, LlmGamingCoexistenceStatus)
        self.assertEqual(status.game_fps_actual, 30.0)
        self.assertEqual(status.frame_budget_ms, 33.33)
        self.assertGreater(status.vsync_blanking_window_ms, 10.0)
        self.assertTrue(status.coexistence_stable)
        self.assertGreater(status.llm_tokens_generated_per_sec, 10.0)
        self.assertEqual(status.vital_max_hp, 6)

    def test_06_community_mvp_configs_generation(self):
        """Tests community dxvk/vkd3d and registry configuration exporters."""
        dxvk_cfg = CommunityMvpArtifactGenerator.generate_dxvk_vkd3d_config("TheDivision2")
        self.assertIn("dxvk.enableSubgroupSizeControl = True", dxvk_cfg)
        self.assertIn("dxvk.requiredSubgroupSize = 16", dxvk_cfg)
        self.assertIn("vkd3d.enableAsyncCompute = True", dxvk_cfg)

        reg_patch = CommunityMvpArtifactGenerator.generate_windows_registry_patch(vram_mb=4096)
        self.assertIn("DedicatedSegmentSize", reg_patch)
        self.assertIn("Tile4Preferred", reg_patch)


if __name__ == "__main__":
    unittest.main()
