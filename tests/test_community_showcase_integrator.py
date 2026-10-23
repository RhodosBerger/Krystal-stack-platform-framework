#!/usr/bin/env python3
# ==============================================================================
# TESTS FOR COMMUNITY SHOWCASE INTEGRATOR & AUDIT GUARD
# ==============================================================================

import unittest
from krystal_kernel.community_showcase_integrator import (
    XdpNetworkDescriptorBridge,
    MultiModelChatInferenceAdapter,
    FractalBilliardManifoldBridge,
    CommunityAuditGuard,
    AuditRequiredException,
    VITAL_MAX_HP
)


class TestCommunityShowcaseIntegrator(unittest.TestCase):
    """Verifies MIT showcase adapters and Apache/GPL audit gates."""

    def test_01_invariant_holds(self):
        """Constitutional invariant must strictly equal 6."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_02_xdp_packet_descriptor_128b_alignment(self):
        """XDP packet descriptor must align to exact 128-byte contract frame."""
        res = XdpNetworkDescriptorBridge.pack_xdp_packet_descriptor(
            src_ip="192.168.1.10",
            dst_ip="10.0.0.1",
            src_port=443,
            dst_port=5201,
            xdp_action=2
        )
        self.assertEqual(res["source_repo"], "gamemann/xdp-firewall")
        self.assertEqual(res["license"], "MIT")
        self.assertEqual(res["frame_size_bytes"], 128)
        self.assertTrue(res["is_128b_aligned"])
        self.assertEqual(res["xdp_action_name"], "XDP_PASS")
        self.assertEqual(res["vital_max_hp"], 6)

    def test_03_multi_model_chat_inference_dispatch(self):
        """Multi-model chat adapter must dispatch across GPU DP4A and CPU streams with physical telemetry text."""
        res = MultiModelChatInferenceAdapter.dispatch_parallel_comparison(
            prompt="Telemetria domácej stanice: Fáza L1 230V / 16A, teplota svorky 45.0°C, UMA zbernica 1.40µs, šum 26.0dB"
        )
        self.assertEqual(res["source_repo"], "seehiong/multi-model-chat")
        self.assertEqual(res["license"], "MIT")
        self.assertGreaterEqual(res["compared_models_count"], 3)
        self.assertEqual(res["vital_max_hp"], 6)

        # Overenie pôvodných nameraných hodnôt pre domáce stanice a počítače v textovej reprezentácii
        first_resp = res["comparison_results"][0]["simulated_completion"]
        self.assertIn("230.0V", first_resp)          # Napätie domácej stanice
        self.assertIn("T_L1=45.0°C", first_resp)      # Prahová teplota svorky fázy
        self.assertIn("I_N=1.20A", first_resp)        # Prúd nulovým vodičom
        self.assertIn("Tj=58.25°C", first_resp)       # Teplota kremíka počítača
        self.assertIn("UMA zbernica=1.40µs", first_resp) # Latencia UMA pamäťovej zbernice
        self.assertIn("Šum RL=26.0dB", first_resp)    # Šum / útlm odrazu napätia
        self.assertIn("945W", first_resp)             # Odľahčenie príkonu GPU
        self.assertIn("Bojler TÚV=65.0°C", first_resp) # Prahová teplota bojlera

    def test_04_fractal_billiard_orbit_generation(self):
        """Fractal billiard bridge must generate discrete orbit points and fractal dimension."""
        res = FractalBilliardManifoldBridge.compute_billiard_orbit(iterations=32)
        self.assertEqual(res["source_repo"], "xcontcom/billiard-fractals")
        self.assertEqual(res["license"], "MIT")
        self.assertEqual(res["orbit_points_count"], 32)
        self.assertTrue(res["sdf_manifold_compatible"])
        self.assertEqual(res["vital_max_hp"], 6)

    def test_05_community_audit_guard_blocks_unauthorized_gpl(self):
        """Accessing GPL or Apache repos without audit token must raise AuditRequiredException."""
        # Test GPL-3.0 repo
        with self.assertRaises(AuditRequiredException) as ctx:
            CommunityAuditGuard.evaluate_integration_eligibility("standardgalactic/standardgalactic")
        self.assertIn("VÝZVA NA AUDIT", str(ctx.exception))
        self.assertIn("GPL-3.0", str(ctx.exception))

        # Test Apache-2.0 repo
        with self.assertRaises(AuditRequiredException) as ctx2:
            CommunityAuditGuard.evaluate_integration_eligibility("seehiong/polyphon-audio")
        self.assertIn("VÝZVA NA AUDIT", str(ctx2.exception))
        self.assertIn("Apache-2.0", str(ctx2.exception))

    def test_06_community_audit_guard_passes_with_clearance(self):
        """Providing the audit approval token permits controlled integration."""
        res = CommunityAuditGuard.evaluate_integration_eligibility(
            "seehiong/polyphon-audio",
            audit_clearance_token="AUDIT_APPROVED_SEEHIONG_POLYPHON-AUDIO"
        )
        self.assertEqual(res["status"], "APPROVED")
        self.assertTrue(res["audit_verified"])
        self.assertEqual(res["vital_max_hp"], 6)

    def test_07_deterministic_execution_and_ipc_enforcement(self):
        """Ensures GPL/Apache approvals enforce IPC isolation and billiard orbits are deterministic."""
        # 1. Overenie, že schválený Apache/GPL modul vynucuje izolovaný proces (Princíp P1)
        clearance = CommunityAuditGuard.evaluate_integration_eligibility(
            "seehiong/polyphon-audio",
            audit_clearance_token="AUDIT_APPROVED_SEEHIONG_POLYPHON-AUDIO"
        )
        self.assertEqual(clearance.get("execution_boundary"), "ISOLATED_IPC_SOCKET")
        self.assertFalse(clearance.get("allow_direct_memory_link", True))

        # 2. Overenie bitového determinizmu pre fraktálny biliard (rovnaký vstup = identický hash)
        orbit_run_1 = FractalBilliardManifoldBridge.compute_billiard_orbit(iterations=32)
        orbit_run_2 = FractalBilliardManifoldBridge.compute_billiard_orbit(iterations=32)
        self.assertEqual(orbit_run_1["state_hash"], orbit_run_2["state_hash"])


if __name__ == "__main__":
    unittest.main()
