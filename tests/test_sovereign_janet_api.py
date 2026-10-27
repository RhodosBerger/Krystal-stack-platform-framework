"""
tests/test_sovereign_janet_api.py

Unit and Invariant Verification Suite for Janet Sovereign System API & Bridge.
Constitutional Invariant: VITAL_MAX_HP == 6 (not Valid(A) => Delta S == 0)
"""

import os
import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_janet.janet_bridge import JanetValidator
from krystal_janet.sovereign_janet_api_bridge import (
    JanetCommandKind,
    JanetCommandReceipt,
    SovereignJanetSystemBridge,
    GLOBAL_SOVEREIGN_JANET_SYSTEM_BRIDGE
)


class TestSovereignJanetApi(unittest.TestCase):

    def setUp(self) -> None:
        self.bridge = SovereignJanetSystemBridge(vital_max_hp=VITAL_MAX_HP)
        self.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.janet_api_path = os.path.join(self.repo_root, "krystal_janet", "sovereign_system_api.janet")

    def test_01_sovereign_system_api_janet_syntax_and_brackets(self) -> None:
        """Verifies that sovereign_system_api.janet exists and has strictly balanced brackets."""
        self.assertTrue(os.path.isfile(self.janet_api_path))
        res = JanetValidator.validate_file(self.janet_api_path)
        self.assertTrue(res["valid"], f"Bracket imbalance detected: {res.get('bracket_counts')}")
        self.assertGreaterEqual(res["line_count"], 50)
        self.assertIn("query-silicon-pdn", res["definitions"])
        self.assertIn("evaluate-chronometric-shadow", res["definitions"])
        self.assertIn("step-symplectic-phase", res["definitions"])
        self.assertIn("evaluate-constitutional-gatekeeper", res["definitions"])

    def test_02_janet_bridge_initialization(self) -> None:
        """Verifies bridge initialization preserves VITAL_MAX_HP == 6."""
        self.assertEqual(self.bridge.vital_max_hp, VITAL_MAX_HP)

    def test_03_l0_silicon_pdn_connection(self) -> None:
        """Tests Janet L0 query evaluates voltage flyback and thermal envelope."""
        receipt = self.bridge.dispatch_janet_command(
            "query-silicon-pdn",
            {"vdd": 0.85, "delta_i_amps": 15.0}
        )
        self.assertTrue(receipt.is_success)
        self.assertEqual(receipt.status_code, "PDN_ENVELOPE_EVALUATED")
        self.assertLessEqual(receipt.output_payload["peak_voltage_v"], 1.00)
        self.assertEqual(receipt.vital_max_hp, VITAL_MAX_HP)

    def test_04_l1_chronometric_shadow_connection(self) -> None:
        """Tests Janet L1 query evaluates 60Hz VBlank shadow window >= 3.50ms."""
        receipt = self.bridge.dispatch_janet_command(
            "evaluate-chronometric-shadow",
            {"refresh_hz": 60.0, "bus_load": 0.55}
        )
        self.assertTrue(receipt.is_success)
        self.assertGreaterEqual(receipt.output_payload["available_shadow_budget_ms"], 3.50)
        self.assertTrue(receipt.output_payload["is_feasible"])

    def test_05_l2_symplectic_phase_connection(self) -> None:
        """Tests Janet L2 query steps Leapfrog particle and verifies det(J) == 1.000000."""
        receipt = self.bridge.dispatch_janet_command(
            "step-symplectic-phase",
            {"q": 1.0, "p": 0.0, "dt": 0.01}
        )
        self.assertTrue(receipt.is_success)
        self.assertEqual(receipt.output_payload["jacobian_determinant"], 1.000000)
        self.assertIn("0x", receipt.output_payload["sector_code"])
        self.assertTrue(receipt.state_mutated)

    def test_06_l2_hex_hsl_sector_mapping(self) -> None:
        """Tests Janet D6 dihedral Hex-HSL sector angle resolution."""
        # 0x10 -> Sector 1 -> 0 deg (Red)
        r1 = self.bridge.dispatch_janet_command("map-hex-hsl-sector", {"sector_byte": 0x10})
        self.assertEqual(r1.output_payload["hue_degrees"], 0.0)

        # 0x40 -> Sector 4 -> 180 deg (Cyan)
        r4 = self.bridge.dispatch_janet_command("map-hex-hsl-sector", {"sector_byte": 0x40})
        self.assertEqual(r4.output_payload["hue_degrees"], 180.0)

    def test_07_l3_dwell_hysteresis_anti_chatter(self) -> None:
        """Tests Janet L3 dwell counter enforces 12-frame stability for upscaling."""
        # Frame 5 (below 12) -> held
        r_hold = self.bridge.dispatch_janet_command(
            "evaluate-dwell-hysteresis",
            {"current_dwell": 5, "threshold": 12, "requested_upscale": True}
        )
        self.assertFalse(r_hold.output_payload["allowed"])
        self.assertTrue(r_hold.output_payload["anti_chatter_active"])

        # Frame 12 -> authorized
        r_auth = self.bridge.dispatch_janet_command(
            "evaluate-dwell-hysteresis",
            {"current_dwell": 12, "threshold": 12, "requested_upscale": True}
        )
        self.assertTrue(r_auth.output_payload["allowed"])

    def test_08_l4_zero_copy_128b_frame_packing(self) -> None:
        """Tests Janet L4 contract frame serialization and verification."""
        receipt = self.bridge.dispatch_janet_command(
            "build-128b-frame-descriptor",
            {"contract_type": 1, "source_node": 1, "target_node": 2, "payload_values": [0.85, 0.08, 62.0]}
        )
        self.assertTrue(receipt.is_success)
        self.assertEqual(receipt.output_payload["frame_byte_length"], 128)
        self.assertTrue(receipt.output_payload["is_valid_128b"])

    def test_09_l5_constitutional_gatekeeper_and_zero_mutation(self) -> None:
        """Tests Janet L5 gatekeeper: rejects dangerous delta with strictly Delta S == 0."""
        # Breach attempt: +0.20V on 0.85V -> 1.05V > 1.00V ceiling
        r_breach = self.bridge.dispatch_janet_command(
            "evaluate-constitutional-gatekeeper",
            {"variable_id": "ELEM_CORE_VOLTAGE", "proposed_delta": 0.20}
        )
        self.assertFalse(r_breach.is_success)
        self.assertFalse(r_breach.state_mutated)
        self.assertEqual(r_breach.delta_s, 0.0)

        # Safe intervention: +0.03V on 0.85V -> 0.88V <= 1.00V
        r_safe = self.bridge.dispatch_janet_command(
            "evaluate-constitutional-gatekeeper",
            {"variable_id": "ELEM_CORE_VOLTAGE", "proposed_delta": 0.03}
        )
        self.assertTrue(r_safe.is_success)
        self.assertTrue(r_safe.state_mutated)
        self.assertEqual(r_safe.delta_s, 0.03)

    def test_10_guidebook_and_alpha_pulse_queries(self) -> None:
        """Tests Janet querying guidebook catalog and 10 Hz alpha cognitive rhythm."""
        r_guide = self.bridge.dispatch_janet_command("query-guidebook-catalog-summary", {})
        self.assertTrue(r_guide.is_success)
        self.assertGreaterEqual(r_guide.output_payload["total_sections"], 8)

        r_pulse = self.bridge.dispatch_janet_command("compute-alpha-harmonic-pulse", {"elapsed_sec": 0.025})
        self.assertTrue(r_pulse.is_success)
        self.assertAlmostEqual(r_pulse.output_payload["pulse_value"], 1.0, delta=1e-3)

    def test_11_direct_sexp_evaluation(self) -> None:
        """Tests parsing and executing raw Janet S-expression strings."""
        sexp = '(query-silicon-pdn :vdd 0.85 :delta_i_amps 12.0)'
        receipt = self.bridge.eval_sexp_string(sexp)
        self.assertTrue(receipt.is_success)
        self.assertEqual(receipt.command, "query-silicon-pdn")

    def test_12_constitutional_invariant_breach_prevention(self) -> None:
        """Verifies ContractViolation is raised if vital_max_hp != 6 is attempted."""
        with self.assertRaises(ContractViolation):
            SovereignJanetSystemBridge(vital_max_hp=4)


if __name__ == "__main__":
    unittest.main()
