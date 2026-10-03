"""
Unit Tests for Microsoft Authenticator Two-Factor Authentication (2FA) Engine
==============================================================================
Validates:
1. RFC 6238 Time-Based One-Time Password (TOTP) token generation & verification.
2. Clock drift tolerance window (+-30s).
3. Pure Python vector SVG QR code generation for otpauth:// URIs.
4. User setup, provisional state, confirmation, and active state transitions.
5. Emergency one-time backup recovery codes and single-use consumption.
6. Microsoft Authenticator 2-digit number matching challenge protocol.
7. Strict 6 Max HP vital invariant across authenticated sessions.
8. Janet DSL definition file integrity.
"""

import unittest
import os
import time
from krystal_web_hub.economic_engine.microsoft_authenticator_2fa import (
    VITAL_MAX_HP,
    TIME_STEP_SECONDS,
    TOKEN_DIGITS,
    compute_totp_token,
    verify_totp_token,
    generate_svg_qr_code,
    MicrosoftAuthenticator2FAEngine,
    GLOBAL_2FA_AUTHENTICATOR
)


class TestMicrosoftAuthenticator2FA(unittest.TestCase):

    def setUp(self):
        self.engine = MicrosoftAuthenticator2FAEngine(issuer_name="KrystalTest")
        self.test_secret = "JBSWY3DPEHPK3PXPJBSWY3DPEHPK3PXP"  # Standard RFC 32-char Base32

    def test_vital_max_hp_constant_is_strictly_six(self):
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_totp_token_computation(self):
        now = 1700000000.0  # Deterministic test timestamp
        token = compute_totp_token(self.test_secret, unix_timestamp=now)
        self.assertEqual(len(token), TOKEN_DIGITS)
        self.assertTrue(token.isdigit())

        # Exact match verification
        self.assertTrue(verify_totp_token(self.test_secret, token, drift_window=0, unix_timestamp=now))
        # Wrong token fails
        self.assertFalse(verify_totp_token(self.test_secret, "000000", drift_window=0, unix_timestamp=now))

    def test_totp_drift_tolerance_window(self):
        now = 1700000000.0
        current_token = compute_totp_token(self.test_secret, unix_timestamp=now)
        prev_token = compute_totp_token(self.test_secret, unix_timestamp=now - TIME_STEP_SECONDS)
        next_token = compute_totp_token(self.test_secret, unix_timestamp=now + TIME_STEP_SECONDS)
        distant_token = compute_totp_token(self.test_secret, unix_timestamp=now - (TIME_STEP_SECONDS * 4))

        # Drift window of 1 allows T-1, T, T+1
        self.assertTrue(verify_totp_token(self.test_secret, current_token, drift_window=1, unix_timestamp=now))
        self.assertTrue(verify_totp_token(self.test_secret, prev_token, drift_window=1, unix_timestamp=now))
        self.assertTrue(verify_totp_token(self.test_secret, next_token, drift_window=1, unix_timestamp=now))

        # Outside window fails
        self.assertFalse(verify_totp_token(self.test_secret, distant_token, drift_window=1, unix_timestamp=now))

    def test_svg_qr_code_generator(self):
        uri = "otpauth://totp/KrystalTest:admin?secret=JBSWY3DPEHPK3PXP&issuer=KrystalTest"
        svg = generate_svg_qr_code(uri, box_size=8, border=4)
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))
        self.assertIn('fill="#06b6d4"', svg)
        self.assertIn('viewBox="0 0', svg)

    def test_user_2fa_setup_and_confirmation_lifecycle(self):
        user = "test_sysadmin"
        setup = self.engine.setup_user_2fa(user)
        self.assertEqual(setup["username"], user)
        self.assertEqual(setup["status"], "pending_confirmation")
        self.assertEqual(setup["digits"], 6)
        self.assertEqual(setup["period_seconds"], 30)
        self.assertTrue(setup["otpauth_uri"].startswith("otpauth://totp/"))
        self.assertEqual(len(setup["backup_codes"]), 8)

        # Confirm with invalid token -> fails
        fail_res = self.engine.confirm_user_2fa(user, "999999")
        self.assertFalse(fail_res["success"])
        self.assertFalse(self.engine.get_user_status(user)["is_active"])

        # Confirm with correct token -> succeeds
        valid_tok = compute_totp_token(setup["secret_base32"])
        ok_res = self.engine.confirm_user_2fa(user, valid_tok)
        self.assertTrue(ok_res["success"])
        self.assertTrue(self.engine.get_user_status(user)["is_active"])

    def test_authentication_with_token_and_emergency_backup_codes(self):
        user = "backup_test_user"
        setup = self.engine.setup_user_2fa(user)
        valid_tok = compute_totp_token(setup["secret_base32"])
        self.engine.confirm_user_2fa(user, valid_tok)

        # Authenticate with current TOTP token
        auth_totp = self.engine.verify_credentials_2fa(user, valid_tok)
        self.assertTrue(auth_totp["authenticated"])
        self.assertEqual(auth_totp["method"], "totp_token")
        self.assertEqual(auth_totp["vital_max_hp"], 6)

        # Authenticate with one-time backup recovery code
        backup_code = setup["backup_codes"][0]
        auth_backup = self.engine.verify_credentials_2fa(user, backup_code)
        self.assertTrue(auth_backup["authenticated"])
        self.assertEqual(auth_backup["method"], "emergency_backup_code")
        self.assertEqual(auth_backup["remaining_backup_codes"], 7)

        # Re-using the same backup code must FAIL (single-use invariant)
        reuse_fail = self.engine.verify_credentials_2fa(user, backup_code)
        self.assertFalse(reuse_fail["authenticated"])

    def test_microsoft_authenticator_number_matching_challenge(self):
        user = "challenge_user"
        chal = self.engine.create_number_matching_challenge(user)
        c_id = chal["challenge_id"]
        c_num = chal["challenge_number"]

        self.assertTrue(10 <= c_num <= 99)
        self.assertEqual(chal["status"], "pending")

        # Wrong number selected -> rejected
        wrong_res = self.engine.verify_number_matching_challenge(c_id, (c_num + 1) % 100)
        self.assertFalse(wrong_res["success"])
        self.assertEqual(wrong_res["status"], "rejected")

        # Correct number selected -> approved
        ok_chal = self.engine.create_number_matching_challenge(user)
        correct_res = self.engine.verify_number_matching_challenge(ok_chal["challenge_id"], ok_chal["challenge_number"])
        self.assertTrue(correct_res["success"])
        self.assertEqual(correct_res["status"], "approved")
        self.assertEqual(correct_res["vital_max_hp"], 6)

    def test_janet_dsl_definition_file_exists(self):
        janet_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "krystal_janet",
            "microsoft_authenticator_totp.janet"
        )
        self.assertTrue(os.path.exists(janet_path))
        with open(janet_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("(def VITAL-MAX-HP 6)", content)
        self.assertIn("TOTP-SPEC", content)
        self.assertIn("NUMBER-MATCHING-SPEC", content)
        self.assertIn("BACKUP-CODES-SPEC", content)
        self.assertIn("compute-time-counter", content)


if __name__ == "__main__":
    unittest.main()
