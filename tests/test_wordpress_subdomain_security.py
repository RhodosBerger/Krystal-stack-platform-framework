"""
Unit & Integration Test Suite for WordPress Subdomain Security Gate & Krystal Reverse Proxy
"""

import unittest
import time
import json
import urllib.request
import urllib.parse
from krystal_web_hub.economic_engine.wordpress_security_and_java_transpiler import (
    WordPressSubdomainSecurityGate,
    SubdomainSession,
    GLOBAL_WORDPRESS_SUBDOMAIN_GATE,
    VITAL_MAX_HP
)

BASE_URL = "http://127.0.0.1:8089"


class TestWordPressSubdomainSecurity(unittest.TestCase):

    def setUp(self):
        self.gate = WordPressSubdomainSecurityGate(shared_secret="test_secret_key_12345")
        self.gate.add_allowed_subdomain_pattern(r"^krystal\.testdomena\.sk$")

    def test_vital_max_hp_invariant(self):
        self.assertEqual(self.gate.vital_max_hp, 6)
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_subdomain_whitelist_matching(self):
        self.assertTrue(self.gate.is_subdomain_allowed("krystal.testdomena.sk"))
        self.assertTrue(self.gate.is_subdomain_allowed("https://krystal.testdomena.sk/path"))
        self.assertTrue(self.gate.is_subdomain_allowed("app.krystal-stack.com"))
        self.assertTrue(self.gate.is_subdomain_allowed("localhost:8089"))
        self.assertTrue(self.gate.is_subdomain_allowed("127.0.0.1"))
        
        # Unauthorized subdomains
        self.assertFalse(self.gate.is_subdomain_allowed("malicious.attacker.com"))
        self.assertFalse(self.gate.is_subdomain_allowed("phishing.org"))

    def test_signed_token_lifecycle_and_verification(self):
        token = self.gate.create_signed_token(
            user_id=42,
            username="krystal_chieftain",
            role="administrator",
            subdomain="krystal.testdomena.sk"
        )
        self.assertIn(".", token)
        
        ok, reason, session = self.gate.verify_signed_token(token)
        self.assertTrue(ok, f"Verification failed with reason: {reason}")
        self.assertEqual(reason, "VERIFIED")
        self.assertIsNotNone(session)
        self.assertEqual(session.user_id, 42)
        self.assertEqual(session.username, "krystal_chieftain")
        self.assertEqual(session.role, "administrator")
        self.assertEqual(session.subdomain, "krystal.testdomena.sk")
        self.assertEqual(session.vital_max_hp, 6)

    def test_replay_attack_prevention(self):
        token = self.gate.create_signed_token(
            user_id=10,
            username="player_one",
            role="subscriber",
            subdomain="krystal.testdomena.sk"
        )
        # First verification succeeds
        ok1, reason1, session1 = self.gate.verify_signed_token(token)
        self.assertTrue(ok1)
        
        # Second verification with exact same token must fail due to nonce reuse
        ok2, reason2, session2 = self.gate.verify_signed_token(token)
        self.assertFalse(ok2)
        self.assertEqual(reason2, "REPLAY_ATTACK_DETECTED")
        self.assertIsNone(session2)

    def test_tampered_signature_rejection(self):
        token = self.gate.create_signed_token(
            user_id=1,
            username="admin",
            role="administrator",
            subdomain="krystal.testdomena.sk"
        )
        parts = token.split(".")
        tampered_token = parts[0] + "." + "deadbeef" * 8
        
        ok, reason, session = self.gate.verify_signed_token(tampered_token)
        self.assertFalse(ok)
        self.assertEqual(reason, "INVALID_HMAC_SIGNATURE")

    def test_expired_token_rejection(self):
        self.gate.token_ttl_seconds = -1.0 # Force immediate expiration
        token = self.gate.create_signed_token(
            user_id=7,
            username="expired_user",
            role="subscriber",
            subdomain="krystal.testdomena.sk"
        )
        ok, reason, session = self.gate.verify_signed_token(token)
        self.assertFalse(ok)
        self.assertEqual(reason, "TOKEN_EXPIRED")

    def test_unauthorized_subdomain_in_token_rejection(self):
        token = self.gate.create_signed_token(
            user_id=99,
            username="hacker",
            role="admin",
            subdomain="evil.malicious-site.com"
        )
        ok, reason, session = self.gate.verify_signed_token(token)
        self.assertFalse(ok)
        self.assertIn("UNAUTHORIZED_SUBDOMAIN", reason)

    def test_security_telemetry(self):
        telemetry = self.gate.get_security_telemetry()
        self.assertEqual(telemetry["vital_max_hp_rule"], 6)
        self.assertIn("verified_sessions_count", telemetry)
        self.assertIn("allowed_subdomain_patterns", telemetry)


class TestWordPressSubdomainHttpEndpoints(unittest.TestCase):

    def test_01_subdomain_security_status_endpoint(self):
        req = urllib.request.Request(f"{BASE_URL}/api/wordpress/subdomain/security-status")
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data.get("success"))
            self.assertEqual(data.get("vital_max_hp_rule"), 6)

    def test_02_generate_and_verify_token_endpoints(self):
        gen_url = f"{BASE_URL}/api/wordpress/subdomain/generate-token"
        req_body = json.dumps({
            "user_id": 105,
            "username": "bohemia_chieftain",
            "role": "player",
            "subdomain": "krystal.poslednikmen.cz"
        }).encode("utf-8")
        
        req = urllib.request.Request(gen_url, data=req_body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            gen_data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(gen_data.get("success"))
            token = gen_data.get("token")
            self.assertIsNotNone(token)

        # Verify the generated token
        verify_url = f"{BASE_URL}/api/wordpress/subdomain/verify-token"
        verify_body = json.dumps({"token": token}).encode("utf-8")
        req_v = urllib.request.Request(verify_url, data=verify_body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req_v, timeout=5) as resp_v:
            self.assertEqual(resp_v.status, 200)
            v_data = json.loads(resp_v.read().decode("utf-8"))
            self.assertTrue(v_data.get("success"))
            self.assertEqual(v_data.get("status"), "VERIFIED")
            self.assertEqual(v_data["session"]["user_id"], 105)
            self.assertEqual(v_data["session"]["username"], "bohemia_chieftain")
            self.assertEqual(v_data["session"]["vital_max_hp_rule"], 6)

    def test_03_invalid_token_rejected_endpoint(self):
        verify_url = f"{BASE_URL}/api/wordpress/subdomain/verify-token"
        verify_body = json.dumps({"token": "invalid.bogus_token"}).encode("utf-8")
        req_v = urllib.request.Request(verify_url, data=verify_body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req_v, timeout=5) as resp:
                self.assertEqual(resp.status, 401)
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 401)
            err_data = json.loads(e.read().decode("utf-8"))
            self.assertFalse(err_data.get("success"))
            self.assertEqual(err_data.get("status"), "REJECTED")


if __name__ == "__main__":
    unittest.main()
