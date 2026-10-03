"""
Unit Tests for WordPress Docker Security, Projector Analog ADC & Java Transpiler Engine
======================================================================================
Validates:
1. BBQ Firewall regex rules (SQLi, path traversal, RCE, XSS).
2. Antispam Bee honeypot and timing trap evaluation.
3. Wordfence Brute Force defense (rate limits, lockouts, attempts remaining).
4. Cryptographic password hashing (Argon2id and PBKDF2).
5. Projector stream analog-to-digital converter (ADC) & DirectML NPU tensor pipeline.
6. Procedural Island Realms (Denmark as Heaven, Finland as Slovakia, Czechia as Latvia,
   Germany as Poland, France as America) with strict 6 Max HP invariant.
7. Java 21 Transpiler generating modern records and parallel pipelines.
8. Janet DSL definition file integrity.
"""

import unittest
import os
import time
from krystal_web_hub.economic_engine.wordpress_security_and_java_transpiler import (
    VITAL_MAX_HP,
    WordPressSecurityEngine,
    ProjectorAnalogNpuBridge,
    IslandRealm,
    ProceduralIslandEngine,
    KrystalJavaTranspiler,
    GLOBAL_WORDPRESS_SECURITY,
    GLOBAL_PROJECTOR_ANALOG_BRIDGE,
    GLOBAL_PROCEDURAL_ISLAND_ENGINE,
    GLOBAL_JAVA_TRANSPILER
)


class TestWordPressSecurityAndJavaTranspiler(unittest.TestCase):

    def setUp(self):
        self.security = WordPressSecurityEngine()
        self.projector = ProjectorAnalogNpuBridge(scanlines=1080, adc_bit_depth=10, fps=60.0)
        self.islands = ProceduralIslandEngine()
        self.transpiler = KrystalJavaTranspiler()

    def test_vital_max_hp_constant_is_strictly_six(self):
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_cryptographic_password_hashing(self):
        raw_pw = "KrystalAdminSecure2026!#"
        argon_hash = self.security.hash_password(raw_pw, method="argon2id")
        self.assertTrue(argon_hash.startswith("$argon2id$v=19$m=65536"))
        self.assertTrue(self.security.verify_password(raw_pw, argon_hash))
        self.assertFalse(self.security.verify_password("WrongPassword123", argon_hash))

        pbkdf2_hash = self.security.hash_password(raw_pw, method="pbkdf2")
        self.assertTrue(pbkdf2_hash.startswith("$pbkdf2-sha512$100000$"))
        self.assertTrue(self.security.verify_password(raw_pw, pbkdf2_hash))
        self.assertFalse(self.security.verify_password("InvalidGuess!", pbkdf2_hash))

    def test_bbq_firewall_inspection(self):
        # 1. SQL Injection blocked
        res_sqli = self.security.inspect_bbq_firewall("id=1' UNION SELECT 1,2,3--")
        self.assertTrue(res_sqli["blocked"])
        self.assertEqual(res_sqli["threat_type"], "SQL_INJECTION")
        self.assertEqual(res_sqli["status_code"], 403)

        # 2. Path Traversal blocked
        res_trav = self.security.inspect_bbq_firewall("file=../../etc/passwd")
        self.assertTrue(res_trav["blocked"])
        self.assertEqual(res_trav["threat_type"], "PATH_TRAVERSAL")
        self.assertEqual(res_trav["status_code"], 403)

        # 3. RCE blocked
        res_rce = self.security.inspect_bbq_firewall("cmd=eval(base64_decode('...'))")
        self.assertTrue(res_rce["blocked"])
        self.assertEqual(res_rce["threat_type"], "REMOTE_CODE_EXECUTION")

        # 4. XSS blocked
        res_xss = self.security.inspect_bbq_firewall("search=<script>alert(1)</script>")
        self.assertTrue(res_xss["blocked"])
        self.assertEqual(res_xss["threat_type"], "CROSS_SITE_SCRIPTING")

        # 5. Clean query allowed
        res_clean = self.security.inspect_bbq_firewall("page=2&sort=name")
        self.assertFalse(res_clean["blocked"])
        self.assertEqual(res_clean["threat_type"], "NONE")
        self.assertEqual(res_clean["status_code"], 200)

    def test_antispam_bee_evaluation(self):
        now_ms = time.time() * 1000.0

        # Honeypot filled -> SPAM
        res_honeypot = self.security.evaluate_antispam_bee(
            form_data={"krystal_trap_honey_bee": "bot_content", "comment": "hello"},
            submission_timestamp_ms=now_ms - 5000.0,
            request_time_ms=now_ms
        )
        self.assertTrue(res_honeypot["is_spam"])
        self.assertEqual(res_honeypot["reason"], "HONEYPOT_TRIGGERED")

        # Too fast (< 3.0s) -> SPAM
        res_fast = self.security.evaluate_antispam_bee(
            form_data={"krystal_trap_honey_bee": "", "comment": "hello"},
            submission_timestamp_ms=now_ms - 1200.0,  # Only 1.2s
            request_time_ms=now_ms
        )
        self.assertTrue(res_fast["is_spam"])
        self.assertEqual(res_fast["reason"], "SUBMISSION_TOO_FAST")

        # Legitimate submission (>= 3.0s, empty honeypot) -> APPROVED
        res_valid = self.security.evaluate_antispam_bee(
            form_data={"krystal_trap_honey_bee": "", "comment": "Legitimate user inquiry."},
            submission_timestamp_ms=now_ms - 6500.0,  # 6.5s
            request_time_ms=now_ms
        )
        self.assertFalse(res_valid["is_spam"])
        self.assertEqual(res_valid["action"], "approved")

    def test_wordfence_brute_force_lockouts(self):
        test_ip = "192.168.1.188"
        # 4 failed attempts -> allowed=True, attempts remaining decreases
        for i in range(4):
            res = self.security.record_login_attempt(test_ip, "admin", success=False)
            self.assertTrue(res["allowed"])
            self.assertEqual(res["remaining_attempts"], 4 - i)

        # 5th failed attempt -> LOCKOUT
        lockout_res = self.security.record_login_attempt(test_ip, "admin", success=False)
        self.assertFalse(lockout_res["allowed"])
        self.assertEqual(lockout_res["action"], "locked_out")
        self.assertEqual(lockout_res["remaining_attempts"], 0)
        self.assertGreater(lockout_res["lockout_remaining_seconds"], 1700.0)

        # Status check while locked out
        status_res = self.security.check_brute_force_status(test_ip)
        self.assertTrue(status_res["is_locked_out"])
        self.assertEqual(status_res["remaining_attempts"], 0)

    def test_projector_analog_npu_bridge(self):
        frame = self.projector.convert_analog_frame_to_npu_tensor(frame_index=42, beam_intensity=0.8)
        self.assertEqual(frame["frame_index"], 42)
        self.assertEqual(frame["scanlines_total"], 1080)
        self.assertEqual(frame["resolution"], "1920x1080")
        self.assertEqual(frame["npu_tensor_shape"], [1, 3, 1080, 1920])
        self.assertGreater(frame["raw_stream_bitrate_gbps"], 2.0)
        self.assertGreater(len(frame["sampled_scanlines"]), 5)
        for sig in frame["sampled_scanlines"]:
            self.assertTrue(0.0 <= sig["digital_tensor_val"] <= 1.0)
            self.assertEqual(sig["sync_pulse_volts"], -0.3)

    def test_five_canonical_island_realms(self):
        realms = self.islands.get_all_realms()
        self.assertEqual(len(realms), 5)

        expected_realms = [
            ("denmark_heaven", "Dánsko ako Nebo"),
            ("finland_slovakia", "Fínsko ako Slovensko"),
            ("czechia_latvia", "Česko ako Lotyšsko"),
            ("germany_poland", "Nemecko ako Poľsko"),
            ("france_america", "Francúzsko ako Amerika")
        ]

        for r_id, expected_name_part in expected_realms:
            self.assertIn(r_id, realms)
            realm = realms[r_id]
            self.assertIn(expected_name_part, realm["display_name"])
            self.assertEqual(realm["vital_max_hp"], 6)
            self.assertGreater(len(realm["resource_nodes"]), 0)

        # Geometry test
        geo = self.islands.generate_island_geometry("denmark_heaven", seed=99)
        self.assertEqual(geo["realm_id"], "denmark_heaven")
        self.assertEqual(len(geo["shoreline_points"]), 24)
        for outpost in geo["outposts"]:
            self.assertEqual(outpost["hp"], 6)
            self.assertEqual(outpost["max_hp"], 6)

    def test_parallel_java_transpiler(self):
        realms_dict = self.islands._realms
        java_src = self.transpiler.generate_java_island_source(realms_dict)
        self.assertIn("package com.krystal.stack.procedural;", java_src)
        self.assertIn("public static final int VITAL_MAX_HP = 6;", java_src)
        self.assertIn("public sealed interface IslandCharacter permits", java_src)
        self.assertIn("record ProceduralIslandRealm(", java_src)
        self.assertIn("CompletableFuture", java_src)
        self.assertIn("Executors.newVirtualThreadPerTaskExecutor()", java_src)

    def test_janet_dsl_definition_file_exists(self):
        janet_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "krystal_janet",
            "wordpress_docker_security_and_java_engine.janet"
        )
        self.assertTrue(os.path.exists(janet_path))
        with open(janet_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("(def VITAL-MAX-HP 6)", content)
        self.assertIn("BBQ-FIREWALL-PATTERNS", content)
        self.assertIn("ANTISPAM-BEE-SPEC", content)
        self.assertIn("CANONICAL-ISLAND-REALMS", content)
        self.assertIn("evaluate-bbq-query", content)
        self.assertIn("denmark-heaven", content)


if __name__ == "__main__":
    unittest.main()
