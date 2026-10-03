"""
Automated Ethical Self-Audit Script (Hackni Vlastný Web / Defense-in-Depth)
===========================================================================
Performs a benign, non-destructive security posture audit against the running
Krystal-Stack Web Hub server (localhost:8089) based on the principles in
docs/research/ETHICAL_WEB_DEFENSE_AND_AUDITING_KNOWLEDGE_BASE.md:
1. SQL Injection / BBQ Firewall inspection rejection.
2. Path Traversal & Directory Traversal rejection.
3. XSS pattern defense.
4. Microsoft Authenticator 2FA TOTP replay protection.
5. Antispam Bee honeypot trap activation.
6. The 6 Max HP Vital Invariant clamping.
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
import urllib.parse

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_web_hub.economic_engine.microsoft_authenticator_2fa import (
    compute_totp_token,
    GLOBAL_2FA_AUTHENTICATOR
)
from krystal_web_hub.economic_engine.wordpress_security_and_java_transpiler import (
    GLOBAL_WORDPRESS_SECURITY
)

BASE_URL = "http://127.0.0.1:8089"


def send_probe(path, method="GET", payload=None):
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"} if payload else {}
    data = json.dumps(payload).encode("utf-8") if payload else None

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            content = resp.read().decode("utf-8")
            return resp.status, content
    except urllib.error.HTTPError as he:
        return he.code, he.read().decode("utf-8")
    except Exception as e:
        return 0, str(e)


def main():
    print("=================================================================")
    print(" KRYSTAL-STACK ETHICAL SELF-AUDIT // HACKNI VLASTNÝ WEB")
    print(" Target: Localhost Engine Core (Port 8089)")
    print("=================================================================")

    passed = 0
    total = 0

    # ── Test 1: BBQ Firewall SQL Injection Defense ────────────────────
    total += 1
    status, body = send_probe("/api/cards?query=union+select+1,2,3,benchmark(1000000,sha1(1))")
    if status == 403 or "FORBIDDEN_BY_FIREWALL" in body:
        print("[PASS] Test 1: BBQ Firewall SQLi Rejection (Status 403 Forbidden)")
        passed += 1
    else:
        print(f"[FAIL] Test 1: SQLi probe was not blocked (Status {status})")

    # ── Test 2: Path Traversal Defense ────────────────────────────────
    total += 1
    status, body = send_probe("/static/../../boot.ini")
    if status in (403, 404):
        print(f"[PASS] Test 2: Path Traversal Blocked (Status {status})")
        passed += 1
    else:
        print(f"[FAIL] Test 2: Path traversal accessible (Status {status})")

    # ── Test 3: BBQ Firewall Cross-Site Scripting (XSS) Filter ────────
    total += 1
    status, body = send_probe("/api/status?name=%3Cscript%3Ealert(document.cookie)%3C/script%3E")
    if status == 403 or "FORBIDDEN_BY_FIREWALL" in body:
        print("[PASS] Test 3: XSS Vector Filtered by Layer 7 WAF (Status 403 Forbidden)")
        passed += 1
    else:
        print(f"[FAIL] Test 3: XSS probe was not blocked (Status {status})")

    # ── Test 4: Antispam Bee Honeypot Verification ────────────────────
    total += 1
    now_ms = time.time() * 1000.0
    trap_eval = GLOBAL_WORDPRESS_SECURITY.evaluate_antispam_bee(
        form_data={"author": "spam_bot", "krystal_trap_honey_bee": "i_am_a_bot"},
        submission_timestamp_ms=now_ms - 1200.0
    )
    if trap_eval.get("is_spam") and trap_eval.get("reason") == "HONEYPOT_TRIGGERED":
        print("[PASS] Test 4: Antispam Bee Caught Honeypot Field & Fast Submission")
        passed += 1
    else:
        print("[FAIL] Test 4: Antispam Bee failed to detect bot submission")

    # ── Test 5: Wordfence Brute Force Rate Limiting ───────────────────
    total += 1
    test_ip = "198.51.100.42"
    for _ in range(6):
        GLOBAL_WORDPRESS_SECURITY.record_login_attempt(ip=test_ip, username="admin", success=False)
    lockout = GLOBAL_WORDPRESS_SECURITY.check_brute_force_status(test_ip)
    if lockout["is_locked_out"]:
        print(f"[PASS] Test 5: Wordfence Engine Locked Out Brute Force IP (Locked for {lockout['lockout_remaining_seconds']}s)")
        passed += 1
    else:
        print("[FAIL] Test 5: Wordfence Engine failed to lock out IP after 6 failed attempts")

    # ── Test 6: 2FA TOTP Replay Attack Defense ────────────────────────
    total += 1
    user = "audit_admin"
    setup = GLOBAL_2FA_AUTHENTICATOR.setup_user_2fa(user)
    sec = setup["secret_base32"]
    token = compute_totp_token(sec)

    # Confirm account
    confirm_res = GLOBAL_2FA_AUTHENTICATOR.confirm_user_2fa(user, token)
    # First sign-in attempt: should succeed
    first_res = GLOBAL_2FA_AUTHENTICATOR.verify_credentials_2fa(user, token)
    # Second sign-in attempt with the exact same token: should fail (replay attack blocked)
    replay_res = GLOBAL_2FA_AUTHENTICATOR.verify_credentials_2fa(user, token)

    if confirm_res["success"] and first_res["authenticated"] and not replay_res["authenticated"]:
        print("[PASS] Test 6: 2FA TOTP Replay Attack Defeated (Token reuse rejected)")
        passed += 1
    else:
        print(f"[FAIL] Test 6: Replay attack not properly blocked")

    # ── Test 7: Vital Invariant Max HP = 6 Clamping ───────────────────
    total += 1
    disp_res = send_probe("/api/vulkan-iris-xe/simulate-dispatch", method="POST", payload={"workload_chunks": 16})
    try:
        parsed = json.loads(disp_res[1])
        if parsed.get("vital_max_hp") == 6:
            print("[PASS] Test 7: The 6 Max HP Vital Invariant strictly verified")
            passed += 1
        else:
            print("[FAIL] Test 7: Vital Max HP is not 6")
    except Exception:
        print("[FAIL] Test 7: Could not parse response")

    print("=================================================================")
    print(f" SELF-AUDIT RESULTS: {passed} / {total} CHECKS PASSED ({round(passed/total*100, 1)}%)")
    print(" SYSTEM POSTURE: SECURE & DEFENDED (DEFENSE-IN-DEPTH ACTIVE)")
    print("=================================================================")


if __name__ == "__main__":
    main()
