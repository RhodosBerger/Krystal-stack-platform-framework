"""
Verification of WordPress Subdomain Security & Reverse Proxy Architecture Flow
"""

import urllib.request
import urllib.parse
import json
import sys

BASE_URL = "http://127.0.0.1:8089"


def main():
    print("=================================================================")
    print(" VERIFYING END-TO-END WORDPRESS SUBDOMAIN SECURITY FLOW")
    print("=================================================================")

    # Step 1: Health & Telemetry
    status_url = f"{BASE_URL}/api/wordpress/subdomain/security-status"
    req = urllib.request.Request(status_url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert data["success"] is True
        assert data["vital_max_hp_rule"] == 6
        print("[PASS] 1. Kernel Security Status: ONLINE, Max HP Invariant = 6")

    # Step 2: BBQ WAF Attack Simulation
    attacks = [
        ("Legitimate Query", "/api/islands/geometry?realm_id=denmark_heaven", True),
        ("SQL Injection", "/api/islands?q=%27%20union%20select%201,2,3--", False),
        ("Path Traversal", "/api/cards?file=../../../../etc/passwd", False),
        ("RCE Exploit", "/api/eval?cmd=eval(base64_decode('test'))", False),
        ("XSS Payload", "/api/view?name=%3Cscript%3Ealert(1)%3C/script%3E", False)
    ]

    for name, test_uri, expected_pass in attacks:
        bbq_url = f"{BASE_URL}/api/wordpress/security/inspect_bbq"
        payload = json.dumps({"query_string": test_uri, "request_uri": test_uri}).encode("utf-8")
        req = urllib.request.Request(bbq_url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            is_blocked = res_data["result"]["blocked"]
            if expected_pass:
                assert not is_blocked, f"Legitimate request incorrectly blocked: {name}"
                print(f"[PASS] 2. BBQ WAF allowed legitimate request: {name}")
            else:
                assert is_blocked, f"Malicious request not blocked: {name}"
                print(f"[PASS] 2. BBQ WAF blocked attack: {name} -> Threat: {res_data['result']['threat_type']}")

    # Step 3: WordPress Token Minting
    gen_url = f"{BASE_URL}/api/wordpress/subdomain/generate-token"
    token_req = json.dumps({
        "user_id": 77,
        "username": "krystal_archon",
        "role": "administrator",
        "subdomain": "krystal.poslednikmen.cz"
    }).encode("utf-8")
    req = urllib.request.Request(gen_url, data=token_req, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        t_data = json.loads(resp.read().decode("utf-8"))
        token = t_data["token"]
        assert token and "." in token
        print(f"[PASS] 3. WordPress Token Minted: {token[:24]}... [Role: {t_data['user_id']}]")

    # Step 4: Token Verification
    ver_url = f"{BASE_URL}/api/wordpress/subdomain/verify-token"
    v_req = json.dumps({"token": token}).encode("utf-8")
    req = urllib.request.Request(ver_url, data=v_req, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        v_data = json.loads(resp.read().decode("utf-8"))
        assert v_data["status"] == "VERIFIED"
        assert v_data["session"]["vital_max_hp_rule"] == 6
        print(f"[PASS] 4. Token Verified by Kernel: User='{v_data['session']['username']}', HP={v_data['session']['vital_max_hp_rule']}")

    # Step 5: Anti-Replay Defense
    try:
        req = urllib.request.Request(ver_url, data=v_req, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            print("[FAIL] Replay attack was not rejected!")
            sys.exit(1)
    except urllib.error.HTTPError as e:
        assert e.code == 401
        err_res = json.loads(e.read().decode("utf-8"))
        assert err_res["reason"] == "REPLAY_ATTACK_DETECTED"
        print(f"[PASS] 5. Replay Attack Detected & Rejected: {err_res['reason']}")

    # Step 6: Verify HTML Studio Route
    studio_url = f"{BASE_URL}/wordpress-subdomain-security"
    with urllib.request.urlopen(studio_url, timeout=5) as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        assert "WORDPRESS SUBDOMAIN SECURITY GATE" in content
        print(f"[PASS] 6. HTML Studio Loaded: {len(content)} bytes")

    print("=================================================================")
    print(" ALL 6 SUBDOMAIN SECURITY CHECKS PASSED PERFECTLY!")
    print("=================================================================")


if __name__ == "__main__":
    main()
