import urllib.request
import json
import sys

BASE_URL = "http://127.0.0.1:8089"

def check_endpoint(method, path, data=None):
    url = f"{BASE_URL}{path}"
    req_data = json.dumps(data).encode('utf-8') if data is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            body = resp.read()
            print(f"[OK] {method} {path} -> {status} ({content_type}, {len(body)} bytes)")
            if "application/json" in content_type:
                parsed = json.loads(body.decode('utf-8'))
                if "vital_max_hp_rule" in parsed:
                    assert parsed["vital_max_hp_rule"] == 6, f"Invalid HP rule: {parsed['vital_max_hp_rule']}"
                    print(f"     => Vital Max HP = {parsed['vital_max_hp_rule']} (PASSED)")
            return True
    except Exception as e:
        print(f"[FAIL] {method} {path} -> {e}")
        return False

def main():
    print("=== VERIFYING KRYSTAL ENGINE ENDPOINTS ===")
    checks = [
        ("GET", "/static/greek_bohemia_pantheon_studio.html", None),
        ("GET", "/static/img/greek_bohemia_pantheon_memory.jpg", None),
        ("GET", "/api/greek-bohemia/pantheon", None),
        ("GET", "/api/greek-bohemia/memory-strategies", None),
        ("POST", "/api/greek-bohemia/simulate-leveling", {"axiom_ids": ["pythagoras_harmonic_stride", "aristotle_golden_mean", "heraclitus_panta_rhei"]}),
        ("POST", "/api/greek-bohemia/form-pact", {"greek_deity_id": "zeus", "bohemian_ally_id": "perun", "pact_title": "Blesk a Hrom", "lore": "Spojenie najvyšších vládcov nebies."}),
        ("GET", "/api/godot/camera/templates", None),
        ("GET", "/api/godot/models/catalog", None),
        ("POST", "/api/godot/camera/configure", {"template_id": "tps_orbit", "overrides": {"fov_deg": 85.0}}),
        ("POST", "/api/godot/package-manager/install", {"package_id": "camera-controller-3d"}),
        ("GET", "/api/health", None)
    ]

    all_passed = True
    for method, path, data in checks:
        if not check_endpoint(method, path, data):
            all_passed = False

    if all_passed:
        print("\nALL ENDPOINT VERIFICATIONS PASSED CLEANLY (100% SUCCESS)!")
        sys.exit(0)
    else:
        print("\nSOME CHECKS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
