"""
Verification script for Greek Bohemia Pantheon & Godot Canvas 3D Arena
=============================================================================
Validates all REST API endpoints and Static Studio Web pages on Port 8089.
=============================================================================
"""

import sys
import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8089"


def test_endpoint(method: str, path: str, payload=None, expected_status=200):
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode('utf-8') if payload else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            status = resp.status
            body = resp.read()
            print(f"[PASS] {method} {path} -> {status} (Bytes: {len(body)})")
            if "application/json" in resp.headers.get("Content-Type", ""):
                parsed = json.loads(body.decode('utf-8'))
                return True, parsed
            return True, body
    except urllib.error.HTTPError as e:
        print(f"[FAIL] {method} {path} -> HTTP {e.code}: {e.reason}")
        return False, None
    except Exception as e:
        print(f"[ERROR] {method} {path} -> {e}")
        return False, None


def main():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE ENDPOINT VERIFICATION ON PORT 8089")
    print("=" * 70)

    all_passed = True

    # 1. Greek Bohemia Pantheon GET
    p_ok, p_data = test_endpoint("GET", "/api/greek-bohemia/pantheon")
    if p_ok and p_data.get("vital_max_hp_rule") == 6:
        g_cnt = p_data.get('greek_deities_count', len(p_data.get('greek_deities', [])))
        b_cnt = p_data.get('bohemian_allies_count', len(p_data.get('bohemian_allies', [])))
        print(f"  -> Greek Bohemia Pantheon OK (Greek Deities: {g_cnt}, Bohemian Allies: {b_cnt})")
    else:
        all_passed = False

    # 2. Greek Bohemia Strategies GET
    s_ok, s_data = test_endpoint("GET", "/api/greek-bohemia/memory-strategies")
    if s_ok and len(s_data.get("strategies", [])) >= 6:
        print(f"  -> Philosophical Memory Axioms OK ({len(s_data['strategies'])} strategies)")
    else:
        all_passed = False

    # 3. Simulate Leveling POST
    l_ok, l_data = test_endpoint("POST", "/api/greek-bohemia/simulate-leveling", payload={"axiom_ids": ["pythagoras_harmonics", "heraclitus_flux"]})
    if l_ok and l_data.get("vital_max_hp_rule") == 6:
        print(f"  -> Memory Leveling Convergence OK (Status: {l_data.get('status')})")
    else:
        all_passed = False

    # 4. Form Pact POST
    pact_ok, pact_data = test_endpoint("POST", "/api/greek-bohemia/form-pact", payload={
        "greek_deity_id": "zeus",
        "bohemian_ally_id": "perun",
        "pact_title": "Blesková Zmluva",
        "lore": "Bleskové urýchlenie vyrovnávania pamäťových blokov"
    })
    if pact_ok and pact_data.get("success") is True and pact_data.get("pact", {}).get("vital_max_hp") == 6:
        print(f"  -> Pantheon Pact Formed OK: {pact_data.get('pact', {}).get('title')}")
    else:
        all_passed = False

    # 5. Greek Bohemia Studio Static HTML
    html_ok, _ = test_endpoint("GET", "/static/greek_bohemia_pantheon_studio.html")
    if not html_ok:
        all_passed = False

    # 6. Greek Bohemia Image
    img_ok, _ = test_endpoint("GET", "/static/img/greek_bohemia_pantheon_memory.jpg")
    if not img_ok:
        all_passed = False

    # 7. Godot Canvas 3D UI Route
    g_ui_ok, _ = test_endpoint("GET", "/godot-canvas-3d")
    if not g_ui_ok:
        all_passed = False

    # 8. Godot Arena Scene GET
    g_scene_ok, g_scene = test_endpoint("GET", "/api/godot/arena-scene")
    if g_scene_ok and g_scene.get("vital_max_hp_rule") == 6 and len(g_scene.get("hexes", [])) == 19:
        print(f"  -> Godot Arena Scene OK (19 Hexes, Player HP: {g_scene['player_hero']['hp']}/6)")
    else:
        all_passed = False

    # 9. Godot Tactical Action POST (Strike)
    act_ok, act_data = test_endpoint("POST", "/api/godot/tactical-action", payload={"action_type": "strike"})
    if act_ok and act_data.get("vital_max_hp_rule") == 6:
        print(f"  -> Tactical Action OK (Action: {act_data.get('action_type')}, Enemy HP: {act_data.get('enemy_hp')}/6)")
    else:
        all_passed = False

    # 10. Godot AI Counter POST
    ai_ok, ai_data = test_endpoint("POST", "/api/godot/ai-counter", payload={})
    if ai_ok and ai_data.get("vital_max_hp_rule") == 6:
        print(f"  -> AI Counter Turn OK (Turn: {ai_data.get('turn_number')}, Player HP: {ai_data.get('player_hp')}/6)")
    else:
        all_passed = False

    # 11. Godot Arena Reset POST
    rst_ok, rst_data = test_endpoint("POST", "/api/godot/reset", payload={})
    if rst_ok and rst_data.get("vital_max_hp_rule") == 6:
        print(f"  -> Arena Reset OK (Player HP: {rst_data['player_hero']['hp']}/6)")
    else:
        all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY (11/11 PASSED)!")
        sys.exit(0)
    else:
        print("SOME VERIFICATIONS FAILED!")
        sys.exit(1)


if __name__ == '__main__':
    main()
