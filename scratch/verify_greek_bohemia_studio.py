import urllib.request
import json

def verify_greek_bohemia():
    base = "http://localhost:8089"

    print("[1] Verifying GET /static/greek_bohemia_pantheon_studio.html...")
    with urllib.request.urlopen(f"{base}/static/greek_bohemia_pantheon_studio.html") as resp:
        assert resp.status == 200
        html = resp.read().decode('utf-8', errors='ignore')
        assert "Bohemia" in html and "Krystal-Stack" in html
        print(" -> OK: Greek-Bohemia studio page serves valid HTML.")

    print("[2] Verifying GET /static/img/greek_bohemia_pantheon_memory.jpg...")
    with urllib.request.urlopen(f"{base}/static/img/greek_bohemia_pantheon_memory.jpg") as resp:
        assert resp.status == 200
        content = resp.read()
        assert len(content) > 1000
        print(f" -> OK: Hero artwork loaded ({len(content)} bytes).")

    print("[3] Verifying GET /api/greek-bohemia/pantheon...")
    with urllib.request.urlopen(f"{base}/api/greek-bohemia/pantheon") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert data.get("vital_max_hp_rule") == 6
        assert len(data.get("greek_deities", [])) >= 8
        assert len(data.get("bohemian_allies", [])) >= 8
        assert len(data.get("pacts", [])) >= 5
        print(f" -> OK: Loaded {len(data['greek_deities'])} deities, {len(data['bohemian_allies'])} bohemian allies, {len(data['pacts'])} active pacts.")

    print("[4] Verifying GET /api/greek-bohemia/memory-strategies...")
    with urllib.request.urlopen(f"{base}/api/greek-bohemia/memory-strategies") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert data.get("vital_max_hp_rule") == 6
        assert len(data.get("strategies", [])) >= 6
        print(f" -> OK: Loaded {len(data['strategies'])} philosophical memory strategies.")

    print("[5] Verifying POST /api/greek-bohemia/simulate-leveling...")
    req_body = json.dumps({"axiom_ids": ["pythagoras_harmonics", "aristotle_golden_mean", "heraclitus_flux"]}).encode('utf-8')
    req = urllib.request.Request(f"{base}/api/greek-bohemia/simulate-leveling", data=req_body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert data.get("vital_max_hp_rule") == 6
        assert data.get("status") == "MEMORY_AUTONOMOUSLY_LEVELLED"
        assert "rebalanced_memory_pools" in data
        assert data.get("combined_latency_reduction_percent", 0) > 0
        print(f" -> OK: Leveling simulation successful! Latency reduction: {data['combined_latency_reduction_percent']}%, Hit rate: {data['achieved_cache_hit_rate_percent']}%")

    print("[6] Verifying POST /api/greek-bohemia/form-pact...")
    pact_body = json.dumps({
        "greek_deity_id": "zeus",
        "bohemian_ally_id": "perun",
        "pact_title": "Blesková Aliancia Hromovládcov",
        "lore": "Zeus a Perun spájajú hromy a blesky na akceleráciu pamäťových zberníc."
    }).encode('utf-8')
    req = urllib.request.Request(f"{base}/api/greek-bohemia/form-pact", data=pact_body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert data.get("success") is True
        pact = data.get("pact", {})
        assert pact.get("synergy_multiplier", 1.0) > 1.0
        assert pact.get("vital_max_hp") == 6
        print(" -> OK: Pact formed successfully with synergy multiplier > 1.0 and vital_max_hp = 6")

    print("\nALL GREEK-BOHEMIA ENDPOINTS & WORKFLOWS VALIDATED 100% OK!")

if __name__ == '__main__':
    verify_greek_bohemia()
