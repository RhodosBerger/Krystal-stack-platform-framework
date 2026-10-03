import urllib.request
import json
import time

def test_endpoints():
    base_url = "http://127.0.0.1:8089"
    time.sleep(1.0)
    
    print("Testing GET /static/greek_bohemia_pantheon_studio.html...")
    req = urllib.request.Request(f"{base_url}/static/greek_bohemia_pantheon_studio.html")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        content = resp.read().decode('utf-8')
        assert "Krystal-Stack" in content and "Bohemia" in content
        print(f"  OK (Length: {len(content)})")
        
    print("Testing GET /static/img/greek_bohemia_pantheon_memory.jpg...")
    req = urllib.request.Request(f"{base_url}/static/img/greek_bohemia_pantheon_memory.jpg")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = resp.read()
        assert len(data) > 1000
        print(f"  OK (Bytes: {len(data)})")
        
    print("Testing GET /api/greek-bohemia/pantheon...")
    req = urllib.request.Request(f"{base_url}/api/greek-bohemia/pantheon")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        greek_count = data.get("greek_deities_count", 0)
        bohemia_count = data.get("bohemian_allies_count", 0)
        pacts_count = data.get("active_pacts_count", 0)
        print(f"  OK: Greek Deities: {greek_count}, Bohemian Allies: {bohemia_count}, Active Pacts: {pacts_count}")
        assert greek_count >= 8
        assert bohemia_count >= 8
        assert pacts_count >= 5
        assert data.get("vital_max_hp_rule") == 6
        
    print("Testing GET /api/greek-bohemia/memory-strategies...")
    req = urllib.request.Request(f"{base_url}/api/greek-bohemia/memory-strategies")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        axioms_count = data.get("axioms_count", 0)
        strategies = data.get("strategies", [])
        print(f"  OK: Axioms Count: {axioms_count}, Strategies: {len(strategies)}")
        assert axioms_count >= 6
        assert len(strategies) >= 6
        assert data.get("vital_max_hp_rule") == 6

    print("Testing POST /api/greek-bohemia/simulate-leveling...")
    req = urllib.request.Request(
        f"{base_url}/api/greek-bohemia/simulate-leveling",
        data=json.dumps({"axiom_ids": ["pythagoras_harmonics", "heraclitus_flux", "aristotle_golden_mean"]}).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert data.get("status") == "MEMORY_AUTONOMOUSLY_LEVELLED"
        print(f"  OK: Status: {data.get('status')}, Applied Axioms: {data.get('applied_axioms_count')}, Latency Reduction: {data.get('combined_latency_reduction_percent')}%")
        assert data.get("vital_max_hp_rule") == 6
        assert len(data.get("rebalanced_memory_pools", {})) >= 4

    print("Testing POST /api/greek-bohemia/form-pact...")
    req = urllib.request.Request(
        f"{base_url}/api/greek-bohemia/form-pact",
        data=json.dumps({
            "greek_deity_id": "athena",
            "bohemian_ally_id": "libuse",
            "pact_title": "Pakt Múdrosti a Proroctva",
            "lore": "Aténina sova múdrosti a Vyšehradská kňažná Libuša."
        }).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert data.get("success") is True
        pact = data.get("pact", {})
        print(f"  OK: Pact formed: {data.get('pact_id')}, Title: {pact.get('title')}, Synergy: {pact.get('synergy_multiplier')}x")
        assert pact.get("vital_max_hp") == 6

    print("\n=======================================================")
    print("ALL 6 GREEK-BOHEMIA VERIFICATIONS PASSED WITH 200 OK!")
    print("VITAL MAX HP = 6 INVARIANT FULLY PRESERVED ACROSS ALL API CALLS!")
    print("=======================================================")

if __name__ == "__main__":
    test_endpoints()
