import urllib.request
import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://127.0.0.1:8089"

def test_get(endpoint):
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200, f"Expected 200 for {endpoint}, got {resp.status}"
        data = json.loads(resp.read().decode('utf-8'))
        assert data.get("success") is True, f"Expected success: True for {endpoint}"
        return data

def test_post(endpoint, payload):
    url = f"{BASE_URL}{endpoint}"
    req_body = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=req_body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200, f"Expected 200 for {endpoint}, got {resp.status}"
        data = json.loads(resp.read().decode('utf-8'))
        assert data.get("success") is True, f"Expected success: True for {endpoint}"
        return data

def main():
    print("=== 1. Testing GET /api/archetypes/representatives?gender=all ===")
    reps_all = test_get("/api/archetypes/representatives?gender=all")
    print(f"Total Male: {reps_all['total_male']}, Total Female: {reps_all['total_female']}, Returned: {reps_all['returned_count']}")
    assert reps_all['total_male'] == 20
    assert reps_all['total_female'] == 20
    assert reps_all['returned_count'] == 40

    print("=== 2. Testing GET /api/archetypes/representatives?gender=male ===")
    reps_m = test_get("/api/archetypes/representatives?gender=male")
    assert reps_m['returned_count'] == 20
    print(f"Male 1: {reps_m['representatives'][0]['name']} ({reps_m['representatives'][0]['archetype_class']})")

    print("=== 3. Testing GET /api/archetypes/representatives?gender=female ===")
    reps_f = test_get("/api/archetypes/representatives?gender=female")
    assert reps_f['returned_count'] == 20
    print(f"Female 1: {reps_f['representatives'][0]['name']} ({reps_f['representatives'][0]['archetype_class']})")

    print("=== 4. Testing GET /api/archetypes/character?id=m_cold_15 ===")
    char_m = test_get("/api/archetypes/character?id=m_cold_15")
    print(f"Loaded: {char_m['character']['name']}, Appearance: {char_m['character']['duel_stats']['appearance']}, Favored: {char_m['character']['favored_weapon']}")
    assert char_m['character']['id'] == "m_cold_15"

    print("=== 5. Testing GET /api/archetypes/character?id=f_cold_15 ===")
    char_f = test_get("/api/archetypes/character?id=f_cold_15")
    print(f"Loaded: {char_f['character']['name']}, Appearance: {char_f['character']['duel_stats']['appearance']}")
    assert char_f['character']['id'] == "f_cold_15"

    print("=== 6. Testing GET /api/archetypes/races ===")
    races_data = test_get("/api/archetypes/races")
    print(f"Races Count: {races_data['count']}")
    assert races_data['count'] == 12

    print("=== 7. Testing GET /api/archetypes/helpers ===")
    helpers_data = test_get("/api/archetypes/helpers")
    print(f"Total Helpers: {helpers_data['total_count']}, Filtered: {helpers_data['filtered_count']}")
    assert helpers_data['total_count'] == 120

    print("=== 8. Testing GET /api/archetypes/helper?index=1 ===")
    h1 = test_get("/api/archetypes/helper?index=1")
    print(f"Helper #1: {h1['helper']['name']} (Race: {h1['helper']['race']}, Aura: {h1['helper']['aura_bonus']})")
    assert h1['helper']['index'] == 1

    print("=== 9. Testing GET /api/archetypes/helper?index=120 ===")
    h120 = test_get("/api/archetypes/helper?index=120")
    print(f"Helper #120: {h120['helper']['name']} (Race: {h120['helper']['race']})")
    assert h120['helper']['index'] == 120

    print("=== 10. Testing POST /api/archetypes/composite_build ===")
    composite = test_post("/api/archetypes/composite_build", {
        "hero_id": "m_cold_15",
        "helper_index": 1,
        "race_id": "crystal",
        "slider_adjustments": {"appearance": 15, "aim": 8},
        "synergy_scale": 1.4
    })
    b = composite['composite_build']
    print(f"Composite Hero: {b['hero_name']}, Appearance: {b['composite_duel_stats']['appearance']}, Synergy Match: {b['racial_synergy_match']}")
    assert b['composite_duel_stats']['appearance'] > 40

    print("=== 11. Testing POST /api/archetypes/duel_round (Aiming Head vs Duck Down) ===")
    duel_evade = test_post("/api/archetypes/duel_round", {
        "hero_id": "m_cold_15",
        "helper_index": 1,
        "race_id": "crystal",
        "attack_zone": "head",
        "defense_stance": "duck_down",
        "weapon_type": "ranged",
        "base_damage": 5,
        "defender_current_hp": 6
    })
    dr = duel_evade['duel_result']
    print(f"Duel Result: Hit={dr['hit']}, Evaded={dr['evaded']}, HP After: {dr['defender_hp_after']}/6")
    assert dr['evaded'] is True
    assert dr['damage_dealt'] == 0

    print("=== 12. Testing POST /api/archetypes/duel_round (Torso Hit vs Stand Firm) ===")
    duel_hit = test_post("/api/archetypes/duel_round", {
        "hero_id": "f_cold_15",
        "helper_index": 61,
        "race_id": "infernal",
        "attack_zone": "torso",
        "defense_stance": "stand_firm",
        "weapon_type": "cold_melee",
        "base_damage": 6,
        "defender_current_hp": 6
    })
    dr2 = duel_hit['duel_result']
    print(f"Duel Result: Hit={dr2['hit']}, Evaded={dr2['evaded']}, Net Damage: {dr2['net_damage_dealt']}, HP After: {dr2['defender_hp_after']}/6")
    assert dr2['hit'] is True
    assert dr2['net_damage_dealt'] > 0
    assert dr2['defender_hp_after'] <= 6

    print("\n>>> ALL 12 MAGICAL ARCHETYPES & 120 HELPERS ENDPOINTS VERIFIED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    main()
