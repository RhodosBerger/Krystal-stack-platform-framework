import urllib.request
import urllib.parse
import json
import sys

def test_get(endpoint):
    url = f"http://127.0.0.1:8089{endpoint}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print(f"[GET] {endpoint} => Status: {resp.status}, Success: {data.get('success', False)}")
        return data

def test_post(endpoint, payload):
    url = f"http://127.0.0.1:8089{endpoint}"
    body = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print(f"[POST] {endpoint} => Status: {resp.status}, Success: {data.get('success', False)}")
        return data

try:
    print("--- 1. Testing GET /api/roster/characters ---")
    all_roster = test_get("/api/roster/characters")
    assert all_roster["total_available"] == 60, f"Expected 60 characters, got {all_roster['total_available']}"
    assert all_roster["matched_count"] == 60

    print("--- 2. Testing GET /api/roster/characters?race=crystal&tier=5 ---")
    crystal_tier5 = test_get("/api/roster/characters?race=crystal&tier=5")
    assert crystal_tier5["matched_count"] == 4, f"Expected 4 tier-5 crystal chars, got {crystal_tier5['matched_count']}"

    print("--- 3. Testing GET /api/roster/character?id=c_archon_17 ---")
    char_apex = test_get("/api/roster/character?id=c_archon_17")
    assert char_apex["character"]["name"] == "Kryštálový Archón"
    assert char_apex["character"]["hud_panel"]["aim_reticle"] == "archon_sovereign_ring"

    print("--- 4. Testing GET /api/duel/the_west_builds ---")
    builds = test_get("/api/duel/the_west_builds")
    assert "odolavac_pure_resistance" in builds["builds"]
    assert "vystupovac_intimidator" in builds["builds"]

    print("--- 5. Testing POST /api/tactical/gradients ---")
    grad_res = test_post("/api/tactical/gradients", {
        "height_map": {"0_0": 0.0, "1_0": 2.0, "-1_0": -1.0, "0_1": 1.0, "0_-1": -1.0},
        "center_hex": [0, 0],
        "base_range": 4.0,
        "base_damage": 3,
        "attacker_elevation": 3.0,
        "target_elevation": 0.0,
        "gradient_dot_fire_dir": -0.5,
        "is_mortar": True,
        "start_pos": [0.0, 3.0, 0.0],
        "end_pos": [4.0, 0.0, 2.0]
    })
    assert grad_res["gradient"]["center_height_m"] == 0.0
    assert grad_res["weapon_scale"]["effective_damage"] >= 3
    assert grad_res["movement_arrow"]["horizontal_distance_m"] > 0

    print("--- 6. Testing POST /api/tactical/orbital_spell ---")
    orb_res = test_post("/api/tactical/orbital_spell", {
        "caster_pos": [0.0, 0.0, 0.0],
        "time_sec": 1.5,
        "harmony_key": "fibonacci_triad",
        "caster_toughness": 20,
        "caster_reflexes": 25
    })
    assert len(orb_res["orbiting_bodies"]) == 3
    assert orb_res["aura_resonance_field"]["aura_ward_shield_points"] > 0

    print("--- 7. Testing POST /api/duel/the_west_round (Aiming Head, Duck Stance => Evaded) ---")
    duel_evaded = test_post("/api/duel/the_west_round", {
        "attacker_stats": {"aim": 30, "appearance": 25},
        "defender_stats": {"dodge": 20, "tactics": 20, "mobility": 20},
        "attack_zone": "head",
        "defense_stance": "duck_down",
        "weapon_type": "ranged",
        "base_weapon_damage": 10,
        "defender_current_hp": 6
    })
    assert duel_evaded["duel_round"]["evaded"] is True
    assert duel_evaded["duel_round"]["defender_hp_after"] == 6

    print("--- 8. Testing POST /api/duel/the_west_round (Toughness soak vs Cold Melee) ---")
    duel_soak = test_post("/api/duel/the_west_round", {
        "attacker_stats": {"aim": 30, "appearance": 10},
        "defender_stats": {"toughness": 25, "dodge": 10, "tactics": 20, "mobility": 10},
        "attack_zone": "torso",
        "defense_stance": "stand_firm",
        "weapon_type": "cold_melee",
        "base_weapon_damage": 12,
        "defender_current_hp": 6
    })
    assert duel_soak["duel_round"]["hit"] is True
    assert duel_soak["duel_round"]["mitigation_stat"] == "Toughness (Húževnatosť)"
    assert duel_soak["duel_round"]["mitigation_amount"] == 10.0 # 25 / 2.5
    assert duel_soak["duel_round"]["net_damage_dealt"] == 2 # 12 - 10
    assert duel_soak["duel_round"]["defender_hp_after"] == 4 # 6 - 2

    print("\nALL 8 LIVE ENDPOINTS VERIFIED SUCCESSFULLY WITH 100% ASSERTIONS PASSING!")
except Exception as e:
    print(f"FAILED: {e}", file=sys.stderr)
    sys.exit(1)
