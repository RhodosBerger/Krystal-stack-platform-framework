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
    print("=== 1. Testing GET /api/cosmic/zodiac_sky ===")
    zodiac_data = test_get("/api/cosmic/zodiac_sky")
    sky = zodiac_data["zodiac_sky"]
    const = sky["active_constellation"]
    print(f"Active Sign: {const['symbol']} {const['name_sk']} ({const['element']}), Zenith: {sky['zenith_transit_active']}")
    assert len(zodiac_data["all_constellations"]) == 12

    print("=== 2. Testing GET /api/system/grid_matrices ===")
    grids_data = test_get("/api/system/grid_matrices")
    print(f"Supported Grids Count: {grids_data['count']}")
    presets = [g["title"] for g in grids_data["grid_matrices"]]
    print("Presets:", ", ".join(presets))
    assert grids_data["count"] == 6

    print("=== 3. Testing GET /api/inventory/plus_status ===")
    inv_data = test_get("/api/inventory/plus_status")
    inv = inv_data["inventory"]
    print(f"Base Slots: {inv['base_slots']}, Plus Tokens: {inv['plus_tokens']}, Total Capacity: {inv['total_capacity']}")
    assert inv["total_capacity"] == inv["base_slots"] + (inv["plus_tokens"] * 5)

    print("=== 4. Testing POST /api/combat/immunity_check (Toxic 100% Acid Immunity) ===")
    imm_data = test_post("/api/combat/immunity_check", {
        "race_id": "toxic",
        "ailment_type": "poison_acid",
        "raw_potency": 8,
        "active_ward": 6
    })
    mit = imm_data["ailment_mitigation"]
    print(f"Ailment Status: {mit['status']}, Absorbed: {mit['absorbed_potency']}, Remaining: {mit['final_potency']}")
    assert mit["status"] == "IMMUNE"
    assert mit["final_potency"] == 0

    print("=== 5. Testing POST /api/equipment/optics_zoom (Tier 5 Cosmic Lens) ===")
    opt_data = test_post("/api/equipment/optics_zoom", {
        "gear_tier": 5,
        "target_distance_hex": 4
    })
    opt = opt_data["optics"]
    print(f"Optics: {opt['optics_name']}, Zoom: {opt['zoom_multiplier']}x, FOV: {opt['field_of_view_degrees']}°, Aim Bonus: +{opt['net_aim_bonus']}")
    assert opt["zoom_multiplier"] == 8.0
    assert opt["celestial_constellations_visible"] is True

    print("=== 6. Testing POST /api/inventory/plus_slots (Add Token & Store) ===")
    token_res = test_post("/api/inventory/plus_slots", {
        "action": "add_token",
        "count": 1
    })
    new_cap = token_res["inventory"]["total_capacity"]
    print(f"New Capacity after '+' token: {new_cap}")

    store_res = test_post("/api/inventory/plus_slots", {
        "action": "store",
        "slot_index": new_cap - 1,
        "item_data": {"name": "Hviezdny Aéterový Puškohľad Sharps", "tier": 5}
    })
    print(f"Stored Items Count: {store_res['inventory']['stored_items_count']}")
    assert store_res["inventory"]["stored_items_count"] >= 1

    print("=== 7. Testing POST /api/system/prerequisites_check ===")
    prereq_res = test_post("/api/system/prerequisites_check", {
        "hero_profile": {
            "tier": 4,
            "race": "crystal",
            "duel_stats": {"toughness": 20, "reflexes": 30, "aim": 38, "appearance": 35}
        },
        "requirements": {
            "attributes": {"aim": 30, "appearance": 25},
            "min_tier": 3,
            "required_race": "crystal"
        }
    })
    p = prereq_res["prerequisites"]
    print(f"Prerequisites Eligible: {p['eligible']}, Fulfillment: {p['fulfillment_percentage']}%")
    assert p["eligible"] is True
    assert p["fulfillment_percentage"] == 100.0

    print("\n>>> ALL 7 ZODIAC, GRIDS, IMMUNITIES & OPTICS ENDPOINTS VERIFIED! <<<")

if __name__ == "__main__":
    main()
