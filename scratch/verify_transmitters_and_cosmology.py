import urllib.request
import urllib.parse
import json
import sys
import traceback

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
    print("--- 1. Testing GET /api/cosmology/planes ---")
    planes = test_get("/api/cosmology/planes")
    assert planes["success"] is True
    assert len(planes["planes"]) == 5
    assert "abyssal_inferno" in planes["planes"]
    assert "empyrean_heaven" in planes["planes"]

    print("--- 2. Testing GET /api/cosmology/apostles ---")
    apostles = test_get("/api/cosmology/apostles")
    assert apostles["success"] is True
    assert apostles["count"] == 12
    peter = [a for a in apostles["apostles"] if a["id"] == "apostle_peter_01"][0]
    assert peter["name"] == "Šimon Peter"
    assert peter["warhammer_stats"]["Sv"] == "2+"

    print("--- 3. Testing GET /api/cosmology/angels ---")
    angels = test_get("/api/cosmology/angels")
    assert angels["success"] is True
    assert angels["count"] == 5
    michael = [a for a in angels["angels"] if a["id"] == "archangel_michael"][0]
    assert "Plamenný Meč" in michael["weapon"]

    print("--- 4. Testing GET /api/cosmology/portals ---")
    portals = test_get("/api/cosmology/portals")
    assert portals["success"] is True
    assert len(portals["portals"]) == 4

    print("--- 5. Testing GET /api/cosmology/tree_network ---")
    trees = test_get("/api/cosmology/tree_network")
    assert trees["success"] is True
    assert trees["network"]["network_status"] == "INTERTWINED_HEALTHY"

    print("--- 6. Testing GET /api/transmitter/status ---")
    st = test_get("/api/transmitter/status")
    assert st["success"] is True
    assert st["transmitter"]["bot_id"] == "bot_prime_explorer"

    print("--- 7. Testing POST /api/transmitter/step ---")
    step1 = test_post("/api/transmitter/step", {"delta_vector": [2.0, 0.0, 3.0]})
    assert step1["success"] is True
    assert step1["step_index"] >= 1
    assert step1["position"] == [2.0, 0.0, 3.0]

    print("--- 8. Testing POST /api/transmitter/rewind ---")
    rew = test_post("/api/transmitter/rewind", {})
    assert rew["success"] is True
    assert rew["position"] == [0.0, 0.0, 0.0]

    print("--- 9. Testing POST /api/transmitter/teleport ---")
    tp = test_post("/api/transmitter/teleport", {"target_position": [10.0, 50.0, 10.0], "plane": "aetheric_sky"})
    assert tp["success"] is True
    assert tp["plane"] == "aetheric_sky"

    print("--- 10. Testing POST /api/transmitter/stimulus (God mode toggle) ---")
    gm = test_post("/api/transmitter/stimulus", {"toggle_god_mode": True})
    assert gm["success"] is True
    assert gm["god_mode"] is True

    print("--- 11. Testing POST /api/nocturnal/atmosphere ---")
    sky = test_post("/api/nocturnal/atmosphere", {"celestial_time_sec": 15.0, "lunar_phase": 0.8})
    assert sky["success"] is True
    assert sky["nocturnal_atmosphere"]["lunar_intensity"] > 0
    assert len(sky["nocturnal_atmosphere"]["spectral_spirit_entities"]) == 3

    print("--- 12. Testing POST /api/environment/weather_entropy ---")
    weath = test_post("/api/environment/weather_entropy", {"pressure_delta": 85.0, "humidity_pct": 90.0, "wind_shear_mps": 80.0})
    assert weath["success"] is True
    assert weath["weather"]["is_catastrophe_active"] is True
    assert weath["weather"]["hazard_damage_per_turn"] >= 2

    print("\nALL 12 LIVE TRANSMITTER, COSMOLOGY, NOCTURNAL SKY & APOSTLES TESTS PASSED PERFECTLY!")
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
