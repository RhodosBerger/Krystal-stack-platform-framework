"""
E2E Verification Script: Epic Campaign Saga, Missions, Canvas Theater & Lore Codex
===================================================================================
Verifies all campaign HTTP endpoints on port 8089 and ensures strict 6 Max HP compliance.
"""

import sys
import json
import urllib.request

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_URL = "http://127.0.0.1:8089"

def http_get(endpoint: str):
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url, headers={"User-Agent": "KrystalVerification/1.0"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, resp.read()

def http_post(endpoint: str, payload: dict):
    url = f"{BASE_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "KrystalVerification/1.0"}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def main():
    print("=== STARTING E2E VERIFICATION: EPIC CAMPAIGN SAGA & THEATER ===")
    
    # 1. GET /api/campaign/chapters
    status, raw = http_get("/api/campaign/chapters")
    assert status == 200
    data = json.loads(raw.decode("utf-8"))
    assert data["success"] is True
    assert data["count"] >= 3
    print(f"[PASS] 1. Campaign Chapters: {data['count']} canonical chapters loaded")
    for chap in data["chapters"]:
        assert chap["enemy_leader_hp"] == 6
        assert len(chap["choices"]) >= 2
    print("       All chapters enforce Vital 6 Max HP on enemy leaders.")

    # 2. GET /api/campaign/codex
    status, raw = http_get("/api/campaign/codex")
    assert status == 200
    data = json.loads(raw.decode("utf-8"))
    assert data["success"] is True
    codex = data["codex"]
    assert "crystal_tribe" in codex["factions"]
    assert "ironclad_mortar_island" in codex["vehicles_and_tech"]
    print(f"[PASS] 2. Lore Codex & Bestiary: {len(codex['factions'])} factions, {len(codex['vehicles_and_tech'])} tech entries")

    # 3. POST /api/campaign/generate_mission
    status, m_res = http_post("/api/campaign/generate_mission", {
        "altitude_tier": "stratospheric_citadel",
        "threat_level": 4,
        "weather_condition": "totem_rift_lightning",
        "enemy_faction": "toxic_tribe"
    })
    assert status == 200 and m_res["success"] is True
    mission = m_res["mission"]
    print(f"[PASS] 3. Procedural Mission Generated: '{mission['mission_name']}'")
    print(f"       Altitude: {mission['altitude_m']}m ({mission['location_name']}), Weather: {mission['weather']}")
    print(f"       Threat: {mission['threat_level']}/5, Enemy Squads: {len(mission['enemy_squads'])}")
    for sq in mission["enemy_squads"]:
        assert sq["squad_hp"] == 6
    assert mission["vital_max_hp_rule_observed"] is True

    # 4. POST /api/campaign/simulate_operation (Victory path)
    status, op_res = http_post("/api/campaign/simulate_operation", {
        "chapter_id": "chapter_1_desert_caravan",
        "chosen_choice_id": "choice_silent_drop",
        "player_squad_hp": 6,
        "artillery_active": True,
        "flight_support_active": True
    })
    assert status == 200 and op_res["success"] is True
    op = op_res["operation"]
    print(f"[PASS] 4. Operation Simulated: Victory={op['victory']}, Choice={op['choice_made']}")
    print(f"       Final HP: Player={op['player_final_hp']}/6, Enemy={op['enemy_final_hp']}/6, Rounds={op['rounds_executed']}")
    print(f"       Rewards: +{op['rewards_awarded']['aether_nuggets']} Nuggets, +{op['rewards_awarded']['experience_xp']} XP")
    assert op["rewards_awarded"]["vital_invariant_honored"] is True
    assert 0 <= op["player_final_hp"] <= 6

    # 5. GET /static/epic_campaign_theater_studio.html
    status, html_raw = http_get("/static/epic_campaign_theater_studio.html")
    assert status == 200
    assert b"EPIC CAMPAIGN THEATER" in html_raw
    assert b"epic_krystal_campaign_showcase.jpg" in html_raw
    print(f"[PASS] 5. Campaign Theater Studio HTML served: {len(html_raw)} bytes")

    # 6. GET /static/img/epic_krystal_campaign_showcase.jpg
    status, img_raw = http_get("/static/img/epic_krystal_campaign_showcase.jpg")
    assert status == 200
    assert len(img_raw) > 500000
    print(f"[PASS] 6. Epic Masterpiece Artwork JPEG served: {len(img_raw)} bytes")

    print("\n>>> ALL 6 CAMPAIGN & THEATER E2E CHECKS PASSED WITH 100% SUCCESS! <<<")

if __name__ == "__main__":
    main()
