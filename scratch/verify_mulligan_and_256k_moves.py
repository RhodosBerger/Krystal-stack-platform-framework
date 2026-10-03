import sys
import json
import urllib.request
import urllib.error

sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8089"

def req_get(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "MulliganVerificationScript/1.0"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def req_post(path, data):
    url = f"{BASE_URL}{path}"
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "MulliganVerificationScript/1.0"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    print("[1/8] Testing GET /api/cards/mulligan/hand...")
    hand_res = req_get("/api/cards/mulligan/hand")
    assert hand_res["success"], "mulligan hand failed"
    assert hand_res["count"] == 4, f"Expected 4 cards, got {hand_res['count']}"
    card_names = [c["name"] for c in hand_res["hand"]]
    print(f" -> OK: 4 Cards in hand: {card_names}")

    print("[2/8] Testing GET /api/cards/mulligan/catalog...")
    cat_res = req_get("/api/cards/mulligan/catalog")
    assert cat_res["success"]
    assert cat_res["count"] == 4
    print(f" -> OK: 4 canonical card archetypes loaded.")

    print("[3/8] Testing POST /api/cards/mulligan/toggle (selecting card index 1: POHYB S JEDNOTKOU)...")
    toggle_res = req_post("/api/cards/mulligan/toggle", {"card_index": 1})
    assert toggle_res["success"]
    assert 1 in toggle_res["selected_indices"]
    print(f" -> OK: Selected card index 1 for replacement: {toggle_res['selected_indices']}")

    print("[4/8] Testing POST /api/cards/mulligan/exchange (confirming mulligan)...")
    exch_res = req_post("/api/cards/mulligan/exchange", {})
    assert exch_res["success"]
    assert exch_res["replaced_count"] == 1
    assert len(exch_res["final_hand"]) == 4
    print(f" -> OK: Mulligan executed! Replaced 1 card, new hand size: {len(exch_res['final_hand'])}")

    print("[5/8] Testing POST /api/cards/mulligan/reset (restore canonical hand)...")
    reset_res = req_post("/api/cards/mulligan/reset", {})
    assert reset_res["success"]
    assert len(reset_res["hand"]) == 4
    print(" -> OK: Mulligan hand reset to initial showcase state.")

    print("[6/8] Testing GET /api/tactical/combinatorial_moves/specs...")
    specs_res = req_get("/api/tactical/combinatorial_moves/specs")
    assert specs_res["success"]
    assert specs_res["total_combinations"] == 262144
    assert specs_res["vital_max_hp_rule"] == 6
    print(f" -> OK: Combinatorial Specs verified: {specs_res['formula']}")

    print("[7/8] Testing POST /api/tactical/combinatorial_moves/simulate (evaluating 256,000 moves)...")
    sim_res = req_post("/api/tactical/combinatorial_moves/simulate", {
        "hero_pos": [0, -2],
        "enemy_pos": [0, 2],
        "crystals": [[0, -1], [1, 0], [0, 0]]
    })
    assert sim_res["success"]
    assert sim_res["total_combinations_evaluated"] == 262144
    print(f" -> OK: 256k combinations evaluated in {sim_res['evaluation_time_ms']} ms! Top move: {sim_res['optimal_turn_recommendation']['name']} (Score: {sim_res['optimal_turn_recommendation']['tactical_utility_score']})")

    print("[8/8] Testing GET /static/mulligan_cards_studio.html...")
    req = urllib.request.Request(f"{BASE_URL}/static/mulligan_cards_studio.html")
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8")
        assert "MULLIGAN PHASE" in html
        assert "SIMULÁCIA 256 TISÍC KOMBINÁCIÍ" in html
        print(f" -> OK: Mulligan Studio HTML served ({len(html)} bytes).")

    print("\nALL 8 END-TO-END MULLIGAN & 256k COMBINATORIAL CHECKS PASSED PERFECTLY!")

if __name__ == "__main__":
    main()
