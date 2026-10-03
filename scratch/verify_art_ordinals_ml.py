import urllib.request
import urllib.parse
import json
import sys

BASE_URL = "http://127.0.0.1:8089"

def req_get(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def req_post(path, data):
    url = f"{BASE_URL}{path}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def verify_all():
    print("==================================================================")
    print(" [VERIFICATION] ART AUCTIONS, ORDINALS & EVOLUTIONARY ML ENDPOINTS")
    print("==================================================================")

    # 1. GET /api/art/auctions
    auctions = req_get("/api/art/auctions?status=active")
    assert auctions["success"] is True, "Failed to get active auctions"
    assert auctions["count"] >= 3, f"Expected at least 3 lots, got {auctions['count']}"
    first_lot = auctions["lots"][0]
    print(f" [+] 1. GET /api/art/auctions: OK ({auctions['count']} lots, First: '{first_lot['title']}')")

    # 2. POST /api/art/auctions/bid
    bid_res = req_post("/api/art/auctions/bid", {
        "lot_id": first_lot["id"],
        "bidder": "Collector_Slovakia",
        "amount": first_lot["current_bid"] + 30
    })
    assert bid_res["success"] is True, "Bid failed"
    print(f" [+] 2. POST /api/art/auctions/bid: OK (Bid placed: {bid_res['current_bid']} by {bid_res['highest_bidder']})")

    # 3. POST /api/ordinals/inscribe
    inscribe_res = req_post("/api/ordinals/inscribe", {
        "art_lot_id": first_lot["id"],
        "content_payload": "Krystal Artifact 777 Ordinal Proof",
        "content_type": "text/plain;charset=utf-8",
        "owner_address": "bc1p_dusan_krystal_vault"
    })
    assert inscribe_res["success"] is True, "Inscription failed"
    insc_id = inscribe_res["inscription"]["id"]
    print(f" [+] 3. POST /api/ordinals/inscribe: OK (Inscribed: {insc_id}, Hash: {inscribe_res['inscription']['content_sha256'][:16]}...)")

    # 4. GET /api/ordinals/export
    export_res = req_get(f"/api/ordinals/export?inscription_id={insc_id}")
    assert export_res["success"] is True, "Export failed"
    witness_hex = export_res["exported_ordinal"]["witness_envelope_hex"]
    assert witness_hex.startswith("0063036f7264"), "Invalid Bitcoin witness script prefix"
    print(f" [+] 4. GET /api/ordinals/export: OK (Witness script hex prefix: {witness_hex[:24]}...)")

    # 5. POST /api/replay/record_level
    replay_res = req_post("/api/replay/record_level", {
        "match_id": "match_verif_01",
        "level_name": "Kryštálové Hory",
        "level_idx": 1,
        "seed": 9999,
        "hero_count": 4,
        "events": [{"tick": 1, "action": "strike_head", "damage": 2}],
        "visual_layers": {"fog": 0.3, "light": "radiant"}
    })
    assert replay_res["success"] is True, "Level recording failed"
    snap_hash = replay_res["level_segment"]["perceptual_snapshot"]["snapshot_hash"]
    print(f" [+] 5. POST /api/replay/record_level: OK (Unique perceptual snapshot: {snap_hash[:16]}...)")

    # 6. GET /api/replay/level
    level_res = req_get("/api/replay/level?match_id=match_verif_01&level_idx=1")
    assert level_res["success"] is True, "Get level segment failed"
    print(f" [+] 6. GET /api/replay/level: OK (Retrieved segment for match_verif_01:1)")

    # 7. POST /api/ml/hero_matrices (5x2, 4x5, 30x20, 90x120)
    mat_res = req_post("/api/ml/hero_matrices", {
        "hero_count": 4,
        "target_dims": "all",
        "battle_intensity": 0.75
    })
    assert mat_res["success"] is True, "Hero matrices calculation failed"
    matrices = mat_res["matrices"]
    assert "5x2" in matrices and "4x5" in matrices and "30x20" in matrices and "90x120" in matrices
    sched = mat_res["sampling_frequency_schedule"]
    print(f" [+] 7. POST /api/ml/hero_matrices: OK (Dims: 5x2, 4x5, 30x20, 90x120 | Frequency: {sched['effective_hz']} Hz // {sched['mode']})")

    # 8. POST /api/ml/evolutionary_physics
    evo_res = req_post("/api/ml/evolutionary_physics", {
        "generations": 3,
        "friction": 0.05,
        "gravity": 9.81,
        "collision_target": [12.0, 0.0, 5.0]
    })
    assert evo_res["success"] is True, "Evolutionary physics failed"
    best_ind = evo_res["best_individual"]
    print(f" [+] 8. POST /api/ml/evolutionary_physics: OK (Generations run: {evo_res['generations_run']}, Best fitness: {best_ind['fitness']} pts)")

    # 9. GET /api/ml/congestion_control & POST /api/ml/congestion_dispatch
    ctrl_res = req_get("/api/ml/congestion_control")
    assert ctrl_res["success"] is True, "Congestion status failed"
    init_win = ctrl_res["congestion_status"]["window_size"]

    dispatch_res = req_post("/api/ml/congestion_dispatch", {})
    assert dispatch_res["success"] is True, "Congestion dispatch failed"
    dispatched_cnt = dispatch_res["dispatch"]["dispatched_count"]
    print(f" [+] 9. GET & POST /api/ml/congestion: OK (Window: {init_win}, Dispatched genetic events: {dispatched_cnt})")

    # 10. GET Static Studio Page
    req = urllib.request.Request(f"{BASE_URL}/static/art_ordinals_and_ml_studio.html")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, "Studio HTML page failed to load"
        print(f" [+] 10. GET /static/art_ordinals_and_ml_studio.html: OK (Status 200)")

    print("==================================================================")
    print(" ALL 10 ENDPOINTS & FEATURES VERIFIED WITH 100% SUCCESS!")
    print("==================================================================")

if __name__ == '__main__':
    verify_all()
