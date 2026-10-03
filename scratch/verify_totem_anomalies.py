import sys
import json
import urllib.request
import urllib.error

sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8089"

def req_get(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "TotemVerificationScript/1.0"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def req_post(path, data):
    url = f"{BASE_URL}{path}"
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "TotemVerificationScript/1.0"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    print("[1/9] Testing GET /api/totems/sector_status...")
    st = req_get("/api/totems/sector_status")
    assert st["success"], "sector_status failed"
    assert st["vital_hp_invariant"] == 6, "vital invariant must be 6"
    assert len(st["totems"]) == 5, f"Expected 5 totems, got {len(st['totems'])}"
    print(f" -> OK: {len(st['totems'])} totems loaded, 6 HP invariant verified.")

    print("[2/9] Testing GET /api/totems/phenomena_catalog...")
    phenom_cat = req_get("/api/totems/phenomena_catalog")
    assert phenom_cat["success"]
    assert len(phenom_cat["phenomena"]) == 5
    print(f" -> OK: {len(phenom_cat['phenomena'])} visual phenomena types available.")

    print("[3/9] Testing GET /api/totems/spell_catalog...")
    spells = req_get("/api/totems/spell_catalog")
    assert spells["success"]
    assert "frost_crystal_nova" in spells["spells"]
    print(f" -> OK: {len(spells['spells'])} spell projection archetypes available.")

    print("[4/9] Testing POST /api/totems/spell_projection (Frost Crystal Nova)...")
    proj_res = req_post("/api/totems/spell_projection", {
        "spell_key": "frost_crystal_nova",
        "caster_origin": [0.0, 0.0],
        "heading_deg": 270.0
    })
    assert proj_res["success"]
    proj = proj_res["projection"]
    print(f" -> OK: Projection ID: {proj['projection_id']}, Spell: {proj['spell_name']}, Heading: {proj['heading_degrees']}°")

    print("[5/9] Testing POST /api/totems/trigger_anomaly...")
    anom_res = req_post("/api/totems/trigger_anomaly", {
        "sector_id": "sector_north_crystal",
        "anomaly_type": "entropy_storm",
        "epicenter": [0.0, -8.66],
        "magnitude": 0.88,
        "duration_sec": 45.0
    })
    assert anom_res["success"]
    anom = anom_res["anomaly"]
    print(f" -> OK: Anomaly ID: {anom['id']}, Drift: {anom['frequency_drift_hz']} Hz, Divergence: {anom['gradient_divergence']}")

    print("[6/9] Testing POST /api/totems/detect_anomalies...")
    scan_res = req_post("/api/totems/detect_anomalies", {
        "sector_id": "sector_north_crystal"
    })
    assert scan_res["success"]
    scan = scan_res["scan"]
    print(f" -> OK: Stability: {scan['sector_stability_index']}%, Anomalies detected: {scan['anomalies_detected_count']}")

    print("[7/9] Testing POST /api/totems/attune (Harmonize & Repair)...")
    attune_res = req_post("/api/totems/attune", {
        "totem_id": "totem_north_crystal",
        "caster_tribe": "crystal",
        "channel_energy": 35.0
    })
    assert attune_res["success"]
    assert attune_res["hp"] <= 6, "Totem HP must not exceed 6"
    print(f" -> OK: Totem {attune_res['totem_id']} attuned! Status: {attune_res['status']}, Res: {attune_res['resonance_charge']}%, HP: {attune_res['hp']}/6")

    print("[8/9] Testing POST /api/totems/phenomenon...")
    p_res = req_post("/api/totems/phenomenon", {
        "phenomenon_type": "st_elmos_plasma_discharge",
        "coordinates": [0.0, 0.0, 3.0],
        "intensity": 0.95,
        "resonance_frequency_hz": 864.0
    })
    assert p_res["success"]
    u = p_res["phenomenon"]["godot_shader_uniforms"]
    print(f" -> OK: Generated St. Elmo's Plasma: {u['plasma_lumens']} lumens, Corona: {u['corona_discharge_radius']}m")

    print("[9/9] Testing GET /static/totem_anomalies_studio.html...")
    req = urllib.request.Request(f"{BASE_URL}/static/totem_anomalies_studio.html")
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8")
        assert "SECTOR TOTEMS & SPELL ANOMALIES" in html
        print(f" -> OK: Studio HTML served ({len(html)} bytes).")

    print("\nALL 9 END-TO-END TOTEM & ANOMALY CHECKS PASSED PERFECTLY!")

if __name__ == "__main__":
    main()
