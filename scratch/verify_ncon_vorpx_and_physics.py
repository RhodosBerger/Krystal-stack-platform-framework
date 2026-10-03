import sys
import json
import urllib.request
import urllib.error

sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8089"

def req_get(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "NconVerificationScript/1.0"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def req_post(path, data):
    url = f"{BASE_URL}{path}"
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "NconVerificationScript/1.0"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    print("[1/9] Testing GET /api/vr/headsets/profiles...")
    vr_res = req_get("/api/vr/headsets/profiles")
    assert vr_res["success"], "profiles failed"
    assert "ncon_by_korrado" in vr_res["profiles"]
    assert "htc_vive_pro" in vr_res["profiles"]
    assert "meta_quest" in vr_res["profiles"]
    print(f" -> OK: 3 Headset Profiles loaded (Ncon FOV: {vr_res['profiles']['ncon_by_korrado']['fov_degrees']}°).")

    print("[2/9] Testing GET /api/marketing/ncon_product/specs...")
    m_res = req_get("/api/marketing/ncon_product/specs")
    assert m_res["success"]
    m = m_res["marketing"]
    assert m["product_name"] == "Ncon by Korrado"
    assert "SWING FREE" in m["tagline"]
    print(f" -> OK: Marketing package verified: '{m['product_name']}' - '{m['tagline']}'")

    print("[3/9] Testing GET /api/graphics/cel_shading/spec...")
    cel_res = req_get("/api/graphics/cel_shading/spec")
    assert cel_res["success"]
    assert cel_res["spec"]["background_palette"]["primary_yellow_hex"] == "#facc15"
    print(" -> OK: Borderlands Cel-shading uniforms verified (Yellow #facc15, Ink contours).")

    print("[4/9] Testing POST /api/vr/stereo_projection...")
    stereo_res = req_post("/api/vr/stereo_projection", {
        "headset_id": "ncon_by_korrado",
        "custom_ipd_mm": 63.5,
        "camera_pos": [0.0, 1.7, 0.0]
    })
    assert stereo_res["success"]
    p = stereo_res["projection"]
    print(f" -> OK: Stereo projection computed for {p['headset_name']} (L: {p['left_eye_camera']['position'][0]}m, R: {p['right_eye_camera']['position'][0]}m)")

    print("[5/9] Testing POST /api/physics/grapple_tether/simulate...")
    tether_res = req_post("/api/physics/grapple_tether/simulate", {
        "origin_pos": [0.0, 1.7, 0.0],
        "target_pos": [12.0, 0.0, 0.0],
        "player_mass_kg": 85.0,
        "target_mass_kg": 45.0,
        "reel_in_force_n": 1400.0
    })
    assert tether_res["success"]
    t = tether_res["tether"]
    assert t["triggers_detonation"], "Must trigger detonation on high kinetic energy"
    print(f" -> OK: Tether Reel-in simulated: {t['target_acceleration_mps2']} m/s² target accel, {t['impact_kinetic_energy_j']} J kinetic energy!")

    print("[6/9] Testing POST /api/physics/slingshot/simulate...")
    sling_res = req_post("/api/physics/slingshot/simulate", {
        "current_velocity_mps": 18.0,
        "tether_tension_n": 1400.0,
        "player_mass_kg": 85.0,
        "release_angle_deg": 25.0
    })
    assert sling_res["success"]
    s = sling_res["slingshot"]
    print(f" -> OK: Slingshot Boost: {s['initial_velocity_mps']} m/s -> {s['boosted_velocity_mps']} m/s (+{s['kinetic_energy_boost_pct']}%)")

    print("[7/9] Testing POST /api/physics/wingsuit_glide/simulate...")
    glide_res = req_post("/api/physics/wingsuit_glide/simulate", {
        "drop_altitude_m": 150.0,
        "airspeed_mps": 35.0,
        "dive_pitch_deg": -12.0
    })
    assert glide_res["success"]
    g = glide_res["glide"]
    assert g["glide_ratio"] == "3.5:1"
    print(f" -> OK: Wingsuit Glide: {g['glide_ratio']} ratio, {g['horizontal_range_m']} m range, {g['flight_duration_sec']} s flight")

    print("[8/9] Testing GET /static/ncon_vorpx_marketing_studio.html...")
    req = urllib.request.Request(f"{BASE_URL}/static/ncon_vorpx_marketing_studio.html")
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8")
        assert "NCON BY KORRADO" in html
        assert "JUST CAUSE" in html
        assert "BORDERLANDS" in html
        print(f" -> OK: Marketing Studio HTML served ({len(html)} bytes).")

    print("[9/9] Testing GET /static/img/ncon_borderlands_justcause_poster.jpg...")
    req_img = urllib.request.Request(f"{BASE_URL}/static/img/ncon_borderlands_justcause_poster.jpg")
    with urllib.request.urlopen(req_img) as resp:
        img_data = resp.read()
        assert len(img_data) > 100000
        print(f" -> OK: Key Art Poster served ({len(img_data)} bytes).")

    print("\nALL 9 END-TO-END NCON VORPX & KINETIC PHYSICS CHECKS PASSED PERFECTLY!")

if __name__ == "__main__":
    main()
