"""
E2E Verification Script: Aerial Balloons, Floating Islands, Rogallo Wings & Plunging Mortar
=============================================================================================
Verifies all HTTP endpoints on port 8089 and ensures strict 6 Max HP compliance.
"""

import sys
import json
import urllib.request
import urllib.error

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
    print("=== STARTING E2E VERIFICATION: AERIAL BALLOONS & PLUNGING MORTAR ===")
    
    # 1. GET /api/aerial/islands/network
    status, raw = http_get("/api/aerial/islands/network")
    assert status == 200, f"Expected 200, got {status}"
    net_data = json.loads(raw.decode("utf-8"))
    assert net_data["success"] is True
    network = net_data["network"]
    print(f"[PASS] 1. Island Network: {network['total_islands']} islands, {network['total_skybridges']} skybridges")
    assert "island_alpha" in network["islands"]
    assert network["islands"]["island_alpha"]["mortar_equipped"] is True
    assert network["islands"]["island_alpha"]["garrison_hp"] == 6

    # 2. GET /api/aerial/balloons/profiles
    status, raw = http_get("/api/aerial/balloons/profiles")
    assert status == 200
    prof_data = json.loads(raw.decode("utf-8"))
    print(f"[PASS] 2. Aerostat Profiles: {prof_data['count']} profiles available")
    profile_ids = [p["id"] for p in prof_data["profiles"]]
    assert "ironclad_siege_island" in profile_ids
    assert "thermal_sun_furnace" in profile_ids

    # 3. POST /api/aerial/balloons/buoyancy
    status, b_data = http_post("/api/aerial/balloons/buoyancy", {
        "profile_id": "ironclad_siege_island",
        "current_payload_kg": 14000.0
    })
    assert status == 200 and b_data["success"] is True
    buoy = b_data["buoyancy"]
    print(f"[PASS] 3. Buoyancy Lift: Gross Lift={buoy['gross_lift_n']} N, Net Force={buoy['net_buoyant_force_n']} N, Buoyant={buoy['is_buoyant']}")
    assert buoy["is_buoyant"] is True

    # 4. POST /api/aerial/mortar/fire_plunge (Elevation Plunge Ballistics + 6 Max HP Invariant)
    status, m_data = http_post("/api/aerial/mortar/fire_plunge", {
        "island_elevation_m": 280.0,
        "muzzle_velocity_mps": 85.0,
        "pitch_angle_deg": 65.0,
        "yaw_angle_deg": 0.0,
        "shell_caliber_mm": 240.0,
        "wind_speed_mps": 4.5,
        "wind_direction_deg": 90.0,
        "target_dist_m": 450.0,
        "target_initial_hp": 6
    })
    assert status == 200 and m_data["success"] is True
    mortar = m_data["mortar_fire"]
    print(f"[PASS] 4. Plunging Mortar: Hit={mortar['hit_quality']}, Dmg={mortar['damage_inflicted']} HP, Remaining={mortar['target_remaining_hp']}/6 HP")
    print(f"       Range={mortar['horizontal_range_m']} m, Impact Speed={mortar['impact_speed_mps']} m/s, Plunge Energy={mortar['plunging_kinetic_energy_kj']} kJ")
    assert mortar["vital_max_hp_rule_observed"] is True
    assert 0 <= mortar["target_remaining_hp"] <= 6

    # 5. POST /api/aerial/flight/simulate (Rogallo Hang Glider)
    status, f_data = http_post("/api/aerial/flight/simulate", {
        "vehicle_type": "rogallo_hang_glider",
        "launch_altitude_m": 340.0,
        "initial_airspeed_mps": 18.0,
        "glide_ratio": 7.5,
        "thermal_updraft_mps": 3.2,
        "flight_duration_sec": 45.0
    })
    assert status == 200 and f_data["success"] is True
    glider = f_data["flight_simulation"]
    print(f"[PASS] 5. Rogallo Hang Glider: Glide Distance={glider['total_glide_distance_m']} m, Final Alt={glider['final_altitude_m']} m, Thermal Active={glider['thermal_lift_active']}")
    assert glider["total_glide_distance_m"] > 500.0

    # 6. POST /api/aerial/flight/simulate (Steerable Parachute)
    status, p_data = http_post("/api/aerial/flight/simulate", {
        "vehicle_type": "steerable_parachute",
        "deployment_altitude_m": 280.0,
        "payload_mass_kg": 85.0,
        "canopy_area_m2": 28.0,
        "drag_coefficient": 1.45,
        "steer_lateral_mps": 3.5,
        "descent_duration_sec": 35.0
    })
    assert status == 200 and p_data["success"] is True
    chute = p_data["flight_simulation"]
    print(f"[PASS] 6. Steerable Parachute: Terminal Velocity={chute['terminal_velocity_mps']} m/s, Soft Landing={chute['soft_landing_guaranteed']}, Lateral Drift={chute['total_lateral_drift_m']} m")
    assert chute["soft_landing_guaranteed"] is True
    assert chute["terminal_velocity_mps"] < 6.0

    # 7. POST /api/aerial/skybridge/traverse
    status, t_data = http_post("/api/aerial/skybridge/traverse", {
        "bridge_id": "bridge_alpha_beta",
        "traveler_weight_kg": 85.0,
        "method": "zipline"
    })
    assert status == 200 and t_data["success"] is True
    trav = t_data["traversal"]
    print(f"[PASS] 7. Skybridge Traversal: {trav['bridge_id']} via {trav['method']}, Speed={trav['speed_mps']} m/s, Duration={trav['traversal_time_sec']} s, Safe={trav['safe_crossing']}")
    assert trav["safe_crossing"] is True

    # 8. POST /api/aerial/grapple/island_board
    status, g_data = http_post("/api/aerial/grapple/island_board", {
        "hero_pos": [0.0, 240.0, 30.0],
        "target_island_id": "island_alpha",
        "target_island_pos": [0.0, 280.0, 0.0],
        "grapple_cable_length_max_m": 80.0
    })
    assert status == 200 and g_data["success"] is True
    board = g_data["boarding"]
    print(f"[PASS] 8. Just Cause Grapple Boarding: Status={board['status']}, Dist={board['distance_m']} m, Slingshot Boost=+{board['slingshot_boost_mps']} m/s")
    assert board["status"] == "hook_attached_and_reeled"

    # 9. GET /static/aerial_balloons_mortar_studio.html
    status, html_raw = http_get("/static/aerial_balloons_mortar_studio.html")
    assert status == 200
    assert b"AERIAL BALLOONS & MORTARS" in html_raw
    assert b"balloon_island_mortar_rogalo_poster.jpg" in html_raw
    print(f"[PASS] 9. Interactive Studio HTML served: {len(html_raw)} bytes")

    # 10. GET /static/img/balloon_island_mortar_rogalo_poster.jpg
    status, img_raw = http_get("/static/img/balloon_island_mortar_rogalo_poster.jpg")
    assert status == 200
    assert len(img_raw) > 500000
    print(f"[PASS] 10. Masterpiece Poster JPEG served: {len(img_raw)} bytes")

    print("\n>>> ALL 10 E2E CHECKS PASSED WITH 100% SUCCESS! <<<")

if __name__ == "__main__":
    main()
