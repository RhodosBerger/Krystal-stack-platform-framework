import urllib.request
import json
import time

def test_endpoints():
    base_url = "http://localhost:8089"
    
    print("[1] Testing GET /cnc-simulator HTML page...")
    req = urllib.request.Request(f"{base_url}/cnc-simulator")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        html = resp.read().decode('utf-8')
        assert "KRYSTAL CNC" in html, "HTML should contain KRYSTAL CNC"
        assert "VITAL_MAX_HP_RULE" in html or "6" in html, "Should have vital max hp reference"
        print(" -> OK: /cnc-simulator served valid HTML")

    print("[2] Testing GET /static/cnc_machining_studio.html...")
    req = urllib.request.Request(f"{base_url}/static/cnc_machining_studio.html")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        print(" -> OK: static file accessible directly")

    print("[3] Testing GET /api/cnc/catalog...")
    req = urllib.request.Request(f"{base_url}/api/cnc/catalog")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert data.get("vital_max_hp_rule") == 6, f"Expected vital_max_hp_rule=6, got {data.get('vital_max_hp_rule')}"
        assert "tools" in data and "materials" in data and "presets" in data
        assert len(data["tools"]) >= 4
        assert len(data["materials"]) >= 4
        print(f" -> OK: Catalog has {len(data['tools'])} tools, {len(data['materials'])} materials, {len(data['presets'])} presets")

    print("[4] Testing GET /api/cnc/presets...")
    req = urllib.request.Request(f"{base_url}/api/cnc/presets")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert len(data["presets"]) >= 3
        print(f" -> OK: Presets endpoint returns {len(data['presets'])} presets")

    print("[5] Testing POST /api/cnc/speeds-and-feeds...")
    payload = json.dumps({"tool_id": "t1_endmill_3mm", "material_id": "al_6061"}).encode('utf-8')
    req = urllib.request.Request(f"{base_url}/api/cnc/speeds-and-feeds", data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert data.get("vital_max_hp_rule") == 6
        assert data["spindle_rpm"] > 1000
        print(f" -> OK: Speeds and feeds calculated: {data['spindle_rpm']} RPM, feed: {data['feed_rate_xy_mm_min']} mm/min")

    print("[6] Testing POST /api/cnc/simulate-toolpath...")
    preset_entities = [
        {"type": "RECT", "x": -25.0, "y": -20.0, "width": 50.0, "height": 40.0, "depth": 1.5, "feed_rate": 450.0},
        {"type": "CIRCLE", "x": 0.0, "y": 0.0, "radius": 15.0, "depth": 2.0, "feed_rate": 350.0}
    ]
    payload = json.dumps({
        "entities": preset_entities,
        "tool_id": "t1_endmill_3mm",
        "material_id": "al_6061",
        "target_depth": 2.0
    }).encode('utf-8')
    req = urllib.request.Request(f"{base_url}/api/cnc/simulate-toolpath", data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert data.get("vital_max_hp_rule") == 6
        assert len(data.get("moves", [])) > 0
        assert data.get("total_machining_time_s", 0) > 0
        print(f" -> OK: Simulation produced {len(data['moves'])} moves, est time: {data['total_machining_time_s']:.2f}s")

    print("[7] Testing POST /api/cnc/generate-gcode...")
    payload = json.dumps({
        "entities": preset_entities,
        "tool_id": "t1_endmill_3mm",
        "material_id": "al_6061",
        "target_depth": 2.0,
        "program_name": "TEST_CNC_JOB"
    }).encode('utf-8')
    req = urllib.request.Request(f"{base_url}/api/cnc/generate-gcode", data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert data.get("vital_max_hp_rule") == 6
        assert "G21" in data.get("gcode", "")
        assert "M30" in data.get("gcode", "")
        print(f" -> OK: G-code generated ({len(data['gcode'].splitlines())} lines)")

    print("\nALL CNC STUDIO ENDPOINTS & WORKFLOWS VALIDATED SUCCESSFULLY!")

if __name__ == '__main__':
    test_endpoints()
