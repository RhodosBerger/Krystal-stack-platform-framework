#!/usr/bin/env python3
"""
VERIFICATION TEST: Achievement & Hardware Telemetry REST Endpoints in Server
==========================================================================
Verifies that all new endpoints in server.py respond correctly:
- GET /api/achievements
- POST /api/achievements/unlock
- GET /api/achievements/codex
- GET /api/telemetry/hardware
- POST /api/narrative/synthesize
"""

import os
import sys
import json
import urllib.request
import urllib.error
import threading
import time

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from krystal_web_hub.server import start_server

def test_endpoints():
    print("=== STARTING SERVER VERIFICATION ON PORT 8089 ===")
    server_thread = threading.Thread(target=start_server, kwargs={"port": 8089}, daemon=True)
    server_thread.start()
    time.sleep(1.0)

    base = "http://127.0.0.1:8089"

    # 1. Test GET /api/achievements
    req = urllib.request.Request(f"{base}/api/achievements")
    with urllib.request.urlopen(req, timeout=5) as res:
        data = json.loads(res.read().decode('utf-8'))
        assert data["status"] == "OK", "Status must be OK"
        assert len(data["achievements"]) >= 5, "Must have at least 5 achievements"
        print(f"✓ GET /api/achievements: OK ({len(data['achievements'])} achievements loaded)")

    # 2. Test POST /api/achievements/unlock
    unlock_payload = json.dumps({
        "achievement_id": "bullet_time_mastery",
        "city": "Praha - Staré Město",
        "tribe": "Kryštálový Kmeň"
    }).encode('utf-8')
    req2 = urllib.request.Request(f"{base}/api/achievements/unlock", data=unlock_payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req2, timeout=5) as res2:
        data2 = json.loads(res2.read().decode('utf-8'))
        assert data2["status"] == "SUCCESS", "Unlock must succeed"
        ach = data2["result"]["achievement"]
        chapter = data2["result"]["story_chapter"]
        telemetry = data2["result"]["telemetry"]
        print(f"✓ POST /api/achievements/unlock: Unlocked '{ach['title']}' (+{ach['reward']['krystal_credits']} KC)")
        print(f"    Story Chapter: '{chapter['chapter_title']}'")
        j_tok = telemetry.get('energy_joules_per_tok') or telemetry.get('joules_per_token')
        print(f"    Intel Telemetry: {telemetry['power_watts']}W, {j_tok} J/tok")

    # 3. Test GET /api/achievements/codex
    req3 = urllib.request.Request(f"{base}/api/achievements/codex")
    with urllib.request.urlopen(req3, timeout=5) as res3:
        data3 = json.loads(res3.read().decode('utf-8'))
        assert data3["status"] == "OK"
        assert data3["chapters_count"] >= 1
        print(f"✓ GET /api/achievements/codex: OK ({data3['chapters_count']} chapters in codex)")

    # 4. Test GET /api/telemetry/hardware
    req4 = urllib.request.Request(f"{base}/api/telemetry/hardware")
    with urllib.request.urlopen(req4, timeout=5) as res4:
        data4 = json.loads(res4.read().decode('utf-8'))
        assert data4["status"] == "OK"
        snap = data4["live_snapshot"]
        print(f"✓ GET /api/telemetry/hardware: Backend: {snap['backend']}, Power: {snap['power_watts']}W, TTFT: {snap['ttft_ms']}ms")

    # 5. Test POST /api/narrative/synthesize
    synth_payload = json.dumps({
        "topic": "Vztýčenie Týnskeho Monolitu",
        "city": "Praha - Staré Město",
        "tribe": "Kryštálový Kmeň",
        "category": "URBAN_ARCHITECTURE"
    }).encode('utf-8')
    req5 = urllib.request.Request(f"{base}/api/narrative/synthesize", data=synth_payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req5, timeout=5) as res5:
        data5 = json.loads(res5.read().decode('utf-8'))
        assert data5["status"] == "SUCCESS"
        ch = data5["chapter"]
        print(f"✓ POST /api/narrative/synthesize: Generated '{ch['chapter_title']}' ({ch['telemetry']['power_watts']}W)")

    print("\n>>> ALL SERVER API ENDPOINTS VERIFIED 100% FUNCTIONAL <<<")

if __name__ == "__main__":
    test_endpoints()
