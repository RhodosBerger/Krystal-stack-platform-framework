#!/usr/bin/env python3
"""
Krystal-Stack Localhost Environment Verification Test Suite
===========================================================
Validates:
  - Multi-threaded HTTP Server launch & shutdown
  - HTML, CSS, JS static asset delivery
  - REST API endpoints (Health, Status, Control, Governor, Director)
  - Server-Sent Events (SSE) live ASCII streaming
  - Closed-loop Director prompt processing
"""

import sys
import time
import json
import threading
import urllib.request
import urllib.error

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

TEST_PORT = 8099
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"

def run_tests():
    print("\n" + "="*70)
    print(" [TEST] KRYSTAL-STACK LOCALHOST ENVIRONMENT TEST SUITE")
    print("="*70)

    # 1. Start Server in Background Thread
    from krystal_web_hub.server import ThreadedHTTPServer, KrystalHubHandler, engine_worker_loop, state
    
    server = ThreadedHTTPServer(("127.0.0.1", TEST_PORT), KrystalHubHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    worker_thread = threading.Thread(target=engine_worker_loop, daemon=True)
    worker_thread.start()

    # Wait for server to warm up
    time.sleep(1.0)
    print("[1/7] Localhost Server started on test port 8099: OK")

    try:
        # 2. Test GET / (HTML)
        req = urllib.request.Request(f"{BASE_URL}/")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200, f"Expected 200, got {resp.status}"
            html = resp.read().decode("utf-8")
            assert "KRYSTAL" in html, "Missing KRYSTAL branding in HTML"
            assert "asciiCanvas" in html, "Missing ASCII canvas element in HTML"
            print("[2/7] Static HTML delivery (GET /): OK")

        # 3. Test GET /static/style.css & app.js
        with urllib.request.urlopen(f"{BASE_URL}/static/style.css") as resp:
            assert resp.status == 200
            assert "cyberpunk" in resp.read().decode("utf-8").lower()
        with urllib.request.urlopen(f"{BASE_URL}/static/app.js") as resp:
            assert resp.status == 200
            assert "EventSource" in resp.read().decode("utf-8")
        print("[3/7] Static CSS and JS assets delivery: OK")

        # 4. Test GET /api/health & /api/status
        with urllib.request.urlopen(f"{BASE_URL}/api/health") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data.get("status") == "HEALTHY"
        with urllib.request.urlopen(f"{BASE_URL}/api/status") as resp:
            status = json.loads(resp.read().decode("utf-8"))
            assert "mode" in status
            assert "entropy" in status
            assert "governor" in status
            print(f"[4/7] REST API Health & Status (Mode: {status['mode']}, Entropy: {status['entropy']['total']:.2f}): OK")

        # 5. Test POST /api/control (Mode Switch)
        req = urllib.request.Request(
            f"{BASE_URL}/api/control",
            data=json.dumps({"mode": "RAYMARCH_ANOMALY"}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            assert res.get("mode") == "RAYMARCH_ANOMALY"
            print("[5/7] Control API Mode Switch to RAYMARCH_ANOMALY: OK")

        # 6. Test POST /api/director (Prompt Injection)
        req = urllib.request.Request(
            f"{BASE_URL}/api/director",
            data=json.dumps({"prompt": "engage stealth mode"}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            assert "Matrix" in res.get("reply", "") or "stealth" in res.get("reply", "").lower()
            print(f"[6/7] Cognitive Scene Director prompt injection: OK -> '{res.get('reply')}'")

        # 7. Test SSE Stream (GET /api/stream)
        print("[7/7] Testing Server-Sent Events (SSE) Live Stream...")
        req = urllib.request.Request(f"{BASE_URL}/api/stream")
        with urllib.request.urlopen(req) as resp:
            frames_received = 0
            for _ in range(15):
                line = resp.readline().decode("utf-8")
                if line.startswith("data:"):
                    raw_json = line[5:].strip()
                    try:
                        frame_data = json.loads(raw_json)
                        if "ascii" in frame_data and len(frame_data["ascii"]) > 50:
                            frames_received += 1
                            if frames_received >= 3:
                                break
                    except Exception:
                        pass
            assert frames_received >= 2, f"Expected at least 2 ASCII frames, received {frames_received}"
            print(f"      Received {frames_received} live ASCII frames via SSE stream: OK")

        print("\n" + "="*70)
        print(" [SUCCESS] ALL PRODUCTION LOCALHOST VERIFICATION TESTS PASSED SUCCESSFULLY!")
        print("="*70 + "\n")
        return True

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        state.running = False
        server.shutdown()
        server.server_close()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
