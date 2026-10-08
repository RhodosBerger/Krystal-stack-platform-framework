"""
KRYSTAL-STACK: VERIFY HTTP ENDPOINTS & ROUTING ALIASES
======================================================
Tests in-process HTTP server on an ephemeral port (e.g. 8099) to verify that:
1. /city and /city/ resolve to 200 OK
2. /city-studio and /city_composer_studio.html resolve to 200 OK
3. /api/city/* endpoints return valid 200 OK responses
4. No 404 "Endpoint not found" occurs for any alias
"""

import sys
import os
import time
import json
import threading
import urllib.request
import urllib.error
from pathlib import Path

# Setup UTF-8 encoding
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from krystal_web_hub.server import KrystalHubHandler, ThreadedHTTPServer


def test_endpoints():
    print("=" * 80)
    print(" [KRYSTAL-STACK] HTTP ENDPOINTS & ROUTING VERIFICATION")
    print("=" * 80)

    test_port = 8099
    server = ThreadedHTTPServer(("127.0.0.1", test_port), KrystalHubHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f">> Ephemeral test server active on http://127.0.0.1:{test_port}/")

    endpoints_to_test = [
        # UI Pages & Aliases
        ("/city", 200, "text/html"),
        ("/city/", 200, "text/html"),
        ("/city-studio", 200, "text/html"),
        ("/city-studio/", 200, "text/html"),
        ("/city_studio", 200, "text/html"),
        ("/city-composer", 200, "text/html"),
        ("/city_composer_studio.html", 200, "text/html"),
        ("/metropolis", 200, "text/html"),
        ("/static/city_composer_studio.html", 200, "text/html"),

        # API Endpoints
        ("/api/city/compose?seed=42", 200, "application/json"),
        ("/api/city/metropolis?seed=101&cols=3&rows=3", 200, "application/json"),
        ("/api/city/ascii_skyline?seed=42", 200, "text/plain"),
        ("/api/city/ascii_plan?seed=42", 200, "text/plain"),
        ("/api/city/export_godot?seed=42", 200, "text/plain"),
        ("/api/city/export_java?seed=42", 200, "text/plain"),
        ("/api/city/export_janet?seed=42", 200, "text/plain"),
    ]

    all_passed = True
    results = []

    for path, expected_code, expected_content_type in endpoints_to_test:
        url = f"http://127.0.0.1:{test_port}{path}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                status = resp.status
                ct = resp.headers.get("Content-Type", "")
                content = resp.read()
                is_ok = (status == expected_code) and (expected_content_type in ct)
                print(f"  [{'PASS' if is_ok else 'FAIL'}] {path:<42} -> {status} (CT: {ct[:25]}, {len(content)} bytes)")
                results.append({"path": path, "status": status, "passed": is_ok})
                if not is_ok:
                    all_passed = False
        except urllib.error.HTTPError as e:
            print(f"  [FAIL] {path:<42} -> HTTP {e.code} (Message: {e.reason})")
            results.append({"path": path, "status": e.code, "passed": False})
            all_passed = False
        except Exception as e:
            print(f"  [ERROR] {path:<42} -> {e}")
            results.append({"path": path, "status": 500, "passed": False})
            all_passed = False

    server.shutdown()
    server.server_close()

    print("\n" + "=" * 80)
    print(f"VERIFICATION RESULT: {'ALL PASS' if all_passed else 'FAILURES DETECTED'}")
    print("=" * 80)

    # Write report
    report_file = Path(ROOT_DIR) / "scratch" / "test_endpoints_report.json"
    report_file.write_text(json.dumps({"all_passed": all_passed, "results": results}, indent=2), encoding="utf-8")
    print(f">> Report saved to: {report_file}")

    return all_passed


if __name__ == "__main__":
    success = test_endpoints()
    sys.exit(0 if success else 1)
