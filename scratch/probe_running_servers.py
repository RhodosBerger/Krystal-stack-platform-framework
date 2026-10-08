"""
Probe all local ports to see what server is running and what routes it responds to.
"""

import urllib.request
import urllib.error
import json
from pathlib import Path

results = {}

ports_to_check = [8080, 8089, 8000, 3000, 5000]
paths_to_check = [
    "/",
    "/api/health",
    "/api/status",
    "/city",
    "/city/",
    "/city-studio",
    "/static/city_composer_studio.html",
    "/static/index.html"
]

for port in ports_to_check:
    results[port] = {"online": False, "paths": {}}
    for path in paths_to_check:
        url = f"http://127.0.0.1:{port}{path}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                results[port]["online"] = True
                results[port]["paths"][path] = {
                    "code": resp.status,
                    "ct": resp.headers.get("Content-Type", "")[:30],
                    "len": len(resp.read())
                }
        except urllib.error.HTTPError as e:
            results[port]["online"] = True
            body = e.read().decode('utf-8', errors='replace')[:200]
            results[port]["paths"][path] = {
                "code": e.code,
                "msg": e.reason,
                "body_snippet": body
            }
        except Exception as e:
            results[port]["paths"][path] = {"error": str(e)}

out_file = Path(__file__).resolve().parent / "probe_results.json"
out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"Probe completed. Written to {out_file}")
