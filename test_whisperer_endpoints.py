import sys
import io
import json
from krystal_web_hub.server import KrystalHubHandler

class DummyHandler(KrystalHubHandler):
    def __init__(self, path, method="GET", body=b""):
        self.path = path
        self.command = method
        self.requestline = f"{method} {path} HTTP/1.1"
        self.request_version = "HTTP/1.1"
        self.headers = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        if method == "GET":
            self.do_GET()
        elif method == "POST":
            self.do_POST()

def verify_endpoints():
    print("Testing GET /api/whisperer/calculator...")
    r1 = DummyHandler("/api/whisperer/calculator")
    r1.wfile.seek(0)
    out1 = r1.wfile.read().decode("utf-8", errors="ignore")
    assert "200 OK" in out1, f"Expected 200 OK in out1, got: {out1[:200]}"
    assert '"status": "OK"' in out1, f"Expected status OK, got: {out1[:200]}"
    print("  -> OK!")

    print("Testing POST /api/whisperer/adaptive_swap...")
    r2 = DummyHandler("/api/whisperer/adaptive_swap", method="POST", body=b"{}")
    r2.wfile.seek(0)
    out2 = r2.wfile.read().decode("utf-8", errors="ignore")
    assert "200 OK" in out2, f"Expected 200 OK in out2, got: {out2[:200]}"
    assert "SWAP_SUCCESS" in out2 or "IDLE" in out2, f"Expected swap success, got: {out2[:200]}"
    print("  -> OK!")

    print("Testing POST /api/whisperer/synthesize_godot...")
    r3 = DummyHandler("/api/whisperer/synthesize_godot", method="POST", body=b'{"scene": "NeoPraha_Test", "seed": 42}')
    r3.wfile.seek(0)
    out3 = r3.wfile.read().decode("utf-8", errors="ignore")
    assert "200 OK" in out3, f"Expected 200 OK in out3, got: {out3[:200]}"
    assert '"status": "SUCCESS"' in out3, f"Expected status SUCCESS, got: {out3[:200]}"
    print("  -> OK!")

    print("Testing GET /api/whisperer/readback_logs...")
    r4 = DummyHandler("/api/whisperer/readback_logs")
    r4.wfile.seek(0)
    out4 = r4.wfile.read().decode("utf-8", errors="ignore")
    assert "200 OK" in out4, f"Expected 200 OK in out4, got: {out4[:200]}"
    assert '"status": "OK"' in out4, f"Expected status OK, got: {out4[:200]}"
    print("  -> OK!")

    print("\nALL SERVER ENDPOINTS VERIFIED AND RESPONDING WITH 200 OK!")

if __name__ == "__main__":
    verify_endpoints()
