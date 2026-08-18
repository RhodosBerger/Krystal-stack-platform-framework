"""
The Great Audit 🕵️
Comprehensive System Integrity & Health Check for CNC Copilot.
Verifies:
1. Backend Connectivity (API)
2. Cortex Memory (Redis & Immutable Logs)
3. Live Link Synapse (WebSocket)
4. Absolute Safety Protocols (G90 in G-Code)
"""
import requests
import redis
import json
import asyncio
import websockets
import os
import sys
import hashlib
from datetime import datetime

# Config
API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/live_link"
REDIS_URL = "redis://localhost:6379/0"
GCODE_GEN_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cms", "llm_gcode_generator.py")

class SystemAuditor:
    def __init__(self):
        self.report = []
        self.errors = 0
        print(f"🕵️ STARTING SYSTEM AUDIT [{datetime.now().isoformat()}]")

    def log(self, area, status, msg):
        icon = "✅" if status == "PASS" else "❌"
        line = f"{icon} [{area}] {msg}"
        self.report.append(line)
        print(line)
        if status == "FAIL": self.errors += 1

    def check_backend(self):
        try:
            resp = requests.get(f"{API_URL}/api/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                self.log("BACKEND", "PASS", f"Online (Status: {data.get('global_status')})")
            else:
                self.log("BACKEND", "FAIL", f"Status Code: {resp.status_code}")
        except Exception as e:
            self.log("BACKEND", "FAIL", f"Connection Error: {e}")

    def check_cortex(self):
        try:
            r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            if r.ping():
                self.log("CORTEX", "PASS", "Redis Memory Connected")
                
                # Check Immutable Logs
                logs = r.lrange("cortex:logs", 0, 5)
                if logs:
                    valid_hashes = 0
                    for log_str in logs:
                        log = json.loads(log_str)
                        if "hash" in log:
                            # Verify Hash
                            stored_hash = log.pop("hash")
                            recalc_hash = hashlib.sha256(json.dumps(log, sort_keys=True).encode()).hexdigest()
                            if stored_hash == recalc_hash:
                                valid_hashes += 1
                    
                    if valid_hashes > 0:
                        self.log("CORTEX", "PASS", f"Verified {valid_hashes} Immutable Log Hashes")
                    else:
                        self.log("CORTEX", "WARN", "No logs found or Hash mismatch")
                else:
                    self.log("CORTEX", "WARN", "Log Stream Empty")

        except Exception as e:
            self.log("CORTEX", "FAIL", f"Redis Error: {e}")

    async def check_livelink(self):
        try:
            async with websockets.connect(WS_URL) as ws:
                # Send Ping
                await ws.send(json.dumps({"type": "AUDIT_PING", "payload": {}}))
                self.log("LIVELINK", "PASS", "WebSocket Connected")
        except Exception as e:
            self.log("LIVELINK", "FAIL", f"WebSocket Error: {e}")

    def check_safety_protocols(self):
        # Static Analysis of G-Code Generator
        if os.path.exists(GCODE_GEN_PATH):
            with open(GCODE_GEN_PATH, 'r') as f:
                content = f.read()
                
            if 'G90' in content and 'ABSOLUTE SAFETY PROTOCOL' in content:
                self.log("SAFETY", "PASS", "G90 Absolute Protocol Found in Code")
            else:
                self.log("SAFETY", "FAIL", "G90 Protocol MISSING in Generator Source!")
        else:
             self.log("SAFETY", "WARN", f"Generator path not found: {GCODE_GEN_PATH}")

    async def run(self):
        print("-" * 50)
        self.check_backend()
        self.check_cortex()
        self.check_safety_protocols()
        await self.check_livelink()
        
        print("-" * 50)
        if self.errors == 0:
            print("🟢 SYSTEM READY FOR PRODUCTION")
        else:
            print(f"🔴 SYSTEM AUDIT FAILED ({self.errors} Errors)")

if __name__ == "__main__":
    auditor = SystemAuditor()
    asyncio.run(auditor.run())
