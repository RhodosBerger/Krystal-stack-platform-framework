#!/usr/bin/env python3
"""
UI CONNECTOR
Client Bridge for fetching dependencies between UI and Backend.
"""

import requests
import json
from typing import Dict, Any, List

class DashboardConnector:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.session = requests.Session()

    def check_health(self) -> bool:
        try:
            resp = self.session.get(f"{self.api_url}/")
            return resp.status_code == 200
        except:
            return False

    def fetch_perception(self, fanuc_data: Dict, sw_data: Dict) -> Dict[str, Any]:
        """
        Sends raw simulation data to the backend -> Returns Brain State.
        """
        payload = {
            "fanuc_data": fanuc_data,
            "sw_data": sw_data
        }
        try:
            resp = self.session.post(f"{self.api_url}/perceive", json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e), "safety": 0, "recommended_action": "NETWORK_ERROR"}

    def run_remote_optimization(self, gcode: List[str], material: str) -> List[str]:
        """
        Offloads optimization to the server.
        """
        try:
            resp = self.session.post(f"{self.api_url}/optimize", params={"material": material}, json=gcode)
            resp.raise_for_status()
            return resp.json().get("optimized_ir", [])
        except Exception as e:
            return [f"; NETWORK ERROR: {e}"]

# Usage
if __name__ == "__main__":
    client = DashboardConnector()
    if client.check_health():
        print(" [OK] Connected to Fanuc Rise API")
        res = client.fetch_perception({"load": 50, "rpm": 8000}, {"curvature": 0.1})
        print("Remote Brain says:", res)
    else:
        print(" [ERR] API Offline")
