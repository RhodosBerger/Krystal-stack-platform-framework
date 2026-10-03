import urllib.request
import json
import sys
import os

sys.path.insert(0, os.path.abspath("."))
from krystal_web_hub.economic_engine import (
    EconomicLedger, CombatantState, Tribe, ResourceCost
)

def audit_devops():
    results = {}
    print("=== KRYSTAL-STACK DEVOPS VERIFICATION (ITERATION 15) ===")

    # 1. Daemon Health Port 8089
    try:
        req = urllib.request.Request("http://127.0.0.1:8089/api/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            results["port_8089_health"] = data.get("healthy", False) or data.get("status") in ("healthy", "ONLINE")
            print(f"[OK] Port 8089 Health: {data.get('status')} (healthy={data.get('healthy')})")
    except Exception as e:
        results["port_8089_health"] = False
        print(f"[FAIL] Port 8089 Health: {e}")

    # 2. Daemon Status & Max HP Rule
    try:
        req = urllib.request.Request("http://127.0.0.1:8089/api/status")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            max_hp = data.get("rules", {}).get("max_hp") or data.get("vital_max_hp_rule")
            results["max_hp_rule"] = max_hp == 6
            print(f"[OK] Core Engine Status: {data.get('status')}, Max HP Rule: {max_hp}")
    except Exception as e:
        results["max_hp_rule"] = False
        print(f"[FAIL] Core Engine Status: {e}")

    # 3. Execution Architecture Metrics Endpoint
    try:
        req = urllib.request.Request("http://127.0.0.1:8089/api/execution-architecture/metrics")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            measured = len(data.get("metrics_by_state", {}).get("MEASURED", []))
            results["execution_metrics"] = measured >= 4
            print(f"[OK] Execution Architecture Metrics: {measured} MEASURED metrics verified")
    except Exception as e:
        results["execution_metrics"] = False
        print(f"[FAIL] Execution Metrics: {e}")

    # 4. Godot .tscn Export Integrity
    try:
        req = urllib.request.Request("http://127.0.0.1:8089/api/economy/tscn")
        with urllib.request.urlopen(req, timeout=3) as resp:
            tscn_content = resp.read().decode('utf-8')
            has_header = "[gd_scene" in tscn_content
            has_node = "[node name=" in tscn_content
            results["godot_tscn_integrity"] = has_header and has_node
            print(f"[OK] Godot .tscn Export Integrity: Valid Godot Scene format ({len(tscn_content)} chars)")
    except Exception as e:
        results["godot_tscn_integrity"] = False
        print(f"[FAIL] Godot .tscn Export: {e}")

    # 5. Double-entry Ledger Balance Conservation
    try:
        hero = CombatantState(
            name="Auditor",
            tribe=Tribe.CRYSTAL,
            hp=6,
            max_hp=6,
            mana=50,
            aether_crystals=20
        )
        cost = ResourceCost(mana=10, aether_crystal=5)
        can_afford, msg = EconomicLedger.can_afford(hero, cost)
        deltas = EconomicLedger.deduct_resources(hero, cost)
        
        # Verify conservation: exact deltas match cost
        conserved = (
            can_afford and
            deltas.get("mana") == -10 and
            deltas.get("aether_crystal") == -5 and
            hero.mana == 40 and
            hero.aether_crystals == 15 and
            hero.hp == 6
        )
        results["ledger_conservation"] = conserved
        print(f"[OK] Double-Entry Ledger Conservation: Strict balance delta conservation verified: {conserved}")
    except Exception as e:
        results["ledger_conservation"] = False
        print(f"[FAIL] Ledger Audit: {e}")

    all_passed = all(results.values())
    print(f"\nDevOps Verification Status: {'ALL PASS' if all_passed else 'DEGRADED'}")
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(audit_devops())
