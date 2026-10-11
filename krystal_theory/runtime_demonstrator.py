"""
==============================================================================
KRYSTAL-STACK TRACK C: DETERMINISTIC RUNTIME DEMONSTRATOR
==============================================================================
Demonstrates the coupled P1 (Strain-Activated Triage Reachability) and
P2 (Smith's Rule / WSPT Scheduling) pipeline across three canonical scenarios:
- Scenario A: Low / Zero Strain (Nominal quiescent regime)
- Scenario B: Moderate Strain (Hypervigilance reached & scheduled)
- Scenario C: Acute Strain & Contention (Multi-mechanism triage & WSPT ranked)
==============================================================================
"""

import os
import sys
import json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from krystal_theory.formal_theory_bridge import FormalTheoryBridge


def run_demonstration():
    bridge = FormalTheoryBridge()
    print("=" * 80)
    print("KRYSTAL-STACK TRACK C: FORMAL THEORY RESOLUTION RUNTIME DEMONSTRATOR")
    print("Coupling P1 (Reachable Mechanisms) + P2 (Smith's WSPT Scheduler)")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # SCENARIO A: Low Strain (Delta >= 0)
    # ------------------------------------------------------------------------
    print("\n--- SCENARIO A: ZERO STRAIN (REAL REGIME, QUIESCENT) ---")
    res_a = bridge.execute_coupled_pipeline(
        a=1.0, b=4.0, c=3.0, # Delta = 16 - 12 = 4 >= 0
        context_entropy=0.15,
        memory_contention=0.8,
        thermal_pressure_pct=52.0
    )
    print(f"  Discriminant Delta:     {res_a['strain']['delta']}")
    print(f"  Normalized Strain:      {res_a['strain']['normalized_strain']:.4f}")
    print(f"  Reachable Mechanisms:   {res_a['reachability']['runtime_reachable']}")
    print(f"  Currently Active:       {res_a['reachability']['currently_active']}")
    print(f"  Selected Action:        {res_a['selected_action']['mechanism'] if res_a['selected_action'] else 'None'}")
    print(f"  Blocked Reasons:        {res_a['reachability']['blocked_reasons']}")
    print(f"  Invariant Gate:         {res_a['invariant_gate']['reason']}")

    # ------------------------------------------------------------------------
    # SCENARIO B: Moderate Strain (Delta < 0, sigma ~= 0.609)
    # ------------------------------------------------------------------------
    print("\n--- SCENARIO B: MODERATE STRAIN (LATENT COMPLEX REGIME) ---")
    res_b = bridge.execute_coupled_pipeline(
        a=1.0, b=2.0, c=1.5, # Delta = 4 - 6 = -2 < 0
        context_entropy=0.30,
        memory_contention=1.2,
        thermal_pressure_pct=62.0,
        epistemic_uncertainty=0.55,
        safety_margin=0.20
    )
    print(f"  Discriminant Delta:     {res_b['strain']['delta']}")
    print(f"  Normalized Strain:      {res_b['strain']['normalized_strain']:.4f}")
    print(f"  Reachable Mechanisms:   {res_b['reachability']['runtime_reachable']}")
    print(f"  Currently Active:       {res_b['reachability']['currently_active']}")
    print(f"  Ranked Actions:         {[a['action']['mechanism'] for a in res_b['ranked_actions']]}")
    print(f"  Selected Action:        {res_b['selected_action']['mechanism'] if res_b['selected_action'] else 'None'}")
    print(f"  Invariant Gate:         {res_b['invariant_gate']['reason']}")

    # ------------------------------------------------------------------------
    # SCENARIO C: Acute Strain + Contention (Multi-mechanism triage & WSPT rank)
    # ------------------------------------------------------------------------
    print("\n--- SCENARIO C: ACUTE STRAIN & CONTENTION (WSPT TRIAGE ARBITRATION) ---")
    res_c = bridge.execute_coupled_pipeline(
        a=1.0, b=2.0, c=5.0, # Delta = 4 - 20 = -16 < 0
        context_entropy=0.45,
        memory_contention=3.5, # Contention exceeds limit
        thermal_pressure_pct=76.0,
        epistemic_uncertainty=0.60,
        safety_margin=0.15,
        loss_tolerance=0.40
    )
    print(f"  Discriminant Delta:     {res_c['strain']['delta']}")
    print(f"  Normalized Strain:      {res_c['strain']['normalized_strain']:.4f}")
    print(f"  Reachable Mechanisms:   {res_c['reachability']['runtime_reachable']}")
    print(f"  Currently Active:       {res_c['reachability']['currently_active']}")
    for r in res_c['ranked_actions']:
        act = r['action']
        print(f"    Rank {r['rank']}: {act['mechanism']} (weight={act['weight']:.2f}, p={act['processing_time']}us, rho={act['smith_ratio']:.6f})")
    print(f"  Selected Action:        {res_c['selected_action']['mechanism']}")
    print(f"  Invariant Gate:         {res_c['invariant_gate']['reason']}")

    print("\n" + "=" * 80)
    print("FORMAL THEORY RESOLUTION REPORT JSON:")
    report = bridge.generate_resolution_report()
    print(json.dumps(report, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    run_demonstration()
