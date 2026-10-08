#!/usr/bin/env python3
"""
VERIFICATION SUITE: NEXTGEN v2 ARCHITECTURE, IRIS XE KISAK OPTIMIZER & ANTI-MINING GOVERNOR
==========================================================================================
Verifies:
  1. Invariant: VITAL_MAX_HP = 6 across all nextgen components.
  2. Iris Xe Kisak Bandwidth Optimizer (Tile4, CCS, SIMD16 reduction).
  3. Anti-Mining Power Governor (detecting disguised PoW vs authentic graphics).
  4. NextGen Divergence-Free Subgroup Raymarch Kernel.
  5. REST API endpoints:
     - GET /api/nextgen/status
     - POST /api/nextgen/audit_energy
     - POST /api/nextgen/optimize_iris_xe

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
"""

import sys
import os
import io
import json

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from krystal_stack_nextgen import (
    VITAL_MAX_HP,
    IrisXeKisakOptimizer,
    AntiMiningPowerGovernor,
    WorkloadCategory,
    SubgroupRaymarchKernel,
    GLOBAL_IRIS_XE_OPTIMIZER,
    GLOBAL_ENERGY_GOVERNOR,
    GLOBAL_SUBGROUP_KERNEL
)
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


def test_invariant():
    print("\n[TEST 1] Verifying Inviolable Architectural Invariant...")
    assert VITAL_MAX_HP == 6, f"Expected VITAL_MAX_HP == 6, got {VITAL_MAX_HP}"
    print(f"  ✓ VITAL_MAX_HP locked at {VITAL_MAX_HP}")


def test_iris_xe_kisak_optimizer():
    print("\n[TEST 2] Verifying Iris Xe Kisak Bandwidth Optimizer...")
    plan = GLOBAL_IRIS_XE_OPTIMIZER.compute_bandwidth_optimization(1920, 1080, 60, 96)
    print(f"  ✓ Baseline Bandwidth:  {plan.baseline_bandwidth_gb_s} GB/s ({plan.baseline_watts} W)")
    print(f"  ✓ Optimized Bandwidth: {plan.optimized_bandwidth_gb_s} GB/s ({plan.optimized_watts} W)")
    print(f"  ✓ Bandwidth Reduction: {plan.bandwidth_reduction_ratio}x")
    print(f"  ✓ Efficiency Multiplier: {plan.energy_efficiency_multiplier}x")
    assert plan.bandwidth_reduction_ratio >= 3.0
    assert plan.optimized_watts < plan.baseline_watts
    assert plan.vital_max_hp == 6


def test_anti_mining_power_governor():
    print("\n[TEST 3] Verifying Anti-Mining Power Governor & Tariff Compliance...")
    gov = GLOBAL_ENERGY_GOVERNOR

    # Clean graphics test
    a1 = gov.audit_workload_telemetry(8.5, 60.0, 0.65, True, 10.0)
    print(f"  ✓ Clean Graphics: {a1.workload_category} (Compliant: {a1.is_tariff_compliant})")
    assert a1.workload_category == WorkloadCategory.AUTHENTIC_GRAPHICS_INTERACTIVE
    assert a1.is_tariff_compliant is True

    # Mining attempt test
    a2 = gov.audit_workload_telemetry(27.0, 120.0, 0.01, False, 60.0)
    print(f"  ✓ Disguised Mining: {a2.workload_category} (Suspicion: {a2.mining_suspicion_score})")
    assert a2.workload_category == WorkloadCategory.DISGUISED_CRYPTO_MINING
    assert a2.is_tariff_compliant is False

    # Pathological memory stall test
    a3 = gov.audit_workload_telemetry(24.0, 16.0, 0.45, True, 14.0)
    print(f"  ✓ Memory Stall: {a3.workload_category} (Remediation: {a3.remediation_action[:25]}...)")
    assert a3.workload_category == WorkloadCategory.PATHOLOGICAL_BOTTLENECK_STALL


def test_subgroup_raymarch_kernel():
    print("\n[TEST 4] Verifying NextGen Subgroup Raymarch Kernel...")
    shader, prof = GLOBAL_SUBGROUP_KERNEL.generate_optimized_godot_shader()
    print(f"  ✓ Kernel Profile: {prof.kernel_id} | Divergence: {prof.eu_warp_divergence_pct}% | Projected FPS: {prof.projected_fps_iris_xe}")
    assert "intersect_aabb" in shader
    assert prof.eu_warp_divergence_pct < 10.0
    assert prof.vital_max_hp == 6


def test_nextgen_endpoints():
    print("\n[TEST 5] Verifying Web Hub NextGen REST Endpoints...")

    # 1. GET /api/nextgen/status
    h1 = DummyHandler("/api/nextgen/status")
    h1.wfile.seek(0)
    out1 = h1.wfile.read().decode("utf-8", errors="ignore")
    assert "200 OK" in out1, f"Expected 200 OK, got: {out1[:150]}"
    assert '"status": "OK"' in out1
    assert '"nextgen_version": "2.0-PRODUCTION"' in out1
    assert '"vital_max_hp": 6' in out1
    print("  ✓ GET /api/nextgen/status -> 200 OK")

    # 2. POST /api/nextgen/audit_energy
    body2 = json.dumps({"watts": 7.5, "fps": 60.0, "entropy": 0.50, "has_display_presentation": True}).encode("utf-8")
    h2 = DummyHandler("/api/nextgen/audit_energy", method="POST", body=body2)
    h2.wfile.seek(0)
    out2 = h2.wfile.read().decode("utf-8", errors="ignore")
    assert "200 OK" in out2, f"Expected 200 OK, got: {out2[:150]}"
    assert '"status": "SUCCESS"' in out2
    assert '"is_tariff_compliant": true' in out2
    print("  ✓ POST /api/nextgen/audit_energy -> 200 OK (Audited & Verified)")

    # 3. POST /api/nextgen/optimize_iris_xe
    body3 = json.dumps({"width": 1920, "height": 1080, "fps": 60, "steps": 64}).encode("utf-8")
    h3 = DummyHandler("/api/nextgen/optimize_iris_xe", method="POST", body=body3)
    h3.wfile.seek(0)
    out3 = h3.wfile.read().decode("utf-8", errors="ignore")
    assert "200 OK" in out3, f"Expected 200 OK, got: {out3[:150]}"
    assert '"status": "SUCCESS"' in out3
    assert '"bandwidth_reduction_ratio":' in out3
    print("  ✓ POST /api/nextgen/optimize_iris_xe -> 200 OK (Kisak Plan emitted)")


def main():
    print("=" * 75)
    print(" KRYSTAL-STACK NEXTGEN (v2 ARCHITECTURE) FULL SYSTEM VERIFICATION")
    print("=" * 75)

    test_invariant()
    test_iris_xe_kisak_optimizer()
    test_anti_mining_power_governor()
    test_subgroup_raymarch_kernel()
    test_nextgen_endpoints()

    print("\n" + "=" * 75)
    print(" ✅ ALL NEXTGEN VERIFICATION TESTS PASSED (100% SUCCESS)")
    print("=" * 75)


if __name__ == "__main__":
    main()
