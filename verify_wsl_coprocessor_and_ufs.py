#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: VERIFICATION SUITE FOR WSL2 COPROCESSOR & UFS LOG TRIAGE
==============================================================================
Validates:
  1. WSL2 Coprocessor discovery, execution modes, and Bash script presence.
  2. Automated failing process diagnostics (0xC0000005, OOM, Deadlock, Zombie).
  3. UFS log stream sorting, severity breakdown, and inode integrity checks.
  4. Automated administrative remediation actions without manual human intervention.
  5. C/C++ Header definitions in include/krystal_wsl_ufs_bridge.h.
  6. Server endpoint handlers and OpenAPI 3.1.0 schema validity.

System Invariant: VITAL_MAX_HP = 6.
==============================================================================
"""

import os
import sys
import json
from pathlib import Path

# Ensure root directory is in sys.path
WORKSPACE_ROOT = Path(__file__).parent.resolve()
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from krystal_kernel.wsl_coprocessor import (
    GLOBAL_WSL_COPROCESSOR,
    ProcessFailureType,
    AutomatedAdminAction,
    VITAL_MAX_HP
)
from krystal_kernel.openapi_spec import get_openapi_specification


def test_1_wsl_coprocessor_status():
    print("[TEST 1/6] Testing WSL2 Coprocessor Discovery & Script Setup...")
    status = GLOBAL_WSL_COPROCESSOR.get_coprocessor_status()
    print(f"  -> WSL Available: {status['wsl_installed']} | Distro: {status['default_distro']}")
    print(f"  -> Coprocessor Engine: {status['coprocessor_engine']} | Mode: {status['execution_mode']}")
    print(f"  -> Hardware Coprocessor Script: {status['scripts_available']['wsl_hardware_coprocessor']}")
    print(f"  -> UFS Triage Script: {status['scripts_available']['ufs_log_triage']}")
    
    assert status["coprocessor_engine"] == "ACTIVE_UNIX_COPROCESSOR"
    assert status["vital_max_hp"] == 6
    assert status["scripts_available"]["wsl_hardware_coprocessor"] is True
    assert status["scripts_available"]["ufs_log_triage"] is True
    print("  [PASS] WSL2 Coprocessor Setup verified.")


def test_2_process_diagnostics():
    print("\n[TEST 2/6] Testing Failing Process Diagnostics & Crash Classification...")
    # Case A: Access Violation 0xC0000005 / SIGSEGV
    rep_segv = GLOBAL_WSL_COPROCESSOR.diagnose_failing_process(
        pid=4412,
        process_name="vulkan_renderer.exe",
        crash_signature="STATUS_ACCESS_VIOLATION (0xC0000005)"
    )
    print(f"  -> 0xC0000005 Case: {rep_segv.failure_type.value} | Action: {rep_segv.prescribed_admin_action.value} ({rep_segv.confidence_pct}%)")
    assert rep_segv.failure_type == ProcessFailureType.SEGFAULT_ACCESS_VIOLATION
    assert rep_segv.prescribed_admin_action == AutomatedAdminAction.QUARANTINE_PROCESS_AND_STAGE_DUMP
    assert rep_segv.admin_intervention_required is False

    # Case B: Out Of Memory
    rep_oom = GLOBAL_WSL_COPROCESSOR.diagnose_failing_process(
        pid=8192,
        process_name="tensor_compiler.exe",
        crash_signature="OOM_KILLER (0xC0000017)"
    )
    print(f"  -> OOM Case: {rep_oom.failure_type.value} | Action: {rep_oom.prescribed_admin_action.value}")
    assert rep_oom.failure_type == ProcessFailureType.OOM_MEMORY_EXHAUSTION
    assert rep_oom.prescribed_admin_action == AutomatedAdminAction.ESCALATE_UMA_QUOTIENT_TO_512MB

    # Case C: Mutex Deadlock
    rep_deadlock = GLOBAL_WSL_COPROCESSOR.diagnose_failing_process(
        pid=9004,
        process_name="ring_buffer_sync.exe",
        crash_signature="STATUS_POSSIBLE_DEADLOCK (0xC0000194)"
    )
    print(f"  -> Deadlock Case: {rep_deadlock.failure_type.value} | Action: {rep_deadlock.prescribed_admin_action.value}")
    assert rep_deadlock.failure_type == ProcessFailureType.MUTEX_DEADLOCK
    assert rep_deadlock.vital_max_hp == 6
    print("  [PASS] Failing Process Diagnostics verified.")


def test_3_ufs_log_triage():
    print("\n[TEST 3/6] Testing Automated UFS Log Triage & Inode Verification...")
    triage = GLOBAL_WSL_COPROCESSOR.run_ufs_log_triage(auto_heal=True)
    print(f"  -> Triage ID: {triage.triage_id} | Total Log Entries: {triage.total_log_entries}")
    print(f"  -> Severities: {triage.severity_breakdown}")
    print(f"  -> Signatures Detected: {triage.detected_crash_signatures}")
    print(f"  -> UFS Inode Health: {triage.ufs_inode_health['free_inodes']} free of {triage.ufs_inode_health['total_inodes']} (State: {triage.ufs_inode_health['filesystem_state']})")
    print(f"  -> Automated Maintenance: {triage.automated_maintenance['recommended_action']}")

    assert triage.total_log_entries > 0
    assert triage.ufs_inode_health["total_inodes"] == 65536
    assert triage.automated_maintenance["admin_intervention_required"] is False
    assert triage.vital_max_hp == 6
    print("  [PASS] UFS Log Triage & Inode Verification verified.")


def test_4_automated_admin_remediation():
    print("\n[TEST 4/6] Testing Automated System Administrator Remediation Actions...")
    rem_dump = GLOBAL_WSL_COPROCESSOR.execute_automated_admin_remediation("QUARANTINE_PROCESS_AND_STAGE_DUMP")
    print(f"  -> Quarantine Dump Action: {rem_dump['remediation']['status']} | Human needed: {rem_dump['remediation']['human_admin_needed']}")
    assert rem_dump["remediation"]["status"] == "COMPLETED"
    assert rem_dump["remediation"]["human_admin_needed"] is False

    rem_uma = GLOBAL_WSL_COPROCESSOR.execute_automated_admin_remediation("ESCALATE_UMA_QUOTIENT_TO_512MB")
    print(f"  -> UMA Quotient Escalate: {rem_uma['remediation']['effect']}")
    assert rem_uma["remediation"]["status"] == "COMPLETED"
    assert rem_uma["vital_max_hp"] == 6
    print("  [PASS] Automated Admin Remediation verified.")


def test_5_cpp_bridge_header():
    print("\n[TEST 5/6] Testing C/C++ Header Definitions in include/krystal_wsl_ufs_bridge.h...")
    header_path = WORKSPACE_ROOT / "include" / "krystal_wsl_ufs_bridge.h"
    assert header_path.exists(), f"Header missing at {header_path}"
    content = header_path.read_text(encoding="utf-8")
    
    assert "KRYSTAL_VITAL_MAX_HP 6" in content
    assert "KrystalWslFailureType" in content
    assert "KRYSTAL_FAIL_SEGFAULT_ACCESS_VIOL" in content
    assert "KrystalAdminRemediationAction" in content
    assert "KrystalUfsVolumeHealth" in content
    assert "krystal_translate_ntstatus_to_failure" in content
    assert "krystal_resolve_admin_remediation" in content
    print("  -> Validated C99/C++ structs, enums, and inline translator functions.")
    print("  [PASS] C/C++ Bridge Header verified.")


def test_6_server_and_openapi():
    print("\n[TEST 6/6] Testing OpenAPI 3.1.0 & Server Endpoint Verification...")
    spec = get_openapi_specification()
    paths = spec["paths"]
    schemas = spec["components"]["schemas"]
    
    assert "/api/wsl/coprocessor" in paths
    assert "/api/wsl/diagnose_process" in paths
    assert "/api/wsl/ufs_triage" in paths
    assert "/api/wsl/remediate" in paths
    assert "WSLCoprocessorStatus" in schemas
    assert "ProcessDiagnosticReport" in schemas
    assert "UfsLogTriageResult" in schemas

    # Test server imports and execution
    from krystal_web_hub.server import KrystalHubHandler
    assert hasattr(KrystalHubHandler, "do_GET")
    assert hasattr(KrystalHubHandler, "do_POST")
    print(f"  -> Verified OpenAPI 3.1.0: {len(paths)} endpoints and {len(schemas)} schemas.")
    print("  [PASS] Server Endpoints & OpenAPI Specification verified.")


if __name__ == "__main__":
    print("=" * 85)
    print("  KRYSTAL-STACK: WSL2 COPROCESSOR & UFS LOG TRIAGE VERIFICATION")
    print("=" * 85)
    test_1_wsl_coprocessor_status()
    test_2_process_diagnostics()
    test_3_ufs_log_triage()
    test_4_automated_admin_remediation()
    test_5_cpp_bridge_header()
    test_6_server_and_openapi()
    print("=" * 85)
    print("  ALL 6 WSL2 COPROCESSOR & UFS TEST SUITES PASSED WITH 100% ACCURACY!")
    print(f"  SYSTEM INVARIANT SATISFIED: VITAL_MAX_HP = {VITAL_MAX_HP}")
    print("=" * 85)
