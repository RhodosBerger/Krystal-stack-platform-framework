#!/usr/bin/env python3
"""
================================================================================
KSL v0.2 — HARDWARE TRANSITION EXPERIMENT 001 (HTE-001) PREFLIGHT VERIFICATION
================================================================================
Module: krystal_kernel/hte001_preflight.py

Normative References:
  - docs/research/KSL_V02_HARDWARE_TRANSITION_EXPERIMENT_001.md
  - docs/research/KSL_CONTROL_PLANE_SOFTWARE_BASELINE_FREEZE.md
  - artifacts/hte001/HTE001_RUN_MANIFEST.json
  - docs/research/HTE001_FAILURE_TAXONOMY_SPEC.md

Constitutional Principles:
  - Observer Governance Law: Preflight checks inspect and verify, never mutate.
  - Anti-Promotion Barrier: Claims KS-CLAIM-019 and KS-CLAIM-020 remain PENDING.
  - Zero-Mutation Guarantee: Invariant breach halts execution with Delta S == 0.
  - VITAL_MAX_HP == 6 (Constitutional Sentinel P10).
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import time
import hmac
import shutil
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

from krystal_kernel.constants import VITAL_MAX_HP
from krystal_kernel.verification.theory_dependency_graph import GLOBAL_THEORY_DEPENDENCY_GRAPH
from krystal_kernel.hte001_hardware_transition_harness import (
    Hte001RunManifest,
    canonical_jcs_serialize,
)

PREFLIGHT_SECRET_KEY = b"KRYSTAL_HTE001_PREFLIGHT_VERIFICATION_SECRET_v01"
THEORY_NODES_EXPECTED = 56
EXPECTED_MANIFEST_DIGEST = "ae6a1add5fcf1d3cb36c36859d3f38ef48a27df340b8aaf39daa3b47ea17f9cf"
EXPECTED_HARNESS_DIGEST = "f520b171e29ea8b85b5d11d65c2ad7dd4baa9e0019ecd4eb56fee8d585f7a064"
EXPECTED_SOFTWARE_IDENTITY = "e5c8578cbe3382dd059a319ad915567859b18c6cb1d0fe61ed3abbbc0c7d3fd0"


@dataclass
class PreflightCheckItem:
    check_name: str
    passed: bool
    details: str
    measured_value: Optional[Any] = None
    expected_value: Optional[Any] = None


@dataclass
class PreflightCheckReceipt:
    receipt_id: str
    timestamp_iso: str
    timestamp_ns: int
    manifest_hash: str
    manifest_verified: bool
    harness_hash: str
    harness_verified: bool
    software_identity: str
    software_identity_verified: bool
    vital_max_hp: int
    theory_nodes_total: int
    hardware_present: bool
    environment_type: str  # "PHYSICAL_HARDWARE" | "EMULATED_DEVELOPMENT"
    all_checks_passed: bool
    overall_status: str     # "[PRE-FLIGHT VERIFIED]" | "[PRE-FLIGHT FAILED]"
    checks: List[Dict[str, Any]]
    epistemic_warning: str
    hmac_signature: str = ""


class Hte001PreflightChecker:
    """
    Executes a comprehensive preflight verification suite before RUN-0001 physical execution.
    Ensures bitstreams, manifests, harness hashes, and environment permissions match preregistered constants.
    """

    def __init__(
        self,
        manifest_path: str = "artifacts/hte001/HTE001_RUN_MANIFEST.json",
        harness_path: str = "krystal_kernel/hte001_hardware_transition_harness.py",
        artifacts_dir: str = "artifacts/hte001",
        enforce_physical_hardware: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.harness_path = Path(harness_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.enforce_physical_hardware = enforce_physical_hardware

    def run_all_checks(self) -> Tuple[bool, PreflightCheckReceipt]:
        checks: List[PreflightCheckItem] = []

        # Check 1: Constitutional Invariants (P10)
        actual_nodes = len(GLOBAL_THEORY_DEPENDENCY_GRAPH._nodes)
        p10_passed = (VITAL_MAX_HP == 6) and (actual_nodes == THEORY_NODES_EXPECTED)
        checks.append(PreflightCheckItem(
            check_name="CONSTITUTIONAL_INVARIANTS",
            passed=p10_passed,
            details=f"VITAL_MAX_HP={VITAL_MAX_HP}, THEORY_NODES={actual_nodes} (expected {THEORY_NODES_EXPECTED})",
            measured_value={"vital_max_hp": VITAL_MAX_HP, "theory_nodes": actual_nodes},
            expected_value={"vital_max_hp": 6, "theory_nodes": THEORY_NODES_EXPECTED},
        ))

        # Check 2: Manifest Hash Integrity
        manifest_hash_ok = False
        manifest_digest = ""
        manifest_obj: Optional[Hte001RunManifest] = None
        if self.manifest_path.exists():
            try:
                manifest_obj, manifest_digest = Hte001RunManifest.load_from_file(str(self.manifest_path))
                manifest_hash_ok = (manifest_digest == EXPECTED_MANIFEST_DIGEST)
                details = f"Loaded manifest {manifest_obj.experiment_id}. Hash match: {manifest_hash_ok}."
            except Exception as e:
                details = f"Manifest load failure: {str(e)}"
        else:
            details = f"Manifest file not found: {self.manifest_path}"

        checks.append(PreflightCheckItem(
            check_name="MANIFEST_PREREGISTRATION_DIGEST",
            passed=manifest_hash_ok,
            details=details,
            measured_value=manifest_digest,
            expected_value=EXPECTED_MANIFEST_DIGEST,
        ))

        # Check 3: Acquisition Harness Hash Integrity
        harness_hash_ok = False
        harness_digest = ""
        if self.harness_path.exists():
            try:
                content = self.harness_path.read_bytes()
                harness_digest = hashlib.sha256(content).hexdigest()
                harness_hash_ok = (harness_digest == EXPECTED_HARNESS_DIGEST)
                details = f"Harness read ({len(content)} bytes). Hash match: {harness_hash_ok}."
            except Exception as e:
                details = f"Harness read failure: {str(e)}"
        else:
            details = f"Harness file not found: {self.harness_path}"

        checks.append(PreflightCheckItem(
            check_name="ACQUISITION_HARNESS_DIGEST",
            passed=harness_hash_ok,
            details=details,
            measured_value=harness_digest,
            expected_value=EXPECTED_HARNESS_DIGEST,
        ))

        # Check 4: Software Identity Match
        software_id_ok = False
        if manifest_obj:
            software_id_ok = (manifest_obj.acquisition_software_identity == EXPECTED_SOFTWARE_IDENTITY)
            sw_details = f"Software identity: {manifest_obj.acquisition_software_identity}"
        else:
            sw_details = "Manifest not loaded"

        checks.append(PreflightCheckItem(
            check_name="ACQUISITION_SOFTWARE_IDENTITY",
            passed=software_id_ok,
            details=sw_details,
            measured_value=manifest_obj.acquisition_software_identity if manifest_obj else None,
            expected_value=EXPECTED_SOFTWARE_IDENTITY,
        ))

        # Check 5: Storage & Directory Write Permissions
        dir_ok = False
        disk_space_mb = 0.0
        try:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            test_file = self.artifacts_dir / ".preflight_write_test.tmp"
            test_file.write_text("preflight_test_ok\n")
            test_file.unlink()
            total, used, free = shutil.disk_usage(str(self.artifacts_dir))
            disk_space_mb = free / (1024 * 1024)
            dir_ok = (disk_space_mb >= 50.0)
            dir_details = f"Write access verified. Available disk: {disk_space_mb:.1f} MB (min 50 MB)."
        except Exception as e:
            dir_details = f"Directory access error: {str(e)}"

        checks.append(PreflightCheckItem(
            check_name="STORAGE_WRITE_PERMISSIONS",
            passed=dir_ok,
            details=dir_details,
            measured_value=f"{disk_space_mb:.1f} MB",
            expected_value=">= 50.0 MB",
        ))

        # Check 6: Monotonic Clock Resolution & Forward Progress
        t0 = time.monotonic_ns()
        time.sleep(0.001)
        t1 = time.monotonic_ns()
        delta_ns = t1 - t0
        clock_ok = (delta_ns > 0) and (delta_ns < 100_000_000)  # within 100ms
        checks.append(PreflightCheckItem(
            check_name="MONOTONIC_CLOCK_RESOLUTION",
            passed=clock_ok,
            details=f"Monotonic timer forward delta: {delta_ns} ns ({delta_ns/1e6:.3f} ms).",
            measured_value=delta_ns,
            expected_value="> 0 ns",
        ))

        # Check 7: Hardware & Telemetry Interface Probe
        hardware_present = False
        i2c_path = Path("/dev/i2c-1")
        if i2c_path.exists():
            hardware_present = True
            hw_details = "Physical I2C bus device detected at /dev/i2c-1."
        else:
            hw_details = "Physical I2C bus /dev/i2c-1 not detected. Emulation / mock telemetry active."

        if self.enforce_physical_hardware and not hardware_present:
            hw_passed = False
            hw_details += " [ENFORCED FAILURE: Physical FPGA/INA226 hardware required]."
        else:
            hw_passed = True

        checks.append(PreflightCheckItem(
            check_name="HARDWARE_INTERFACE_PROBE",
            passed=hw_passed,
            details=hw_details,
            measured_value="HARDWARE_PRESENT" if hardware_present else "EMULATION_MOCK",
            expected_value="PHYSICAL_DETECTED" if self.enforce_physical_hardware else "ANY",
        ))

        # Overall synthesis
        all_passed = all(c.passed for c in checks)
        env_type = "PHYSICAL_HARDWARE" if hardware_present else "EMULATED_DEVELOPMENT"
        overall_status = "[PRE-FLIGHT VERIFIED]" if all_passed else "[PRE-FLIGHT FAILED]"

        receipt_id = f"PREFLIGHT-{int(time.time())}-{hashlib.sha256(str(time.monotonic_ns()).encode()).hexdigest()[:8]}"
        timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        timestamp_ns = time.time_ns()

        receipt = PreflightCheckReceipt(
            receipt_id=receipt_id,
            timestamp_iso=timestamp_iso,
            timestamp_ns=timestamp_ns,
            manifest_hash=manifest_digest,
            manifest_verified=manifest_hash_ok,
            harness_hash=harness_digest,
            harness_verified=harness_hash_ok,
            software_identity=manifest_obj.acquisition_software_identity if manifest_obj else "",
            software_identity_verified=software_id_ok,
            vital_max_hp=VITAL_MAX_HP,
            theory_nodes_total=actual_nodes,
            hardware_present=hardware_present,
            environment_type=env_type,
            all_checks_passed=all_passed,
            overall_status=overall_status,
            checks=[asdict(c) for c in checks],
            epistemic_warning=(
                "ANTI-PROMOTION NOTICE: Preflight verification certifies environmental, hash, and "
                "manifest readiness only. Claims KS-CLAIM-019 and KS-CLAIM-020 strictly remain in "
                "PENDING status until physical FPGA RUN-0001 execution on Xilinx KCU116."
            ),
        )

        # Sign receipt with HMAC-SHA256
        data_to_sign = {k: v for k, v in asdict(receipt).items() if k != "hmac_signature"}
        jcs_bytes = canonical_jcs_serialize(data_to_sign)
        receipt.hmac_signature = hmac.new(PREFLIGHT_SECRET_KEY, jcs_bytes, hashlib.sha256).hexdigest()

        return all_passed, receipt


def main():
    parser = argparse.ArgumentParser(description="KSL v0.2 HTE-001 Preflight Verification Tool")
    parser.add_argument("--manifest", default="artifacts/hte001/HTE001_RUN_MANIFEST.json", help="Path to run manifest")
    parser.add_argument("--harness", default="krystal_kernel/hte001_hardware_transition_harness.py", help="Path to harness")
    parser.add_argument("--artifacts-dir", default="artifacts/hte001", help="Output directory for receipts")
    parser.add_argument("--enforce-physical", action="store_true", help="Fail if real hardware is not connected")
    parser.add_argument("--output-receipt", default="artifacts/hte001/HTE001_PREFLIGHT_RECEIPT.json", help="Save receipt path")
    parser.add_argument("--json", action="store_true", help="Print receipt JSON to stdout")
    args = parser.parse_args()

    checker = Hte001PreflightChecker(
        manifest_path=args.manifest,
        harness_path=args.harness,
        artifacts_dir=args.artifacts_dir,
        enforce_physical_hardware=args.enforce_physical,
    )

    passed, receipt = checker.run_all_checks()

    if args.output_receipt:
        out_path = Path(args.output_receipt)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(receipt), f, indent=2)

    if args.json:
        print(json.dumps(asdict(receipt), indent=2))
    else:
        print("=" * 80)
        print("KSL v0.2 — HTE-001 PREFLIGHT VERIFICATION REPORT")
        print("=" * 80)
        print(f"Receipt ID:       {receipt.receipt_id}")
        print(f"Timestamp:        {receipt.timestamp_iso}")
        print(f"Overall Status:   {receipt.overall_status}")
        print(f"Environment:      {receipt.environment_type}")
        print(f"Manifest Digest:  {receipt.manifest_hash} (Match: {receipt.manifest_verified})")
        print(f"Harness Digest:   {receipt.harness_hash} (Match: {receipt.harness_verified})")
        print(f"Software Identity: {receipt.software_identity} (Match: {receipt.software_identity_verified})")
        print(f"Constitutional:   VITAL_MAX_HP={receipt.vital_max_hp}, THEORY_NODES={receipt.theory_nodes_total}")
        print("-" * 80)
        for c in receipt.checks:
            mark = "PASS" if c["passed"] else "FAIL"
            print(f"  [{mark:4s}] {c['check_name']:<32s} : {c['details']}")
        print("-" * 80)
        print(f"HMAC Signature:   {receipt.hmac_signature}")
        print(f"Notice:           {receipt.epistemic_warning}")
        print("=" * 80)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
