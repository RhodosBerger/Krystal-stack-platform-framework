#!/usr/bin/env python3
"""
GAMESA Stack Verification - Ubuntu TPU/OpenVINO Checker

Verifies Intel GPU, OpenVINO, Mesa driver, and kernel prerequisites.
"""

import subprocess
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    fix_hint: Optional[str] = None

def run_cmd(cmd: List[str], timeout: int = 10) -> Tuple[int, str, str]:
    """Run command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -2, "", "Command timed out"

def check_i915_module() -> CheckResult:
    """Check if i915 kernel module is loaded."""
    rc, out, _ = run_cmd(["lsmod"])
    if "i915" in out:
        return CheckResult("i915 module", True, "Loaded")
    return CheckResult("i915 module", False, "Not loaded", "sudo modprobe i915")

def check_dri_access() -> CheckResult:
    """Check /dev/dri access."""
    dri = Path("/dev/dri")
    if not dri.exists():
        return CheckResult("/dev/dri", False, "Missing", "Check kernel config")
    cards = list(dri.glob("card*"))
    renders = list(dri.glob("renderD*"))
    if cards and renders:
        readable = os.access(renders[0], os.R_OK | os.W_OK)
        if readable:
            return CheckResult("/dev/dri", True, f"{len(cards)} card(s), {len(renders)} render node(s)")
        return CheckResult("/dev/dri", False, "Permission denied", "Add user to 'render' group")
    return CheckResult("/dev/dri", False, "No devices", "Check i915 driver")

def check_cpufreq() -> CheckResult:
    """Check cpufreq sysfs access."""
    cpufreq = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    if cpufreq.exists():
        governor = cpufreq / "scaling_governor"
        if governor.exists():
            current = governor.read_text().strip()
            return CheckResult("cpufreq", True, f"Governor: {current}")
    return CheckResult("cpufreq", False, "Not available", "Check kernel config or container perms")

def check_clinfo() -> CheckResult:
    """Check OpenCL via clinfo."""
    rc, out, err = run_cmd(["clinfo", "-l"])
    if rc != 0:
        return CheckResult("clinfo", False, err.strip(), "apt install clinfo intel-opencl-icd")
    platforms = [l for l in out.split("\n") if "Platform" in l or "Device" in l]
    if platforms:
        return CheckResult("clinfo", True, f"{len(platforms)} platform/device entries")
    return CheckResult("clinfo", False, "No platforms", "Install intel-opencl-icd")

def check_openvino() -> CheckResult:
    """Check OpenVINO runtime."""
    try:
        from openvino import Core
        core = Core()
        devices = core.available_devices
        return CheckResult("OpenVINO", True, f"Devices: {', '.join(devices)}")
    except ImportError:
        return CheckResult("OpenVINO", False, "Not installed", "pip install openvino")
    except Exception as e:
        return CheckResult("OpenVINO", False, str(e), "Check OpenVINO installation")

def check_level_zero() -> CheckResult:
    """Check Level Zero / SYCL."""
    rc, out, _ = run_cmd(["sycl-ls"])
    if rc == 0 and out.strip():
        lines = [l for l in out.split("\n") if l.strip()]
        return CheckResult("Level Zero/SYCL", True, f"{len(lines)} device(s)")
    return CheckResult("Level Zero/SYCL", False, "Not available", "Install level-zero runtime")

def check_thermal_headroom() -> CheckResult:
    """Check thermal sensors accessible."""
    thermal = Path("/sys/class/thermal")
    if thermal.exists():
        zones = list(thermal.glob("thermal_zone*"))
        if zones:
            return CheckResult("Thermal sensors", True, f"{len(zones)} zone(s)")
    return CheckResult("Thermal sensors", False, "Not available", "Check ACPI thermal support")

def check_mesa_gamesa() -> CheckResult:
    """Check custom Mesa-GAMESA driver."""
    mesa_path = Path("/opt/gamesa/lib/dri")
    if mesa_path.exists():
        return CheckResult("Mesa-GAMESA", True, "Custom driver installed")
    # Check standard mesa
    rc, out, _ = run_cmd(["glxinfo", "-B"])
    if rc == 0 and "Mesa" in out:
        version = [l for l in out.split("\n") if "OpenGL version" in l]
        return CheckResult("Mesa-GAMESA", True, version[0] if version else "Mesa present")
    return CheckResult("Mesa-GAMESA", False, "Not found", "Build mesa-gamesa-drivers/")

def run_all_checks() -> List[CheckResult]:
    """Run all verification checks."""
    checks = [
        check_i915_module,
        check_dri_access,
        check_cpufreq,
        check_thermal_headroom,
        check_clinfo,
        check_openvino,
        check_level_zero,
        check_mesa_gamesa,
    ]
    return [check() for check in checks]

def print_report(results: List[CheckResult]) -> int:
    """Print verification report. Returns exit code."""
    print("\n" + "=" * 60)
    print("GAMESA Stack Verification Report")
    print("=" * 60 + "\n")

    passed = 0
    failed = 0

    for r in results:
        status = "\033[92mPASS\033[0m" if r.passed else "\033[91mFAIL\033[0m"
        print(f"[{status}] {r.name}: {r.message}")
        if not r.passed and r.fix_hint:
            print(f"       Fix: {r.fix_hint}")
        if r.passed:
            passed += 1
        else:
            failed += 1

    print("\n" + "-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("-" * 60 + "\n")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    results = run_all_checks()
    sys.exit(print_report(results))
