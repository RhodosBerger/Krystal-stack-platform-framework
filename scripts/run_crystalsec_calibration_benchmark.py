"""
Regenerates artifacts/CRYSTALSEC_V3_CALIBRATION_AND_REGRET_BENCHMARK.json (Phase 5.1, evidence revision 5.2).

Usage (from repository root):
    python scripts/run_crystalsec_calibration_benchmark.py [--iterations 25] [--size 128] [--no-gpu]

The report separates evidence classes explicitly:
  - empirical_distributions : HOST_CPU_REFERENCE (T0) and HOST_CPU_EMULATED (T1/T2) wall-clock
  - hardware_evidence       : real Vulkan SDF raymarch dispatch (HARDWARE_MEASURED) if available
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from krystal_kernel.verification.crystalsec_v3_placement_calibration import (  # noqa: E402
    PlacementCalibrationController,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--no-gpu", action="store_true", help="skip the Vulkan hardware evidence probe")
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "artifacts", "CRYSTALSEC_V3_CALIBRATION_AND_REGRET_BENCHMARK.json"),
    )
    args = parser.parse_args()

    ctrl = PlacementCalibrationController()
    report = ctrl.run_calibration_suite(
        iterations_per_class=args.iterations,
        workload_size=args.size,
        repeats_per_measurement=args.repeats,
        include_hardware_evidence=not args.no_gpu,
    )
    report["host"] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    rm = report["regret_metrics"]
    cm = report["calibration_metrics"]
    print(f"wrote {args.out}")
    print(f"placement_quality={rm['placement_quality']} selected={rm['selected_tier_distribution']} "
          f"best={rm['best_tier_distribution']}")
    print(f"calibration_status={cm['calibration_status']} mace={cm['mace_composite']}")
    hw = report.get("hardware_evidence")
    if hw:
        print(f"hardware_evidence={hw['status']} device={hw.get('device_name')}")
        for row in hw.get("rows", []):
            print(f"  {row['resolution']:>8}  T2={row['t2_median_ms']:.3f}ms ({row['t2_evidence']})  "
                  f"T0={row['t0_median_ms']:.3f}ms  ratio={row['t0_over_t2_ratio']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
