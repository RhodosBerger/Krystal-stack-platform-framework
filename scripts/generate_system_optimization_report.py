#!/usr/bin/env python3
"""
CLI Command Script:
Generates detailed markdown output regarding:
  1. Memory bus optimization (Optimalizácia pamäťovej zbernice)
  2. Impacts of reducing precision to float16 (Dopady znižovania presnosti na float16)
  3. Pre-staging prediction in NPU (Predikcia pre-stagingu v NPU)
  4. Cross-platform thermal management proposals (Návrhy na riadenie teploty naprieč platformami)

Usage:
    python scripts/generate_system_optimization_report.py
    python scripts/generate_system_optimization_report.py --save docs/research/SYSTEM_REPORT.md

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
"""

import sys
import os

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from krystal_stack_nextgen import GLOBAL_THERMAL_BUS_GOVERNOR, VITAL_MAX_HP


def main():
    assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP must remain 6"
    report = GLOBAL_THERMAL_BUS_GOVERNOR.generate_comprehensive_markdown_report()

    # If --save argument provided, also persist report to disk
    if len(sys.argv) > 2 and sys.argv[1] == "--save":
        out_path = sys.argv[2]
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[REPORT SAVED] -> {out_path}", file=sys.stderr)

    print(report)


if __name__ == "__main__":
    main()
