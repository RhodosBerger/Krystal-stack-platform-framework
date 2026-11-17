#!/usr/bin/env python3
"""CLI runner executing the 8-domain Stock vs. Krystal Platform Benchmark Suite
and writing JSON, CSV, and Markdown difference reports.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.trigonometric_hybrid_platform_framework.stock_vs_krystal_platform_benchmark_suite import (
    run_stock_vs_krystal_benchmark_suite,
)


def main() -> int:
    res = run_stock_vs_krystal_benchmark_suite()
    print(
        f"[OK] Stock vs Krystal Suite | VITAL_MAX_HP={res['vital_max_hp']} | "
        f"Krystal wins={res['summary']['krystal_wins']}/{res['summary']['total_benchmarks']} | "
        f"GeomMean={res['summary']['geometric_mean_ratio_x']}x"
    )
    for b in res["benchmarks"]:
        print(
            f"  - {b['benchmark_id']}: Stock={b['stock_stats']['median']} {b['unit']} "
            f"vs Krystal={b['krystal_stats']['median']} {b['unit']} "
            f"(diff={b['difference_krystal_minus_stock']:+} {b['unit']}, {b['speedup_or_gain_x']}x, winner={b['winner']})"
        )
    print(f"JSON: {res['json_path']}")
    print(f"CSV:  {res['csv_path']}")
    print(f"MD:   {res['md_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
