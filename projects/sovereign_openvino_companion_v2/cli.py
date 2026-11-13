"""CLI Runner & Paired A/B Verifier for `projects/sovereign_openvino_companion_v2`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from projects.sovereign_openvino_companion_v2.smart_tuner import SovereignOpenVinoCompanionV2


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="sovereign-openvino-companion-v2")
    parser.add_argument("--run-session", action="store_true")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--ab-benchmark", type=int, default=0)
    parser.add_argument("--candidate-nodes", action="store_true")
    parser.add_argument("--log-dir", type=str, default="")
    parser.add_argument("--reset-ledger", action="store_true")
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir) if args.log_dir else None
    v2 = SovereignOpenVinoCompanionV2(log_dir=log_dir, reset_ledger=args.reset_ledger)
    out: Dict[str, Any] = {}
    if args.run_session or (args.ab_benchmark <= 0 and not args.candidate_nodes):
        out["session"] = v2.run_smart_session(num_frames=args.frames)
    if args.ab_benchmark > 0:
        out["ab_benchmark"] = v2.run_v1_vs_v2_ab_benchmark(iterations=args.ab_benchmark)
    if args.candidate_nodes:
        out["candidate_nodes"] = v2.prototype_candidate_runtime_nodes()
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
