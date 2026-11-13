"""CLI Runner & Telemetry Verifier for `projects/sovereign_openvino_companion`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from projects.sovereign_openvino_companion.companion_engine import SovereignOpenVinoCompanionProject


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sovereign-openvino-companion",
        description="Run OpenVINO + oneAPI companion cycles with cryptographic computation logging",
    )
    parser.add_argument("--validate-env", action="store_true", help="Verify OpenVINO devices & invariants")
    parser.add_argument("--run-session", action="store_true", help="Run a multi-phase 120Hz VBlank companion session")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames for --run-session")
    parser.add_argument("--supervise-coordinates", action="store_true", help="Run supervisor co-worker microservice coordinate maintenance")
    parser.add_argument("--benchmark", type=int, default=0, help="Run N warm iterations across CPU/GPU crossover")
    parser.add_argument("--verify-logs", action="store_true", help="Verify SHA-256 chain integrity of the ledger")
    parser.add_argument("--log-dir", type=str, default="", help="Optional custom directory for JSONL logs")
    parser.add_argument("--reset-ledger", action="store_true", help="Reset JSONL ledger before session")
    parser.add_argument("--json", action="store_true", help="Always output compact JSON")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    log_path = Path(args.log_dir) if args.log_dir else None
    project = SovereignOpenVinoCompanionProject(log_dir=log_path, reset_ledger=args.reset_ledger)

    out: Dict[str, Any] = {}
    if args.validate_env:
        out["environment"] = project.validate_environment()
    if args.run_session:
        out["session"] = project.run_companion_session(num_frames=args.frames)
    if args.supervise_coordinates:
        out["supervisor_community"] = project.run_supervised_community_maintenance(cycles=args.frames)
    if args.benchmark > 0:
        out["benchmark"] = project.run_statistical_benchmark(iterations=args.benchmark)
    if args.verify_logs:
        ok, audit = project.logger.verify_chain_integrity()
        out["log_verification"] = audit
        if not ok:
            print(json.dumps(out, indent=2))
            return 1

    if not out:
        out["environment"] = project.validate_environment()

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
