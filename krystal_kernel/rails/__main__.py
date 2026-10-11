"""CLI: ``python -m krystal_kernel.rails {list|validate|run} [preset-or-path]``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .rail import RailValidationError, check_rail, list_presets, load_preset, run_rail, summarize_result


def _load(target: str):
    path = Path(target)
    if path.suffix == ".json" and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return load_preset(target)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m krystal_kernel.rails")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list presets")
    for name in ("validate", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("target", help="preset name or path to a .rail.json file")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for p in list_presets():
            print(f"{p['rail']:<34} {p['description'][:90]}")
        return 0
    doc = _load(args.target)
    if args.cmd == "validate":
        _, errors = check_rail(doc)
        print("valid" if not errors else "\n".join(errors))
        return 0 if not errors else 1
    try:
        result = run_rail(doc)
    except RailValidationError as exc:
        print("\n".join(exc.errors), file=sys.stderr)
        return 1
    view = summarize_result(result)
    shown = {k: view["values"][k] for k in (result["outputs"] or view["values"])}
    print(json.dumps({"ok": view["ok"], "error": view["error"], "outputs": shown,
                      "trace": view["trace"], "total_wall_ms": view["total_wall_ms"]},
                     indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
