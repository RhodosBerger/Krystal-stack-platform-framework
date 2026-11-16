"""Fact synthesis engine: perception over logs, fact extraction, induction, bridging,
module distance, allocation strategy and a narrator — all from local data with provenance.

Layers
  L1 perception   computation ledgers (artifacts/logs/**/computation_ledger.jsonl) -> utilization map
  L2 facts        bot report + profile sweep -> atomic facts, each with source file + key path
  L3 induction    per-profile facts -> general rules with support / counterexamples
  L4 bridge       two independent measurements of the same mechanism -> consistent or CONFLICT
  L5 allocation   measured latencies + compile costs -> derived device-allocation rule
  L6 distance     AST import graph -> pipeline.fetch_distance_between_modules(a, b)
  L7 narrator     sentences generated only from facts; every sentence cites fact IDs

Nothing here fetches data from the internet. Derived values are labeled DERIVED.
"""

from __future__ import annotations

import ast
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from krystal_kernel.constants import VITAL_MAX_HP

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
REPORTS = REPO_ROOT / "reports"


# ---------------------------------------------------------------------------
# L1 perception over logs
# ---------------------------------------------------------------------------

def perceive_ledgers(root: Path = REPO_ROOT / "artifacts" / "logs") -> Dict[str, Any]:
    files = sorted(root.rglob("computation_ledger.jsonl")) if root.exists() else []
    by_stratum: Dict[str, float] = defaultdict(float)
    by_operation: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "duration_ms": 0.0})
    patterns: Dict[str, int] = defaultdict(int)
    chains: List[Dict[str, Any]] = []
    total = 0.0
    for f in files:
        prev: Optional[str] = None
        links_ok, entries, bad_lines = True, 0, 0
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            entries += 1
            if prev is not None and e.get("prev_entry_sha256") != prev:
                links_ok = False
            prev = e.get("entry_sha256")
            d = float(e.get("duration_ms", 0.0))
            total += d
            by_stratum[e.get("stratum", "UNKNOWN")] += d
            op = by_operation[e.get("operation", "UNKNOWN")]
            op["count"] += 1
            op["duration_ms"] += d
            for p in e.get("detected_patterns", []):
                patterns[str(p)] += 1
        chains.append({"file": str(f.relative_to(REPO_ROOT)), "entries": entries, "prev_hash_links_consistent": links_ok,
                       "unparseable_lines": bad_lines})
    share = {s: round(v / total, 4) for s, v in by_stratum.items()} if total else {}
    return {
        "ledger_files": len(files),
        "total_logged_duration_ms": round(total, 3),
        "utilization_share_by_stratum": share,
        "top_operations": sorted(({"operation": k, **{kk: round(vv, 3) for kk, vv in v.items()}} for k, v in by_operation.items()),
                                 key=lambda x: -x["duration_ms"])[:8],
        "detected_patterns": dict(patterns),
        "chains": chains,
        "evidence_level": "LOG_REPORTED (durations as written by the logging code; hash links checked, entry hashes not recomputed)",
    }


# ---------------------------------------------------------------------------
# L2 facts
# ---------------------------------------------------------------------------

def _fact(fid: str, statement: str, source: str, path: str, evidence: str, value: Any) -> Dict[str, Any]:
    return {"fact_id": fid, "statement": statement, "source": source, "key_path": path, "evidence": evidence, "value": value}


def synthesize_facts(bot_path: Path = REPORTS / "consistency_bot_latest.json",
                     sweep_path: Path = REPORTS / "conv_fingerprint_profile_sweep.json") -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    if bot_path.exists():
        bot = json.loads(bot_path.read_text(encoding="utf-8"))
        src = str(bot_path.relative_to(REPO_ROOT))
        for c in bot["checks"]:
            facts.append(_fact(f"F_{c['check_id']}_verdict", f"{c['claim']}: {c['verdict']}", src,
                               f"checks[{c['check_id']}].verdict", c["evidence_class"], c["verdict"]))
            m = c["measured"]
            if c["check_id"] == "C03" and "devices" in m:
                for dev, d in m["devices"].items():
                    for k in ("median_ms", "p95_ms", "compile_ms", "max_rel_error"):
                        if d.get(k) is not None:
                            facts.append(_fact(f"F_C03_{dev}_{k}", f"{dev} conv pipeline {k} = {d[k]}", src,
                                               f"checks[C03].measured.devices.{dev}.{k}", "LOCAL_MEASURED", d[k]))
            if c["check_id"] == "C06" and m.get("threshold_k"):
                facts.append(_fact("F_C06_threshold_k", f"NumPy BLAS leaves sequential-FMA order at K = {m['threshold_k']}",
                                   src, "checks[C06].measured.threshold_k", "LOCAL_MEASURED", m["threshold_k"]))
            if c["check_id"] == "C08":
                facts.append(_fact("F_C08_mismatched_upstreams", f"nginx upstream ports not matching app port: {m.get('mismatched_upstreams')}",
                                   src, "checks[C08].measured.mismatched_upstreams", "STATIC_VALIDATION", m.get("mismatched_upstreams")))
    if sweep_path.exists():
        sw = json.loads(sweep_path.read_text(encoding="utf-8"))
        src = str(sweep_path.relative_to(REPO_ROOT))
        for r in sw["rows"]:
            if "error" in r:
                continue
            pid, k = r["profile"]["profile_id"], r["K"]
            facts.append(_fact(f"F_SW_{pid}_blas_fma", f"Profile {pid} (K={k}): BLAS == sequential FMA is {r['blas_equals_fma_natural']}",
                               src, f"rows[{pid}].blas_equals_fma_natural", "LOCAL_MEASURED",
                               {"K": k, "holds": r["blas_equals_fma_natural"]}))
            for dev, d in r["devices"].items():
                facts.append(_fact(f"F_SW_{pid}_{dev}", f"Profile {pid} (K={k}): {dev} kernel {d['kernel']} bit-identical to {d['bit_identical_to']}",
                                   src, f"rows[{pid}].devices.{dev}", "LOCAL_MEASURED",
                                   {"K": k, "device": dev, "kernel": d["kernel"], "identical_to": d["bit_identical_to"]}))
    return facts


# ---------------------------------------------------------------------------
# L3 induction + L4 bridge + L5 allocation
# ---------------------------------------------------------------------------

def induce_rules(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rules: List[Dict[str, Any]] = []
    by_id = {f["fact_id"]: f for f in facts}

    for dev in ("GPU", "CPU"):
        dev_facts = [f for f in facts if f["fact_id"].startswith("F_SW_") and f["fact_id"].endswith(f"_{dev}")]
        if not dev_facts:
            continue
        if dev == "GPU":
            holds = [f for f in dev_facts if "seq_natural_k_order_fma" in f["value"]["identical_to"]]
            stmt = "GPU conv kernel computes sequential FMA in natural k order"
        else:
            holds = [f for f in dev_facts if not any(v.startswith("seq_") for v in f["value"]["identical_to"])]
            stmt = "CPU conv kernel matches no sequential summation recipe (blocked accumulation)"
        counter = [f["fact_id"] for f in dev_facts if f not in holds]
        ks = sorted(f["value"]["K"] for f in dev_facts)
        rules.append({"rule_id": f"R_{dev}_RECIPE", "statement": stmt,
                      "verdict": "SUPPORTED_WITH_SCOPE" if not counter else "CONFLICT",
                      "support": len(holds), "counterexamples": counter, "scope_K": [ks[0], ks[-1]],
                      "derived_from": [f["fact_id"] for f in dev_facts]})

    blas = [f for f in facts if f["fact_id"].endswith("_blas_fma")]
    thr = by_id.get("F_C06_threshold_k")
    if blas and thr:
        t = thr["value"]
        consistent = [f for f in blas if f["value"]["holds"] == (f["value"]["K"] < t)]
        contradict = [f["fact_id"] for f in blas if f not in consistent]
        rules.append({"rule_id": "R_BRIDGE_BLAS_BOUNDARY",
                      "statement": f"BLAS == sequential FMA iff K < {t} (matmul bisection) agrees with conv profile sweep",
                      "verdict": "SUPPORTED_WITH_SCOPE" if not contradict else "CONFLICT",
                      "support": len(consistent), "counterexamples": contradict,
                      "note": "two independent measurements (conv sweep vs. matmul bisection) of one mechanism",
                      "derived_from": ["F_C06_threshold_k"] + [f["fact_id"] for f in blas]})

    g, c = by_id, "F_C03_"
    need = [f"{c}GPU_compile_ms", f"{c}CPU_compile_ms", f"{c}GPU_median_ms", f"{c}CPU_median_ms"]
    if all(n in g for n in need):
        dc = g[need[0]]["value"] - g[need[1]]["value"]
        dm = g[need[3]]["value"] - g[need[2]]["value"]
        if dm > 0:
            n_star = dc / dm
            stmt = f"Allocate the conv pipeline to GPU only when > {n_star:.0f} inferences run per compiled model; otherwise CPU"
        else:
            n_star = None
            stmt = "GPU median is not faster than CPU median here; CPU allocation dominates"
        rules.append({"rule_id": "R_ALLOC_BREAK_EVEN", "statement": stmt, "verdict": "DERIVED",
                      "break_even_inferences": None if n_star is None else round(n_star, 1),
                      "formula": "(compile_gpu - compile_cpu) / (median_cpu - median_gpu)", "derived_from": need})
    return rules


# ---------------------------------------------------------------------------
# L6 module distance
# ---------------------------------------------------------------------------

def _import_graph(package_dir: Path = PACKAGE_DIR) -> Dict[str, set]:
    pkg = "projects.trigonometric_hybrid_platform_framework"
    graph: Dict[str, set] = defaultdict(set)
    for f in sorted(package_dir.glob("*.py")):
        mod = f.stem
        graph[mod]
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            targets: List[str] = []
            if isinstance(n, ast.ImportFrom):
                if n.level == 1 and n.module:
                    targets = [n.module.split(".")[0]]
                elif n.module and n.module.startswith(pkg + "."):
                    targets = [n.module[len(pkg) + 1:].split(".")[0]]
                elif n.module and n.module.startswith("krystal_kernel"):
                    targets = [n.module]
            elif isinstance(n, ast.Import):
                targets = [a.name for a in n.names if a.name.startswith("krystal_kernel")]
            for t in targets:
                if t != mod:
                    graph[mod].add(t)
                    graph[t].add(mod)
    return graph


def fetch_distance_between_modules(
    a: str,
    b: str,
    package_dir: Path = PACKAGE_DIR,
    graph: Optional[Dict[str, set]] = None,
) -> Dict[str, Any]:
    """Shortest path length in the undirected import graph (hops) + Jaccard distance of neighbour sets."""
    g = graph if graph is not None else _import_graph(package_dir)
    if a not in g or b not in g:
        return {"a": a, "b": b, "hops": None, "path": None, "jaccard_distance": None, "note": "module not in graph"}
    prev = {a: None}
    q = deque([a])
    while q:
        cur = q.popleft()
        if cur == b:
            break
        for nxt in sorted(g[cur]):
            if nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    path = None
    if b in prev:
        path, cur = [], b
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
    na, nb = g[a] | {a}, g[b] | {b}
    return {"a": a, "b": b, "hops": None if path is None else len(path) - 1, "path": path,
            "jaccard_distance": round(1 - len(na & nb) / len(na | nb), 4)}


def module_distance_matrix(package_dir: Path = PACKAGE_DIR) -> Dict[str, Any]:
    g = _import_graph(package_dir)
    mods = sorted(p.stem for p in package_dir.glob("*.py") if p.stem != "__init__")
    return {a: {b: fetch_distance_between_modules(a, b, package_dir, graph=g)["hops"] for b in mods} for a in mods}


# ---------------------------------------------------------------------------
# L7 narrator
# ---------------------------------------------------------------------------

def narrate(facts: List[Dict[str, Any]], rules: List[Dict[str, Any]], perception: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for r in rules:
        cite = ", ".join(r["derived_from"][:3]) + (" …" if len(r["derived_from"]) > 3 else "")
        extra = f" Counterexamples: {r['counterexamples']}." if r.get("counterexamples") else ""
        lines.append(f"[{r['verdict']}] {r['statement']}.{extra} (sources: {cite})")
    conflicts = [f for f in facts if f["fact_id"].endswith("_verdict") and f["value"] in ("CONFLICT", "FALSIFIED")]
    for f in conflicts:
        lines.append(f"[{f['value']}] {f['statement']} (source: {f['fact_id']})")
    if perception.get("ledger_files"):
        top = max(perception["utilization_share_by_stratum"].items(), key=lambda kv: kv[1], default=None)
        if top:
            lines.append(f"[LOG_REPORTED] Of {perception['total_logged_duration_ms']} ms logged in {perception['ledger_files']} ledgers, "
                         f"stratum {top[0]} took the largest share ({top[1]:.0%}).")
    return lines


def run_fact_synthesis(write: bool = True) -> Dict[str, Any]:
    perception = perceive_ledgers()
    facts = synthesize_facts()
    rules = induce_rules(facts)
    result = {
        "vital_max_hp": VITAL_MAX_HP,
        "fact_count": len(facts),
        "rule_count": len(rules),
        "perception": perception,
        "rules": rules,
        "narrative": narrate(facts, rules, perception),
        "module_distance_hops": module_distance_matrix(),
        "facts": facts,
    }
    if write:
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / "fact_synthesis_latest.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


if __name__ == "__main__":
    out = run_fact_synthesis()
    print("FACTS", out["fact_count"], "RULES", out["rule_count"])
    for line in out["narrative"]:
        print(" -", line)
    print("PERCEPTION", json.dumps({k: out["perception"][k] for k in ("ledger_files", "total_logged_duration_ms", "utilization_share_by_stratum")}))
    print("CHAINS", out["perception"]["chains"])
    print("DIST consistency_bot->virtual_convolution_pipeline", fetch_distance_between_modules("consistency_bot", "virtual_convolution_pipeline"))
    print("DIST simulation_grid_3d->platform_engine", fetch_distance_between_modules("simulation_grid_3d", "platform_engine"))
