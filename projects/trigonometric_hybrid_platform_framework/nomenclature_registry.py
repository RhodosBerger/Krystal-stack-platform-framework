"""Nomenclature propedeutics node: every term of the framework must bind to real code.

A term is accepted only if
  - its binding "module:qualname" imports and resolves to a callable/class,
  - its test file exists,
and the registry reports public symbols of the package that have no term (undocumented).
This keeps vocabulary from drifting away from what the code actually does.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any, Dict, List

PKG = "projects.trigonometric_hybrid_platform_framework"
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]

TERMS: tuple = (
    ("Connection registry", "Typed nodes and edges of the sync architecture; only the gate may write state.",
     "adaptive_cross_genre_sync:validate_connection_graph", "tests/test_adaptive_cross_genre_sync.py"),
    ("Strictest-policy join", "When two genres touch one entity, the stricter sync mode wins; observers cannot write.",
     "adaptive_cross_genre_sync:join_sync_policy", "tests/test_adaptive_cross_genre_sync.py"),
    ("Adaptive horizon controller", "Rollback horizon = ceil((EWMA + 2 sigma_upper) / tick), immediate up, down after 12 ticks.",
     "adaptive_cross_genre_sync:AdaptiveHorizonController", "tests/test_adaptive_cross_genre_sync.py"),
    ("Rollback simulation", "Deterministic client/kernel replay with SHA-256 checksum on every tick.",
     "adaptive_cross_genre_sync:simulate_adaptive_rollback", "tests/test_adaptive_cross_genre_sync.py"),
    ("Card grid", "Design-manual card view of the sync architecture (Grid / List / Hex / Hex 3D).",
     "adaptive_sync_card_grid:render_adaptive_sync_card_grid_html", "tests/test_adaptive_sync_card_grid.py"),
    ("Simulation grid 3D", "Babylon.js replay of kernel state vs. client prediction on a 19-hex board.",
     "simulation_grid_3d:render_simulation_grid_3d_html", "tests/test_simulation_grid_3d_and_production.py"),
    ("Virtual convolution pipeline", "In-memory OpenVINO Conv-ReLU-Conv-ReLU checked against NumPy before timing.",
     "virtual_convolution_pipeline:run_virtual_convolution_pipeline", "tests/test_virtual_convolution_pipeline.py"),
    ("Summation-order fingerprint", "Compares a device against references that differ only in summation order and rounding.",
     "virtual_convolution_pipeline:diagnose_bit_identity", "tests/test_virtual_convolution_pipeline.py"),
    ("Contra modules", "Adjoint (transposed conv), CG inverse and ReLU partial inverse of the convolution.",
     "virtual_convolution_contra:run_contra_modules", "tests/test_virtual_convolution_contra.py"),
    ("Pattern-over-profiles sweep", "Runs the fingerprint over geometries/seeds; verdict per hypothesis, no averaging.",
     "pattern_profile_environment:run_pattern_sweep", "tests/test_pattern_profile_environment.py"),
    ("Consistency bot", "Executable claim checks published as JSON, Markdown and PNG.",
     "consistency_bot:run_bot", "tests/test_consistency_bot.py"),
    ("BLAS K threshold", "Smallest reduction length where NumPy BLAS leaves sequential-FMA order.",
     "consistency_bot:blas_sequential_fma_threshold", "tests/test_consistency_bot.py"),
    ("Fact synthesis", "Facts with provenance from reports, induced rules, bridges and a citing narrator.",
     "fact_synthesis_engine:run_fact_synthesis", "tests/test_fact_synthesis_engine.py"),
    ("Log perception", "Utilization share per stratum from computation ledgers, with hash-link check.",
     "fact_synthesis_engine:perceive_ledgers", "tests/test_fact_synthesis_engine.py"),
    ("Module distance", "Hops in the AST import graph plus Jaccard distance of neighbours.",
     "fact_synthesis_engine:fetch_distance_between_modules", "tests/test_fact_synthesis_engine.py"),
    ("Fact-driven allocator", "Picks the device with minimal compile + n * median cost under a p95 budget.",
     "custom_allocator:FactDrivenAllocator", "tests/test_custom_allocator.py"),
    ("Ontology integrity", "Unique fact IDs, existing sources, no rule citing a missing fact.",
     "custom_allocator:check_ontology_integrity", "tests/test_custom_allocator.py"),
    ("Neuro-symbolic bridge", "LLM proposes cited sentences; the fact base keeps only GROUNDED ones.",
     "neurosymbolic_bridge:verify_answer", "tests/test_neurosymbolic_bridge.py"),
)


def _public_symbols() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for f in sorted(PACKAGE_DIR.glob("*.py")):
        if f.stem.startswith("_"):
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        out[f.stem] = [n.name for n in tree.body
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_")]
    return out


def validate_nomenclature(terms: tuple = TERMS) -> Dict[str, Any]:
    rows, bound = [], set()
    for term, definition, binding, test in terms:
        mod_name, _, qual = binding.partition(":")
        status = "RESOLVED"
        try:
            obj = importlib.import_module(f"{PKG}.{mod_name}")
            for part in qual.split("."):
                obj = getattr(obj, part)
            if not callable(obj):
                status = "NOT_CALLABLE"
        except (ImportError, AttributeError) as exc:
            status = f"UNRESOLVED: {type(exc).__name__}"
        if status == "RESOLVED" and not (REPO_ROOT / test).exists():
            status = "NO_TEST_FILE"
        bound.add((mod_name, qual.split(".")[0]))
        rows.append({"term": term, "definition": definition, "binding": binding, "test": test, "status": status})
    undocumented = {m: [s for s in syms if (m, s) not in bound] for m, syms in _public_symbols().items()}
    undocumented = {m: s for m, s in undocumented.items() if s}
    unresolved = [r["term"] for r in rows if r["status"] != "RESOLVED"]
    return {"verdict": "VERIFIED_WITH_SCOPE" if not unresolved else "FALSIFIED",
            "terms": rows, "unresolved_terms": unresolved, "undocumented_public_symbols": undocumented}


def export_glossary(path: Path = REPO_ROOT / "reports" / "nomenclature_propedeutics.md") -> Dict[str, Any]:
    res = validate_nomenclature()
    lines = ["# Nomenclature propedeutics", "", f"Verdict: **{res['verdict']}**", "",
             "| Term | Meaning | Binds to | Test | Status |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {r['term']} | {r['definition']} | `{r['binding']}` | `{r['test']}` | {r['status']} |" for r in res["terms"]]
    n_undoc = sum(len(v) for v in res["undocumented_public_symbols"].values())
    lines += ["", f"Public symbols without a term: {n_undoc}", ""]
    lines += [f"- `{m}`: {', '.join(s)}" for m, s in res["undocumented_public_symbols"].items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    res["path"] = str(path)
    return res


if __name__ == "__main__":
    r = export_glossary()
    print(r["verdict"], "terms:", len(r["terms"]), "unresolved:", r["unresolved_terms"],
          "undocumented:", sum(len(v) for v in r["undocumented_public_symbols"].values()), r["path"])
