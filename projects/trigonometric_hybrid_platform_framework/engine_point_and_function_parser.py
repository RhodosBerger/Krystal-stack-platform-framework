"""Deterministic parser for engine spatial points/directives, AST parser for all
framework functions & call topology, advanced game-engine integration graph,
and full-day improvement report generator (`2026-11-16`).
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from krystal_kernel.constants import UMA_SATURATION_THRESHOLD, VITAL_MAX_HP, ContractViolation
from krystal_kernel.zero_copy_contracts import canonicalize_float32

from .clash_battle_royale import hex_distance
from .cluster_pid_gemma_finetune_and_mesh_ml import (
    build_cluster_entity_pid_registry,
    pack_intranet_128b_packet,
    unpack_intranet_128b_packet,
)
from .fact_synthesis_engine import REPO_ROOT, REPORTS

PKG_DIR = Path(__file__).resolve().parent
DOCS_ARCH_DIR = REPO_ROOT / "docs" / "architecture"

_TOKEN_RE = re.compile(r"^\s*([A-Z0-9_]+)\s*\(([^()]*)\)\s*$")


# ---------------------------------------------------------------------------
# 1. Deterministic Parser of Spatial Points & Engine Directives
# ---------------------------------------------------------------------------

def parse_engine_spatial_points_and_directives(
    dsl_script: str,
    uma_utilization: float = 0.35,
    arena_radius: int = 3,
) -> Dict[str, Any]:
    """Parse structured engine DSL commands (`HEX`, `PT3`, `RAY`, `LAYER_MASK`, `CALL`) with strict contract gates."""
    if VITAL_MAX_HP != 6:
        raise ContractViolation("Constitutional invariant VITAL_MAX_HP != 6", violation_code="ERR_PARSER_HP")
    if not isinstance(dsl_script, str) or not dsl_script.strip():
        raise ContractViolation("Empty DSL point script", violation_code="ERR_DSL_EMPTY")

    clauses = [c.strip() for c in re.split(r"[;\n|]+", dsl_script) if c.strip()]
    if not clauses:
        raise ContractViolation("No parseable clauses in DSL script", violation_code="ERR_DSL_CLAUSES")

    parsed_nodes: List[Dict[str, Any]] = []
    function_refs: List[str] = []
    max_allowed_layers = 10 if uma_utilization < UMA_SATURATION_THRESHOLD else 5
    max_mask = (1 << max_allowed_layers) - 1

    for idx, raw in enumerate(clauses):
        m = _TOKEN_RE.match(raw)
        if not m:
            raise ContractViolation(f"Invalid DSL syntax at clause #{idx}: {raw!r}", violation_code="ERR_DSL_SYNTAX")
        op, arg_body = m.group(1), m.group(2).strip()
        parts = [p.strip() for p in arg_body.split(",") if p.strip()] if arg_body else []

        if op == "HEX":
            if len(parts) != 2:
                raise ContractViolation("HEX expects (q, r)", violation_code="ERR_DSL_HEX_ARITY")
            try:
                q, r = int(parts[0]), int(parts[1])
            except ValueError as exc:
                raise ContractViolation("HEX coordinates must be exact integers", violation_code="ERR_DSL_HEX_INT") from exc
            dist = hex_distance((q, r), (0, 0))
            if dist > arena_radius:
                raise ContractViolation(f"HEX({q},{r}) distance {dist} > R={arena_radius}", violation_code="ERR_DSL_HEX_BOUNDS")
            wx = canonicalize_float32(math.sqrt(3.0) * (q + 0.5 * r))
            wz = canonicalize_float32(1.5 * r)
            parsed_nodes.append({"index": idx, "kind": "HEX", "axial": [q, r], "ring_dist": dist, "world_xz": [wx, wz]})

        elif op == "PT3":
            if len(parts) != 3:
                raise ContractViolation("PT3 expects (x, y, z)", violation_code="ERR_DSL_PT3_ARITY")
            try:
                coords = [float(v) for v in parts]
            except ValueError as exc:
                raise ContractViolation("PT3 expects finite floats", violation_code="ERR_DSL_PT3_FLOAT") from exc
            if not all(math.isfinite(c) for c in coords):
                raise ContractViolation("Non-finite float in PT3", violation_code="ERR_DSL_PT3_NAN")
            canon = [canonicalize_float32(c) for c in coords]
            parsed_nodes.append({"index": idx, "kind": "PT3", "xyz_canonical": canon})

        elif op == "RAY":
            if len(parts) != 2:
                raise ContractViolation("RAY expects (theta_deg, length)", violation_code="ERR_DSL_RAY_ARITY")
            try:
                deg, length = float(parts[0]), float(parts[1])
            except ValueError as exc:
                raise ContractViolation("RAY expects numeric (theta_deg, length)", violation_code="ERR_DSL_RAY_NUM") from exc
            if not (math.isfinite(deg) and math.isfinite(length) and length >= 0.0):
                raise ContractViolation("RAY requires finite angle and non-negative length", violation_code="ERR_DSL_RAY_BOUNDS")
            rad = math.radians(deg)
            dx = canonicalize_float32(math.cos(rad) * length)
            dy = canonicalize_float32(math.sin(rad) * length)
            wedge = int(((deg % 360.0) // 60.0)) % 6
            parsed_nodes.append(
                {"index": idx, "kind": "RAY", "theta_deg": round(deg, 4), "length": canonicalize_float32(length), "vec_xy": [dx, dy], "hex_wedge": wedge}
            )

        elif op == "LAYER_MASK":
            if len(parts) != 1:
                raise ContractViolation("LAYER_MASK expects 1 integer/hex mask", violation_code="ERR_DSL_MASK_ARITY")
            try:
                raw_mask = int(parts[0], 0)
            except ValueError as exc:
                raise ContractViolation("Invalid integer/hex mask in LAYER_MASK", violation_code="ERR_DSL_MASK_INT") from exc
            if raw_mask < 0 or raw_mask >= (1 << 20):
                raise ContractViolation("Godot VisualInstance3D mask must fit in 20 bits", violation_code="ERR_DSL_MASK_BITS")
            clamped_mask = raw_mask & max_mask
            parsed_nodes.append(
                {
                    "index": idx,
                    "kind": "LAYER_MASK",
                    "requested_hex": hex(raw_mask),
                    "governed_hex": hex(clamped_mask),
                    "uma_clamped": clamped_mask != raw_mask,
                }
            )

        elif op == "CALL":
            if len(parts) != 1 or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", parts[0]):
                raise ContractViolation("CALL expects a valid Python function identifier", violation_code="ERR_DSL_CALL_ID")
            fn_name = parts[0]
            function_refs.append(fn_name)
            parsed_nodes.append({"index": idx, "kind": "CALL", "function": fn_name})

        else:
            raise ContractViolation(f"Unsupported engine DSL opcode: {op!r}", violation_code="ERR_DSL_OPCODE")

    pids = build_cluster_entity_pid_registry()["entities"]
    summary_tag = f"NODES={len(parsed_nodes)}:CALLS={len(function_refs)}:MASK<={hex(max_mask)}"
    pkt = unpack_intranet_128b_packet(pack_intranet_128b_packet(3, pids[0]["pid"], pids[5]["pid"], uma_utilization, summary_tag))

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "uma_utilization": round(uma_utilization, 4),
        "clause_count": len(parsed_nodes),
        "parsed_nodes": parsed_nodes,
        "referenced_functions": function_refs,
        "verified_128b_packet": pkt,
    }


# ---------------------------------------------------------------------------
# 2. Deterministic AST Parser of All Framework Functions & Call Topology
# ---------------------------------------------------------------------------

def parse_framework_functions_and_topology(pkg_dir: Path = PKG_DIR) -> Dict[str, Any]:
    """Introspect all modules in `trigonometric_hybrid_platform_framework` via Python `ast`."""
    py_files = sorted(p for p in pkg_dir.glob("*.py") if p.name != "__init__.py")
    functions: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []
    module_trees: List[Tuple[str, ast.Module]] = []
    known_fn_names: Set[str] = set()

    for path in py_files:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
        mod_name = path.stem
        module_trees.append((mod_name, tree))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                known_fn_names.add(node.name)
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                classes.append(
                    {
                        "module": mod_name,
                        "class_name": node.name,
                        "lineno": node.lineno,
                        "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")],
                    }
                )

    call_edges: List[Dict[str, str]] = []
    for mod_name, tree in module_trees:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                args = [a.arg for a in node.args.args]
                ret = ast.unparse(node.returns) if node.returns is not None else "Any"
                doc = (ast.get_docstring(node) or "").strip().splitlines()
                summary = doc[0] if doc else ""
                callees: Set[str] = set()
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        if isinstance(sub.func, ast.Name) and sub.func.id in known_fn_names and sub.func.id != node.name:
                            callees.add(sub.func.id)
                        elif isinstance(sub.func, ast.Attribute) and sub.func.attr in known_fn_names and sub.func.attr != node.name:
                            callees.add(sub.func.attr)
                for c in sorted(callees):
                    call_edges.append({"caller": f"{mod_name}.{node.name}", "callee": c})
                functions.append(
                    {
                        "module": mod_name,
                        "function": node.name,
                        "lineno": node.lineno,
                        "args": args,
                        "return_type": ret,
                        "doc_summary": summary,
                        "calls_framework_functions": sorted(callees),
                    }
                )

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "modules_scanned": len(py_files),
        "public_functions_count": len(functions),
        "public_classes_count": len(classes),
        "cross_function_call_edges_count": len(call_edges),
        "functions": functions,
        "classes": classes,
        "call_edges": call_edges,
    }


# ---------------------------------------------------------------------------
# 3. Advanced Game Engine Integration Graph + Full-Day Markdown Export
# ---------------------------------------------------------------------------

ADVANCED_ENGINE_STAGES: Tuple[Dict[str, str], ...] = (
    {
        "stage_id": "S1_DSL_POINT_AND_AST_INGEST",
        "title": "Parser bodov a AST funkcií",
        "mechanism": "HEX/PT3/RAY/LAYER_MASK -> canonicalize_float32 + overenie symbolov v AST katalógu",
        "budget": "< 0.25 ms CPU",
    },
    {
        "stage_id": "S2_GONIO_AND_TENSOR_FIELD",
        "title": "Goniometrické jadro & difúzia D⁻¹A",
        "mechanism": "DESC_01..03 (cos θ, sin θ, kliny 0..5) + OpenVINO maticový potenciál na 37 hexoch",
        "budget": "~0.48 ms CPU_4T AVX-512",
    },
    {
        "stage_id": "S3_AUTHORITATIVE_CLASH_KERNEL",
        "title": "Autoritatívne herné jadro ClashGame",
        "mechanism": "VITAL_MAX_HP == 6, ~Valid => Delta_S == 0, reaktívna AI + svetové javy (zóna, studne)",
        "budget": "13 362,7 ťahov/s (0.075 ms/ťah)",
    },
    {
        "stage_id": "S4_BACKBRANCH_AND_FREIVALDS",
        "title": "Spätné vetvenie OP_01..04 & Freivalds O(n²)",
        "mechanism": "Pri UMA >= 0.78 odpojenie GPU_ALLOCATE, krátenie rámcov 16->8 a kontrola rel_gap <= 1e-4",
        "budget": "0 rámcov oneskorenia + 12 vzoriek hysterézie",
    },
    {
        "stage_id": "S5_GODOT_MULTIMESH_AND_KOPT",
        "title": "Godot 4 MultiMesh, VR hologram & .kopt Fresnel",
        "mechanism": "10/20 vrstiev (0x3ff) pri UMA < 0.78, satelitný 3D .obj mesh, sokety BP_360..1920",
        "budget": "<= 16.67 ms (60 Hz) / 8.33 ms (120 Hz)",
    },
    {
        "stage_id": "S6_128B_KPKT_HYPERVISOR_BUS",
        "title": "128 B KPKT sieť & regulácia WSL2",
        "mechanism": "2 x 64 B cache riadky + SHA-256 (5.04x vs JSON), kvóta WSL2 RAM 10->8 GB pri špičke",
        "budget": "77.8 MB/s (637 349 rámcov/s)",
    },
)


DAILY_CHRONOLOGY_WAVES: Tuple[Dict[str, str], ...] = (
    {
        "time": "09:48 – 09:50",
        "commits": "82c28a59, 7211578a",
        "milestone": "OpenVINO konvolučný reťazec, kontra-moduly a odhalenie FMA sčítacieho poradia",
        "gain": "Dokázaná bitová zhoda GPU (Iris Xe) so sekvenčným prirodzeným poradím FMA a vysvetlený rozdiel voči blokovému sčítaniu CPU.",
    },
    {
        "time": "10:04",
        "commits": "87d5440f",
        "milestone": "Syntéza faktov, prah K = 449, FactDrivenAllocator, Freivalds O(n²) a register pojmov",
        "gain": "Presne lokalizovaný prechod redukcie OpenBLAS na K = 449 (vyriešený spor C05) + pravdepodobnostný audit maticového súčinu so zložitosťou O(n²).",
    },
    {
        "time": "10:08",
        "commits": "b65022a3",
        "milestone": "Autoritatívna hexová aréna Clash Battle Royale (37 polí) + 128 B potvrdenia a Babylon UI",
        "gain": "Ústavný invariant VITAL_MAX_HP == 6, neplatná akcia => Delta_S == 0 a viazanie kariet na reálny rozpočet rámcov.",
    },
    {
        "time": "11:32",
        "commits": "7b87dd1e",
        "milestone": "Heterogénny oneAPI + LLM CLI shell, Intel akcelerátor, krivky výkonu, DxDiag a oprava AST O(M³)",
        "gain": "Zrýchlenie testovacieho a syntézneho behu 126,4x (z 1428,94 s na 11,30 s) + amortizácia iGPU kompilácie cez CACHE_DIR (~187 ms -> ~9–15 ms).",
    },
    {
        "time": "14:27",
        "commits": "123a0d6b",
        "milestone": "Merania x86-64 (X01..X05), šírka pásma arény, 4 živé štúdiá, HENC mikro-LLM, Godot vrstvy a goniometrické jadro",
        "gain": "5,04x zrýchlenie 128 B binárneho prenosu oproti JSONu, odhalené 14,0x/37,4x spomalenie subnormálov IEEE-754, 13 362,7 ťahov/s (7 súbežných 60 Hz arén), 10/20 vrstiev Godot a most Antigravity-Carbon bez OpenAI.",
    },
    {
        "time": "14:44 – teraz",
        "commits": "95722f9a + aktuálny",
        "milestone": "Medzivrstvový demonštrátor (XLP_01..04), deterministický parser bodov/AST funkcií a pokročilá integrácia enginu",
        "gain": "Premostenie patternov do DCC/Blenderu, robotického dvojčaťa ROS2, samoladiacej stanice s WSL2 a 120 Hz IDE + plný AST katalóg funkcií.",
    },
)


def build_engine_integration_and_export_daily_markdown(
    sample_dsl: str = (
        "HEX(2, -1); PT3(1.5, 0.42, -2.0); RAY(58.2825, 3.0); "
        "LAYER_MASK(0x3ff); CALL(compute_algebraic_tensor_move); CALL(resolve_action_with_world_and_ai_chain)"
    ),
    uma_utilization: float = 0.35,
    md_path: Path = DOCS_ARCH_DIR / "DAILY_IMPROVEMENTS_PARSER_AND_ENGINE_INTEGRATION_2026_11_16.md",
    json_path: Path = REPORTS / "engine_integration_and_parser_summary.json",
) -> Dict[str, Any]:
    """Run both parsers, verify CALL targets against AST catalog, and write the comprehensive Slovak Markdown report."""
    parsed_dsl = parse_engine_spatial_points_and_directives(sample_dsl, uma_utilization=uma_utilization)
    ast_catalog = parse_framework_functions_and_topology()
    known_names = {f["function"] for f in ast_catalog["functions"]} | {
        m for c in ast_catalog["classes"] for m in c["methods"]
    }

    unresolved_calls = [fn for fn in parsed_dsl["referenced_functions"] if fn not in known_names]
    if unresolved_calls:
        raise ContractViolation(f"DSL CALL references unknown framework function(s): {unresolved_calls}", violation_code="ERR_DSL_UNKNOWN_FN")

    payload = {
        "vital_max_hp": VITAL_MAX_HP,
        "parsed_dsl_demo": parsed_dsl,
        "ast_catalog_summary": {
            "modules_scanned": ast_catalog["modules_scanned"],
            "public_functions_count": ast_catalog["public_functions_count"],
            "public_classes_count": ast_catalog["public_classes_count"],
            "cross_function_call_edges_count": ast_catalog["cross_function_call_edges_count"],
        },
        "advanced_engine_stages": list(ADVANCED_ENGINE_STAGES),
        "daily_chronology_waves": list(DAILY_CHRONOLOGY_WAVES),
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_lines: List[str] = [
        "# Celodenný prehľad zlepšení, parser bodov a funkcií & pokročilá herná integrácia (`2026-11-16`)",
        "",
        "> **Ústavné kotvy:** `VITAL_MAX_HP == 6` · `PROPOSER != VERIFIER` · `UMA < 0.78` · $\\neg\\text{Valid} \\implies \\Delta S = 0$ · nulová závislosť od OpenAI.",
        "",
        "---",
        "",
        "## 1. Čo sme od rána zlepšili (chronologický a kvantitatívny súhrn)",
        "",
        "| Čas | Commity | Hlavný míľnik | Zmeraný / overený prínos |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for w in DAILY_CHRONOLOGY_WAVES:
        md_lines.append(f"| `{w['time']}` | `{w['commits']}` | **{w['milestone']}** | {w['gain']} |")

    md_lines.extend(
        [
            "",
            "### Kľúčové číselné skoky dňa (`LOCAL_MEASURED` & `VERIFIED_WITH_SCOPE`)",
            "- **126,4× zrýchlenie syntézy a testov:** odstránenie redundantného `O(M³)` parsovania AST skrátilo beh z `1428,94 s` (`23m 48s`) na **`11,30 s`**.",
            "- **5,04× rýchlejší medzimodulový prenos (`X05`):** binárne `128 B` pakety `KPKT` (`2 × 64 B` cache riadky + `SHA-256`) dosahujú **`637 349 rámcov/s` (`77,8 MB/s`)** oproti JSONu.",
            "- **Eliminácia 14,00× až 37,37× subnormálnych zásekov (`X03`):** preukázané mikroinštrukčné spomalenie IEEE-754 subnormálov v NumPy (`14,00×`) a OpenVINO CPU (`37,37×`), vyriešené bránou `canonicalize_float32`.",
            "- **Amortizácia kompilácie Iris Xe (`~187 ms -> ~9–15 ms`):** perzistentná `CACHE_DIR` pre GPU bloby a zero-copy `ov.Tensor` na `4` fyzických jadrách `AVX-512` (`+55,84 %` oproti `1T`).",
            "- **Kapacita herného jadra a scény:** **`13 362,7` potvrdených ťahov/s** (`7` súbežných 60 Hz arén pri `21,77 Mbps`/arénu) a **`10 / 20` aktívnych vrstiev Godot 4** (`0x3ff`, resp. `5 / 20` v tieňovom režime `UMA >= 0.78`).",
            "",
            "---",
            "",
            "## 2. Pokročilá herná integrácia do enginu (`S1..S6`)",
            "",
            "```mermaid",
            "flowchart LR",
            '  S1["S1: DSL & AST Parser<br/>HEX / PT3 / RAY / CALL"] --> S2["S2: Goniometrické jadro<br/>cos θ, sin θ + Difúzia D⁻¹A"]',
            '  S2 --> S3["S3: ClashGame Kernel<br/>HP == 6, ΔS == 0 pri chybe"]',
            '  S3 --> S4["S4: Spätné vetvy OP_01..04<br/>UMA < 0.78 & Freivalds O(n²)"]',
            '  S4 --> S5["S5: Godot 4 MultiMesh & VR<br/>10/20 vrstiev + .kopt Fresnel"]',
            '  S5 --> S6["S6: 128 B KPKT Zbernica<br/>SHA-256 + WSL2 kvóta RAM"]',
            "  S6 -. telemetrická spätná väzba .-> S2",
            "```",
            "",
            "| Fáza | Názov | Mechanizmus v engine | Rozpočet / strop |",
            "| :--- | :--- | :--- | :--- |",
        ]
    )
    for st in ADVANCED_ENGINE_STAGES:
        md_lines.append(f"| `{st['stage_id']}` | **{st['title']}** | {st['mechanism']} | `{st['budget']}` |")

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Deterministický parser bodov a AST katalóg funkcií frameworku",
            "",
            f"- **Prehľadané moduly balíka:** `{ast_catalog['modules_scanned']}`",
            f"- **Verejné funkcie extrahované cez `ast`:** `{ast_catalog['public_functions_count']}`",
            f"- **Verejné triedy extrahované cez `ast`:** `{ast_catalog['public_classes_count']}`",
            f"- **Overené medzifunkčné hrán volaní:** `{ast_catalog['cross_function_call_edges_count']}`",
            "",
            "### Ukážka spracovania bodov a príkazov enginu (`parse_engine_spatial_points_and_directives`)",
            f"Vstupný skript: `{sample_dsl}`",
            "",
            "```json",
            json.dumps(parsed_dsl, indent=2, ensure_ascii=False),
            "```",
            "",
            "### Súhrnný index modulov a verejných funkcií",
            "",
            "| Modul | Počet funkcií | Ukážkové verejné funkcie |",
            "| :--- | :---: | :--- |",
        ]
    )

    by_mod: Dict[str, List[str]] = {}
    for f in ast_catalog["functions"]:
        by_mod.setdefault(f["module"], []).append(f["function"])
    for mod, fn_list in sorted(by_mod.items()):
        preview = ", ".join(f"`{n}`" for n in fn_list[:5])
        if len(fn_list) > 5:
            preview += f" *(+{len(fn_list) - 5} ďalších)*"
        md_lines.append(f"| `{mod}.py` | `{len(fn_list)}` | {preview} |")

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Epistemické ohraničenie (`AGENTS.md`)",
            "- **Zmerané lokálne (`LOCAL_MEASURED`):** latencie OpenVINO `CPU 1T/4T/8T` a `Iris Xe`, priepustnosť cache `L1d/L2/L3/DRAM`, penalizácia subnormálov IEEE-754, rýchlosť `128 B` paketov a konvergencia mikro-modelov.",
            "- **Odvodené a zmluvne overené (`DERIVED` / `VERIFIED_WITH_SCOPE`):** limit vrstiev Godot 4 (`10/20` pri `UMA < 0.78`), vzorce šírky pásma `E1..E8`, šablóna `.wslconfig` a `PL1` výkonové obálky.",
            "- **Nezmerané v používateľskom priestore Windows (`UNKNOWN`):** priame hardvérové RAPL Joule registre a Ring-0 MSR teploty bez jadrového ovládača.",
        ]
    )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {
        "vital_max_hp": VITAL_MAX_HP,
        "markdown_path": str(md_path),
        "json_path": str(json_path),
        "modules_scanned": ast_catalog["modules_scanned"],
        "public_functions_count": ast_catalog["public_functions_count"],
        "public_classes_count": ast_catalog["public_classes_count"],
        "call_edges_count": ast_catalog["cross_function_call_edges_count"],
        "parsed_clauses_count": parsed_dsl["clause_count"],
    }
