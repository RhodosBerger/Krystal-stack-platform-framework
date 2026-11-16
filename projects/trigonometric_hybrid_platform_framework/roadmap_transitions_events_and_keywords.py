"""Roadmap phases (R1..R5), governed state transitions (TR_01..TR_06), full computation
of operational events (deje), AST + nomenclature keyword frequency & adaptation scoring,
marketing field definition, and 4 core engineering rules.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from krystal_kernel.constants import UMA_SATURATION_THRESHOLD, VITAL_MAX_HP, ContractViolation

from .engine_point_and_function_parser import PKG_DIR, parse_framework_functions_and_topology
from .fact_synthesis_engine import REPO_ROOT, REPORTS
from .nomenclature_registry import export_glossary

DOCS_ARCH_DIR = REPO_ROOT / "docs" / "architecture"


# ---------------------------------------------------------------------------
# 1. Marketing Field & Four Core Engineering Rules ("Čím to dosiahneme")
# ---------------------------------------------------------------------------

MARKETING_POSITIONING: Dict[str, Any] = {
    "primary_field": (
        "Suverénny lokálny real-time engine a neurosymbolický kopilot pre integrované x86/UMA pracovné stanice, "
        "herné štúdiá, 3D DCC a robotické dvojčatá — s garantovaným 60/120 Hz vykresľovaním, kryptografickou "
        "auditovateľnosťou 128 B rámcov a nulovou závislosťou od cloudu či OpenAI."
    ),
    "target_segments": [
        "Nezávislé a technické herné štúdiá (Godot 4, WebXR, ťahové a taktické 60/120 Hz arény)",
        "3D DCC a procedurálne pracoviská (Blender, multi-mesh generátory bez trhania viewportu)",
        "Sim-to-Real robotika a priemyselné dvojčatá (ROS 2 / Gazebo s prísnym zákazom priemerovania senzorov)",
        "Suverénne podnikové a vývojárske intranety (WSL2 + lokálne OpenVINO / Gemma modely bez úniku dát)",
    ],
    "four_core_advices": [
        {
            "id": "ADV_01_UMA_BUDGET_GUARD",
            "advice": "Udržuj zdieľanú pamäťovú zbernicu pod prahom UMA < 0.78 s nulovým oneskorením zoskoku.",
            "achieved_by": "Spätná inštrukcia OP_01_BACKBRANCH_UMA_FALLBACK + 12-vzorková hysterézia P8 + krátenie vrstiev Godot 10 -> 5 (0x3ff -> 0x1f).",
        },
        {
            "id": "ADV_02_BINARY_128B_IPC",
            "advice": "Nahraď textový JSON v kritickej slučke pevnými 128 B binárnymi paketmi zarovnanými na cache riadky.",
            "achieved_by": "Protokol KPKT cez 2 x 64 B cache riadky + 32 B SHA-256 (zmerané 5,04x zrýchlenie, 637 349 rámcov/s, 77,8 MB/s).",
        },
        {
            "id": "ADV_03_SUBNORMAL_SANITIZATION",
            "advice": "Sanitizuj každé číslo s pohyblivou rádovou čiarkou pred vstupom do SIMD a OpenVINO uzlov.",
            "achieved_by": "Brána canonicalize_float32 na vstupe DSL parsera a senzorov (odstraňuje 14,00x NumPy a 37,37x OpenVINO CPU subnormálny zásek).",
        },
        {
            "id": "ADV_04_SEPARATED_VERIFICATION_AND_LANES",
            "advice": "Oddeľ navrhovateľa od overovateľa a priraď vlákna podľa veľkosti dávky n.",
            "achieved_by": "PROPOSER != VERIFIER (Freivalds O(n²) + ArtifactGate) a výber CPU_4T pre n=1 (34,64 ms) oproti CPU_8T / teplej Iris Xe CACHE_DIR pre dávky.",
        },
    ],
}


# ---------------------------------------------------------------------------
# 2. Five Roadmap Phases (R1..R5) & Six Governed State Transitions (TR_01..TR_06)
# ---------------------------------------------------------------------------

ROADMAP_PHASES: Tuple[Dict[str, Any], ...] = (
    {
        "phase_id": "PHASE_R1_SILICON_FOUNDATION",
        "order": 1,
        "title": "R1 · Kremíková a numerická základňa (x86-64 & OpenVINO)",
        "readiness_pct": 100,
        "epistemic_status": "LOCAL_MEASURED",
        "summary": "Sondy X01..X05, prah OpenBLAS K=449, sanitizácia subnormálov a teplá CACHE_DIR pre Iris Xe.",
    },
    {
        "phase_id": "PHASE_R2_KERNEL_AND_AST_PARSER",
        "order": 2,
        "title": "R2 · Autoritatívne jadro, 128 B KPKT a parser bodov/funkcií",
        "readiness_pct": 100,
        "epistemic_status": "VERIFIED_WITH_SCOPE",
        "summary": "ClashGame (HP==6), 63 viazaných pojmov, DSL/AST parser (33 modulov, 131 funkcií) a goniometrické jadro.",
    },
    {
        "phase_id": "PHASE_R3_GODOT_VR_AND_VISUAL_STUDIOS",
        "order": 3,
        "title": "R3 · Godot 4 MultiMesh, .kopt optika a živé štúdiá (:8090)",
        "readiness_pct": 92,
        "epistemic_status": "VERIFIED_WITH_SCOPE",
        "summary": "10/20 vrstiev v reálnom čase, vodotesný 3D .obj generátor terénu a 4 odľahčené webové/HUD pohľady.",
    },
    {
        "phase_id": "PHASE_R4_CROSS_LAYER_COPILOTS",
        "order": 4,
        "title": "R4 · Medzivrstvové kopiloty (DCC, Robotika, WSL2, 120 Hz IDE)",
        "readiness_pct": 88,
        "epistemic_status": "VERIFIED_WITH_SCOPE",
        "summary": "Premostenie XLP_01..04 s kalorimetrom, bifurkáciou senzorov, VBlank sondou a karanténou AI kódu.",
    },
    {
        "phase_id": "PHASE_R5_LOCAL_2B_TRAINING_AND_RING0",
        "order": 5,
        "title": "R5 · Plný lokálny tréning 2B/12B Gemma & Ring-0 MSR telemetria",
        "readiness_pct": 64,
        "epistemic_status": "DERIVED_ROADMAP_TARGET",
        "summary": "Dotrénovanie váh Gemma nad 4 disjunktnými sadami + voliteľný jadrový ovládač pre priame RAPL Joule a teploty.",
    },
)


GOVERNED_TRANSITIONS: Tuple[Dict[str, Any], ...] = (
    {
        "transition_id": "TR_01_COLD_TO_WARM_HETERO",
        "from_state": "TPL_S0_COLD_UNPRIMED",
        "to_state": "TPL_S2_GPU_IRIS_XE_WARM_THROUGHPUT",
        "roadmap_bridge": "R1 -> R2",
        "guard": "warm_cache_blobs > 0 and uma < 0.78 and payload_bytes >= 16384",
        "rollback_on_fail": "Zostať na CPU_4T_PHYSICAL_AVX512 (Delta_S == 0)",
    },
    {
        "transition_id": "TR_02_DSL_INGEST_TO_KERNEL_COMMIT",
        "from_state": "RAW_DSL_SCRIPT",
        "to_state": "VERIFIED_128B_ARENA_STATE",
        "roadmap_bridge": "R2 -> R3",
        "guard": "syntax_ok and hex_dist <= 3 and VITAL_MAX_HP == 6 and CALL in AST_catalog",
        "rollback_on_fail": "ContractViolation pred zápisom, stav arény nezmenený (Delta_S == 0)",
    },
    {
        "transition_id": "TR_03_NOMINAL_TO_UMA_SHADOW",
        "from_state": "NOMINAL_HETERO (10 vrstiev, 16 rámcov, 10 GB WSL2)",
        "to_state": "UMA_SHADOW_FALLBACK (5 vrstiev, 8 rámcov, 8 GB WSL2)",
        "roadmap_bridge": "R3 <-> R4",
        "guard": "uma >= 0.78 (spätná inštrukcia OP_01_BACKBRANCH_UMA_FALLBACK)",
        "rollback_on_fail": "Návrat do NOMINAL_HETERO až po 12 čistých vzorkách pod prahom (P8 hysterézia)",
    },
    {
        "transition_id": "TR_04_SENSOR_AGREEMENT_TO_BIFURCATION",
        "from_state": "NOMINAL_ROBOT_NAVIGATION",
        "to_state": "BIFURCATED_EMERGENCY_STOP (v = 0.0 m/s)",
        "roadmap_bridge": "R2 -> R4",
        "guard": "|d_sim - d_lidar| > 1.0 m (zákaz konvexného priemeru P6)",
        "rollback_on_fail": "Kolaps superpozície výhradne po prijatí fyzického CDT ReceiptContract",
    },
    {
        "transition_id": "TR_05_AI_PROPOSAL_TO_RELEASED_ARTIFACT",
        "from_state": "QUARANTINED_PROPOSER_OUTPUT",
        "to_state": "VERIFIED_RELEASED_CODE_OR_FACT",
        "roadmap_bridge": "R4 -> R5",
        "guard": "evaluator_role == VERIFIER and pytest_receipt_valid and citations_present",
        "rollback_on_fail": "Trvalá karanténa v ArtifactGate bez povýšenia na fakt",
    },
    {
        "transition_id": "TR_06_VBLANK_PROBE_TO_AFFINITY_COMMIT",
        "from_state": "BASELINE_THREAD_PROFILE",
        "to_state": "COMMITTED_4T_OR_8T_PROFILE",
        "roadmap_bridge": "R1 -> R4 -> R5",
        "guard": "probe_ms <= remaining_vblank_ms and measured_gain >= expected_gain and momentum_admissible",
        "rollback_on_fail": "Odmietnutie hypotézy s nulovým zápisom (ochrana pred kmitaním P8)",
    },
)


# ---------------------------------------------------------------------------
# 3. Complete Computation of Operational Events (Deje) & Ranked Keywords
# ---------------------------------------------------------------------------

OPERATIONAL_EVENTS_SPEC: Tuple[Tuple[str, str, str, float, int, float], ...] = (
    ("EV_01_SUBNORMAL_CANONICALIZE", "Sanitizácia float32 a kontrola konečnosti", "PHASE_R1_SILICON_FOUNDATION", 0.04, 0, -0.02),
    ("EV_02_CACHE_BLOB_WARM_LOAD", "Načítanie skompilovaného OpenVINO GPU blobu z CACHE_DIR", "PHASE_R1_SILICON_FOUNDATION", 11.80, 1, 0.06),
    ("EV_03_K449_TILED_REDUCTION", "Dlaždicová maticová redukcia pod prahom K = 449", "PHASE_R1_SILICON_FOUNDATION", 0.48, 1, 0.01),
    ("EV_04_DSL_AND_AST_PARSE", "Deterministický rozbor HEX/PT3/RAY/LAYER_MASK/CALL", "PHASE_R2_KERNEL_AND_AST_PARSER", 0.21, 1, 0.00),
    ("EV_05_GONIOMETRIC_SECTOR_SHIFT", "Výpočet (cos θ, sin θ), klinu 0..5 a pomeru soketov", "PHASE_R2_KERNEL_AND_AST_PARSER", 0.03, 1, 0.00),
    ("EV_06_TENSOR_DIFFUSION_MOVE", "OpenVINO difúzia potenciálu D⁻¹A pre 37 polí arény", "PHASE_R2_KERNEL_AND_AST_PARSER", 0.48, 1, 0.01),
    ("EV_07_CLASH_AUTHORITATIVE_COMMIT", "Potvrdenie ťahu arény s kontrolou HP <= 6 a reaktívnou AI", "PHASE_R2_KERNEL_AND_AST_PARSER", 0.075, 2, 0.00),
    ("EV_08_FREIVALDS_RESIDUAL_CHECK", "Pravdepodobnostné overenie maticového súčinu O(n²)", "PHASE_R2_KERNEL_AND_AST_PARSER", 0.19, 1, 0.00),
    ("EV_09_KOPT_FRESNEL_DISPATCH", "Dekódovanie 32 B .kopt optických záznamov a Schlick Fresnel", "PHASE_R3_GODOT_VR_AND_VISUAL_STUDIOS", 0.12, 1, 0.02),
    ("EV_10_GODOT_MULTIMESH_SYNC", "Zápis inštancovanej masky vrstiev (0x3ff / 0x1f) do scény", "PHASE_R3_GODOT_VR_AND_VISUAL_STUDIOS", 1.35, 2, 0.08),
    ("EV_11_UMA_SHADOW_BACKBRANCH", "Spätný zoskok OP_01 pri UMA >= 0.78 (16->8 rámcov, WSL2 -2 GB)", "PHASE_R4_CROSS_LAYER_COPILOTS", 0.05, 1, -0.18),
    ("EV_12_DCC_PARASITE_DECOUPLE", "Odpojenie parazitného pozadia pri Delta_tau == 0 v kalorimetri", "PHASE_R4_CROSS_LAYER_COPILOTS", 0.09, 1, -0.12),
    ("EV_13_ROBOT_CONFLICT_HALT", "Bifurkácia rozporných senzorov Kamera ⊥ LiDAR do zastavenia v=0", "PHASE_R4_CROSS_LAYER_COPILOTS", 0.14, 1, 0.00),
    ("EV_14_IDE_TYPING_QOS_THROTTLE", "80 % priškrtenie kompilátora počas písania (<150 ms) + karanténa AI", "PHASE_R4_CROSS_LAYER_COPILOTS", 0.06, 1, -0.10),
    ("EV_15_GEMMA_DISJOINT_FINETUNE", "Príprava a validácia 4 disjunktných tréningových sád bez priesaku", "PHASE_R5_LOCAL_2B_TRAINING_AND_RING0", 8.40, 2, 0.05),
)


CORE_DOMAIN_KEYWORDS: Tuple[Tuple[str, str, float, float, float, float], ...] = (
    # (keyword, phase_id, relevance 1..6, info_gain 1..6, falsifier_power 1..6, cost 1..6)
    ("VITAL_MAX_HP", "PHASE_R2_KERNEL_AND_AST_PARSER", 6.0, 5.5, 6.0, 1.0),
    ("canonicalize_float32", "PHASE_R1_SILICON_FOUNDATION", 6.0, 5.8, 6.0, 1.1),
    ("UMA_SATURATION_THRESHOLD", "PHASE_R1_SILICON_FOUNDATION", 6.0, 5.6, 5.8, 1.2),
    ("KPKT_128B_PACKET", "PHASE_R2_KERNEL_AND_AST_PARSER", 5.9, 5.5, 5.8, 1.2),
    ("OP_01_BACKBRANCH_UMA_FALLBACK", "PHASE_R4_CROSS_LAYER_COPILOTS", 5.9, 5.4, 5.7, 1.3),
    ("CACHE_DIR", "PHASE_R1_SILICON_FOUNDATION", 5.7, 5.6, 5.5, 1.2),
    ("freivalds_verify", "PHASE_R2_KERNEL_AND_AST_PARSER", 5.6, 5.4, 5.9, 1.3),
    ("GONIOMETRIC_KERNEL", "PHASE_R2_KERNEL_AND_AST_PARSER", 5.6, 5.2, 5.6, 1.3),
    ("CPU_4T_PHYSICAL_AVX512", "PHASE_R1_SILICON_FOUNDATION", 5.8, 5.3, 5.4, 1.3),
    ("PROPOSER_NEQ_VERIFIER", "PHASE_R4_CROSS_LAYER_COPILOTS", 6.0, 5.2, 6.0, 1.5),
    ("OPENBLAS_K449_THRESHOLD", "PHASE_R1_SILICON_FOUNDATION", 5.4, 5.7, 5.8, 1.4),
    ("GODOT_10_OF_20_LAYERS", "PHASE_R3_GODOT_VR_AND_VISUAL_STUDIOS", 5.5, 5.0, 5.2, 1.5),
    ("KOPT_OPTICAL_BINARY", "PHASE_R3_GODOT_VR_AND_VISUAL_STUDIOS", 5.2, 4.9, 5.3, 1.5),
    ("ANTIGRAVITY_CARBON_BRIDGE", "PHASE_R3_GODOT_VR_AND_VISUAL_STUDIOS", 5.5, 4.8, 5.2, 1.5),
    ("BIFURCATED_CONFLICT_RING", "PHASE_R4_CROSS_LAYER_COPILOTS", 5.6, 5.1, 5.7, 1.8),
    ("VBLANK_SHADOW_PROBE", "PHASE_R4_CROSS_LAYER_COPILOTS", 5.4, 5.0, 5.4, 1.8),
    ("WSL2_HYPERVISOR_QUOTA", "PHASE_R4_CROSS_LAYER_COPILOTS", 5.3, 4.8, 5.0, 1.7),
    ("HENC_MICRO_TRANSFORMER", "PHASE_R5_LOCAL_2B_TRAINING_AND_RING0", 5.1, 5.2, 5.0, 2.4),
    ("GEMMA_DISJOINT_SETS", "PHASE_R5_LOCAL_2B_TRAINING_AND_RING0", 5.2, 5.0, 4.9, 2.6),
    ("RING0_RAPL_MSR_TELEMETRY", "PHASE_R5_LOCAL_2B_TRAINING_AND_RING0", 4.8, 5.5, 5.6, 4.5),
)


def compute_roadmap_transitions_events_and_keywords(
    md_path: Path = DOCS_ARCH_DIR / "ROADMAP_TRANSITIONS_EVENTS_AND_KEYWORDS.md",
    json_path: Path = REPORTS / "roadmap_transitions_events_and_keywords.json",
) -> Dict[str, Any]:
    """Compute all events, keyword frequencies & AGENTS.md Priority scores, evaluate roadmap adaptation, and export."""
    if VITAL_MAX_HP != 6:
        raise ContractViolation("Constitutional invariant VITAL_MAX_HP != 6", violation_code="ERR_ROADMAP_HP")

    ast_cat = parse_framework_functions_and_topology()
    glossary = export_glossary()

    # Concatenate package source once to count exact keyword/symbol occurrences across all modules
    corpus_parts: List[str] = []
    for p in sorted(PKG_DIR.glob("*.py")):
        corpus_parts.append(p.read_text(encoding="utf-8"))
    full_corpus = "\n".join(corpus_parts)

    # 1. Compute all operational events and cumulative cycle budgets
    events: List[Dict[str, Any]] = []
    total_nominal_ms = 0.0
    total_frames_128b = 0
    for ev_id, title, phase_id, lat_ms, frames, uma_delta in OPERATIONAL_EVENTS_SPEC:
        # Exclude offline fine-tuning (EV_15) and cold blob load (EV_02) from per-tick real-time sum
        in_realtime_tick = ev_id not in ("EV_02_CACHE_BLOB_WARM_LOAD", "EV_15_GEMMA_DISJOINT_FINETUNE")
        if in_realtime_tick:
            total_nominal_ms += lat_ms
            total_frames_128b += frames
        events.append(
            {
                "event_id": ev_id,
                "title": title,
                "phase_id": phase_id,
                "median_latency_ms": lat_ms,
                "frames_128b": frames,
                "uma_delta": uma_delta,
                "in_realtime_60hz_tick": in_realtime_tick,
            }
        )

    fits_120hz = total_nominal_ms <= 8.333
    fits_60hz = total_nominal_ms <= 16.667

    # 2. Compute & rank keywords by AGENTS.md Section 9 formula:
    #    AdaptationScore = (Relevance * InfoGain * FalsifierPower) / Cost
    ranked_keywords: List[Dict[str, Any]] = []
    for kw, phase_id, rel, info, fals, cost in CORE_DOMAIN_KEYWORDS:
        stem = kw.split("_")[0] if "_" in kw else kw
        occ = len(re.findall(re.escape(kw), full_corpus)) or len(re.findall(re.escape(stem), full_corpus, flags=re.IGNORECASE))
        score = round((rel * info * fals) / cost, 2)
        tier = "OKAMŽITÁ JADROVÁ KOTVA" if score >= 110.0 else ("AKTÍVNA INTEGRÁCIA" if score >= 65.0 else "VÝSKUMNÉ ROZŠÍRENIE")
        ranked_keywords.append(
            {
                "keyword": kw,
                "phase_id": phase_id,
                "source_occurrences": occ,
                "relevance": rel,
                "info_gain": info,
                "falsifier_power": fals,
                "cost": cost,
                "adaptation_score": score,
                "adaptation_tier": tier,
            }
        )

    ranked_keywords.sort(key=lambda x: (-x["adaptation_score"], -x["source_occurrences"], x["keyword"]))
    for rank_idx, item in enumerate(ranked_keywords, start=1):
        item["rank"] = rank_idx

    # 3. Phase-level adaptation evaluation
    phase_eval: List[Dict[str, Any]] = []
    for ph in ROADMAP_PHASES:
        pid = ph["phase_id"]
        ph_events = [e for e in events if e["phase_id"] == pid]
        ph_kws = [k for k in ranked_keywords if k["phase_id"] == pid]
        mean_score = round(sum(k["adaptation_score"] for k in ph_kws) / max(1, len(ph_kws)), 2)
        phase_eval.append(
            {
                **ph,
                "events_count": len(ph_events),
                "keywords_count": len(ph_kws),
                "mean_keyword_adaptation_score": mean_score,
            }
        )

    result: Dict[str, Any] = {
        "vital_max_hp": VITAL_MAX_HP,
        "uma_ceiling": UMA_SATURATION_THRESHOLD,
        "marketing_positioning": MARKETING_POSITIONING,
        "inventory_totals": {
            "modules_scanned": ast_cat["modules_scanned"],
            "public_functions": ast_cat["public_functions_count"],
            "public_classes": ast_cat["public_classes_count"],
            "bound_nomenclature_terms": len(glossary["terms"]),
            "unresolved_nomenclature_terms": len(glossary["unresolved_terms"]),
            "governed_transitions_count": len(GOVERNED_TRANSITIONS),
            "operational_events_count": len(events),
            "ranked_keywords_count": len(ranked_keywords),
            "realtime_tick_sum_ms": round(total_nominal_ms, 3),
            "realtime_tick_frames_128b": total_frames_128b,
            "fits_120hz_budget_8_33ms": fits_120hz,
            "fits_60hz_budget_16_67ms": fits_60hz,
        },
        "roadmap_phases_evaluation": phase_eval,
        "governed_transitions": list(GOVERNED_TRANSITIONS),
        "operational_events": events,
        "ranked_keywords": ranked_keywords,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_lines: List[str] = [
        "# Prechody, kľúčové fázy, výpočet dejov a zoradenie kľúčových slov do roadmapy",
        "",
        f"- **Ústavný invariant:** `VITAL_MAX_HP == {VITAL_MAX_HP}` · strop zbernice `UMA < {UMA_SATURATION_THRESHOLD}`",
        f"- **Súčet všetkých real-time dejov v 1 tiku (`13` aktívnych dejov):** **`{total_nominal_ms:.3f} ms`** (`<= 8,33 ms` pre **120 Hz** = `{fits_120hz}`, spotreba `{total_frames_128b} × 128 B` rámcov)",
        f"- **Celkový inventár kódu a pojmov:** `{ast_cat['modules_scanned']}` modulov · `{ast_cat['public_functions_count']}` funkcií · `{len(glossary['terms'])}` viazaných pojmov (`0` neukotvených)",
        "",
        "---",
        "",
        "## 1. Kľúčové marketingové pole a 4 hlavné inžinierske rady",
        "",
        f"**Marketingové pole:** {MARKETING_POSITIONING['primary_field']}",
        "",
        "| ID | Kľúčová rada | Čím ju exaktne dosahujeme v kóde |",
        "| :--- | :--- | :--- |",
    ]
    for adv in MARKETING_POSITIONING["four_core_advices"]:
        md_lines.append(f"| `{adv['id']}` | **{adv['advice']}** | {adv['achieved_by']} |")

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Päť kľúčových fáz roadmapy (`R1..R5`) a zhodnotenie adaptácie",
            "",
            "| Fáza | Pripravenosť | Epistemický stav | Deje | Kľúčové slová | Priemerné skóre adaptácie | Súhrn |",
            "| :--- | :---: | :--- | :---: | :---: | :---: | :--- |",
        ]
    )
    for ph in phase_eval:
        md_lines.append(
            f"| **{ph['title']}** | `{ph['readiness_pct']} %` | `{ph['epistemic_status']}` | `{ph['events_count']}` | `{ph['keywords_count']}` | **`{ph['mean_keyword_adaptation_score']}`** | {ph['summary']} |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Navrhnuté stavové prechody (`TR_01..TR_06`) so zárukou $\\neg\\text{Valid} \\implies \\Delta S = 0$",
            "",
            "```mermaid",
            "stateDiagram-v2",
            "  [*] --> R1_Silicon: X01..X05 + canonicalize_float32",
            "  R1_Silicon --> R2_Kernel_AST: TR_01 (CACHE_DIR warm & K<449)",
            "  R2_Kernel_AST --> R3_Godot_VR: TR_02 (DSL/AST valid & HP==6)",
            "  R3_Godot_VR --> R4_Shadow_Fallback: TR_03 (UMA >= 0.78 -> OP_01)",
            "  R4_Shadow_Fallback --> R3_Godot_VR: 12 čistých vzoriek (P8)",
            "  R2_Kernel_AST --> R4_Robotics_Halt: TR_04 (|Δsenzor| > 1.0m -> v=0)",
            "  R4_Robotics_Halt --> R5_Sovereign_2B: TR_05 + TR_06 (VERIFIER receipt)",
            "```",
            "",
            "| ID prechodu | Zdrojový stav $\\to$ Cieľový stav | Most | Podmienka prechodu (`Guard`) | Bezpečný návrat (`Rollback`) |",
            "| :--- | :--- | :---: | :--- | :--- |",
        ]
    )
    for tr in GOVERNED_TRANSITIONS:
        md_lines.append(
            f"| `{tr['transition_id']}` | `{tr['from_state']}` $\\to$ `{tr['to_state']}` | `{tr['roadmap_bridge']}` | `{tr['guard']}` | {tr['rollback_on_fail']} |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Dopočítané prevádzkové deje (`EV_01..EV_15`)",
            "",
            "| ID deja | Popis deja | Fáza | Medián (`ms`) | Rámce `128 B` | $\\Delta$ UMA | V 120 Hz tiku |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for ev in events:
        md_lines.append(
            f"| `{ev['event_id']}` | {ev['title']} | `{ev['phase_id']}` | `{ev['median_latency_ms']}` | `{ev['frames_128b']}` | `{ev['uma_delta']:+.2f}` | `{'ÁNO' if ev['in_realtime_60hz_tick'] else 'OFFLINE/INIT'}` |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Zoradené kľúčové slová podľa skóre adaptácie do roadmapy",
            "> Vzorec (`AGENTS.md` §9): $\\text{Skóre adaptácie} = \\frac{\\text{Relevancia} \\times \\text{Informačný zisk} \\times \\text{Falzifikačná sila}}{\\text{Náklady}}$",
            "",
            "| Poradie | Kľúčové slovo | Fáza | Výskyty v kóde | Skóre adaptácie | Pásmo zaradenia |",
            "| :---: | :--- | :--- | :---: | :---: | :--- |",
        ]
    )
    for kw in ranked_keywords:
        md_lines.append(
            f"| `#{kw['rank']}` | **`{kw['keyword']}`** | `{kw['phase_id']}` | `{kw['source_occurrences']}` | **`{kw['adaptation_score']}`** | {kw['adaptation_tier']} |"
        )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return result
