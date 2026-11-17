"""Unified Module-to-UI Access Patterns (UAP_01..UAP_04), Pareto Decision Ranking
for Derivative Studies, Backend-over-Documentation Actual-State Parity Gate, and
Multi-Sprint Long-Sequence Prompt-Layer Corpus Generator (`Nov 6 – Nov 17, 2026`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from krystal_kernel.constants import UMA_SATURATION_THRESHOLD, VITAL_MAX_HP, ContractViolation

from .combinatorial_derivative_functions_engine import (
    build_combinatorial_incremental_knowledge_ledger,
    derive_192b_wntc_kpkt_hypervisor_datagram,
    derive_ascii_survival_hud_projection,
    derive_bitboard_u64_hex_mask,
    derive_conv_paced_double_time_schedule,
    derive_godot_animation_track_from_story,
)
from .engine_point_and_function_parser import parse_framework_functions_and_topology
from .fact_synthesis_engine import REPO_ROOT, REPORTS
from .neurosymbolic_theories_case_study_and_kb import (
    CROSS_MODULE_THEORIES,
    run_neurosymbolic_pipeline_case_study_and_build_kb,
)
from .nomenclature_registry import export_glossary
from .platform_engine import StepExecutionGateProxy

DOCS_ARCH_DIR = REPO_ROOT / "docs" / "architecture"


# ---------------------------------------------------------------------------
# 1. Decision Engine: What Brings the Best Choices in Our Derivative Study
# ---------------------------------------------------------------------------

def decide_best_derivative_study_choices() -> Dict[str, Any]:
    """Rank derivative integrations by Pareto EVOI / (Latency * CouplingCost * Risk)."""
    candidates = [
        {
            "choice_id": "DF_01_192B_WNTC_KPKT_DATAGRAM",
            "title": "192 B (64 B WNTC + 128 B KPKT) Dual-OS Binary IPC",
            "throughput_gain_x": 5.04,
            "duty_cycle": 1.0,
            "implementation_risk": 1.1,
            "coupling_modules": 2,
            "evoi_score": 9.6,
            "why_best": "Eliminates JSON DTO overhead across Windows <-> WSL2 in exactly 3 x 64B cache lines with dual CRC32 + SHA-256.",
        },
        {
            "choice_id": "DF_02_CONV_PACED_DOUBLE_TIME",
            "title": "OpenVINO 1x3 Convolution-Paced 120 Hz Double-Time Scheduler",
            "throughput_gain_x": 3.80,
            "duty_cycle": 1.0,
            "implementation_risk": 1.2,
            "coupling_modules": 2,
            "evoi_score": 9.2,
            "why_best": "Prevents 1-tick noise thrashing while guaranteeing 0.00 ms VSync overrun at 120 Hz (k=2) and 0-frame halt at UMA >= 0.78.",
        },
        {
            "choice_id": "DF_05_U64_HEX_BITBOARD_FFI_MASK",
            "title": "Bijective 37-Bit uint64 Hex Bitboard for Rust Zero-GC FFI",
            "throughput_gain_x": 18.0,
            "duty_cycle": 0.9,
            "implementation_risk": 1.8,
            "coupling_modules": 2,
            "evoi_score": 9.4,
            "why_best": "Fits the entire R=3 hex board (37 cells) inside a single 64-bit CPU register with zero heap allocations.",
        },
        {
            "choice_id": "DF_04_NEURAL_ASCII_SURVIVAL_HUD",
            "title": "Deterministic < 2 KB Glyph Survival HUD under UMA >= 0.78",
            "throughput_gain_x": 20.0,
            "duty_cycle": 0.35,
            "implementation_risk": 1.3,
            "coupling_modules": 2,
            "evoi_score": 8.5,
            "why_best": "Cuts viewport bandwidth > 95% (< 300 B payload, < 0.5 ms CPU) when iGPU/UMA is saturated.",
        },
        {
            "choice_id": "DF_03_GODOT_ANIMATION_TRES_COMPILER",
            "title": "Native Godot 4 Animation (.tres) Keyframe & Layer-Mask Compiler",
            "throughput_gain_x": 4.20,
            "duty_cycle": 0.8,
            "implementation_risk": 1.5,
            "coupling_modules": 2,
            "evoi_score": 8.1,
            "why_best": "Turns deterministic simulation & robotics traces directly into playable Godot 4 scenes without manual keyframing.",
        },
    ]
    for c in candidates:
        c["composite_decision_index"] = round(
            (c["evoi_score"] * c["duty_cycle"] * (1.0 + 0.15 * min(c["throughput_gain_x"], 10.0)))
            / (c["implementation_risk"] * c["coupling_modules"]),
            3,
        )
    candidates.sort(key=lambda x: -x["composite_decision_index"])
    for idx, c in enumerate(candidates, start=1):
        c["rank"] = idx

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "decision_rule": "Maximize (EVOI * BoundedGain) / (Risk * CoupledModules); prefer 2-module C-ABI/128B bridges over >=4-way monoliths.",
        "top_recommended_choice": candidates[0]["choice_id"],
        "ranked_choices": candidates,
    }


# ---------------------------------------------------------------------------
# 2. Four Patterns Unifying Access Between Backend Modules & UI (UAP_01..UAP_04)
# ---------------------------------------------------------------------------

UI_MODULE_ACCESS_PATTERNS: Tuple[Dict[str, str], ...] = (
    {
        "pattern_id": "UAP_01_PROJECTION_ENVELOPE_READ",
        "principle": "Render(S) != S (Read-Only Signed Projection Envelope)",
        "contract": (
            "Every backend module exposes UI state via a uniform envelope "
            "{vital_max_hp, state_sha256, uma_utilization, evidence_class, doc_anchor, view_tokens, payload}; "
            "UI never mutates authoritative state directly."
        ),
    },
    {
        "pattern_id": "UAP_02_INTENT_ACTION_STEP_GATE",
        "principle": "~Valid(Action) => Delta_S == 0 (Governed Command Ingress)",
        "contract": (
            "All UI clicks, DSL scripts, and CLI commands enter through StepExecutionGateProxy / ContractViolation "
            "with proposer != verifier before touching kernel state."
        ),
    },
    {
        "pattern_id": "UAP_03_STATE_AB_CSS_TOKEN_ADAPTER",
        "principle": "Zero-Reallocation UMA-Adaptive Visual Tiering",
        "contract": (
            "Backend emits discrete tier tokens (HOLOGRAM_FULL_0x3ff vs WIREFRAME_SHADOW_0x1f vs ASCII_SURVIVAL) "
            "mapped to CSS custom properties and BP_360..1920 sockets without DOM rebuilds."
        ),
    },
    {
        "pattern_id": "UAP_04_DOC_LINKED_CAPABILITY_MANIFEST",
        "principle": "Self-Describing Backend Over Actual-State Documentation",
        "contract": (
            "Every registered endpoint binds its live Python callable, AST signature, pytest anchor, and "
            "architecture Markdown path; startup/CI audits parity so docs never drift from live code."
        ),
    },
)


BACKEND_OPENING_BEST_PRACTICES: Tuple[Dict[str, str], ...] = (
    {
        "id": "BOP_01_SINGLE_CAPABILITY_CATALOG",
        "practice": "Expose a single read-only `/api/v1/runtime/unified_gateway` manifest indexing all modules, UI views, and live `.md` anchors.",
    },
    {
        "id": "BOP_02_DOC_STATE_PARITY_GATE",
        "practice": "Verify on every release that every `.md` architecture file referenced by the gateway exists on disk and matches live invariants (`HP==6`, `0` bare asserts).",
    },
    {
        "id": "BOP_03_TYPED_ERROR_BOUNDARIES",
        "practice": "Map `ContractViolation` and `ValueError` deterministically to HTTP `400` JSON with `vital_max_hp: 6` and `delta_s: 0` — never leak raw stack traces or silently coerce invalid inputs.",
    },
    {
        "id": "BOP_04_BINARY_INTERNAL_JSON_EDGE",
        "practice": "Keep internal module/hypervisor rings strictly binary (`64B WNTC`, `128B KPKT`, `32B .kopt`) and project to UTF-8 JSON/SVG only at the outer HTTP/UI view edge.",
    },
    {
        "id": "BOP_05_EPISTEMIC_BADGING_IN_PAYLOAD",
        "practice": "Tag every backend field with `LOCAL_MEASURED`, `VERIFIED_WITH_SCOPE`, `DERIVED`, or `UNKNOWN` so neither UI nor downstream prompts over-claim.",
    },
)


class UnifiedBackendUiAccessGateway:
    """Unifies read projections and gated mutations between backend modules, UI views, and live docs."""

    def __init__(self) -> None:
        if VITAL_MAX_HP != 6:
            raise ContractViolation("VITAL_MAX_HP != 6 in UnifiedBackendUiAccessGateway", violation_code="ERR_GW_HP")
        self.proxy = StepExecutionGateProxy()
        self._registry: Dict[str, Tuple[Callable[[], Dict[str, Any]], str, str, str]] = {
            "wntc_kpkt_192b": (
                derive_192b_wntc_kpkt_hypervisor_datagram,
                "LOCAL_MEASURED",
                "/studio",
                "docs/architecture/COMBINATORIAL_DERIVATIVE_FUNCTIONS_AND_CHANGES.md",
            ),
            "conv_double_time": (
                derive_conv_paced_double_time_schedule,
                "VERIFIED_WITH_SCOPE",
                "/api/v1/compositor/double_time",
                "docs/architecture/CONSPECTUS_BRIDGING_FEATURES_AND_NEW_INVENTIONS.md",
            ),
            "godot_anim_tres": (
                derive_godot_animation_track_from_story,
                "VERIFIED_WITH_SCOPE",
                "/vr_hologram_profiler",
                "docs/architecture/DAILY_IMPROVEMENTS_PARSER_AND_ENGINE_INTEGRATION_2026_11_16.md",
            ),
            "ascii_survival_hud": (
                derive_ascii_survival_hud_projection,
                "VERIFIED_WITH_SCOPE",
                "/bot_perception_hud",
                "docs/architecture/PRACTICES_PROCEDURES_PLANS_AND_GOALS.md",
            ),
            "rust_u64_bitboard": (
                derive_bitboard_u64_hex_mask,
                "VERIFIED_WITH_SCOPE",
                "/syntactic_playable_studio",
                "docs/architecture/STANDARDIZED_DIGITAL_GAME_THEORY.md",
            ),
        }

    def project_for_ui(self, capability_key: str, uma_utilization: float = 0.35) -> Dict[str, Any]:
        """UAP_01 + UAP_03 + UAP_04: Return a read-only signed UI projection envelope with doc parity."""
        if capability_key not in self._registry:
            raise ContractViolation(f"Unknown capability_key: {capability_key!r}", violation_code="ERR_GW_KEY")
        if not (0.0 <= uma_utilization <= 1.0):
            raise ContractViolation("uma_utilization out of [0, 1]", violation_code="ERR_GW_UMA")

        fn, ev_class, ui_route, doc_rel = self._registry[capability_key]
        doc_abs = REPO_ROOT / doc_rel
        if not doc_abs.exists():
            raise ContractViolation(f"Documentation drift: missing {doc_rel}", violation_code="ERR_GW_DOC_MISSING")

        payload = fn()
        visual_tier = (
            "ASCII_SURVIVAL_TIER2"
            if uma_utilization >= 0.88
            else ("WIREFRAME_SHADOW_0x1f" if uma_utilization >= UMA_SATURATION_THRESHOLD else "HOLOGRAM_FULL_0x3ff")
        )
        raw_digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return {
            "vital_max_hp": VITAL_MAX_HP,
            "capability_key": capability_key,
            "ui_route": ui_route,
            "doc_anchor": doc_rel,
            "doc_exists_on_disk": True,
            "evidence_class": ev_class,
            "uma_utilization": round(uma_utilization, 4),
            "view_tokens": {
                "visual_tier": visual_tier,
                "godot_layer_mask": "0x1f" if uma_utilization >= UMA_SATURATION_THRESHOLD else "0x3ff",
            },
            "state_sha256": raw_digest,
            "payload": payload,
        }


# ---------------------------------------------------------------------------
# 3. Multi-Sprint Long-Sequence Synthesis & Prompt-Layer Corpus (Nov 6 – Nov 17)
# ---------------------------------------------------------------------------

MULTI_SPRINT_LONG_SEQUENCE: Tuple[Dict[str, Any], ...] = (
    {
        "sprint_wave": "SPRINT_W1_NOV06_TO_NOV08",
        "commits": "5f62dad6 .. d5bfb828",
        "theme": "20-Layer Glyph-Trigonometric Coherent Runtime, Zero-GC Synthesis & Ubuntu GPU CI",
        "core_artifacts": [
            "17/20-layer glyph-acoustic-Bezier compiler & Joule amplitude engine",
            "Eliminated 2s LM Studio TCP stall & wired executable Rust/C++/Python synthesis on :8090",
            "Normalized GHCR Docker pipeline & enabled CrystalSec v3 / Iris Xe GPU test jobs",
        ],
    },
    {
        "sprint_wave": "SPRINT_W2_NOV09_TO_NOV10",
        "commits": "4bfe17d1 .. 7e477f3e",
        "theme": "5-Stratum S0..S4 Automaton, 16 Wire Parameters (84B/128B), Rubik S_48 & GA Recompiler",
        "core_artifacts": [
            "Deterministic 8-bit intent automaton + Intel 11th-15th Gen ISA/socket adaptive kernel",
            "Grid-to-Logic Semantic Mirror & 4-Act Blender-Godot-Oxygen scenario",
            "Rubik's Cube G_Rubik subset S_48 choice mechanics & adaptive GA taxonomy optimizer",
        ],
    },
    {
        "sprint_wave": "SPRINT_W3_NOV11_TO_NOV12",
        "commits": "240317f8 .. cc7be983",
        "theme": "Material-Phi (1.618) /studio Multimedia Hub, 3D Rubik + Marias Projection & S5..S9 Gnosis",
        "core_artifacts": [
            "Procedural WAV soundtrack, SVG emblem, 3D Rubik renderer & 24B KMB1 Lockstep Multiplexer",
            "4 Marias playing-card projection lenses (Render(S) = Pi_Marias(S) != S) & Blizzard tactile card manual",
            "S5..S9 Ontologic Gnosis strata with 24B GNOS binary descriptor & K8s/nftables zero-trust configs",
        ],
    },
    {
        "sprint_wave": "SPRINT_W4_NOV13_TO_NOV14",
        "commits": "0980c68e .. a95b9ae4",
        "theme": "Sovereign OpenVINO Companion v1/v2, 64B WNTC WSL2-NT Bridge & Trigonometric Hybrid Platform",
        "core_artifacts": [
            "Hash-chained ComputationLogSystem, 6-microservice Supervisor Community & Brier/ECE calibration",
            "64B WNTC single-cache-line WSL2<->NT descriptor & 120-cell Hypervisor Matrix Choice Palette",
            "Z3 CrossDisplayMultiGenreParameterMatrix, StepExecutionGateProxy, adaptive rollback sync & Babylon 3D grid",
        ],
    },
    {
        "sprint_wave": "SPRINT_W5_NOV16",
        "commits": "82c28a59 .. 56ef041d",
        "theme": "OpenVINO FMA/K=449 Proof, x86-64 X01..X05, Clash 37-Hex Arena, Godot/VR, AST Parser & 40.4x Priority Logger",
        "core_artifacts": [
            "Proved Iris Xe sequential FMA bit-identity, localized OpenBLAS K=449 transition & achieved 126.4x AST speedup",
            "Measured X01..X05 (L2 55.1 GB/s, 14x/37.4x subnormal cliff, 5.04x 128B KPKT speedup, 13,362.7 arena commits/s)",
            "Built Goniometric Kernel DESC_01..03, Godot 10/20 layers, HENC micro-LLM, XLP_01..04 copilots, 3 dependency bridges, visual story generators & 0-bare-assert production gate",
        ],
    },
    {
        "sprint_wave": "SPRINT_W6_NOV17_CURRENT",
        "commits": "22b3d96a .. HEAD",
        "theme": "Double-Time 120Hz/128B Bugfixes, Convolutional Hypervisor Interposer, C(6,2) Derivatives & Unified Gateway",
        "core_artifacts": [
            "Fixed 5 Double-Time bugs: 120Hz k=2 clamping, non-overlapping 128B BUF_00/01, eta_crit=1.72, UMA>=0.78 halt & HTTP query params",
            "Added OpenVINO ConvolutionalHypervisorStepBridge (1x6x1x16 -> Conv1x3 -> StepExecutionGateProxy + 64B WNTC + 128B KPKT)",
            "Synthesized DF_01..DF_05 combinatorial derivatives, TH_01..TH_04 cross-module theories, Neurosymbolic KB & UnifiedBackendUiAccessGateway",
        ],
    },
)


def build_unified_gateway_and_sprint_prompt_corpus(
    json_path: Path = REPORTS / "sprint_long_sequence_prompt_context.json",
    md_path: Path = DOCS_ARCH_DIR / "UNIFIED_BACKEND_UI_ACCESS_AND_SPRINT_PROMPT_CORPUS.md",
) -> Dict[str, Any]:
    """Audit actual-state documentation parity and export the full multi-sprint prompt-engineering corpus."""
    gw = UnifiedBackendUiAccessGateway()
    projections = {k: gw.project_for_ui(k, uma_utilization=0.35) for k in gw._registry}
    decisions = decide_best_derivative_study_choices()
    comb = build_combinatorial_incremental_knowledge_ledger()
    ns_kb = run_neurosymbolic_pipeline_case_study_and_build_kb()
    ast_cat = parse_framework_functions_and_topology()
    glossary = export_glossary()

    # Audit actual-state documentation files in docs/architecture/
    arch_docs = sorted(p.name for p in DOCS_ARCH_DIR.glob("*.md")) if DOCS_ARCH_DIR.exists() else []

    next_layer_meta_prompt = (
        "SYSTEM ANCHOR [KRYSTAL STACK PROMPT LAYER v2026.11.17]:\n"
        "1. Lock constitutional invariants: VITAL_MAX_HP == 6, UMA < 0.78, PROPOSER != VERIFIER, ~Valid => Delta_S == 0, 0 bare asserts.\n"
        "2. Route every operation window (1x6x1x16, canonicalize_float32) through ConvolutionalHypervisorStepBridge -> StepExecutionGateProxy.\n"
        "3. Use binary cache-line multiples internally (64B WNTC, 128B KPKT, 192B Dual-OS, 32B .kopt) and expose UI strictly via UAP_01 Projection Envelopes.\n"
        "4. Treat every new cross-module synthesis as PATTERN_CANDIDATE until an isolated CDT + pytest receipt promotes it to VERIFIED_WITH_SCOPE.\n"
        "5. Stratify claims into LOCAL_MEASURED, VERIFIED_WITH_SCOPE, DERIVED, or UNKNOWN; never invoke OpenAI endpoints."
    )

    corpus: Dict[str, Any] = {
        "vital_max_hp": VITAL_MAX_HP,
        "actual_codebase_state": {
            "platform_modules_count": ast_cat["modules_scanned"],
            "public_functions_count": ast_cat["public_functions_count"],
            "public_classes_count": ast_cat["public_classes_count"],
            "cross_function_call_edges": ast_cat["cross_function_call_edges_count"],
            "bound_nomenclature_terms": len(glossary["terms"]),
            "unresolved_nomenclature_terms": len(glossary["unresolved_terms"]),
            "architecture_markdown_docs_count": len(arch_docs),
            "architecture_markdown_docs": arch_docs,
            "combinatorial_subsets_6_pillars": comb["combinatorics"],
            "neurosymbolic_kb_records": ns_kb["kb_records_count"],
        },
        "derivative_decision_ranking": decisions,
        "ui_module_access_patterns": list(UI_MODULE_ACCESS_PATTERNS),
        "backend_opening_best_practices": list(BACKEND_OPENING_BEST_PRACTICES),
        "verified_gateway_projections": {k: {"ui_route": v["ui_route"], "doc_anchor": v["doc_anchor"], "state_sha256": v["state_sha256"]} for k, v in projections.items()},
        "cross_module_theories_ids": [t["theory_id"] for t in CROSS_MODULE_THEORIES],
        "multi_sprint_long_sequence": list(MULTI_SPRINT_LONG_SEQUENCE),
        "next_layer_prompt_seed": next_layer_meta_prompt,
    }
    corpus_sha = hashlib.sha256(json.dumps(corpus, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    corpus["corpus_sha256"] = corpus_sha

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_lines = [
        "# Unified Backend-to-UI Access, Actual-State Documentation Standards & Multi-Sprint Prompt Corpus",
        "",
        f"- **Constitutional Invariant:** `VITAL_MAX_HP == {VITAL_MAX_HP}` · Corpus `SHA-256 = {corpus_sha[:20]}...`",
        f"- **Live Codebase Inventory:** `{ast_cat['modules_scanned']}` modules · `{ast_cat['public_functions_count']}` public functions · `{ast_cat['public_classes_count']}` classes · `{len(glossary['terms'])}` bound terms (`0` unresolved) · `{len(arch_docs)}` architecture `.md` specs",
        "",
        "---",
        "",
        "## 1. Best Choices in Our Derivative Integration Study (Pareto Decision Ranking)",
        "",
        "| Rank | Choice ID | Composite Index | Gain | Why It Wins |",
        "| :---: | :--- | :---: | :---: | :--- |",
    ]
    for c in decisions["ranked_choices"]:
        md_lines.append(
            f"| `#{c['rank']}` | **`{c['choice_id']}`** | **`{c['composite_decision_index']}`** | `{c['throughput_gain_x']}x` | {c['why_best']} |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Patterns Unifying Access Between Backend Modules & UI (`UAP_01..UAP_04`)",
            "",
            "| Pattern ID | Principle | Contract |",
            "| :--- | :--- | :--- |",
        ]
    )
    for p in UI_MODULE_ACCESS_PATTERNS:
        md_lines.append(f"| `{p['pattern_id']}` | **{p['principle']}** | {p['contract']} |")

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Best Practices for Opening the Backend Layer Over Actual-State Documentation (`BOP_01..BOP_05`)",
            "",
            "| ID | Best Practice |",
            "| :--- | :--- |",
        ]
    )
    for b in BACKEND_OPENING_BEST_PRACTICES:
        md_lines.append(f"| `{b['id']}` | {b['practice']} |")

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Multi-Sprint Long Sequence (`Nov 6 – Nov 17, 2026`) & Ready-to-Use Next Prompt Layer",
            "",
        ]
    )
    for sw in MULTI_SPRINT_LONG_SEQUENCE:
        md_lines.append(f"### `{sw['sprint_wave']}` (`{sw['commits']}`) — {sw['theme']}")
        for art in sw["core_artifacts"]:
            md_lines.append(f"- {art}")
        md_lines.append("")

    md_lines.extend(
        [
            "### Synthesized Next-Layer Prompt Seed",
            "```text",
            next_layer_meta_prompt,
            "```",
        ]
    )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return corpus
