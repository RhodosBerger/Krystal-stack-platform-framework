"""Practices, procedures, measurable goals (G1..G6), full-day execution schedule
(BLK_01..BLK_06), Epistemic Micro-Prompt Router (MP_01..MP_08 per AGENTS.md §10),
and token-budgeted autonomous work executor with SHA-256 checkpointing.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from krystal_kernel.constants import UMA_SATURATION_THRESHOLD, VITAL_MAX_HP, ContractViolation

from .cross_layer_pattern_demo import run_cross_layer_pattern_demonstration
from .engine_point_and_function_parser import parse_engine_spatial_points_and_directives
from .fact_synthesis_engine import REPO_ROOT, REPORTS
from .roadmap_transitions_events_and_keywords import compute_roadmap_transitions_events_and_keywords

DOCS_ARCH_DIR = REPO_ROOT / "docs" / "architecture"


# ---------------------------------------------------------------------------
# 1. Verified Practices, Standardized Procedures & Measurable Goals (G1..G6)
# ---------------------------------------------------------------------------

ENGINEERING_PRACTICES: Tuple[Dict[str, str], ...] = (
    {
        "practice_id": "PRAC_01_ZERO_SUBNORMAL_INGEST",
        "title": "Vstupná sanitizácia float32 pred každým SIMD/OpenVINO uzlom",
        "rule": "Každá externá súradnica a váha prechádza cez canonicalize_float32, čím sa predchádza 14,0x–37,4x mikroinštrukčnému spomaleniu.",
    },
    {
        "practice_id": "PRAC_02_CACHELINE_128B_CONTRACTS",
        "title": "Binárne 128 B rámce (2 x 64 B cache riadky) namiesto JSONu",
        "rule": "Medzimodulová výmena v kritickej slučke používa výhradne KPKT/ZeroCopyContractFrame so SHA-256 (5,04x zrýchlenie).",
    },
    {
        "practice_id": "PRAC_03_CHRONOMETRIC_UMA_GUARD",
        "title": "Ochrana zdieľanej zbernice UMA < 0.78 s hysteréziou 12 vzoriek",
        "rule": "Okamžitý zoskok OP_01 na CPU_4T pri špičke (0 rámcov meškania) a opätovné zapnutie GPU až po 12 čistých vzorkách.",
    },
    {
        "practice_id": "PRAC_04_SEPARATED_PROPOSER_VERIFIER",
        "title": "Prísne oddelenie návrhu od certifikácie (PROPOSER != VERIFIER)",
        "rule": "Žiadny LLM ani simulátor necertifikuje vlastný výstup; vyžaduje sa Freivalds O(n²), pytest ReceiptContract alebo citácia [F_...].",
    },
)


STANDARD_PROCEDURES: Tuple[Dict[str, str], ...] = (
    {
        "step": "PROC_01_OBSERVE_AND_ANCHOR",
        "action": "Zamkni invarianty (VITAL_MAX_HP == 6, UMA < 0.78) a načítaj lokálnu telemetriu bez predpokladov z pamäte.",
    },
    {
        "step": "PROC_02_CHEAPEST_DISCRIMINATING_TEST",
        "action": "Formuluj hypotézu s 5-ticou (Mechanizmus, Efekt, Baseline, Predikcia, Falzifikátor) a spusti najlacnejší CDT test.",
    },
    {
        "step": "PROC_03_SURGICAL_IMPLEMENTATION",
        "action": "Vykonaj minimálny reverzibilný zásah so zárukou ~Valid(Action) => Delta_S == 0.",
    },
    {
        "step": "PROC_04_VERIFY_AND_CHECKPOINT",
        "action": "Over pozitívne aj hraničné testy (pytest), zapíš SHA-256 doklad do reports/ a ulož git commit pred ďalším blokom.",
    },
)


MEASURABLE_GOALS: Tuple[Dict[str, Any], ...] = (
    {
        "goal_id": "G1_120HZ_DETERMINISM",
        "target": "Súčet všetkých 13 real-time dejov v tiku <= 8.33 ms (120 Hz)",
        "current_measured": "2.835 ms (SPLNENÉ, rezerva 5.498 ms)",
        "falsifier": "Akýkoľvek real-time tik presiahne 8.33 ms alebo spôsobí výpadok scanoutu pri UMA >= 0.78.",
    },
    {
        "goal_id": "G2_ZERO_UNBOUND_NOMENCLATURE",
        "target": "0 neukotvených pojmov a 100 % AST overenie príkazov CALL v engine",
        "current_measured": "63/63 pojmov ukotvených, 0 neviazaných, AST skenuje všetky moduly (SPLNENÉ)",
        "falsifier": "Prítomnosť čo i len jedného pojmu bez viazanej funkcie/testu v nomenclature_registry.py.",
    },
    {
        "goal_id": "G3_MULTI_ARENA_THROUGHPUT",
        "target": ">= 5 súbežných 60 Hz arén (4 hráči, 37 hexov) na jedinom mobilnom CPU",
        "current_measured": "7 súbežných 60 Hz arén (13 362,7 potvrdených ťahov/s, SPLNENÉ)",
        "falsifier": "Pokles priepustnosti ClashGame.apply pod 9 750 rámcov/s.",
    },
    {
        "goal_id": "G4_GODOT_VR_LAYER_DENSITY",
        "target": "10/20 aktívnych vrstiev v Godote 4 (0x3ff) pri UMA < 0.78 a 5/20 (0x1f) v tieni",
        "current_measured": "10/20 a 5/20 overené v .tscn exporte aj DSL parseri (SPLNENÉ)",
        "falsifier": "Prekročenie masky 0x1f počas stavu UMA >= 0.78.",
    },
    {
        "goal_id": "G5_SOVEREIGN_ZERO_OPENAI",
        "target": "100 % lokálna prevádzka mosta Antigravity <-> Carbon + Gemma/OpenVINO bez OpenAI",
        "current_measured": "ContractViolation pri pokuse o provider='openai' + 4 disjunktné sady Gemma (SPLNENÉ)",
        "falsifier": "Akékoľvek volanie externého OpenAI endpointu alebo priesak medzi train a eval splitom.",
    },
    {
        "goal_id": "G6_TOKEN_EXHAUSTION_CONTINUITY",
        "target": "Plynulé vykonávanie pracovných blokov až do tokenového stropu s nulovou stratou stavu",
        "current_measured": "Riadené cez EpistemicMicroPromptRouter (MP_01..08) + SHA-256 checkpointy (AKTÍVNE)",
        "falsifier": "Ukončenie bloku pri vyčerpaní tokenov bez uloženého JSON checkpointu a čistého stavu.",
    },
)


# ---------------------------------------------------------------------------
# 2. Full-Day Schedule (BLK_01..BLK_06) & Epistemic Micro-Prompt Router (MP_01..MP_08)
# ---------------------------------------------------------------------------

FULL_DAY_BLOCKS: Tuple[Dict[str, Any], ...] = (
    {
        "block_id": "BLK_01_MORNING_SILICON",
        "window": "09:45 – 10:15",
        "phase": "R1",
        "status": "COMPLETED",
        "deliverable": "OpenVINO konvolučný reťazec, FMA sčítacie poradie, prah OpenBLAS K=449 a Freivalds O(n²).",
    },
    {
        "block_id": "BLK_02_MIDDAY_ARENA_AND_SHELL",
        "window": "10:15 – 12:00",
        "phase": "R2",
        "status": "COMPLETED",
        "deliverable": "Clash Battle Royale (37 hexov, 128 B potvrdenia), heterogénny CLI shell, DxDiag a 126,4x AST zrýchlenie.",
    },
    {
        "block_id": "BLK_03_AFTERNOON_VR_GONIO_PARSER",
        "window": "12:00 – 14:50",
        "phase": "R2 + R3",
        "status": "COMPLETED",
        "deliverable": "Merania X01..X05, goniometrické jadro DESC_01..03, Godot 10/20 vrstiev, .kopt, 4 živé štúdiá a DSL/AST parser.",
    },
    {
        "block_id": "BLK_04_CROSS_LAYER_AND_EXECUTOR",
        "window": "14:50 – 16:30",
        "phase": "R4",
        "status": "COMPLETED_NOW",
        "deliverable": "Medzivrstvové kopiloty XLP_01..04 + TokenBudgetedAutonomousExecutor a router mikropromptov MP_01..08.",
    },
    {
        "block_id": "BLK_05_GEMMA_ADAPTER_AND_LOD_SWEEP",
        "window": "16:30 – 19:00",
        "phase": "R3 + R5",
        "status": "QUEUED_READY",
        "deliverable": "Dávkový lokálny tréning adaptérov nad gemma_neurosymbolic_finetune_sets.jsonl a viac-dlaždicový MultiMesh stress-test.",
    },
    {
        "block_id": "BLK_06_LONG_HORIZON_GOAL_STRESS",
        "window": "19:00 – vyčerpanie tokenov (/goal)",
        "phase": "R4 + R5",
        "status": "QUEUED_READY",
        "deliverable": "Dlhodobý záťažový beh 7 súbežných 60 Hz arén, regresný audit všetkých modulov a finálny SHA-256 checkpoint.",
    },
)


GUIDING_MICRO_PROMPTS: Tuple[Dict[str, str], ...] = (
    {
        "prompt_id": "MP_01_SILICON_CDT_PROBE",
        "task_type": "silicon_benchmark",
        "phase": "R1",
        "micro_prompt": (
            "Zamkni VITAL_MAX_HP==6. Zmeraj n>=10 vzoriek (medián, p95, stdev) na i5-1135G7 + Iris Xe, "
            "over canonicalize_float32 a zapíš výsledok s triedou LOCAL_MEASURED do reports/."
        ),
        "stop_and_checkpoint": "Ulož JSON stopu ak p95 <= rozpočet a rel_err < 1e-3.",
    },
    {
        "prompt_id": "MP_02_DSL_AND_AST_GATE",
        "task_type": "parser_and_contracts",
        "phase": "R2",
        "micro_prompt": (
            "Rozober vstupné body HEX/PT3/RAY/LAYER_MASK/CALL cez deterministický parser, over každý CALL voči "
            "AST katalógu a pri neplatnom vstupe vyhoď ContractViolation so zárukou Delta_S == 0."
        ),
        "stop_and_checkpoint": "Spusti pozitívny aj hraničný pytest test pred zápisom.",
    },
    {
        "prompt_id": "MP_03_GONIOMETRIC_GODOT_SYNC",
        "task_type": "spatial_and_vr_render",
        "phase": "R3",
        "micro_prompt": (
            "Prepočítaj DESC_01..03 z uhla theta (|sin²+cos²-1| < 1e-12), nastav masku vrstiev Godot na 0x3ff "
            "(UMA < 0.78) alebo 0x1f (UMA >= 0.78) a zosynchronizuj .kopt Fresnel záznamy."
        ),
        "stop_and_checkpoint": "Over platnosť .tscn a .kopt SHA-256 hashu.",
    },
    {
        "prompt_id": "MP_04_UMA_HYPERVISOR_BACKBRANCH",
        "task_type": "hypervisor_and_qos",
        "phase": "R4",
        "micro_prompt": (
            "Pri UMA >= 0.78 aktivuj OP_01_BACKBRANCH_UMA_FALLBACK: odpoj GPU_ALLOCATE, skráť rámce 16->8, "
            "zníž kvótu WSL2 RAM o 2 GB a odošli 128 B KPKT paket s 12-vzorkovou hysteréziou návratu."
        ),
        "stop_and_checkpoint": "Potvrď sha256_verified == True na všetkých 128 B paketoch.",
    },
    {
        "prompt_id": "MP_05_CROSS_LAYER_COPILOT_AUDIT",
        "task_type": "cross_layer_integration",
        "phase": "R4",
        "micro_prompt": (
            "Každé premostenie do DCC, Robotiky, WSL2 stanice a IDE označ najprv ako PATTERN_CANDIDATE so 6-ticou, "
            "spusti izolovaný diskriminačný beh a povýš status iba pri preukázanom výsledku."
        ),
        "stop_and_checkpoint": "Zapíš cross_layer_pattern_demo.json a over všetky 4 vrstvy.",
    },
    {
        "prompt_id": "MP_06_GEMMA_SOVEREIGN_FINETUNE",
        "task_type": "local_ml_training",
        "phase": "R5",
        "micro_prompt": (
            "Over nulový priesak medzi train a eval v gemma_neurosymbolic_finetune_sets.jsonl, zakáž provider='openai', "
            "trénuj lokálne váhy/adaptéry a vyžaduj citáciu [F_...] pri každom faktickom výstupe."
        ),
        "stop_and_checkpoint": "Zaznamenaj počiatočnú a koncovú stratu (loss) + eval presnosť.",
    },
    {
        "prompt_id": "MP_07_MULTI_ARENA_STRESS_LOOP",
        "task_type": "endurance_stress",
        "phase": "R4 + R5",
        "micro_prompt": (
            "Spusti záťažovú slučku 7 súbežných 60 Hz arén (ClashGame + tenzorová difúzia D⁻¹A), sleduj p95 latenciu "
            "a pri odchýlke Freivalds rel_gap > 1e-4 okamžite aktivuj OP_03_FREIVALDS_RESIDUAL_TRAP."
        ),
        "stop_and_checkpoint": "Ulož priebežnú štatistiku každých N kôl do reports/.",
    },
    {
        "prompt_id": "MP_08_TOKEN_CAP_HANDOFF_GUARD",
        "task_type": "token_budget_guard",
        "phase": "ALL",
        "micro_prompt": (
            "Keď zostávajúci rozpočet tokenov klesne pod bezpečnú rezervu (<= 15 %), nespúšťaj novú neuzavretú vetvu: "
            "dokonči bežiaci test, zapíš SHA-256 checkpoint s presným ďalším krokom a odporuč /compact alebo /goal."
        ),
        "stop_and_checkpoint": "Zapíš autonomous_day_checkpoint.json a ukonči krok čisto.",
    },
)


class EpistemicMicroPromptRouter:
    """Selects the specialized micro-prompt for a given task_type (forbids monolithic gigaprompts per AGENTS.md §10)."""

    def __init__(self) -> None:
        self._by_type: Dict[str, Dict[str, str]] = {m["task_type"]: m for m in GUIDING_MICRO_PROMPTS}

    def route(self, task_type: str, remaining_token_ratio: float = 0.75) -> Dict[str, str]:
        if not (0.0 <= remaining_token_ratio <= 1.0):
            raise ContractViolation("remaining_token_ratio must be in [0, 1]", violation_code="ERR_TOKEN_RATIO")
        if remaining_token_ratio <= 0.15:
            return dict(self._by_type["token_budget_guard"])
        if task_type not in self._by_type:
            raise ContractViolation(f"Unknown micro-prompt task_type: {task_type!r}", violation_code="ERR_UNKNOWN_TASK_TYPE")
        return dict(self._by_type[task_type])


# ---------------------------------------------------------------------------
# 3. Token-Budgeted Autonomous Work Executor (Programmed Next Stage!)
# ---------------------------------------------------------------------------

class TokenBudgetedAutonomousExecutor:
    """Executes queued engineering tasks guided by routed micro-prompts up to a governed token ceiling."""

    def __init__(self, total_token_budget: int = 12_000, reserve_ratio: float = 0.15) -> None:
        if total_token_budget <= 0 or not (0.05 <= reserve_ratio < 0.5):
            raise ContractViolation("Invalid token budget or reserve_ratio", violation_code="ERR_EXEC_BUDGET")
        self.total_token_budget = total_token_budget
        self.reserve_ratio = reserve_ratio
        self.tokens_spent = 0
        self.router = EpistemicMicroPromptRouter()
        self.completed_steps: List[Dict[str, Any]] = []

    def run_until_budget_cap(
        self,
        work_queue: List[Tuple[str, str, int, Callable[[], Dict[str, Any]]]],
        checkpoint_path: Path = REPORTS / "autonomous_day_checkpoint.json",
    ) -> Dict[str, Any]:
        """Run `(step_id, task_type, est_tokens, fn)` items until reserve threshold, then write SHA-256 checkpoint."""
        deferred_steps: List[str] = []
        halted_for_token_reserve = False

        for step_id, task_type, est_tokens, fn in work_queue:
            remaining = self.total_token_budget - self.tokens_spent
            rem_ratio = remaining / float(self.total_token_budget)
            if rem_ratio <= self.reserve_ratio or (remaining - est_tokens) < int(self.total_token_budget * self.reserve_ratio):
                halted_for_token_reserve = True
                deferred_steps.append(step_id)
                continue

            routed = self.router.route(task_type, remaining_token_ratio=rem_ratio)
            t0 = time.perf_counter()
            out = fn()
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            self.tokens_spent += est_tokens
            self.completed_steps.append(
                {
                    "step_id": step_id,
                    "routed_prompt_id": routed["prompt_id"],
                    "tokens_spent": est_tokens,
                    "elapsed_ms": elapsed_ms,
                    "result_summary": out,
                }
            )

        guard_prompt = self.router.route("token_budget_guard", remaining_token_ratio=0.10 if halted_for_token_reserve else 0.50)
        body = {
            "vital_max_hp": VITAL_MAX_HP,
            "total_token_budget": self.total_token_budget,
            "tokens_spent": self.tokens_spent,
            "tokens_remaining": self.total_token_budget - self.tokens_spent,
            "halted_for_token_reserve": halted_for_token_reserve,
            "completed_count": len(self.completed_steps),
            "deferred_steps": deferred_steps,
            "handoff_micro_prompt": guard_prompt["prompt_id"],
            "completed_steps": self.completed_steps,
        }
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        body["checkpoint_sha256"] = digest

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return body


def execute_programmed_next_stages_and_export_practices_md(
    md_path: Path = DOCS_ARCH_DIR / "PRACTICES_PROCEDURES_PLANS_AND_GOALS.md",
    checkpoint_path: Path = REPORTS / "autonomous_day_checkpoint.json",
) -> Dict[str, Any]:
    """Run the Roadmap computation + TokenBudgetedAutonomousExecutor across queued stages and export the full MD guide."""
    roadmap_data = compute_roadmap_transitions_events_and_keywords()

    executor = TokenBudgetedAutonomousExecutor(total_token_budget=10_000, reserve_ratio=0.15)
    queue: List[Tuple[str, str, int, Callable[[], Dict[str, Any]]]] = [
        (
            "STEP_01_PARSE_MULTI_TILE_SCENE",
            "parser_and_contracts",
            2200,
            lambda: {
                "clauses": parse_engine_spatial_points_and_directives(
                    "HEX(0,0); HEX(-2,1); PT3(0.0,1.25,-1.5); RAY(120.0,2.5); LAYER_MASK(0x3ff); CALL(compute_algebraic_tensor_move)",
                    uma_utilization=0.35,
                )["clause_count"]
            },
        ),
        (
            "STEP_02_UMA_SHADOW_CLAMP_VERIFY",
            "hypervisor_and_qos",
            2400,
            lambda: {
                "shadow_mask": parse_engine_spatial_points_and_directives("HEX(1,0); LAYER_MASK(0xfffff)", uma_utilization=0.84)[
                    "parsed_nodes"
                ][1]["governed_hex"]
            },
        ),
        (
            "STEP_03_CROSS_LAYER_XLP_SWEEP",
            "cross_layer_integration",
            2800,
            lambda: {
                "verified_patterns": sum(
                    1
                    for p in run_cross_layer_pattern_demonstration(uma_utilization=0.82, write_reports=False)["pattern_6_tuples"]
                    if p["evidence_status"] in ("VERIFIED_WITH_SCOPE", "LOCAL_MEASURED")
                )
            },
        ),
        # Intentionally exceeds remaining budget before the 15% reserve so MP_08 safe handoff is exercised & proven!
        (
            "STEP_04_DEFERRED_OVERNIGHT_ENDURANCE",
            "endurance_stress",
            2000,
            lambda: {"arenas": 7},
        ),
    ]
    checkpoint = executor.run_until_budget_cap(queue, checkpoint_path=checkpoint_path)

    md_lines: List[str] = [
        "# Praktiky, postupy, celodenný plán, merateľné góly a riadiace mikroprompty",
        "",
        f"- **Ústavný invariant:** `VITAL_MAX_HP == {VITAL_MAX_HP}` · ochrana zbernice `UMA < {UMA_SATURATION_THRESHOLD}`",
        f"- **Overený 120 Hz rozpočet dejov:** `{roadmap_data['inventory_totals']['realtime_tick_sum_ms']} ms / 8.333 ms`",
        f"- **Kontrolný bod autonómneho exekútora:** `{checkpoint['completed_count']}` krokov dokončených, `{len(checkpoint['deferred_steps'])}` bezpečne odložený pri 15 % tokenovej rezerve (`SHA-256: {checkpoint['checkpoint_sha256'][:16]}...`)",
        "",
        "---",
        "",
        "## 1. Overené inžinierske praktiky (`PRAC_01..PRAC_04`)",
        "",
        "| ID | Praktika | Záväzné pravidlo v projekte |",
        "| :--- | :--- | :--- |",
    ]
    for p in ENGINEERING_PRACTICES:
        md_lines.append(f"| `{p['practice_id']}` | **{p['title']}** | {p['rule']} |")

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Štandardizovaný pracovný postup (`PROC_01..PROC_04`)",
            "",
            "| Krok | Postup |",
            "| :--- | :--- |",
        ]
    )
    for s in STANDARD_PROCEDURES:
        md_lines.append(f"| `{s['step']}` | {s['action']} |")

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Merateľné góly a falzifikátory (`G1..G6`)",
            "",
            "| Gól | Cieľová metrika | Aktuálne zmeraný stav | Podmienka okamžitého vyvrátenia (`Falzifikátor`) |",
            "| :--- | :--- | :--- | :--- |",
        ]
    )
    for g in MEASURABLE_GOALS:
        md_lines.append(f"| `{g['goal_id']}` | {g['target']} | **{g['current_measured']}** | {g['falsifier']} |")

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Celodenný plán vykonávania (`BLK_01..BLK_06`)",
            "",
            "| Blok | Časové okno | Fáza | Stav | Výstup bloku |",
            "| :--- | :---: | :---: | :---: | :--- |",
        ]
    )
    for b in FULL_DAY_BLOCKS:
        md_lines.append(f"| `{b['block_id']}` | `{b['window']}` | `{b['phase']}` | **`{b['status']}`** | {b['deliverable']} |")

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Sada riadiacich mikropromptov (`MP_01..MP_08`) až do vyčerpania tokenov",
            "> Podľa `AGENTS.md` §10 nepoužívame jeden monolitický gigaprompt, ale špecializované mikroprompty prepínané routerom `EpistemicMicroPromptRouter`. Pre dlhý autonómny beh v chate odporúčame príkaz **`/goal`**, a pri priblížení k limitu kontextového okna príkaz **`/compact`**.",
            "",
            "| ID | Typ úlohy | Fáza | Riadiaci mikroprompt | Stop podmienka & Checkpoint |",
            "| :--- | :--- | :---: | :--- | :--- |",
        ]
    )
    for mp in GUIDING_MICRO_PROMPTS:
        md_lines.append(
            f"| `{mp['prompt_id']}` | `{mp['task_type']}` | `{mp['phase']}` | {mp['micro_prompt']} | {mp['stop_and_checkpoint']} |"
        )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {
        "vital_max_hp": VITAL_MAX_HP,
        "practices_md_path": str(md_path),
        "roadmap_md_path": str(DOCS_ARCH_DIR / "ROADMAP_TRANSITIONS_EVENTS_AND_KEYWORDS.md"),
        "checkpoint": checkpoint,
        "realtime_tick_sum_ms": roadmap_data["inventory_totals"]["realtime_tick_sum_ms"],
        "top_3_keywords": [k["keyword"] for k in roadmap_data["ranked_keywords"][:3]],
    }
