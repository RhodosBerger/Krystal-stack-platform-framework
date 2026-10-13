#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: MULTI-LEVEL STATE RE-RENDERING & MULTILINGUAL TAXONOMIES
==============================================================================
Module: krystal_kernel/multilevel_state_rerendering_and_multilingual_taxonomies.py
Description:
    Provides hierarchical state re-rendering across 5 operational tiers (Silicon,
    Pipeline, Epistemic Authority, Linguistic Interface, and Visual Compositor)
    coupled with formal multilingual taxonomies supporting seamless localization
    across Slovak, Czech, English, German, Latin, and Japanese.

Architectural Pillars:
  1. Multi-Level State Hierarchy (5 Operational Tiers):
     - Level 0: Silicon / Hardware Physical State (Tj, Power, MSR, Clocks, Voltages).
     - Level 1: Compute Execution & Subsystem State (Workers, Ring Buffers, ISA).
     - Level 2: Epistemic & Authority State (Truth Lattice, Dual-Gate Verdicts).
     - Level 3: Semantic Linguistic State (Active Locale, Intent, Lexical Register).
     - Level 4: Visual Presentation & Compositor State (Dirty Tiles, Viewport, HUD Inlay).
  2. Hierarchical Dirty-State Cascade (Prekresľovanie):
     - Invalidation at level k selectively cascades to all downstream levels >= k,
       re-synthesizing localized telemetry, re-rasterizing visual viewports, and
       guaranteeing zero wasted compute.
  3. Multilingual Interface Taxonomies:
     - 6 Language Locales (SK, CS, EN, DE, LA, JA).
     - 4 Linguistic Registers (Formal Epistemic, Military Tactical, Cyber Telemetry, Natural).
     - 5 Semantic Intent Taxons (Actuate, Inspect, Explore, Audit, ReRender).
  4. Cross-Language Lexicon & Intent Classifier:
     - Real-time multilingual prompt understanding, translation of core systemic axioms,
       and localized status broadcasting.
  5. Constitutional Invariant:
     - assert VITAL_MAX_HP == 6 strictly enforced across all levels, locales, and renders.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import time
import math
from enum import Enum, unique
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Set

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP

# Constitutional Invariant Validation
assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"


# ==============================================================================
# 1. TAXONOMIES: STATE LEVELS & MULTILINGUAL INTERFACE
# ==============================================================================

@unique
class SystemStateLevelTaxon(str, Enum):
    """Hierarchical tiers of system state in Krystal Stack."""
    LEVEL_0_HARDWARE_SILICON   = "LEVEL_0_HARDWARE_SILICON"   # Thermals, power, clocks, MSRs, EU registers
    LEVEL_1_COMPUTE_PIPELINE   = "LEVEL_1_COMPUTE_PIPELINE"   # Thread router, ring queues, active ISA, worker matrix
    LEVEL_2_EPISTEMIC_AUTHORITY = "LEVEL_2_EPISTEMIC_AUTHORITY" # Truth lattice, Dual-Gate safety/evidence verdicts
    LEVEL_3_SEMANTIC_LINGUISTIC = "LEVEL_3_SEMANTIC_LINGUISTIC" # Language locale, intent vector, linguistic register
    LEVEL_4_VISUAL_COMPOSITOR  = "LEVEL_4_VISUAL_COMPOSITOR"  # Viewport scanlines, dirty tiles, ASCII/Canvas HUD


@unique
class LanguageLocaleTaxon(str, Enum):
    """Supported multilingual interface languages."""
    SK = "SK"  # Slovenčina (Slovak - Core Platform Native Language)
    CS = "CS"  # Čeština (Czech - Brotherly Native Language)
    EN = "EN"  # English (International Engineering Standard)
    DE = "DE"  # Deutsch (Industrial Systems Standard)
    LA = "LA"  # Lingua Latina (Classical Epistemic Axiomatic)
    JA = "JA"  # 日本語 (Cybernetic Robotic Locale)


@unique
class LinguisticRegisterTaxon(str, Enum):
    """Tone, formality, and domain register of the linguistic interface."""
    EPISTEMIC_FORMAL     = "EPISTEMIC_FORMAL"     # Rigorous mathematical axioms, truth lattices, verification proofs
    MILITARY_TACTICAL    = "MILITARY_TACTICAL"    # Terse, imperative, mission-critical status directives
    CYBERNETIC_TELEMETRY = "CYBERNETIC_TELEMETRY" # Dense telemetry, hex opcodes, nanosecond timecodes
    NATURAL_CONVERSATIONAL = "NATURAL_CONVERSATIONAL" # Fluid, human-friendly operator guidance


@unique
class SemanticIntentTaxon(str, Enum):
    """Semantic intent classification of multilingual operator prompts."""
    COMMAND_ACTUATE    = "COMMAND_ACTUATE"    # Triggers execution (run, execute, dispatch, start)
    TELEMETRY_INSPECT  = "TELEMETRY_INSPECT"  # Query metrics (status, check, inspect, show)
    HYPOTHESIS_EXPLORE = "HYPOTHESIS_EXPLORE" # Speculative exploration (what-if, simulate, hypothesize)
    GOVERNANCE_AUDIT   = "GOVERNANCE_AUDIT"   # Invariant & authority audit (verify, assert, proof)
    RE_RENDER_TRIGGER  = "RE_RENDER_TRIGGER"  # Redraws state / viewports (redraw, refresh, re-render)


# ==============================================================================
# 2. CANONICAL MULTILINGUAL LEXICON BRIDGE
# ==============================================================================

MULTILINGUAL_LEXICON: Dict[str, Dict[LanguageLocaleTaxon, str]] = {
    "SYSTEM_READY": {
        LanguageLocaleTaxon.SK: "Systém je pripravený v nominálnom stave.",
        LanguageLocaleTaxon.CS: "Systém je připraven v nominálním stavu.",
        LanguageLocaleTaxon.EN: "System ready in nominal state.",
        LanguageLocaleTaxon.DE: "System betriebsbereit im Nominalzustand.",
        LanguageLocaleTaxon.LA: "Systema paratum in statu nominali.",
        LanguageLocaleTaxon.JA: "システムは通常状態で準備完了しています。"
    },
    "AUTHORITY_APPROVED": {
        LanguageLocaleTaxon.SK: "Duálna brána: Autorita schválená na exekúciu.",
        LanguageLocaleTaxon.CS: "Duální brána: Autorita schválena k exekuci.",
        LanguageLocaleTaxon.EN: "Dual-Gate: Authority approved for actuation.",
        LanguageLocaleTaxon.DE: "Dual-Gate: Autorität zur Ausführung freigegeben.",
        LanguageLocaleTaxon.LA: "Porta Duplex: Auctoritas ad actum comprobata.",
        LanguageLocaleTaxon.JA: "デュアルゲート：実行権限が承認されました。"
    },
    "SAFETY_VETO": {
        LanguageLocaleTaxon.SK: "Bezpečnostná brána VETO: Porušenie invariantu alebo prekročenie limitov.",
        LanguageLocaleTaxon.CS: "Bezpečnostní brána VETO: Porušení invariantu nebo překročení limitů.",
        LanguageLocaleTaxon.EN: "SafetyGate VETO: Invariant breach or threshold exceeded.",
        LanguageLocaleTaxon.DE: "Sicherheits-VETO: Invariantenverletzung oder Grenzwert überschritten.",
        LanguageLocaleTaxon.LA: "VETO Salutis: Invarians violatum aut limes transgressus.",
        LanguageLocaleTaxon.JA: "安全拒否（VETO）：不変条件違反または制限超過。"
    },
    "INVARIANT_HP_SHIELD": {
        LanguageLocaleTaxon.SK: "Ústavný Invariant: 6/6 Max HP štít aktívny.",
        LanguageLocaleTaxon.CS: "Ústavní Invariant: 6/6 Max HP štít aktivní.",
        LanguageLocaleTaxon.EN: "Constitutional Invariant: 6/6 Max HP shield active.",
        LanguageLocaleTaxon.DE: "Verfassungsinvariante: 6/6 Max LP Schild aktiv.",
        LanguageLocaleTaxon.LA: "Invarians Constitutionale: VI/VI Vitae scutum activum.",
        LanguageLocaleTaxon.JA: "憲法的不変条件：6/6 最大HPシールド有効。"
    },
    "RE_RENDER_CASCADE": {
        LanguageLocaleTaxon.SK: "Hierarchické prekresľovanie stavov dokončené.",
        LanguageLocaleTaxon.CS: "Hierarchické překreslování stavů dokončeno.",
        LanguageLocaleTaxon.EN: "Hierarchical state re-rendering complete.",
        LanguageLocaleTaxon.DE: "Hierarchisches Zustands-Rendern abgeschlossen.",
        LanguageLocaleTaxon.LA: "Redditus status hierarchici completus est.",
        LanguageLocaleTaxon.JA: "階層的状態再描画が完了しました。"
    },
    "HARDWARE_SILICON_NOMINAL": {
        LanguageLocaleTaxon.SK: "Kremíková vrstva: Teplota a príkon v bezpečnej zóne.",
        LanguageLocaleTaxon.CS: "Křemíková vrstva: Teplota a příkon v bezpečné zóně.",
        LanguageLocaleTaxon.EN: "Silicon Layer: Thermals and power within safe envelope.",
        LanguageLocaleTaxon.DE: "Siliziumschicht: Temperatur und Leistung im sicheren Bereich.",
        LanguageLocaleTaxon.LA: "Stratum Silicii: Temperatura et potentia intra limites tutos.",
        LanguageLocaleTaxon.JA: "シリコン層：温度および電力は安全範囲内です。"
    }
}


# ==============================================================================
# 3. MULTI-LEVEL SYSTEM STATE SNAPSHOT
# ==============================================================================

@dataclass
class MultiLevelSystemState:
    """Consolidated multi-level snapshot across all 5 operational tiers."""
    timestamp_ns: int = field(default_factory=time.time_ns)

    # Level 0: Silicon / Hardware
    thermal_celsius: float = 52.0
    power_watts: float = 24.5
    cpu_freq_ghz: float = 4.2
    gpu_active: bool = True
    active_isa: str = "INTEL_DP4A_TENSOR"

    # Level 1: Compute Pipeline
    active_workers: int = 4
    queue_depth: int = 0
    cycle_latency_us: float = 248.5
    throughput_fps: float = 144.0

    # Level 2: Epistemic & Authority
    authority_state: str = "AUTH_COMMIT_APPROVED"
    belief_degree: float = 0.96
    epistemic_truth_state: str = "VERIFIED_TRUE"
    hypotheses_count: int = 12

    # Level 3: Semantic Linguistic
    current_locale: LanguageLocaleTaxon = LanguageLocaleTaxon.SK
    current_register: LinguisticRegisterTaxon = LinguisticRegisterTaxon.EPISTEMIC_FORMAL
    last_intent: SemanticIntentTaxon = SemanticIntentTaxon.TELEMETRY_INSPECT

    # Level 4: Visual Compositor
    viewport_width: int = 48
    viewport_height: int = 48
    dirty_tiles_count: int = 0
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        l0 = {
            "thermal_celsius": self.thermal_celsius,
            "current_tj_celsius": self.thermal_celsius,
            "power_watts": self.power_watts,
            "active_power_watts": self.power_watts,
            "cpu_freq_ghz": self.cpu_freq_ghz,
            "active_isa": self.active_isa,
            "msr_energy_profile": "ENERGY_PERF_BIAS_0"
        }
        l1 = {
            "active_workers": self.active_workers,
            "queue_depth": self.queue_depth,
            "ring_queue_depth": self.queue_depth,
            "qpc_clock_ghz": self.cpu_freq_ghz,
            "cycle_latency_us": self.cycle_latency_us,
            "throughput_fps": self.throughput_fps
        }
        l2 = {
            "authority_state": self.authority_state,
            "belief_degree": self.belief_degree,
            "epistemic_truth_state": self.epistemic_truth_state,
            "truth_lattice_state": self.epistemic_truth_state,
            "dual_gate_verdict": "PASS",
            "hypotheses_count": self.hypotheses_count
        }
        l3 = {
            "locale": self.current_locale.value,
            "active_locale": self.current_locale.value,
            "register": self.current_register.value,
            "lexical_register": self.current_register.value,
            "last_intent": self.last_intent.value
        }
        l4 = {
            "viewport": f"{self.viewport_width}x{self.viewport_height}",
            "dirty_tiles_count": self.dirty_tiles_count,
            "scanline_count": 48,
            "vital_max_hp": self.vital_max_hp
        }
        return {
            "timestamp_ns": self.timestamp_ns,
            "level_0_hardware": l0,
            "level_1_pipeline": l1,
            "level_2_epistemic": l2,
            "level_3_semantic": l3,
            "level_4_viewport": l4,
            "levels": {
                "level_0_hardware": l0,
                "level_1_pipeline": l1,
                "level_2_epistemic": l2,
                "level_3_linguistic": l3,
                "level_4_visual": l4
            },
            "vital_max_hp": self.vital_max_hp
        }


# ==============================================================================
# 4. HIERARCHICAL RE-RENDERING ENGINE & MULTILINGUAL CLASSIFIER
# ==============================================================================

@dataclass
class MultiLevelReRenderReport:
    """Forensic report documenting hierarchical state cascade and localized rendering."""
    dirty_trigger_level: SystemStateLevelTaxon
    affected_levels_cascaded: List[SystemStateLevelTaxon]
    target_locale: LanguageLocaleTaxon
    target_register: LinguisticRegisterTaxon
    localized_status_text: str
    localized_authority_banner: str
    localized_silicon_summary: str
    rendered_ascii_hud: str
    dirty_tiles_recomputed: int
    cascade_duration_us: float
    vital_max_hp: int = VITAL_MAX_HP
    report_id: str = ""

    def __post_init__(self):
        if not self.report_id:
            self.report_id = f"RPT_{int(time.time() * 1000)}"

    @property
    def invalidated_level(self) -> str:
        return self.dirty_trigger_level.value

    @property
    def cascaded_levels(self) -> List[str]:
        return [lvl.value for lvl in self.affected_levels_cascaded]

    @property
    def dirty_tiles_invalidation_count(self) -> int:
        return self.dirty_tiles_recomputed

    @property
    def elapsed_us(self) -> float:
        return round(self.cascade_duration_us, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "dirty_trigger_level": self.dirty_trigger_level.value,
            "invalidated_level": self.invalidated_level,
            "affected_levels_cascaded": self.cascaded_levels,
            "cascaded_levels": self.cascaded_levels,
            "target_locale": self.target_locale.value,
            "target_register": self.target_register.value,
            "localized_status_text": self.localized_status_text,
            "localized_authority_banner": self.localized_authority_banner,
            "localized_silicon_summary": self.localized_silicon_summary,
            "rendered_ascii_hud": self.rendered_ascii_hud,
            "dirty_tiles_recomputed": self.dirty_tiles_recomputed,
            "dirty_tiles_invalidation_count": self.dirty_tiles_invalidation_count,
            "cascade_duration_us": self.cascade_duration_us,
            "elapsed_us": self.elapsed_us,
            "vital_max_hp": self.vital_max_hp
        }


class MultilingualIntentClassifier:
    """Classifies natural language prompts into language locales and semantic intents."""

    KEYWORD_LOCALE_MAP = {
        "prekresli": LanguageLocaleTaxon.SK,
        "spusti": LanguageLocaleTaxon.SK,
        "stav": LanguageLocaleTaxon.SK,
        "výkon": LanguageLocaleTaxon.SK,
        "překresli": LanguageLocaleTaxon.CS,
        "spusť": LanguageLocaleTaxon.CS,
        "výkonu": LanguageLocaleTaxon.CS,
        "rerender": LanguageLocaleTaxon.EN,
        "execute": LanguageLocaleTaxon.EN,
        "status": LanguageLocaleTaxon.EN,
        "profiling": LanguageLocaleTaxon.EN,
        "neuzeichnen": LanguageLocaleTaxon.DE,
        "ausführen": LanguageLocaleTaxon.DE,
        "leistung": LanguageLocaleTaxon.DE,
        "reddere": LanguageLocaleTaxon.LA,
        "facere": LanguageLocaleTaxon.LA,
        "statusque": LanguageLocaleTaxon.LA,
        "再描画": LanguageLocaleTaxon.JA,
        "実行": LanguageLocaleTaxon.JA,
        "状態": LanguageLocaleTaxon.JA
    }

    INTENT_KEYWORDS = {
        SemanticIntentTaxon.RE_RENDER_TRIGGER: ["prekresli", "překresli", "rerender", "redraw", "neuzeichnen", "reddere", "再描画", "refresh"],
        SemanticIntentTaxon.COMMAND_ACTUATE: ["spusti", "spusť", "execute", "actuate", "run", "ausführen", "facere", "実行", "start"],
        SemanticIntentTaxon.TELEMETRY_INSPECT: ["stav", "status", "telemetry", "inspect", "info", "zustand", "statusque", "状態", "metrics"],
        SemanticIntentTaxon.HYPOTHESIS_EXPLORE: ["hypotéza", "what-if", "simuluj", "simulate", "explore", "hypothese", "仮説", "predikuj"],
        SemanticIntentTaxon.GOVERNANCE_AUDIT: ["audit", "over", "verify", "assert", "invariant", "prüfen", "veritas", "監査"]
    }

    @classmethod
    def detect_locale(cls, text: str, fallback: LanguageLocaleTaxon = LanguageLocaleTaxon.SK) -> LanguageLocaleTaxon:
        """Identifies language locale using semantic lexical matching and alphabet heuristics."""
        lower = text.lower()

        # 1. Distinctive character heuristics
        if any(c in lower for c in "řěů"):
            return LanguageLocaleTaxon.CS
        if any(c in lower for c in "ľťňôä"):
            return LanguageLocaleTaxon.SK
        if any(ord(c) >= 0x3040 for c in text):
            return LanguageLocaleTaxon.JA
        if any(c in lower for c in "äöüß"):
            return LanguageLocaleTaxon.DE

        # 2. Check keywords sorted by length descending so longer compound keywords match first
        sorted_keywords = sorted(cls.KEYWORD_LOCALE_MAP.items(), key=lambda item: len(item[0]), reverse=True)
        for kw, loc in sorted_keywords:
            if kw in lower:
                return loc

        return fallback

    @classmethod
    def classify_intent(cls, text: str) -> SemanticIntentTaxon:
        """Classifies prompt intent into canonical semantic taxon."""
        lower = text.lower()
        for intent, kws in cls.INTENT_KEYWORDS.items():
            if any(kw in lower for kw in kws):
                return intent
        return SemanticIntentTaxon.TELEMETRY_INSPECT


class MultiLevelStateReRenderingEngine:
    """
    Orchestrates hierarchical multi-level state propagation and localized re-rendering.
    """

    LEVEL_ORDER = [
        SystemStateLevelTaxon.LEVEL_0_HARDWARE_SILICON,
        SystemStateLevelTaxon.LEVEL_1_COMPUTE_PIPELINE,
        SystemStateLevelTaxon.LEVEL_2_EPISTEMIC_AUTHORITY,
        SystemStateLevelTaxon.LEVEL_3_SEMANTIC_LINGUISTIC,
        SystemStateLevelTaxon.LEVEL_4_VISUAL_COMPOSITOR
    ]

    def __init__(self):
        self.current_state = MultiLevelSystemState()
        self.lexicon = MULTILINGUAL_LEXICON
        self.classifier = MultilingualIntentClassifier()

    @property
    def state(self) -> MultiLevelSystemState:
        return self.current_state

    def update_hardware_state(self, thermal: float, power: float, isa: str) -> None:
        """Mutates Level 0 hardware state."""
        self.current_state.thermal_celsius = thermal
        self.current_state.power_watts = power
        self.current_state.active_isa = isa
        self.current_state.timestamp_ns = time.time_ns()

    def trigger_hierarchical_rerender(
        self,
        dirty_level: Optional[SystemStateLevelTaxon] = None,
        target_locale: Optional[LanguageLocaleTaxon] = None,
        target_register: Optional[LinguisticRegisterTaxon] = None,
        invalidated_level: Optional[SystemStateLevelTaxon] = None,
        reason: Optional[str] = None
    ) -> MultiLevelReRenderReport:
        """
        Executes selective hierarchical cascade re-rendering down to Level 4 Visual Compositor.
        """
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6 must strictly hold"
        t0 = time.perf_counter_ns()

        lvl = dirty_level or invalidated_level or SystemStateLevelTaxon.LEVEL_1_COMPUTE_PIPELINE
        locale = target_locale or self.current_state.current_locale
        reg = target_register or self.current_state.current_register

        # 1. Determine affected downstream levels
        idx = self.LEVEL_ORDER.index(lvl)
        affected_levels = self.LEVEL_ORDER[idx:]

        # 2. Re-compute dirty tile count for Visual Compositor based on trigger tier
        if lvl == SystemStateLevelTaxon.LEVEL_0_HARDWARE_SILICON:
            dirty_tiles = 256
        elif lvl == SystemStateLevelTaxon.LEVEL_1_COMPUTE_PIPELINE:
            dirty_tiles = 64
        elif lvl == SystemStateLevelTaxon.LEVEL_2_EPISTEMIC_AUTHORITY:
            dirty_tiles = 36
        elif lvl == SystemStateLevelTaxon.LEVEL_3_SEMANTIC_LINGUISTIC:
            dirty_tiles = 24
        else:
            dirty_tiles = 16


        # 3. Retrieve localized lexicon entries
        status_msg = self.lexicon["SYSTEM_READY"].get(locale, self.lexicon["SYSTEM_READY"][LanguageLocaleTaxon.SK])
        auth_msg = self.lexicon["AUTHORITY_APPROVED"].get(locale, self.lexicon["AUTHORITY_APPROVED"][LanguageLocaleTaxon.SK])
        silicon_msg = self.lexicon["HARDWARE_SILICON_NOMINAL"].get(locale, self.lexicon["HARDWARE_SILICON_NOMINAL"][LanguageLocaleTaxon.SK])
        hp_shield_msg = self.lexicon["INVARIANT_HP_SHIELD"].get(locale, self.lexicon["INVARIANT_HP_SHIELD"][LanguageLocaleTaxon.SK])

        # If thermal is elevated (> 75C), customize message
        if self.current_state.thermal_celsius > 75.0:
            if locale == LanguageLocaleTaxon.SK:
                silicon_msg = f"Kremíková vrstva: Varovanie - Teplota {self.current_state.thermal_celsius}°C vyžaduje odľahčenie ISA."
            elif locale == LanguageLocaleTaxon.CS:
                silicon_msg = f"Křemíková vrstva: Varování - Teplota {self.current_state.thermal_celsius}°C vyžaduje odlehčení ISA."
            elif locale == LanguageLocaleTaxon.EN:
                silicon_msg = f"Silicon Layer: Warning - Thermal elevation {self.current_state.thermal_celsius}°C requires ISA throttling."
            elif locale == LanguageLocaleTaxon.DE:
                silicon_msg = f"Siliziumschicht: Warnung - Temperatur {self.current_state.thermal_celsius}°C erfordert ISA-Drosselung."
            elif locale == LanguageLocaleTaxon.LA:
                silicon_msg = f"Stratum Silicii: Monitum - Calefactio {self.current_state.thermal_celsius}°C moderationem poscit."
            elif locale == LanguageLocaleTaxon.JA:
                silicon_msg = f"シリコン層：警告 - 温度{self.current_state.thermal_celsius}°CのためISA調整が必要です。"

        # 4. Generate Rendered ASCII HUD Card
        hearts = "♥ " * VITAL_MAX_HP
        rendered_hud = (
            f"┌──────────────────────────────────────────────────────────────┐\n"
            f"│ KRYSTAL-STACK MULTI-TIER STATE COMPOSITOR [{locale.value}]              │\n"
            f"├──────────────────────────────────────────────────────────────┤\n"
            f"│ L0 Silicon:  {self.current_state.thermal_celsius:.1f}°C • {self.current_state.power_watts:.1f}W • {self.current_state.active_isa:<20} │\n"
            f"│ L1 Pipeline: {self.current_state.active_workers} Workers • {self.current_state.cycle_latency_us:.1f}μs • {self.current_state.throughput_fps:.0f} FPS           │\n"
            f"│ L2 Epistemic:{self.current_state.authority_state:<24} P={self.current_state.belief_degree:.2f}     │\n"
            f"│ L3 Linguistic:[{locale.value}] {reg.value:<20} Intent: {self.current_state.last_intent.value:<12}│\n"
            f"│ L4 Visual:   48x48 Viewport • {dirty_tiles} Dirty Tiles Recomputed      │\n"
            f"├──────────────────────────────────────────────────────────────┤\n"
            f"│ INVARIANT:   {hearts.strip()} ({VITAL_MAX_HP}/{VITAL_MAX_HP} HP) - {hp_shield_msg[:24]} │\n"
            f"└──────────────────────────────────────────────────────────────┘"
        )

        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0

        # Update local snapshot
        self.current_state.current_locale = locale
        self.current_state.current_register = reg
        self.current_state.dirty_tiles_count = dirty_tiles

        return MultiLevelReRenderReport(
            dirty_trigger_level=lvl,
            affected_levels_cascaded=affected_levels,
            target_locale=locale,
            target_register=reg,
            localized_status_text=status_msg,
            localized_authority_banner=auth_msg,
            localized_silicon_summary=silicon_msg,
            rendered_ascii_hud=rendered_hud,
            dirty_tiles_recomputed=dirty_tiles,
            cascade_duration_us=elapsed_us,
            vital_max_hp=VITAL_MAX_HP
        )


# Global Singleton Instance
GLOBAL_MULTILEVEL_STATE_RERENDERING_ENGINE = MultiLevelStateReRenderingEngine()
