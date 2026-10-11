#!/usr/bin/env python3
"""
KRYSTAL-STACK // MULTI-DISCIPLINARY MEASUREMENT & WEIGHT SUMMARIZER
=============================================================================
Universal radar / multi-axial aspect & weight summarizer inspired by the
Poslední Kmen & Krystal-Stack measurement paradigm.

Transforms high-dimensional empirical observations across diverse domains into:
  1. Multi-axial Aspect Radar Vectors (OBT, KON, POŠ, DOS / Latency, Quality, etc.)
  2. Normalized Attribute Weight Profiles (0.0 - 5.0 scale, sum(w_i) == 1.0)
  3. Domain-Specific Component Distribution Breakdown (e.g., Akcie vs. Pomocníci,
     Compute vs. I/O, Dynamic vs. Boundary safety)
  4. Composite Weighted Quality/Performance Index

Supported Disciplines:
  - CYBER_PHYSICAL_CNC       : Spindle thermal gradient, dynamic RPM, cutting load, collapse reserve
  - CRYSTALSEC_CAUSAL_LLM    : Latency (TTFT+ITL), throughput (TPS), KV pressure, robustness
  - SOVEREIGN_KERNEL         : Health vector H, audit integrity, resource quota, flow control
  - GAME_WORLD_SIMULATION    : Poslední Kmen (Difficulty, Control, Damage, Range, Deck Distribution)

Invariant:
  VITAL_MAX_HP == 6 strictly enforced.
=============================================================================
"""

import os
import sys
import math
import uuid
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP


class MeasurementDiscipline(str, Enum):
    GAME_WORLD_SIMULATION = "GAME_WORLD_SIMULATION"      # Poslední Kmen: OBT, KON, POŠ, DOS
    CYBER_PHYSICAL_CNC = "CYBER_PHYSICAL_CNC"            # Spindle & Thermal Dynamics
    CRYSTALSEC_CAUSAL_LLM = "CRYSTALSEC_CAUSAL_LLM"      # Causal LLM & OpenVINO Execution Plane
    SOVEREIGN_KERNEL = "SOVEREIGN_KERNEL"                # Linux Dispatcher & Telemetry Fabric


@dataclass
class MeasurementAspect:
    code: str                   # e.g., "OBT", "KON", "POŠ", "DOS" or "LAT", "TPS"
    name: str                   # Full name, e.g., "Obtížnost", "Kontrola", "Latencia"
    normalized_score: float     # [0.0 - 5.0] scale (matching the 5-point bar scale)
    relative_weight: float      # [0.0 - 1.0], sum of weights in subject == 1.0
    unit: str = "pts"           # e.g., "pts", "ms", "°C", "tps", "%"
    raw_observed_value: Optional[float] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "normalized_score": round(self.normalized_score, 2),
            "relative_weight": round(self.relative_weight, 3),
            "unit": self.unit,
            "raw_observed_value": round(self.raw_observed_value, 2) if self.raw_observed_value is not None else None,
            "description": self.description
        }


@dataclass
class ComponentCategoryRatio:
    category_a_name: str        # e.g. "Akce", "Compute", "Dynamic"
    category_a_count: int
    category_a_pct: float
    category_b_name: str        # e.g. "Pomocníci", "I/O", "Boundary"
    category_b_count: int
    category_b_pct: float
    total_items: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category_a": {
                "name": self.category_a_name,
                "count": self.category_a_count,
                "pct": round(self.category_a_pct, 1)
            },
            "category_b": {
                "name": self.category_b_name,
                "count": self.category_b_count,
                "pct": round(self.category_b_pct, 1)
            },
            "total_items": self.total_items
        }


@dataclass
class MeasurementSubjectProfile:
    subject_id: str
    discipline: MeasurementDiscipline
    title: str                  # e.g. "CRYSTAL (Vládci Mrazu)", "CNC Spindle 24k RPM", "OpenVINO FP16"
    sub_title: str
    narrative_summary: str
    aspects: List[MeasurementAspect]
    composition_ratio: ComponentCategoryRatio
    vital_hp: int = VITAL_MAX_HP
    conduct_badge: str = "[ACT] [!LCK] [CNF] [!HAZ] [VIT 6HP]"

    @property
    def weighted_composite_index(self) -> float:
        """Computes weighted aggregate score: sum(score_i * weight_i) on a 0 - 5 scale."""
        total_w = sum(a.relative_weight for a in self.aspects) or 1.0
        weighted_sum = sum(a.normalized_score * a.relative_weight for a in self.aspects)
        return round(weighted_sum / total_w, 2)

    def compute_radar_polygon_coordinates(self, radius: float = 100.0, center_x: float = 120.0, center_y: float = 120.0) -> List[Tuple[float, float]]:
        """
        Calculates 2D SVG polygon vertices for the radar chart.
        Axes are placed uniformly on a circle starting from top (angle = -pi/2).
        Scores [0.0 - 5.0] map to [0.0 - radius].
        """
        num_axes = len(self.aspects)
        if num_axes == 0:
            return []

        coords = []
        angle_step = (2.0 * math.pi) / num_axes
        # Start at top (-pi/2)
        base_angle = -math.pi / 2.0

        for i, asp in enumerate(self.aspects):
            angle = base_angle + i * angle_step
            # Fraction of full scale (max 5.0)
            fraction = min(1.0, max(0.0, asp.normalized_score / 5.0))
            r = fraction * radius
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            coords.append((round(x, 1), round(y, 1)))

        return coords

    def to_dict(self) -> Dict[str, Any]:
        polygon = self.compute_radar_polygon_coordinates()
        return {
            "subject_id": self.subject_id,
            "discipline": self.discipline.value,
            "title": self.title,
            "sub_title": self.sub_title,
            "narrative_summary": self.narrative_summary,
            "aspects": [a.to_dict() for a in self.aspects],
            "composition_ratio": self.composition_ratio.to_dict(),
            "weighted_composite_index": self.weighted_composite_index,
            "radar_polygon_svg": " ".join(f"{x},{y}" for x, y in polygon),
            "vital_hp": self.vital_hp,
            "conduct_badge": self.conduct_badge
        }


class MultiDisciplinaryMeasurementSummarizer:
    """Master registry and synthesizer for multi-axial measurements across domains."""

    def __init__(self):
        self.profiles: Dict[str, MeasurementSubjectProfile] = {}
        self._seed_archetype_profiles()

    def _seed_archetype_profiles(self):
        """Initializes canonical measurement subjects representing all key disciplines."""
        # -------------------------------------------------------------
        # 1. POSLEDNÍ KMEN GAME WORLD // CRYSTAL (VLÁDCI MRAZU)
        # Directly derived from the user reference image (media_1791532588669.png)
        # -------------------------------------------------------------
        crystal_profile = MeasurementSubjectProfile(
            subject_id="PK-CRYSTAL",
            discipline=MeasurementDiscipline.GAME_WORLD_SIMULATION,
            title="CRYSTAL",
            sub_title="VLÁDCI MRAZU // SEVERNÍ ŠTÍTY · KAPITOLA I",
            narrative_summary="Z hlubin zamrzlých štítů přichází kmen, jehož magické krystaly dokáží zpomalit čas a zakořenit i nejrychlejšího protivníka. Jejich královna velí bouřím.",
            aspects=[
                MeasurementAspect(code="OBT", name="Obtížnost", normalized_score=4.0, relative_weight=0.25, description="Vysoké nároky na precizní plánování tahů a časování zmrazení."),
                MeasurementAspect(code="KON", name="Kontrola", normalized_score=5.0, relative_weight=0.35, description="Absolutní dominance v crowd-control a manipulaci s časem."),
                MeasurementAspect(code="POŠ", name="Poškození", normalized_score=3.0, relative_weight=0.15, description="Střední kontinuální poškození prostřednictvím mrazových střepů."),
                MeasurementAspect(code="DOS", name="Dosah", normalized_score=4.0, relative_weight=0.25, description="Vynikající dosah projektilů a plošných ledových bouří.")
            ],
            composition_ratio=ComponentCategoryRatio(
                category_a_name="Akce",
                category_a_count=16,
                category_a_pct=64.0,
                category_b_name="Pomocníci",
                category_b_count=9,
                category_b_pct=36.0,
                total_items=25
            )
        )
        self.profiles[crystal_profile.subject_id] = crystal_profile

        # TOXIC (Hnijící slatiny)
        toxic_profile = MeasurementSubjectProfile(
            subject_id="PK-TOXIC",
            discipline=MeasurementDiscipline.GAME_WORLD_SIMULATION,
            title="TOXIC",
            sub_title="HNĚV MOČÁLŮ // HNIJÍCÍ SLATINY · KAPITOLA II",
            narrative_summary="Zamořené slatiny plné jedovatých spor a kyselých roztoků. Pomalá eroze životů protivníka doprovázená masivním plošným oslabováním.",
            aspects=[
                MeasurementAspect(code="OBT", name="Obtížnost", normalized_score=3.0, relative_weight=0.20, description="Intuitivní aplikace stacků jedu a debuffů."),
                MeasurementAspect(code="KON", name="Kontrola", normalized_score=3.0, relative_weight=0.25, description="Spomalení a rozleptání brnění nepřátel."),
                MeasurementAspect(code="POŠ", name="Poškození", normalized_score=5.0, relative_weight=0.35, description="Devastující kumulativní poškození v čase (DoT)."),
                MeasurementAspect(code="DOS", name="Dosah", normalized_score=3.0, relative_weight=0.20, description="Střední dosah aerosolu a bažinných výparů.")
            ],
            composition_ratio=ComponentCategoryRatio(
                category_a_name="Akce",
                category_a_count=18,
                category_a_pct=72.0,
                category_b_name="Pomocníci",
                category_b_count=7,
                category_b_pct=28.0,
                total_items=25
            )
        )
        self.profiles[toxic_profile.subject_id] = toxic_profile

        # DRUID (Pradávný les)
        druid_profile = MeasurementSubjectProfile(
            subject_id="PK-DRUID",
            discipline=MeasurementDiscipline.GAME_WORLD_SIMULATION,
            title="DRUID",
            sub_title="OCHRÁNCI HVOZDU // PRADÁVNÝ LES · KAPITOLA III",
            narrative_summary="Strážci pradávné biosféry čerpající sílu z kořenů a zvířecích totemů. Schopnost regenerace, přeměny a vyvolávání mocných lesních spojenců.",
            aspects=[
                MeasurementAspect(code="OBT", name="Obtížnost", normalized_score=4.0, relative_weight=0.25, description="Komplexní správa esencí a rotace totemů."),
                MeasurementAspect(code="KON", name="Kontrola", normalized_score=4.0, relative_weight=0.25, description="Zakořenění a manipulace s terénem bojiště."),
                MeasurementAspect(code="POŠ", name="Poškození", normalized_score=3.0, relative_weight=0.20, description="Fyzické a přírodní výpady šelem a větví."),
                MeasurementAspect(code="DOS", name="Dosah", normalized_score=3.0, relative_weight=0.30, description="Kombinace kontaktního boje medvěda s dálkovým hvozdem.")
            ],
            composition_ratio=ComponentCategoryRatio(
                category_a_name="Akce",
                category_a_count=11,
                category_a_pct=44.0,
                category_b_name="Pomocníci",
                category_b_count=14,
                category_b_pct=56.0,
                total_items=25
            )
        )
        self.profiles[druid_profile.subject_id] = druid_profile

        # -------------------------------------------------------------
        # 2. CYBER-PHYSICAL CNC // HIGH-SPEED SPINDLE THERMAL PROFILE
        # -------------------------------------------------------------
        cnc_spindle_profile = MeasurementSubjectProfile(
            subject_id="CNC-SPINDLE-24K",
            discipline=MeasurementDiscipline.CYBER_PHYSICAL_CNC,
            title="CNC SPINDLE 24K RPM",
            sub_title="THERMAL DYNAMICS // LINEAR REDUCTION & COLLAPSE DIFFERENTIAL",
            narrative_summary="Vysokootáčkové elektrovreteno s adaptívnym riadením otáčok podľa tepelného gradientu a dynamického diferenciálu zaťaženia.",
            aspects=[
                MeasurementAspect(code="GRAD", name="Tepelný gradient", normalized_score=4.5, relative_weight=0.35, unit="°C/s", raw_observed_value=2.4, description="dT/dt odozva ložísk pri zrýchlení z 12k na 24k RPM."),
                MeasurementAspect(code="DRPM", name="Dynamické RPM", normalized_score=4.8, relative_weight=0.25, unit="kRPM", raw_observed_value=22.8, description="Schopnosť kompenzovať pokles otáčok pri vstupe frézy do materiálu."),
                MeasurementAspect(code="ODP", name="Rezný odpor", normalized_score=3.2, relative_weight=0.20, unit="N", raw_observed_value=480.0, description="Normálové a tangenciálne zložky reznej sily na ostrie."),
                MeasurementAspect(code="REZ", name="Rezerva kolapsu", normalized_score=4.2, relative_weight=0.20, unit="pts", raw_observed_value=84.0, description="Odstup od kritickej teploty zmäknutia maziva a deštrukcie vretena.")
            ],
            composition_ratio=ComponentCategoryRatio(
                category_a_name="Dynamické riadenie",
                category_a_count=68,
                category_a_pct=68.0,
                category_b_name="Bezpečnostné poistky",
                category_b_count=32,
                category_b_pct=32.0,
                total_items=100
            )
        )
        self.profiles[cnc_spindle_profile.subject_id] = cnc_spindle_profile

        # -------------------------------------------------------------
        # 3. CRYSTALSEC 2.0 // OPENVINO CAUSAL LLM EXECUTION PLANE
        # -------------------------------------------------------------
        llm_profile = MeasurementSubjectProfile(
            subject_id="CRYSTALSEC-OPENVINO-GPU",
            discipline=MeasurementDiscipline.CRYSTALSEC_CAUSAL_LLM,
            title="OPENVINO INTEL GPU RUNTIME",
            sub_title="CAUSAL LLM EXECUTION PLANE // 8-PHASE FOOTPRINT",
            narrative_summary="Causal runtime modelujúci životný cyklus promptu, KV-cache alokáciu, Time-to-First-Token a tepelnú disipáciu integrovaného GPU i915.",
            aspects=[
                MeasurementAspect(code="LAT", name="Latencia (TTFT/ITL)", normalized_score=4.6, relative_weight=0.30, unit="ms", raw_observed_value=24.5, description="Time to first token 24.5 ms a inter-token latency 6.2 ms."),
                MeasurementAspect(code="TPS", name="Priepustnosť", normalized_score=4.2, relative_weight=0.25, unit="tps", raw_observed_value=161.2, description="Efektívna priepustnosť pri batch_size=1 a 4 OpenVINO streamoch."),
                MeasurementAspect(code="MEM", name="Pamäťový tlak", normalized_score=3.8, relative_weight=0.25, unit="%", raw_observed_value=28.5, description="Saturácia KV-cache a systémovej RAM bez page-faultov."),
                MeasurementAspect(code="ROB", name="Robustnosť (Zero Throttle)", normalized_score=4.9, relative_weight=0.20, unit="pts", raw_observed_value=98.0, description="Stabilita bez throttlingu s plnou deterministickou overiteľnosťou.")
            ],
            composition_ratio=ComponentCategoryRatio(
                category_a_name="Generatívne fázy",
                category_a_count=65,
                category_a_pct=65.0,
                category_b_name="Validačná & Telemetrická vrstva",
                category_b_count=35,
                category_b_pct=35.0,
                total_items=100
            )
        )
        self.profiles[llm_profile.subject_id] = llm_profile

        # -------------------------------------------------------------
        # 4. SOVEREIGN SYSTEM INTEGRITY // KRYSTAL LINUX DISPATCHER
        # -------------------------------------------------------------
        kernel_profile = MeasurementSubjectProfile(
            subject_id="KRYSTAL-SOVEREIGN-KERNEL",
            discipline=MeasurementDiscipline.SOVEREIGN_KERNEL,
            title="SOVEREIGN ARCH LINUX KERNEL",
            sub_title="COMMUNICATION INTEGRITY & FORENSIC LEDGER PLANE",
            narrative_summary="Základná platformová vrstva garantujúca nemennosť systémových invariantov, deterministickú validáciu kontraktov a auditovateľný stavový vektor H.",
            aspects=[
                MeasurementAspect(code="STV", name="Integrita stavu H", normalized_score=5.0, relative_weight=0.35, unit="pts", raw_observed_value=6.0, description="VITAL_MAX_HP == 6 prísne dodržané na všetkých kontrolných bodoch."),
                MeasurementAspect(code="AUD", name="Audit ledger", normalized_score=4.8, relative_weight=0.25, unit="pts", raw_observed_value=100.0, description="Kryptografické reťazenie auditných záznamov bez možnosti prepísania."),
                MeasurementAspect(code="KVT", name="Dodržiavanie kvót", normalized_score=4.5, relative_weight=0.20, unit="%", raw_observed_value=92.0, description="Presné dodržiavanie CPU CFS kvót a pamäťových limitov cgroups v2."),
                MeasurementAspect(code="TOK", name="Kontrola tokov", normalized_score=4.7, relative_weight=0.20, unit="pts", raw_observed_value=96.0, description="Nulový únik dát mimo deklarovaný Communication Integrity Plane.")
            ],
            composition_ratio=ComponentCategoryRatio(
                category_a_name="Deterministická exekúcia",
                category_a_count=80,
                category_a_pct=80.0,
                category_b_name="Forenzná telemetria",
                category_b_count=20,
                category_b_pct=20.0,
                total_items=100
            )
        )
        self.profiles[kernel_profile.subject_id] = kernel_profile

    def register_profile(self, profile: MeasurementSubjectProfile) -> None:
        """Registers or updates a measurement subject profile."""
        self.profiles[profile.subject_id] = profile

    def get_profile(self, subject_id: str) -> Optional[MeasurementSubjectProfile]:
        return self.profiles.get(subject_id)

    def list_profiles_by_discipline(self, discipline: Optional[MeasurementDiscipline] = None) -> List[MeasurementSubjectProfile]:
        if discipline:
            return [p for p in self.profiles.values() if p.discipline == discipline]
        return list(self.profiles.values())

    def generate_discipline_summary_report(self) -> Dict[str, Any]:
        """Summarizes all measurement profiles, weights, and distributions across disciplines."""
        grouped = {}
        for disc in MeasurementDiscipline:
            items = self.list_profiles_by_discipline(disc)
            grouped[disc.value] = {
                "count": len(items),
                "subjects": [p.to_dict() for p in items]
            }

        return {
            "status": "OK",
            "disciplines_count": len(MeasurementDiscipline),
            "total_profiles_count": len(self.profiles),
            "vital_hp_invariant": VITAL_MAX_HP,
            "disciplines": grouped
        }


GLOBAL_MEASUREMENT_SUMMARIZER = MultiDisciplinaryMeasurementSummarizer()
