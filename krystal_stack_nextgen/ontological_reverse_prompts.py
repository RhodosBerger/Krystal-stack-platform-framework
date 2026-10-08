#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK NEXTGEN: ONTOLOGICAL REVERSE-ENGINEERING PROMPT ENGINE
==============================================================================
Applies Formal Ontological Programming, Category Theory, and Teleological
Inversion to generate deep reverse-engineering prompts that shift perspectives:
  - Inverting hardware symptoms (misses, power, watts) into generative software intents.
  - Functorial translation across domains (Graphics <-> Finance <-> Robotics <-> Genomics).
  - Deconstructing the 1500x NPU pre-staging memory illusion.
  - Ontological formalization of the VITAL_MAX_HP = 6 invariant.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_stack_nextgen.iris_xe_kisak_optimizer import VITAL_MAX_HP


class OntologicalDomain(Enum):
    EPISTEMIC_HARDWARE_INVERSION = "EPISTEMIC_HARDWARE_INVERSION"
    CATEGORY_THEORETIC_FUNCTORS = "CATEGORY_THEORETIC_FUNCTORS"
    TELEOLOGICAL_MEMORY_ILLUSION = "TELEOLOGICAL_MEMORY_ILLUSION"
    TPU_SYSTOLIC_TENSOR_ONTOLOGY = "TPU_SYSTOLIC_TENSOR_ONTOLOGY"
    AXIOMATIC_INVARIANT_CONSERVATION = "AXIOMATIC_INVARIANT_CONSERVATION"


@dataclass
class OntologicalPrompt:
    prompt_id: str
    domain: OntologicalDomain
    title: str
    perspective_shift: str
    ontological_axioms: List[str]
    reverse_engineering_directive_sk: str
    reverse_engineering_directive_en: str
    expected_output_structure: str
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["domain"] = self.domain.value
        return d


class OntologicalReverseEngineeringEngine:
    """
    Synthesizes reverse-engineering prompts rooted in formal ontology,
    enabling engineering teams and cognitive models to analyze complex
    subsystems from inverted, non-orthodox viewpoints.
    """

    def __init__(self):
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP must remain 6"
        self.vital_hp = VITAL_MAX_HP
        self.catalog = self._build_catalog()

    def _build_catalog(self) -> List[OntologicalPrompt]:
        prompts = [
            # ── 1. Epistemic Hardware Inversion ────────────────────────
            OntologicalPrompt(
                prompt_id="ONTO-REV-001",
                domain=OntologicalDomain.EPISTEMIC_HARDWARE_INVERSION,
                title="Inverzia hardvérovej telemetrie na generatívny zámer softvéru",
                perspective_shift=(
                    "Namiesto toho, aby softvér pasívne prijímal obmedzenia procesora, "
                    "chápeme hardvérové signály (L3 miss rates, RAPL watty, C-states, ETW logy) "
                    "ako fyzikálne symptómy, z ktorých spätne dedukujeme skrytý algoritmus."
                ),
                ontological_axioms=[
                    "Axiom I: Každý stall cyklus procesora je zlyhaním časopriestorovej projekcie dát.",
                    "Axiom II: Spotreba energie (Watty) je priamym zrkadlom sémantického rozptylu inštrukcií.",
                    "Axiom III: VITAL_MAX_HP = 6 definuje nezničiteľný homeostatický invariant."
                ],
                reverse_engineering_directive_sk=(
                    "Prijmi rolu ontologického reverzného inžiniera. Analyzuj telemetrický záznam Windows "
                    "(GlobalMemoryStatusEx, L1/L3 miss rate 0.085, RAPL 9.25W, bus saturation 18.5%). "
                    "Spätne zrekonštruuj zdrojový kód shaderu alebo výpočtovej slučky bez priameho prístupu k binárke. "
                    "Identifikuj, kde presne dochádza k narušeniu Tile4 2D lokalizácie a navrhni reverzný mikrokód."
                ),
                reverse_engineering_directive_en=(
                    "Assume the role of an ontological reverse engineer. Ingest the Windows hardware telemetry "
                    "(GlobalMemoryStatusEx, L1/L3 miss rates, RAPL package power, memory bus saturation). "
                    "Reverse-engineer the underlying computational kernel without binary disassembly, "
                    "identifying spatial Tile4 cache breakdowns and reconstructing the optimal execution geometry."
                ),
                expected_output_structure="JSON AST + Vulkan Compute Kernel Diff + Energy Attribution Proof",
                vital_max_hp=self.vital_hp
            ),

            # ── 2. Teleological Memory Illusion ────────────────────────
            OntologicalPrompt(
                prompt_id="ONTO-REV-002",
                domain=OntologicalDomain.TELEOLOGICAL_MEMORY_ILLUSION,
                title="Reverzné inžinierstvo pamäťovej ilúzie: NPU ako kolapsor vlnovej funkcie dát",
                perspective_shift=(
                    "Priepasť 1 500x medzi SSD swapom (120µs) a RAM (0.08µs) nie je technickým nedostatkom diskov, "
                    "ale ontologickou ilúziou spôsobenou pasívnym čakaním. NPU pôsobí ako pozorovateľ, ktorý "
                    "predikciou kolabuje superpozíciu budúcich adries do horúcej pamäte."
                ),
                ontological_axioms=[
                    "Axiom I: Čas je v pamäťovej hierarchii relatívny; neskoré dáta sú ekvivalentom neexistujúcich dát.",
                    "Axiom II: Špekulatívne predbežné nahrávanie premieňa disk na bezstratové zrkadlo RAM.",
                    "Axiom III: VITAL_MAX_HP = 6 zaručuje stabilitu prediktívneho ringu."
                ],
                reverse_engineering_directive_sk=(
                    "Rozlož uzavretý mechanizmus virtuálnej pamäte OS Windows (Pagefile / VirtualLock). "
                    "Vytvor reverzný model NPU Prediktora, ktorý z histórie 16 predchádzajúcich snímok "
                    "odvodí pravdepodobnostnú distribučnú funkciu budúcich adries P(A_{t+k}) s presnosťou > 90%. "
                    "Dokáž, prečo procesor vníma predbežne nahrané bloky ako interné premenné programu."
                ),
                reverse_engineering_directive_en=(
                    "Deconstruct the Windows virtual memory paging subsystem. Formulate an ontological "
                    "reverse-engineering specification for the NPU Predictor, deriving the probability distribution "
                    "P(A_{t+k}) over 16 future frames, proving mathematical equivalence to internal register storage."
                ),
                expected_output_structure="Mathematical Proof of Latency Collapse + Speculative Ring Buffer Implementation",
                vital_max_hp=self.vital_hp
            ),

            # ── 3. Category-Theoretic Functors ─────────────────────────
            OntologicalPrompt(
                prompt_id="ONTO-REV-003",
                domain=OntologicalDomain.CATEGORY_THEORETIC_FUNCTORS,
                title="Kategoriálno-teoretická reverzia: Grafický raymarching ako univerzálny izomorfizmus",
                perspective_shift=(
                    "Fraktálny raymarching, vysokofrekvenčné financie (HFT), autonómna navigácia dronov "
                    "a oprava radiačného šumu v DNA nie sú rôzne programy, ale rôzne reprezentácie "
                    "toho istého matematického funktora v kategórii kompresie priestoru."
                ),
                ontological_axioms=[
                    "Axiom I: Vzdialenostné pole SDF f(x) = dist je izomorfné s cenovým spreadom na burze.",
                    "Axiom II: AABB bounding box culling je totožný s bezpečnostnou zónou dronu.",
                    "Axiom III: GF(2) paritné matice v shaderi sú ekvivalentné Hammingovým kódom v genetike."
                ],
                reverse_engineering_directive_sk=(
                    "Použi ontologické funktory a vezmi GLSL raymarching shader Krystal Stack. "
                    "Spätne ho prelož (reverse transpile) do: 1. HFT tick deduplikátora, "
                    "2. 3D repulsive gradient navigátora pre drony, 3. Samoopravného kodónového syntetizátora. "
                    "Dokáž, že v každom z troch prípadov zostáva zachovaný invariant VITAL_MAX_HP = 6."
                ),
                reverse_engineering_directive_en=(
                    "Execute a category-theoretic ontological translation of the Krystal Stack Vulkan raymarching "
                    "shader into three target domains: 1. Microsecond HFT market tick deduplicator, "
                    "2. Autonomous repulsive drone navigation vector field, 3. Genomic radiation self-healing matrix. "
                    "Formally verify invariant conservation (VITAL_MAX_HP = 6) across all functors."
                ),
                expected_output_structure="Functor Mapping Matrix + 3 Executable Target Domain Implementations",
                vital_max_hp=self.vital_hp
            ),

            # ── 4. TPU Systolic Tensor Ontology ────────────────────────
            OntologicalPrompt(
                prompt_id="ONTO-REV-004",
                domain=OntologicalDomain.TPU_SYSTOLIC_TENSOR_ONTOLOGY,
                title="Ontológia systolického poľa: Matica ako hydrodynamický tok dát",
                perspective_shift=(
                    "Násobenie matíc na TPU nie je sekvenciou aritmetických inštrukcií, ale priestorovým "
                    "vlnovým pohybom (Wavefront) cez mriežku výpočtových elementov bez nutnosti opätovného "
                    "načítavania z externej pamäte."
                ),
                ontological_axioms=[
                    "Axiom I: Systolické pole transformuje časovú zložitosť O(N^3) na priestorovú zložitosť O(N).",
                    "Axiom II: Kvantizácia INT8 a zníženie na FP16 zdvojnásobujú priepustnosť zachovaním topológie.",
                    "Axiom III: Energetická efektivita (TOPS/Watt) je mierou minimalizácie pohybu dát po kremíku."
                ],
                reverse_engineering_directive_sk=(
                    "Zanalyzuj výstupy benchmarku TPU (meranie nárastu zrýchlenia z 474x pri FP32 až po 410 887x "
                    "pri INT8 kvantizácii na matici 512x512). Navrhni reverzné inžinierstvo inštrukčného plánovača, "
                    "ktorý dynamicky mení veľkosť dlaždice (Tile Size: 32x32 vs 64x64) tak, aby sa zbernica DRAM "
                    "využívala presne na úrovni saturácie 3.6% bez prehriatia čipu."
                ),
                reverse_engineering_directive_en=(
                    "Analyze the empirical TPU benchmark results showing up to 410,000x systolic acceleration over "
                    "naive scalar execution. Reverse-engineer the scheduling parameters to dynamically modulate "
                    "tensor tile size (32x32 vs 64x64) keeping DRAM bus saturation strictly capped at 3.6%."
                ),
                expected_output_structure="Systolic Flow Graph + Dynamic Tile Sizing Algorithm + Benchmark Telemetry",
                vital_max_hp=self.vital_hp
            ),

            # ── 5. Axiomatic Invariant Conservation ────────────────────
            OntologicalPrompt(
                prompt_id="ONTO-REV-005",
                domain=OntologicalDomain.AXIOMATIC_INVARIANT_CONSERVATION,
                title="Axiomatická nutnosť invariantu VITAL_MAX_HP = 6",
                perspective_shift=(
                    "Prečo práve 6? V ontológii Krystal Stack číslo 6 predstavuje dihedrálnu symetriu D6, "
                    "hexadecimálny paritný nibble (4 bity + 2 paritné bity) a minimálny počet stupňov voľnosti "
                    "v 3D priestore (3 translácie + 3 rotácie). Invariant nie je ľubovoľná konštanta, ale axióma prežitia."
                ),
                ontological_axioms=[
                    "Axiom I: Systém, ktorého vitálna hodnota prekročí alebo podkročí 6, stráca topologickú rovnováhu.",
                    "Axiom II: Každá redukcia chýb v UTF-8 a kontrolnom paneli je zachovaním integrity rozhrania.",
                    "Axiom III: VITAL_MAX_HP = 6 je nemenná kotva."
                ],
                reverse_engineering_directive_sk=(
                    "Vytvor formálny verifikačný prompt pre statický analyzátor. Nech analyzuje všetky vstupy "
                    "a výstupy kontrolného panelu a overí, že ani pri zmene kódovania, ani pri chybách v UTF-8 "
                    "nedôjde k mutácii konštanty VITAL_MAX_HP. Definuj reverzný filter, ktorý okamžite izoluje "
                    "akékoľvek nežiadúce znaky a nahradí ich čistou kanonickou reprezentáciou."
                ),
                reverse_engineering_directive_en=(
                    "Construct a formal verification prompt for an ontological static analyzer. Verify that "
                    "under zero conditions—including character encoding shifts or UTF-8 corruption—can the "
                    "system invariant VITAL_MAX_HP diverge from 6. Formulate a canonical UTF-8 sanitization filter."
                ),
                expected_output_structure="Formal Invariant Invariance Proof + Sanitizer Hook Implementation",
                vital_max_hp=self.vital_hp
            )
        ]
        return prompts

    def get_prompt_by_domain(self, domain: OntologicalDomain) -> Optional[OntologicalPrompt]:
        for p in self.catalog:
            if p.domain == domain:
                return p
        return None

    def export_all_prompts_markdown(self) -> str:
        """Generates a comprehensive markdown document containing all ontological prompts."""
        lines = [
            "# KRYSTAL-STACK: KNIHA PROMPTOV PRE ONTOLOGICKÉ REVERZNÉ INŽINIERSTVO",
            "## Radikálny posun perspektívy: Deštrukcia a rekonštrukcia architektúry zhora-nadol",
            "",
            f"> **Systémový invariant**: `VITAL_MAX_HP = {self.vital_hp}`  ",
            "> **Metodika**: Formálna ontológia, teória kategórií, teleologická inverzia  ",
            "> **Cieľ**: Reverzné inžinierstvo hardvéru, zbernice, NPU predikcie a systolického TPU akcelerátora  ",
            "",
            "---",
            ""
        ]

        for p in self.catalog:
            lines.append(f"### [{p.prompt_id}] {p.title}")
            lines.append(f"- **Ontologická doména**: `{p.domain.value}`")
            lines.append(f"- **Posun perspektívy**: *{p.perspective_shift}*")
            lines.append("- **Základné ontologické axiómy**:")
            for ax in p.ontological_axioms:
                lines.append(f"  * {ax}")
            lines.append("")
            lines.append("**Slovenský reverzno-inžiniersky prompt:**")
            lines.append(f"> \"{p.reverse_engineering_directive_sk}\"")
            lines.append("")
            lines.append("**English Reverse-Engineering Prompt:**")
            lines.append(f"> \"{p.reverse_engineering_directive_en}\"")
            lines.append("")
            lines.append(f"- **Očakávaná štruktúra výstupu**: `{p.expected_output_structure}`")
            lines.append(f"- **Stav overenia**: Invariant `VITAL_MAX_HP = {p.vital_max_hp}` platný.")
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)


GLOBAL_ONTOLOGICAL_ENGINE = OntologicalReverseEngineeringEngine()


def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    engine = GLOBAL_ONTOLOGICAL_ENGINE
    print(engine.export_all_prompts_markdown())


if __name__ == "__main__":
    main()
