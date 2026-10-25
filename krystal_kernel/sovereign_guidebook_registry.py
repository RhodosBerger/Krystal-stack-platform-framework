"""
krystal_kernel/sovereign_guidebook_registry.py

Sovereign Guidebook & Interactive Manual Registry Engine.
=========================================================
Aggregates, indexes, and serves structured, machine-readable architectural
guidebooks and educational manuals across all Krystal Stack domains:
  1. Architecture Foundations & Cybernetic Model ("Hardware as a Living Organism")
  2. The 6 Sovereign Layers (L0 Silicon PDN to L5 Constitutional Gatekeeper)
  3. The 8 Textbook Inferable Patterns (Chronometric Shadows, Symplectic Physics, etc.)
  4. Sovereign Operations Control Center & Aggregator Compiler
  5. Godot Krystal Edition Packaging & 3D Interactive Showcase
  6. Real-time Intervention Safety Matrix & Zero-Mutation Boundary Invariant

Constitutional Invariant: VITAL_MAX_HP == 6 (not Valid(A) => Delta S == 0)
"""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

from .constants import VITAL_MAX_HP, ContractViolation


class GuidebookCategory(str, Enum):
    ARCHITECTURE_FOUNDATIONS = "ARCHITECTURE_FOUNDATIONS"
    SOVEREIGN_6_LAYERS = "SOVEREIGN_6_LAYERS"
    TEXTBOOK_INFERABLE_PATTERNS = "TEXTBOOK_INFERABLE_PATTERNS"
    OPERATIONS_CONTROL_CENTER = "OPERATIONS_CONTROL_CENTER"
    GODOT_KRYSTAL_EDITION = "GODOT_KRYSTAL_EDITION"
    INTERVENTION_SAFETY_MATRIX = "INTERVENTION_SAFETY_MATRIX"


class ReadingDifficulty(str, Enum):
    INTRODUCTORY = "INTRODUCTORY"
    INTERMEDIATE = "INTERMEDIATE"
    ADVANCED_SYSTEMS = "ADVANCED_SYSTEMS"
    MATHEMATICAL_FORMAL = "MATHEMATICAL_FORMAL"


@dataclass(frozen=True)
class InteractiveSandboxPreset:
    """Preset parameters for running live browser simulations."""
    preset_id: str
    name: str
    target_layer_or_pattern: str
    initial_params: Dict[str, Any]
    expected_observable: str
    vital_max_hp: int = VITAL_MAX_HP


@dataclass(frozen=True)
class GuidebookSection:
    """A formal chapter or section within the Sovereign Guidebook."""
    section_id: str
    category: GuidebookCategory
    title: str
    subtitle: str
    summary_markdown: str
    reading_difficulty: ReadingDifficulty
    order_index: int
    tags: Tuple[str, ...]
    constitutional_invariants: Tuple[str, ...]
    theory_reference_id: Optional[str]
    code_snippet: Optional[str]
    code_language: str = "python"
    sandbox_preset: Optional[InteractiveSandboxPreset] = None
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(
                f"GuidebookSection invariant breach: vital_max_hp={self.vital_max_hp} != {VITAL_MAX_HP}"
            )


class SovereignGuidebookRegistry:
    """
    Authoritative catalog of Krystal Stack overview guidebooks, educational
    manuals, and interactive laboratory specifications.
    """

    def __init__(self, vital_max_hp: int = VITAL_MAX_HP) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Registry initialized with invalid vital_max_hp: {vital_max_hp}")
        self.vital_max_hp = vital_max_hp
        self._sections: Dict[str, GuidebookSection] = {}
        self._init_canonical_sections()

    def _init_canonical_sections(self) -> None:
        """Populates the canonical set of educational and architectural chapters."""
        
        # 1. Architecture Foundations
        self._register(GuidebookSection(
            section_id="arch-01-living-organism",
            category=GuidebookCategory.ARCHITECTURE_FOUNDATIONS,
            title="Hardv\u00e9r ako \u017div\u00fd Organizmus",
            subtitle="Prechod od mechanick\u00e9ho schedulingu k organick\u00e9mu kybernetick\u00e9mu v\u00fdpo\u010dtu",
            summary_markdown=(
                "V\u00e4\u010d\u0161ina s\u00fa\u010dasn\u00fdch opera\u010dn\u00fdch syst\u00e9mov pristupuje k hardv\u00e9ru mechanicky. "
                "Krystal-Stack pristupuje k po\u010d\u00edta\u010du ako k dynamick\u00e9mu biologick\u00e9mu organizmu: "
                "ka\u017ed\u00fd watt elektrickej energie, ka\u017ed\u00fd cyklus GPU a megabajt VRAM je limitovan\u00fd zdroj. "
                "Syst\u00e9m disponuje nervovou s\u00fastavou (sub-50ns telemetria), reflexmi (miechov\u00e1 ochrana pred "
                "tepeln\u00fdm runawayom do 1 ms) a kognit\u00edvnou k\u00f4rou (Shadow Council: Tvorca, Aud\u00edtor, \u00da\u010dtovn\u00edk)."
            ),
            reading_difficulty=ReadingDifficulty.INTRODUCTORY,
            order_index=1,
            tags=("cybernetics", "philosophy", "energy", "telemetry"),
            constitutional_invariants=("VITAL_MAX_HP == 6", "P3_CHRONOMETRIC_SHADOW"),
            theory_reference_id="THEORY-COGNITIVE-RHYTHM-CADENCE",
            code_snippet=(
                "from krystal_kernel.sovereign_runtime import GLOBAL_SOVEREIGN_RUNTIME\n"
                "telemetry = {'vdd_volts': 0.85, 'tj_celsius': 64.2, 'power_watts': 18.4}\n"
                "receipt = GLOBAL_SOVEREIGN_RUNTIME.execute_cycle(telemetry)\n"
                "print(f'Reflex status: {receipt.telemetry_accepted}, VITAL_HP: {receipt.vital_max_hp}')"
            )
        ))

        self._register(GuidebookSection(
            section_id="arch-02-heterogeneous-triad",
            category=GuidebookCategory.ARCHITECTURE_FOUNDATIONS,
            title="Heterog\u00e9nna Tri\u00e1da (CPU + GPU + NPU)",
            subtitle="Trojzlo\u017ekov\u00e1 d\u013be\u013ba pr\u00e1ce bez zdr\u017eania hern\u00e9ho FPS",
            summary_markdown=(
                "Modern\u00e9 \u010dipy (Intel Core Ultra, Ryzen AI, Apple Silicon) obsahuj\u00fa tri v\u00fdpo\u010dtov\u00e9 svety: "
                "1. CPU (Generalista) pre sekven\u010dn\u00fa logiku, "
                "2. GPU (Paralelista) pre 3D rendering a rastrov\u00e9 shadery, "
                "3. NPU (Maticov\u00fd akceler\u00e1tor) s pr\u00edkonom 1-2 W pre neur\u00e1lne siete. "
                "Krystal Stack pres\u00fava neur\u00e1lnu anal\u00fdzu a ASCII kompoz\u00edciu na NPU, v\u010faka \u010domu GPU nestr\u00e1ca FPS."
            ),
            reading_difficulty=ReadingDifficulty.INTERMEDIATE,
            order_index=2,
            tags=("triad", "npu", "gpu", "cpu", "acceleration"),
            constitutional_invariants=("VITAL_MAX_HP == 6", "P1_ZERO_TOUCH_INTERPOSITION"),
            theory_reference_id="THEORY-PERIPHERAL-HW-ACCELERATION-SYNTHESIS",
            code_snippet=(
                "from krystal_kernel.llm_hw_acceleration_bridge import LlmHwAccelerationBridge\n"
                "bridge = LlmHwAccelerationBridge()\n"
                "status = bridge.evaluate_acceleration_envelope()\n"
                "print('Active Device:', status.active_device, '| Tensor Saturation:', status.saturation_ratio)"
            )
        ))

        # 2. The 6 Sovereign Layers
        self._register(GuidebookSection(
            section_id="layer-00-silicon-pdn",
            category=GuidebookCategory.SOVEREIGN_6_LAYERS,
            title="Vrstva L0: Silicon PDN & Indukt\u00edvny Guvern\u00e9r",
            subtitle="Sub-mikrosekundov\u00e9 tlmenie preletov nap\u00e4tia (flyback overshoot <= 0.15V)",
            summary_markdown=(
                "Fyzik\u00e1lna vrstva nap\u00e1jacej siete (Power Delivery Network). Dynamick\u00fd RLC tlmi\u010d zabra\u0148uje "
                "prestreleniam nap\u00e4tia pri prudk\u00fdch prechodoch z\u00e1\u0165a\u017ee (di/dt a\u017e 80 A/\u03bcs). "
                "Nomin\u00e1lne jadrov\u00e9 nap\u00e4tie Vdd = 0.85V, trval\u00fd strop 1.00V, maxim\u00e1lny flyback 0.15V. "
                "Pri prekro\u010den\u00ed Tj > 85\u00b0C okam\u017eite iniciuje term\u00e1lne zhadzovanie z\u00e1\u0165a\u017ee (workload shedding)."
            ),
            reading_difficulty=ReadingDifficulty.ADVANCED_SYSTEMS,
            order_index=10,
            tags=("L0", "silicon", "pdn", "voltage", "thermal", "rlc"),
            constitutional_invariants=("Vdd <= 1.00V", "Tj <= 85.0C", "VITAL_MAX_HP == 6"),
            theory_reference_id="THEORY-UMA-INDUCTIVE-VOLTAGE-GOVERNANCE",
            code_snippet=(
                "from krystal_kernel.sovereign_layer_optimizer import OptimizedSiliconPdnGovernor\n"
                "gov = OptimizedSiliconPdnGovernor()\n"
                "eval_res = gov.evaluate_pdn_envelope(vdd_volts=0.85, delta_current_amps=15.0, dt_microseconds=0.25)\n"
                "print(f'Flyback: {eval_res.flyback_overshoot_v:.4f}V | Safe: {eval_res.is_safe}')"
            ),
            sandbox_preset=InteractiveSandboxPreset(
                preset_id="pdn-stress-test",
                name="PDN RLC Transient Pulse",
                target_layer_or_pattern="L0_SILICON_PDN",
                initial_params={"vdd_volts": 0.85, "delta_i_amps": 25.0, "rise_time_us": 0.2, "l_ph": 45.0},
                expected_observable="Overshoot <= 0.15V, Vdd_peak <= 1.00V"
            )
        ))

        self._register(GuidebookSection(
            section_id="layer-01-chronometric-compositor",
            category=GuidebookCategory.SOVEREIGN_6_LAYERS,
            title="Vrstva L1: Chronometrick\u00fd Tie\u0148ov\u00fd Kompoz\u00edtor",
            subtitle="Pl\u00e1novanie v\u00fdpo\u010dtov v tie\u0148ovom VBlank okne (T_shadow >= 3.50 ms, \u03bc_UMA < 0.78)",
            summary_markdown=(
                "Zabezpe\u010duje KSPP Princ\u00edp P3 (Chronometric Shadow Multiplexing): akt\u00edvny scanout r\u00e1mu "
                "monopolizuje zdie\u013ean\u00fa zbernicu UMA. V\u00fdpo\u010dtov\u00e9 a kognit\u00edvne \u00falohy sa sm\u00fa sp\u00fa\u0161\u0165a\u0165 "
                "v\u00fdhradne v tie\u0148ovom okne VBlanku s rozpo\u010dtom aspo\u0148 3.50 ms pri vy\u0165a\u017een\u00ed zbernice pod 78 %."
            ),
            reading_difficulty=ReadingDifficulty.ADVANCED_SYSTEMS,
            order_index=11,
            tags=("L1", "chronometric", "vblank", "shadow", "uma", "timing"),
            constitutional_invariants=("T_shadow >= 3.50ms", "mu_UMA < 0.78", "VITAL_MAX_HP == 6"),
            theory_reference_id="THEORY-CHRONO-SYMPLECTIC-PROCESSOR-ARCHITECTURE",
            code_snippet=(
                "from krystal_kernel.sovereign_layer_optimizer import OptimizedChronometricShadowScheduler\n"
                "sched = OptimizedChronometricShadowScheduler()\n"
                "win = sched.calculate_shadow_window(refresh_rate_hz=60.0, scanout_duration_ms=12.5, bus_utilization=0.62)\n"
                "print(f'Shadow Budget: {win.available_shadow_budget_ms:.2f}ms | Feasible: {win.is_feasible}')"
            ),
            sandbox_preset=InteractiveSandboxPreset(
                preset_id="vblank-shadow-eval",
                name="60Hz / 120Hz VBlank Window",
                target_layer_or_pattern="L1_CHRONOMETRIC",
                initial_params={"refresh_hz": 60.0, "scanout_ms": 12.5, "bus_load": 0.65},
                expected_observable="Budget >= 3.50ms, Dispatch Allowed"
            )
        ))

        self._register(GuidebookSection(
            section_id="layer-02-symplectic-integrator",
            category=GuidebookCategory.SOVEREIGN_6_LAYERS,
            title="Vrstva L2: Symplektick\u00fd Integr\u00e1tor & Hex-HSL Mrie\u017eka",
            subtitle="Konzerv\u00e1cia f\u00e1zov\u00e9ho objemu (det(J) == 1.000000) a 3D orbit\u00e1lna geometria D6",
            summary_markdown=(
                "Leapfrog Verlet integr\u00e1tor zaru\u010duje nulov\u00fd n\u00e1rast umelej energie v dynamickom modeli "
                "(|Delta H| < 10^-3 po 10 000 krokoch). Z\u00e1rove\u0148 integruje diskr\u00e9tnu 3D Hex-HSL "
                "mrie\u017eku, mapuj\u00facu 6 orbit\u00e1lnych sektorov D6 (0x10 a\u017e 0x60) bez pl\u00e1vaj\u00facich ch\u00fdb."
            ),
            reading_difficulty=ReadingDifficulty.MATHEMATICAL_FORMAL,
            order_index=12,
            tags=("L2", "symplectic", "hamiltonian", "leapfrog", "hex_hsl", "phase_space"),
            constitutional_invariants=("det(J) == 1.000000", "|Delta H| < 1e-3", "VITAL_MAX_HP == 6"),
            theory_reference_id="THEORY-SYMPLECTIC-HAMILTONIAN-CONSERVATION",
            code_snippet=(
                "from krystal_kernel.sovereign_layer_optimizer import OptimizedSymplecticPhaseSpaceEngine\n"
                "engine = OptimizedSymplecticPhaseSpaceEngine()\n"
                "state, res = engine.step_particle(q=1.0, p=0.0, dt=0.01)\n"
                "print(f'det(J): {res.jacobian_determinant:.6f} | Delta H: {res.energy_drift:.8f}')"
            ),
            sandbox_preset=InteractiveSandboxPreset(
                preset_id="symplectic-orbit",
                name="Harmonic Phase-Space Orbit",
                target_layer_or_pattern="L2_SPATIAL_SYMPLECTIC",
                initial_params={"q0": 1.0, "p0": 0.0, "dt": 0.01, "steps": 500},
                expected_observable="det(J) == 1.000000, Elliptical invariant orbit"
            )
        ))

        self._register(GuidebookSection(
            section_id="layer-03-inertial-governor",
            category=GuidebookCategory.SOVEREIGN_6_LAYERS,
            title="Vrstva L3: Kognit\u00edvny a Inerci\u00e1lny Guvern\u00e9r",
            subtitle="12-sn\u00edmkov\u00e1 dwell hyster\u00e9zia (P8) potl\u00e1\u010daj\u00faca oscil\u00e1cie a 10 Hz alfa rytmus",
            summary_markdown=(
                "Implementuje KSPP Princ\u00edp P8 (Inertial Stability). Zamedzuje nepr\u00edpustn\u00e9mu kmitaniu "
                "dynamick\u00e9ho rozl\u00ed\u0161enia (DRS) a throttling stavov. Zatia\u013e \u010do podtaktovanie je okam\u017eit\u00e9, "
                "n\u00e1vrat na vy\u0161\u0161ie rozl\u00ed\u0161enie vy\u017eaduje stabiln\u00fd stav po dobu 12 po sebe id\u00facich sn\u00edmok. "
                "V\u00fdpo\u010det be\u017e\u00ed s p\u00e4\u0165-faktorov\u00fdm Merkle-Momentom s tlmen\u00edm gamma = 0.85."
            ),
            reading_difficulty=ReadingDifficulty.ADVANCED_SYSTEMS,
            order_index=13,
            tags=("L3", "inertial", "hysteresis", "dwell", "merkle_momentum", "anti_chatter"),
            constitutional_invariants=("Dwell >= 12 frames", "Suppression >= 80%", "VITAL_MAX_HP == 6"),
            theory_reference_id="THEORY-COGNITIVE-RHYTHM-CADENCE",
            code_snippet=(
                "from krystal_kernel.sovereign_layer_optimizer import OptimizedCognitiveCadenceGovernor\n"
                "gov = OptimizedCognitiveCadenceGovernor(dwell_frames=12)\n"
                "eval_res = gov.evaluate_cadence(instantaneous_load=0.88, dt_ms=16.67)\n"
                "print(f'DRS Scale: {eval_res.drs_scale:.2f} | Alpha Pulse: {eval_res.alpha_harmonic_pulse:.3f}')"
            )
        ))

        self._register(GuidebookSection(
            section_id="layer-04-zero-copy-bridge",
            category=GuidebookCategory.SOVEREIGN_6_LAYERS,
            title="Vrstva L4: Univerz\u00e1lny 128B SPSC Zero-Copy R\u00e1mec",
            subtitle="KSPP P2 & P9: 128B prirodzene zarovnan\u00e9 bin\u00e1rne zmluvy s HMAC-SHA256 integritou",
            summary_markdown=(
                "V\u0161etka kritick\u00e1 medziprocesov\u00e1 komunik\u00e1cia prebieha cez lock-free SPSC prstence s fixn\u00fdm "
                "128B r\u00e1mcom (_Alignas(64) ZeroCopyContractFrame). R\u00e1mec obsahuje 96B d\u00e1tov\u00fd n\u00e1klad "
                "a 32B HMAC-SHA256 podpis. Garantuje nulov\u00e9 alok\u00e1cie po\u010das z\u00e1kresu a priepustnos\u0165 vy\u0161\u0161iu "
                "ne\u017e 130 000 r\u00e1mcov za sekundu pri p50 < 2.5 \u03bcs."
            ),
            reading_difficulty=ReadingDifficulty.MATHEMATICAL_FORMAL,
            order_index=14,
            tags=("L4", "zero_copy", "spsc", "128B", "hmac_sha256", "memory_barrier"),
            constitutional_invariants=("FrameSize == 128B", "Alignment == 64B", "VITAL_MAX_HP == 6"),
            theory_reference_id="THEORY-ZERO-COPY-128B-CONTRACT-FRAME",
            code_snippet=(
                "from krystal_kernel.sovereign_layer_optimizer import OptimizedZeroCopyContractSerializer\n"
                "ser = OptimizedZeroCopyContractSerializer()\n"
                "raw = ser.serialize_128b_frame(seq=42, stream_id=1, payload_values=[0.85, 0.08, 62.0])\n"
                "print(f'Frame bytes: {len(raw)}B | Validated: {ser.validate_128b_frame(raw)}')\n"
                "assert len(raw) == 128"
            )
        ))

        self._register(GuidebookSection(
            section_id="layer-05-constitutional-gatekeeper",
            category=GuidebookCategory.SOVEREIGN_6_LAYERS,
            title="Vrstva L5: \u00dastavn\u00fd Vr\u00e1tnik & Bezpe\u010dn\u00fd Z\u00e1sah",
            subtitle="Separation of Concerns: Proposer != Evaluator a z\u00e1ruka Delta S == 0 pri odmietnut\u00ed",
            summary_markdown=(
                "Najvy\u0161\u0161ia autorita v hierarchii. Ka\u017ed\u00fd n\u00e1vrh z\u00e1sahu (zmena frekvencie, nap\u00e4tia, rozpo\u010dtu) "
                "mus\u00ed prejs\u0165 dvojf\u00e1zov\u00fdm auditom: 1. Polyhedr\u00e1lna kontrola bezpe\u010dn\u00e9ho konvexn\u00e9ho obalu, "
                "2. Valid\u00e1cia autority. Ak\u00e1ko\u013evek neplatn\u00e1 akcia je bezpe\u010dne odmietnut\u00e1 a zanech\u00e1va autoritat\u00edvny "
                "stav v absol\u00fatnej nemennosti: not Valid(A) => Delta S == 0."
            ),
            reading_difficulty=ReadingDifficulty.MATHEMATICAL_FORMAL,
            order_index=15,
            tags=("L5", "constitutional", "gatekeeper", "delta_s_zero", "safety", "two_phase"),
            constitutional_invariants=("not Valid(A) => Delta S == 0", "VITAL_MAX_HP == 6"),
            theory_reference_id="THEORY-CONSTITUTIONAL-STATE-INVARIANTS",
            code_snippet=(
                "from krystal_kernel.sovereign_layer_optimizer import OptimizedConstitutionalSafetyActuator\n"
                "act = OptimizedConstitutionalSafetyActuator()\n"
                "rcpt = act.evaluate_intervention(variable_id='ELEM_CORE_VOLTAGE', proposed_delta=0.05)\n"
                "print(f'Intervention authorized: {rcpt.is_authorized}, Delta S: {rcpt.delta_s_applied}')"
            )
        ))

        # 3. Textbook Inferable Patterns
        self._register(GuidebookSection(
            section_id="pattern-01-asymmetric-crossover",
            category=GuidebookCategory.TEXTBOOK_INFERABLE_PATTERNS,
            title="Vzor 1: Asymetrick\u00fd F\u00e1zov\u00fd Crossover",
            subtitle="Separ\u00e1cia v\u00fdpo\u010dtov\u00fdch f\u00e1z bez interferencie so scanoutom",
            summary_markdown=(
                "Form\u00e1lny vzor odvoden\u00fd z KSPP P3: scanout a v\u00fdpo\u010det s\u00fa ortogon\u00e1lne "
                "rozf\u00e1zovan\u00e9, \u010d\u00edm sa eliminuje interferencia na zdie\u013eanej zbernici pam\u00e4te UMA."
            ),
            reading_difficulty=ReadingDifficulty.INTERMEDIATE,
            order_index=20,
            tags=("pattern", "crossover", "uma", "orthogonal"),
            constitutional_invariants=("Phi_Scanout * Phi_Compute == 0", "VITAL_MAX_HP == 6"),
            theory_reference_id="THEORY-TEXTBOOK-INFERABLE-PATTERNS",
            code_snippet=(
                "from krystal_kernel.textbook_inferable_patterns import infer_asymmetric_phase_crossover\n"
                "res = infer_asymmetric_phase_crossover(vblank_remaining_ms=4.2, bus_load=0.6)\n"
                "print(f'Allowed: {res.is_crossover_allowed}, Phase: {res.active_phase}')"
            )
        ))

        self._register(GuidebookSection(
            section_id="pattern-04-12frame-dwell",
            category=GuidebookCategory.TEXTBOOK_INFERABLE_PATTERNS,
            title="Vzor 4: 12-Sn\u00edmkov\u00e1 Dwell Hyster\u00e9zia",
            subtitle="Matematick\u00e9 tlmenie oscil\u00e1ci\u00ed cez diskr\u00e9tny po\u010d\u00edtadlovy dwell register",
            summary_markdown=(
                "Odvodzuje hyster\u00e9zny stav pre dynamick\u00e9 rozl\u00ed\u0161enie (DRS) a zmenu prev\u00e1dzkov\u00fdch "
                "bodov, \u010d\u00edm garantuje aspo\u0148 80 % potla\u010denie chatteringu pri kol\u00edsavom za\u0165a\u017een\u00ed."
            ),
            reading_difficulty=ReadingDifficulty.INTERMEDIATE,
            order_index=23,
            tags=("pattern", "hysteresis", "dwell", "anti_chatter"),
            constitutional_invariants=("Dwell >= 12", "VITAL_MAX_HP == 6"),
            theory_reference_id="THEORY-TEXTBOOK-INFERABLE-PATTERNS",
            code_snippet=(
                "from krystal_kernel.textbook_inferable_patterns import infer_12frame_dwell_hysteresis\n"
                "res = infer_12frame_dwell_hysteresis(dwell_counter=8, threshold=12, requested_upscale=True)\n"
                "print('Can Upscale:', res.can_upscale, '| Anti-Chatter Active:', res.is_anti_chatter_active)"
            )
        ))

        # 4. Operations Control Center & Compiler
        self._register(GuidebookSection(
            section_id="ops-01-compiler-aggregator",
            category=GuidebookCategory.OPERATIONS_CONTROL_CENTER,
            title="Riadiace Centrum Oper\u00e1ci\u00ed & Kompil\u00e1tor",
            subtitle="6-f\u00e1zov\u00e1 hermetick\u00e1 agreg\u00e1cia cel\u00e9ho prev\u00e1dzkov\u00e9ho stavu s HMAC certifik\u00e1tom",
            summary_markdown=(
                "Zabezpe\u010duje audit v\u0161etk\u00fdch te\u00f3ri\u00ed, kontraktov, interpol\u00e1torov a tie\u0148ov\u00fdch cyklov. "
                "Spustite\u013en\u00e9 cez suver\u00e9nne bash/bat skripty s okam\u017eit\u00fdm vyhodnoten\u00edm v\u0161etk\u00fdch 6 f\u00e1z za < 2 ms."
            ),
            reading_difficulty=ReadingDifficulty.ADVANCED_SYSTEMS,
            order_index=30,
            tags=("operations", "compiler", "aggregator", "hmac", "receipt"),
            constitutional_invariants=("All 6 Stages PASS", "VITAL_MAX_HP == 6"),
            theory_reference_id="THEORY-SOVEREIGN-OPERATIONS-COMPILER",
            code_snippet=(
                "from krystal_kernel.sovereign_operations_compiler import GLOBAL_SOVEREIGN_OPERATIONS_COMPILER\n"
                "rcpt = GLOBAL_SOVEREIGN_OPERATIONS_COMPILER.compile_and_aggregate_all()\n"
                "print(f'Receipt ID: {rcpt.receipt_id}, Operational: {rcpt.is_fully_operational}')"
            )
        ))

        # 5. Godot Krystal Edition Packaging
        self._register(GuidebookSection(
            section_id="godot-01-custom-distribution",
            category=GuidebookCategory.GODOT_KRYSTAL_EDITION,
            title="Godot Krystal Edition: Vlastn\u00e9 Balenie",
            subtitle="Integr\u00e1cia 6 vrstiev, master Autoload singleton a 3D demon\u0161tra\u010dn\u00fd stage",
            summary_markdown=(
                "Vlastn\u00e1 ed\u00edcia hern\u00e9ho enginu Godot 4.x obohacen\u00e1 o 6 suver\u00e9nnych vrstiev (L0..L5). "
                "Obsahuje skripty Silicon PDN Bridge, Chronometric Compositor, Symplectic Integrator, "
                "Inertial Governor, Zero-Copy Bridge a Realtime Intervention Client s 3D showcase javiskom."
            ),
            reading_difficulty=ReadingDifficulty.INTERMEDIATE,
            order_index=40,
            tags=("godot", "distribution", "packaging", "autoload", "3d"),
            constitutional_invariants=("39 assets verified", "VITAL_MAX_HP == 6"),
            theory_reference_id="THEORY-GODOT-KRYSTAL-EDITION-PACKAGING",
            code_snippet=(
                "from krystal_kernel.godot_krystal_edition_packager import GLOBAL_GODOT_KRYSTAL_EDITION_PACKAGER\n"
                "rcpt = GLOBAL_GODOT_KRYSTAL_EDITION_PACKAGER.assemble_package()\n"
                "print(f'Packaged {rcpt.total_files_packaged} assets, Manifest: {rcpt.manifest_path}')"
            )
        ))

    def _register(self, section: GuidebookSection) -> None:
        self._sections[section.section_id] = section

    def get_section(self, section_id: str) -> Optional[GuidebookSection]:
        return self._sections.get(section_id)

    def list_sections(
        self,
        category: Optional[GuidebookCategory] = None,
        tag: Optional[str] = None
    ) -> List[GuidebookSection]:
        """Returns sections sorted by order_index, optionally filtered."""
        res = list(self._sections.values())
        if category:
            res = [s for s in res if s.category == category]
        if tag:
            res = [s for s in res if tag.lower() in [t.lower() for t in s.tags]]
        res.sort(key=lambda s: s.order_index)
        return res

    def search_sections(self, query: str) -> List[GuidebookSection]:
        """Full-text search across section titles, subtitles, tags, and summary markdown."""
        q = query.strip().lower()
        if not q:
            return self.list_sections()

        matches = []
        for s in self._sections.values():
            score = 0
            if q in s.title.lower():
                score += 10
            if q in s.subtitle.lower():
                score += 5
            if any(q in t.lower() for t in s.tags):
                score += 4
            if q in s.summary_markdown.lower():
                score += 2
            if score > 0:
                matches.append((score, s))

        matches.sort(key=lambda item: (-item[0], item[1].order_index))
        return [item[1] for item in matches]

    def export_catalog_json(self) -> Dict[str, Any]:
        """Exports the entire guidebook catalog formatted for the web UI."""
        return {
            "title": "Krystal Stack: Sovereign Overview Guidebooks & Manuals",
            "version": "1.0.0-SOVEREIGN",
            "total_sections": len(self._sections),
            "vital_max_hp": self.vital_max_hp,
            "categories": [c.value for c in GuidebookCategory],
            "sections": [asdict(s) for s in self.list_sections()]
        }


# Global Singleton instance
GLOBAL_SOVEREIGN_GUIDEBOOK_REGISTRY: SovereignGuidebookRegistry = (
    SovereignGuidebookRegistry()
)
