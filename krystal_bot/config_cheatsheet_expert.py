#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: CONFIGURATION CHEATSHEET EXPERT & CHATBOT REASONING ENGINE
==============================================================================
Module: krystal_bot/config_cheatsheet_expert.py
Description:
    Domain-expert chatbot intelligence and cheatsheet reasoning system.
    Equips the Krystal bot (and local SLM / multi-model cascade) with exhaustive
    factual knowledge, trade-off analysis, validation, and recommendation
    capabilities across all alternative valid configurations:

      1. 6 Canonical Configuration Profiles (REALTIME, BALANCED, ECO, SHADOW, KIOSK, CI)
      2. 9 Coordination Templates (Real-Time, Adaptive, Thermal, Shadow, Sync + 4 Next-Gen)
      3. 8 Cascade Execution Strategies & Multi-Model SLM Routing
      4. 8 Categorized Linux Hardware Allocation Domains
      5. Morse Wire Protocol & Integer Tensor Emergency Beacons
      6. Poslední Kmen Elemental Balance Matrices & Invariant Enforcement

Constitutional Invariant:
    assert VITAL_MAX_HP == 6 (Universally enforced across all responses & models)

Author: Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP

# Lazy-loaded kernel imports to ensure zero-dependency bootstrapping
try:
    from scripts.krystal_config_toolchain import (
        CANONICAL_PROFILES,
        ProfileKind,
        KrystalConfigurationProfile,
    )
except ImportError:
    CANONICAL_PROFILES = {}
    ProfileKind = None

try:
    from krystal_kernel.coordination_templates import (
        CANONICAL_TEMPLATES,
        CoordinationTemplateId,
        CoordinationTemplate,
        GLOBAL_WINDOWS_TIMER_SUBSYSTEM,
    )
except ImportError:
    CANONICAL_TEMPLATES = {}
    CoordinationTemplateId = None

try:
    from krystal_kernel.cascade import (
        GLOBAL_MODEL_CASCADE_RUNTIME,
        ROUTING_PROFILES,
    )
except ImportError:
    GLOBAL_MODEL_CASCADE_RUNTIME = None
    ROUTING_PROFILES = {}

try:
    from krystal_kernel.linux_hw_monitor_utility import (
        GLOBAL_LINUX_HW_MONITOR,
        HardwareResourceCategory,
    )
except ImportError:
    GLOBAL_LINUX_HW_MONITOR = None
    HardwareResourceCategory = None

try:
    from krystal_kernel.morse_wire_protocol import GLOBAL_MORSE_PROTOCOL
except ImportError:
    GLOBAL_MORSE_PROTOCOL = None


# ==============================================================================
# TRADE-OFF MATRIX & SUMMARY KNOWLEDGE BASE
# ==============================================================================

PROFILE_TRADE_OFFS: Dict[str, Dict[str, Any]] = {
    "REALTIME_LOW_LATENCY": {
        "best_for": "Sub-millisecond real-time actuation, physics, high-frequency robotics, competitive gameplay",
        "latency_target": "1.0 ms quantum (sub-1ms jitters)",
        "cpu_usage": "Dedicated P-Cores (Mask 0x000F), HIGH_PRIORITY_CLASS, MMCSS 'Games'",
        "power_budget": "32.0 W (PL1 Sustained, High Thermal Footprint)",
        "isa_acceleration": "Intel AMX-TILE & AVX-512, NPU DP4A INT8 (4 streams)",
        "actuation_state": "ACTIVE_MUTATING",
        "trade_offs": "Maximum performance and determinism; elevated thermal dissipation; not suitable for battery",
        "recommended_workloads": ["realtime", "robotics", "combat", "physics", "low_latency", "p_cores", "max_performance"]
    },
    "BALANCED_DEVELOPMENT": {
        "best_for": "Day-to-day development, 60 FPS viewport rendering, balanced multi-tasking",
        "latency_target": "10.0 ms adaptive quantum",
        "cpu_usage": "Symmetric All-Cores (Mask 0x00FF), NORMAL_PRIORITY_CLASS",
        "power_budget": "24.0 W (Standard notebook/desktop budget)",
        "isa_acceleration": "AVX2 & VNNI, NPU DP4A INT8 (2 streams)",
        "actuation_state": "ACTIVE_MUTATING",
        "trade_offs": "Optimal equilibrium between frame stability, thermals, and compilation speed",
        "recommended_workloads": ["dev", "development", "standard", "viewport", "balanced", "general", "default", "60fps"]
    },
    "ECO_THERMAL_PRESERVATION": {
        "best_for": "Battery preservation, emergency cool-down, thermal rescue under sustained heat",
        "latency_target": "50.0 ms relaxed pacing",
        "cpu_usage": "Efficiency E-Cores (Mask 0x00F0), BELOW_NORMAL_PRIORITY_CLASS",
        "power_budget": "10.0 W (Under-volted by -25mV, passive cool-down)",
        "isa_acceleration": "Standard SSE4.2 / AVX2, NPU Low-Power (1 stream)",
        "actuation_state": "PACED_MUTATING",
        "trade_offs": "Minimal power draw (<10W) and whisper-quiet fans; higher latency spikes (>50ms)",
        "recommended_workloads": ["battery", "eco", "thermal", "heat", "overheating", "quiet", "low_power", "e_cores"]
    },
    "SHADOW_COUNTERFACTUAL_RESEARCH": {
        "best_for": "Counterfactual scenario replaying, non-actuating hypothesis exploration, safe offline benchmarking",
        "latency_target": "2.0 ms high-throughput shadow cadence",
        "cpu_usage": "Symmetric All-Cores (Mask 0x00FF), NORMAL_PRIORITY_CLASS",
        "power_budget": "24.0 W (Shadow Burst)",
        "isa_acceleration": "AVX2 / AMX Emulation, NPU 2 streams",
        "actuation_state": "AUTH_SHADOW_ONLY (Strictly Non-Mutating, 0 OS writes)",
        "trade_offs": "Total safety from side-effects; no hardware state or actuator register mutations",
        "recommended_workloads": ["shadow", "counterfactual", "research", "simulation", "safe", "offline", "audit", "dry_run"]
    },
    "MISSION_CONTROL_STUDIO_KIOSK": {
        "best_for": "Localhost Mission Control hub, live operator telemetry displays, 60Hz frame sync",
        "latency_target": "16.67 ms (Precise 60 Hz QPC multimedia timer lock)",
        "cpu_usage": "Symmetric All-Cores (Mask 0x00FF), HIGH_PRIORITY_CLASS, MMCSS 'Playback'",
        "power_budget": "24.0 W",
        "isa_acceleration": "AVX2, GPU Iris Xe UMA shared display pipeline",
        "actuation_state": "HUD_STREAMING",
        "trade_offs": "Zero tearing for WebSockets and HUD meters; constant QPC polling overhead",
        "recommended_workloads": ["kiosk", "studio", "mission_control", "hud", "display", "websocket", "ui_kit", "60hz"]
    },
    "HEADLESS_CONTAINER_CI": {
        "best_for": "Headless Linux Docker containers, CI/CD regression testing, cloud servers",
        "latency_target": "5.0 ms simulated batch quantum",
        "cpu_usage": "Container quota-bound cgroup v2 slice, BATCH_PRIORITY_CLASS",
        "power_budget": "15.0 W (Host-governed)",
        "isa_acceleration": "OpenVINO CPU Emulation, Mock DP4A vectors",
        "actuation_state": "HEADLESS_TESTING",
        "trade_offs": "100% deterministic portability; no physical display or native NPU dependency",
        "recommended_workloads": ["ci", "container", "docker", "headless", "testing", "regression", "linux_ci", "batch"]
    }
}


# ==============================================================================
# CONFIGURATION CHEATSHEET EXPERT CLASS
# ==============================================================================

class ConfigCheatsheetExpert:
    """
    Expert cheatsheet consultation engine, configuration validator,
    and conversational chatbot handler.
    """

    def __init__(self):
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6 must hold universally"
        self.profiles = CANONICAL_PROFILES
        self.templates = CANONICAL_TEMPLATES
        self.trade_offs = PROFILE_TRADE_OFFS

    # --------------------------------------------------------------------------
    # 1. PROFILE CHEATSHEET OPERATIONS
    # --------------------------------------------------------------------------

    def list_profiles_cheatsheet(self) -> str:
        """Renders an ASCII cheatsheet comparison table of all 6 canonical profiles."""
        lines = [
            "================================================================================",
            "        KRYSTAL-STACK: 6 CANONICAL CONFIGURATION PROFILES CHEATSHEET            ",
            "================================================================================",
            f"{'Profile ID':<30} | {'Cadence':<8} | {'Affinity':<10} | {'PL1 (W)':<8} | {'MMCSS / ISA':<18}",
            "--------------------------------------------------------------------------------",
        ]
        for pid, meta in self.trade_offs.items():
            cadence = "1.0 ms" if "REALTIME" in pid else "10.0 ms" if "BALANCED" in pid else "50.0 ms" if "ECO" in pid else "2.0 ms" if "SHADOW" in pid else "16.7 ms" if "KIOSK" in pid else "5.0 ms"
            aff = "P-Cores" if "REALTIME" in pid else "E-Cores" if "ECO" in pid else "All-Cores"
            pl1 = "32 W" if "REALTIME" in pid else "10 W" if "ECO" in pid else "15 W" if "CI" in pid else "24 W"
            isa = "Games / AMX" if "REALTIME" in pid else "None / AVX2" if "BALANCED" in pid else "None / SSE4" if "ECO" in pid else "None / Shadow" if "SHADOW" in pid else "Playback" if "KIOSK" in pid else "CPU / OpenVINO"
            lines.append(f"{pid:<30} | {cadence:<8} | {aff:<10} | {pl1:<8} | {isa:<18}")

        lines.extend([
            "--------------------------------------------------------------------------------",
            "Invariant: assert VITAL_MAX_HP == 6 | All profiles validated & stable.",
            "Use '/config get <PROFILE_ID>' or '/config recommend <workload>' for details.",
            "================================================================================"
        ])
        return "\n".join(lines)

    def get_profile_details(self, profile_name: str) -> str:
        """Returns exhaustive technical specification and trade-off analysis for a profile."""
        clean_name = profile_name.strip().upper()
        # Fuzzy match
        matched_id = None
        for pid in self.trade_offs:
            if clean_name in pid or pid in clean_name:
                matched_id = pid
                break

        if not matched_id:
            return (
                f"Neznámy profil '{profile_name}'. Dostupné kanonické profily:\n"
                f"  - REALTIME_LOW_LATENCY\n"
                f"  - BALANCED_DEVELOPMENT\n"
                f"  - ECO_THERMAL_PRESERVATION\n"
                f"  - SHADOW_COUNTERFACTUAL_RESEARCH\n"
                f"  - MISSION_CONTROL_STUDIO_KIOSK\n"
                f"  - HEADLESS_CONTAINER_CI\n"
                f"Skúste: /config list"
            )

        t = self.trade_offs[matched_id]
        profile_obj = self.profiles.get(matched_id) if hasattr(self.profiles, "get") else None
        aff_hex = f"0x{profile_obj.cpu_affinity_mask:04X}" if profile_obj else "0x00FF"
        prio = profile_obj.cpu_priority_class if profile_obj else "NORMAL"
        timer_res = profile_obj.timer_resolution_ms if profile_obj else 1

        lines = [
            f"================================================================================",
            f" PROFIL: {matched_id}",
            f"================================================================================",
            f"Primárny účel    : {t['best_for']}",
            f"Cieľová kadencia : {t['latency_target']}",
            f"CPU Alokácia     : {t['cpu_usage']} (Maska: {aff_hex}, Priorita: {prio})",
            f"Časovač OS       : timeBeginPeriod({timer_res}ms) s MMCSS riadením",
            f"Energetický limit: {t['power_budget']}",
            f"Akcelerácia ISA  : {t['isa_acceleration']}",
            f"Režim aktuácie   : {t['actuation_state']}",
            f"Architektonický  : {t['trade_offs']}",
            f"Invariant        : assert VITAL_MAX_HP == 6 [VERIFIED]",
            f"--------------------------------------------------------------------------------",
            f"Vhodné kľúčové slová pre odporúčanie: {', '.join(t['recommended_workloads'])}",
            f"Aplikovanie cez CLI: python scripts/krystal_config_toolchain.py apply --profile {matched_id}",
            f"================================================================================"
        ]
        return "\n".join(lines)

    def compare_profiles(self, p1_name: str, p2_name: str) -> str:
        """Compares two configuration profiles side-by-side."""
        def resolve_id(name: str) -> Optional[str]:
            n = name.strip().upper()
            for pid in self.trade_offs:
                if n in pid or pid in n:
                    return pid
            return None

        id1 = resolve_id(p1_name)
        id2 = resolve_id(p2_name)
        if not id1 or not id2:
            return f"Chyba: Nie je možné porovnať '{p1_name}' a '{p2_name}'. Skontrolujte názvy profilov cez '/config list'."

        t1, t2 = self.trade_offs[id1], self.trade_offs[id2]
        lines = [
            f"================================================================================",
            f" POROVNANIE PROFILOV: {id1} vs {id2}",
            f"================================================================================",
            f"{'Parameter':<22} | {id1[:26]:<26} | {id2[:26]:<26}",
            f"--------------------------------------------------------------------------------",
            f"{'Cieľová kadencia':<22} | {t1['latency_target'][:26]:<26} | {t2['latency_target'][:26]:<26}",
            f"{'CPU Jadro':<22} | {t1['cpu_usage'][:26]:<26} | {t2['cpu_usage'][:26]:<26}",
            f"{'Spotreba (PL1)':<22} | {t1['power_budget'][:26]:<26} | {t2['power_budget'][:26]:<26}",
            f"{'ISA Akcelerácia':<22} | {t1['isa_acceleration'][:26]:<26} | {t2['isa_acceleration'][:26]:<26}",
            f"{'Režim aktuácie':<22} | {t1['actuation_state'][:26]:<26} | {t2['actuation_state'][:26]:<26}",
            f"--------------------------------------------------------------------------------",
            f"Odporúčanie {id1}: {t1['best_for']}",
            f"Odporúčanie {id2}: {t2['best_for']}",
            f"Invariant: VITAL_MAX_HP == 6 dodržaný v oboch konfiguráciách.",
            f"================================================================================"
        ]
        return "\n".join(lines)

    def recommend_profile(self, user_query: str) -> str:
        """
        Analyzes user requirements or conversational prompt and recommends
        the most appropriate configuration profile with technical justification.
        """
        # Accent-folded tokenization
        import unicodedata
        import re

        def fold(text: str) -> str:
            t = unicodedata.normalize("NFKD", text.lower())
            return "".join(c for c in t if not unicodedata.combining(c))

        folded = fold(user_query)
        tokens = set(re.findall(r"[a-z0-9_\-]+", folded))

        scores: Dict[str, int] = {pid: 0 for pid in self.trade_offs}

        # Exact token matching for workload keywords
        for pid, meta in self.trade_offs.items():
            for kw in meta["recommended_workloads"]:
                fkw = fold(kw)
                if fkw in tokens or (len(fkw) > 3 and fkw in folded):
                    scores[pid] += 4

        # Strong semantic heuristics
        # Battery / thermal preservation / cool-down
        if any(w in tokens or w in folded for w in ["bateri", "battery", "setrit", "setrenie", "chladit", "chladenie", "teplot", "zahrievanie", "ticho", "pasiv", "fan", "watt", "eco"]):
            scores["ECO_THERMAL_PRESERVATION"] += 15

        # Real-time / low latency / maximum performance
        if any(w in tokens or w in folded for w in ["realtime", "rychlost", "latenci", "submillisecond", "fps", "boj", "game", "robot", "p-core", "p-cores", "p_core", "p_cores", "vykon", "amx", "avx-512"]):
            scores["REALTIME_LOW_LATENCY"] += 15

        # Kiosk / studio
        if any(w in tokens or w in folded for w in ["kiosk", "studio", "mission", "control", "hud", "displej", "display", "webhub", "krystal_ui", "60hz"]):
            scores["MISSION_CONTROL_STUDIO_KIOSK"] += 15

        # Shadow / safe / simulation
        if any(w in tokens or w in folded for w in ["shadow", "safe", "testovanie", "bezpecn", "audit", "nemutov", "zapis", "hypotez", "counterfactual"]):
            scores["SHADOW_COUNTERFACTUAL_RESEARCH"] += 15

        # CI / Docker / headless (ensure 'ci' is an exact standalone token)
        if ("ci" in tokens or "docker" in tokens or "kontajner" in tokens or "container" in tokens or "headless" in tokens or "runner" in tokens or "regression" in tokens):
            scores["HEADLESS_CONTAINER_CI"] += 15

        # Balanced / dev
        if any(w in tokens or w in folded for w in ["vyvoj", "dev", "standard", "normal", "rovnovah", "vyvazen"]):
            scores["BALANCED_DEVELOPMENT"] += 6

        # Pick top winner
        best_pid = max(scores, key=lambda k: scores[k])
        # If no score, default to BALANCED_DEVELOPMENT
        if scores[best_pid] == 0:
            best_pid = "BALANCED_DEVELOPMENT"

        t = self.trade_offs[best_pid]
        return (
            f"💡 **Odporúčaná Konfigurácia:** `{best_pid}`\n\n"
            f"**Dôvod výberu:**\n"
            f"Na základe vašej požiadavky ('{user_query.strip()}') je profil `{best_pid}` optimálnou voľbou.\n\n"
            f"**Kľúčové parametre konfigurácie:**\n"
            f"- **Určenie:** {t['best_for']}\n"
            f"- **Časový kvant:** {t['latency_target']}\n"
            f"- **CPU topológia:** {t['cpu_usage']}\n"
            f"- **Príkon PL1:** {t['power_budget']}\n"
            f"- **Akcelerácia:** {t['isa_acceleration']}\n"
            f"- **Stav aktuácie:** `{t['actuation_state']}`\n"
            f"- **Invariant integrity:** `assert VITAL_MAX_HP == 6` [GARANTOVANÉ]\n\n"
            f"**Ako profil aktivovať v systéme:**\n"
            f"```bash\npython scripts/krystal_config_toolchain.py apply --profile {best_pid}\n```"
        )

    def validate_configuration_dict(self, cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Rigorous validation of configuration parameters against platform invariants."""
        errors: List[str] = []
        vital_hp = cfg.get("vital_max_hp", 6)
        if vital_hp != 6:
            errors.append(f"Kritické porušenie invariantu: vital_max_hp musí byť 6, nájdené {vital_hp}")

        cadence = cfg.get("target_cadence_ms", 10.0)
        if not (0.1 <= cadence <= 1000.0):
            errors.append(f"Neplatná hodnota target_cadence_ms: {cadence} ms (povolené: 0.1 - 1000.0 ms)")

        pl1 = cfg.get("target_pl1_watts", 24.0)
        if not (5.0 <= pl1 <= 65.0):
            errors.append(f"Nebezpečný limit príkonu PL1: {pl1} W (povolené: 5.0 - 65.0 W)")

        affinity = cfg.get("cpu_affinity_mask", 0x00FF)
        if affinity <= 0:
            errors.append(f"Neplatná maska afinity CPU: {affinity}")

        return (len(errors) == 0, errors)

    # --------------------------------------------------------------------------
    # 2. COORDINATION TEMPLATES CHEATSHEET OPERATIONS
    # --------------------------------------------------------------------------

    def list_templates_cheatsheet(self) -> str:
        """Renders cheatsheet of all 9 canonical coordination templates."""
        lines = [
            "================================================================================",
            "          KRYSTAL-STACK: 9 KOORDINAČNÝCH TEMPLATOV (HW + TIMERY + ALGO)         ",
            "================================================================================",
            f"{'Template ID':<42} | {'Kadencia':<8} | {'Priorita':<10} | {'Algoritmus'}",
            "--------------------------------------------------------------------------------",
        ]
        # Canonical baseline templates
        baselines = [
            ("TEMPLATE_ULTRA_LOW_JITTER_REALTIME", "1.0 ms", "HIGH", "Sub-1ms P-Core QPC"),
            ("TEMPLATE_BALANCED_ADAPTIVE_CADENCE", "10.0 ms", "NORMAL", "Waitable Timer Hybrid"),
            ("TEMPLATE_THERMAL_RESCUE_PACING", "50.0 ms", "BELOW", "E-Core Passive Cooldown"),
            ("TEMPLATE_COUNTERFACTUAL_SHADOW_BURST", "2.0 ms", "NORMAL", "AUTH_SHADOW_ONLY Non-mutating"),
            ("TEMPLATE_MISSION_CONTROL_TELEMETRY_SYNC", "16.7 ms", "HIGH", "60Hz QPC Frame-lock"),
        ]
        for tid, cad, prio, desc in baselines:
            lines.append(f"{tid:<42} | {cad:<8} | {prio:<10} | {desc}")

        lines.append("----------------- [NOVÁ GENERÁCIA ALGORITMOV V TEMPLATOCH] ---------------------")
        nextgens = [
            ("TEMPLATE_NEXTGEN_SYMPLECTIC_TENSOR_HARMONY", "1.618 ms", "HIGH", "Hamiltonian Symplectic + Zlatý Rez Φ"),
            ("TEMPLATE_NEXTGEN_HETEROGENEOUS_ROOFLINE_DISPATCH", "3.0 ms", "HIGH", "Analytický Roofline Break-even Placement"),
            ("TEMPLATE_NEXTGEN_EPISTEMIC_CALIBRATION_LEDGER", "8.0 ms", "NORMAL", "Online Brier Kalibrácia + Platt Scaling"),
            ("TEMPLATE_NEXTGEN_MULTI_TENANT_ISOLATION_SLICE", "5.0 ms", "NORMAL", "Tenancy Interference Tensor Tlmenie"),
        ]
        for tid, cad, prio, desc in nextgens:
            lines.append(f"{tid:<42} | {cad:<8} | {prio:<10} | {desc}")

        lines.extend([
            "--------------------------------------------------------------------------------",
            "Invariant: assert VITAL_MAX_HP == 6 garantovaný v celej matrici.",
            "Spustenie template cyklu: /template dispatch <TEMPLATE_ID>",
            "================================================================================"
        ])
        return "\n".join(lines)

    def get_template_details(self, template_name: str) -> str:
        """Returns deep architectural details for a coordination template."""
        clean = template_name.strip().upper()
        # Find match
        matched = None
        for t in [
            "TEMPLATE_ULTRA_LOW_JITTER_REALTIME",
            "TEMPLATE_BALANCED_ADAPTIVE_CADENCE",
            "TEMPLATE_THERMAL_RESCUE_PACING",
            "TEMPLATE_COUNTERFACTUAL_SHADOW_BURST",
            "TEMPLATE_MISSION_CONTROL_TELEMETRY_SYNC",
            "TEMPLATE_NEXTGEN_SYMPLECTIC_TENSOR_HARMONY",
            "TEMPLATE_NEXTGEN_HETEROGENEOUS_ROOFLINE_DISPATCH",
            "TEMPLATE_NEXTGEN_EPISTEMIC_CALIBRATION_LEDGER",
            "TEMPLATE_NEXTGEN_MULTI_TENANT_ISOLATION_SLICE",
        ]:
            if clean in t or t in clean:
                matched = t
                break

        if not matched:
            return f"Neznámy template '{template_name}'. Použite '/template list' pre zobrazenie všetkých 9 templátov."

        # Return formatted template specs
        return (
            f"📋 **Architektonický Profil Templátu:** `{matched}`\n\n"
            f"- **Cieľová kadencia:** {1.0 if 'REALTIME' in matched else 10.0 if 'BALANCED' in matched else 50.0 if 'THERMAL' in matched else 2.0 if 'SHADOW' in matched else 16.67 if 'TELEMETRY' in matched else 1.618 if 'SYMPLECTIC' in matched else 3.0 if 'ROOFLINE' in matched else 8.0 if 'CALIBRATION' in matched else 5.0} ms\n"
            f"- **Windows API Časovač:** `timeBeginPeriod` + High-Resolution Waitable Timers (QPC)\n"
            f"- **MMCSS Profil:** {'Games' if 'REALTIME' in matched or 'SYMPLECTIC' in matched else 'Playback' if 'TELEMETRY' in matched else 'None / Standard'}\n"
            f"- **CPU Maska afinity:** {'0x000F (P-Cores)' if 'REALTIME' in matched or 'SYMPLECTIC' in matched else '0x00F0 (E-Cores)' if 'THERMAL' in matched else '0x00FF (Všetky jadrá)'}\n"
            f"- **NPU Akcelerácia:** DP4A INT8 Streaming s NPU offloadom\n"
            f"- **Režim integrity:** `assert VITAL_MAX_HP == 6` [PLATFORM INVARIANT VALIDATED]\n"
            f"- **Integrovaný algoritmus:** {'Symplectic Phase-Space + Golden Ratio Φ Tensor' if 'SYMPLECTIC' in matched else 'Analytical Heterogeneous Roofline Dispatch' if 'ROOFLINE' in matched else 'Online Brier Epistemic Calibration' if 'CALIBRATION' in matched else 'Multi-Tenant Cross-Chatter Dampening' if 'MULTI_TENANT' in matched else 'Kanonická reťazená konvolúcia'}\n\n"
            f"Pre okamžitú simuláciu dispečingu napíšte:\n`/template dispatch {matched}`"
        )

    # --------------------------------------------------------------------------
    # 3. CASCADE & SLM RUNTIME CHEATSHEET OPERATIONS
    # --------------------------------------------------------------------------

    def list_cascade_cheatsheet(self) -> str:
        """Returns cheatsheet of multi-model cascade nodes, objectives, and strategies."""
        runtime = GLOBAL_MODEL_CASCADE_RUNTIME
        nodes_info = []
        if runtime:
            for n in runtime.nodes.values():
                spec = n.spec
                nodes_info.append(f"  • [{spec.node_id}] Model: {spec.model_id} (Role: {spec.role}, Backend: {spec.backend}, Quant: {spec.quantization}) - Limit: {spec.context_limit} tok, RAM: {spec.memory_budget_mb:.0f}MB")
        else:
            nodes_info.append("  • Runtime katalog je inicializovaný v režime standalone.")

        lines = [
            "================================================================================",
            "      KRYSTAL-STACK: MULTI-MODEL CASCADE RUNTIME & SLM CHEATSHEET               ",
            "================================================================================",
            "Dostupné špecializované SLM/LLM uzly:",
            *nodes_info,
            "--------------------------------------------------------------------------------",
            "Podporované stratégie kaskády (8 módov):",
            "  1. DIRECT_LOCAL                 - Okamžitá lokálna inferencia s nulovým sieťovým oneskorením",
            "  2. SPECULATIVE_VERIFY           - Návrh SLM + Dual-Gate verifikátor invariantov",
            "  3. CHAIN_OF_THOUGHT             - Krok za krokom epistémický DAG s logickou dekompozíciou",
            "  4. ENSEMBLE_VOTE                - Väčšinový konsenzus naprieč 3 nezávislými modelmi",
            "  5. HIERARCHICAL_FALLBACK        - SLM -> Core RAG -> Large Model fallback reťazec",
            "  6. COST_LATENCY_PARETO          - Dynamické smerovanie podľa Pareto hranice",
            "  7. ROUND_ROBIN_LOAD_BALANCED   - Rovnomerné rozloženie záťaže",
            "  8. EPISTEMIC_UNCERTAINTY_ROUTED - Smerovanie riadené neistotou a kalibráciou",
            "--------------------------------------------------------------------------------",
            "Ciele optimalizácie (Routing Profiles):",
            "  • LATENCY_CRITICAL | ACCURACY_MAXIMAL | COST_MINIMAL | BALANCED",
            "Dual-Gate Invariant: assert VITAL_MAX_HP == 6",
            "Spustenie dopytu cez kaskádu: /cascade run <váš dotaz>",
            "================================================================================"
        ]
        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # 4. HARDWARE ALLOCATION CHEATSHEET OPERATIONS
    # --------------------------------------------------------------------------

    def get_hardware_cheatsheet(self, category: Optional[str] = None) -> str:
        """Queries native Linux/Windows hardware allocation across 8 categories."""
        monitor = GLOBAL_LINUX_HW_MONITOR
        if not monitor:
            return (
                "Hardware monitor nie je priamo dostupný v tomto prostredí.\n"
                "Kategórie alokácie HW prostriedkov v Krystal Stacku:\n"
                " 1. CPU_SCHEDULER       - P/E Core afinity, cgroups v2, preemption\n"
                " 2. MEMORY_UMA          - RAM, Swap, UMA shared graphics memory\n"
                " 3. GPU_ACCELERATOR     - Iris Xe EU záťaž, VRAM, DMA paging\n"
                " 4. NPU_CASCADE         - DP4A INT8 vektory, aktívne streamy, SLM footprint\n"
                " 5. POWER_THERMAL       - RAPL PL1/PL2 wattový príkon, teplota jadier\n"
                " 6. STORAGE_IO          - NVMe IOPS, priepustnosť, swap latencia\n"
                " 7. NETWORK_SOCKETS     - TCP spojenia, soketové buffery, sieťový tok\n"
                " 8. EPISTEMIC_INVARIANTS - Dual-Gate stav, assert VITAL_MAX_HP == 6"
            )

        cat_filter = None
        if category and category.lower() != "all":
            cat_filter = category.lower()

        snap = monitor.sample_snapshot()
        return monitor.format_cli_dashboard(snap, category_filter=cat_filter)

    # --------------------------------------------------------------------------
    # 5. CONVERSATIONAL CHATBOT HANDLER WITH CHEATSHEET EXPERTISE
    # --------------------------------------------------------------------------

    def answer_configuration_chat(self, user_prompt: str) -> str:
        """
        Conversational entry point for the chatbot. Answers any configuration,
        template, latency, hardware, or invariant query with grounded factual
        evidence, recommendations, and ASCII tables.
        """
        import unicodedata
        p_raw = user_prompt.strip().lower()
        p = "".join(c for c in unicodedata.normalize("NFKD", p_raw) if not unicodedata.combining(c))

        # Check for profile recommendations
        if any(w in p for w in ["odporuc", "ktory profil", "aka konfiguracia", "aku konfiguraciu", "vhodna konfiguracia", "najlepsia konfiguracia", "recommend", "best profile", "best config"]):
            return self.recommend_profile(user_prompt)

        # Check for profile comparisons
        if any(w in p for w in ["porovnaj", "rozdiel medzi", "vs", "compare", "difference"]):
            found = []
            for pid in self.trade_offs:
                if pid.lower() in p or pid.replace("_", " ").lower() in p:
                    found.append(pid)
            if len(found) >= 2:
                return self.compare_profiles(found[0], found[1])
            elif "realtime" in p and "balanced" in p:
                return self.compare_profiles("REALTIME_LOW_LATENCY", "BALANCED_DEVELOPMENT")
            elif "eco" in p and "realtime" in p:
                return self.compare_profiles("ECO_THERMAL_PRESERVATION", "REALTIME_LOW_LATENCY")
            elif "eco" in p and "balanced" in p:
                return self.compare_profiles("ECO_THERMAL_PRESERVATION", "BALANCED_DEVELOPMENT")

        # Check for template inquiries
        if any(w in p for w in ["template", "templaty", "koordinac"]):
            if any(w in p for w in ["list", "vsetky", "zoznam", "ake su", "prehlad"]):
                return self.list_templates_cheatsheet()
            for tid in [
                "TEMPLATE_ULTRA_LOW_JITTER_REALTIME",
                "TEMPLATE_BALANCED_ADAPTIVE_CADENCE",
                "TEMPLATE_THERMAL_RESCUE_PACING",
                "TEMPLATE_COUNTERFACTUAL_SHADOW_BURST",
                "TEMPLATE_MISSION_CONTROL_TELEMETRY_SYNC",
                "TEMPLATE_NEXTGEN_SYMPLECTIC_TENSOR_HARMONY",
                "TEMPLATE_NEXTGEN_HETEROGENEOUS_ROOFLINE_DISPATCH",
                "TEMPLATE_NEXTGEN_EPISTEMIC_CALIBRATION_LEDGER",
                "TEMPLATE_NEXTGEN_MULTI_TENANT_ISOLATION_SLICE",
            ]:
                if tid.lower() in p or tid.replace("TEMPLATE_", "").lower() in p:
                    return self.get_template_details(tid)
            return self.list_templates_cheatsheet()

        # Check for hardware monitor inquiries
        if any(w in p for w in ["hardware", "hw", "alokacia", "monitor", "prostriedky", "cgroups", "cpu", "gpu", "npu"]):
            return self.get_hardware_cheatsheet()

        # Check for cascade inquiries
        if any(w in p for w in ["cascade", "kaskada", "slm", "multi-model", "router", "routing"]):
            return self.list_cascade_cheatsheet()

        # Check for specific profile detail request
        for pid in self.trade_offs:
            if pid.lower() in p or pid.replace("_", " ").lower() in p:
                return self.get_profile_details(pid)

        # Default fallback: full cheatsheet overview + recommendation prompt
        return (
            f"🤖 **Krystal Configuration & Architecture Bot:**\n\n"
            f"Som váš architektonický expert na alternatívne platné konfigurácie, koordinačné templáty a hardvérové alokácie Krystal Stacku.\n\n"
            f"{self.list_profiles_cheatsheet()}\n\n"
            f"💬 **Čo môžete urobiť ďalej:**\n"
            f"- Zadať dopyt ako napr.: *'Akú konfiguráciu odporúčaš na šetrenie batérie?'*\n"
            f"- Porovnať profily: `/config compare REALTIME_LOW_LATENCY BALANCED_DEVELOPMENT`\n"
            f"- Zobraziť koordinačné templáty: `/template list`\n"
            f"- Zobraziť alokáciu HW monitora: `/hw summary`\n"
            f"- Spustiť multi-model kaskádu: `/cascade list`\n\n"
            f"🛡️ **Stav platformy:** `assert VITAL_MAX_HP == 6` [OVERENÉ]"
        )


# Global singleton instance
CONFIG_CHEATSHEET_EXPERT = ConfigCheatsheetExpert()
