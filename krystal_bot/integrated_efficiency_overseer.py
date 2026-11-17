"""Integrovaný systém dohľadu nad efektivitou (`IntegratedEfficiencyOverseer`) pre `krystal_bot`:
automaticky vyberá:
  1. Stratégiu zapínania a parkovania jadier (`CORE_WAKE_01..06`),
  2. OpenVINO multithreading preset (`OV_MT_01..05`),
  3. Konkrétny vykonávací skript a modul platformy Krystal Stack,
pod dohľadom ústavného invariantu `VITAL_MAX_HP == 6` (`\\neg Valid => \\Delta S = 0`) a 12-krokovej hysterézie.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from krystal_kernel.constants import UMA_SATURATION_THRESHOLD, VITAL_MAX_HP, ContractViolation


CORE_ACTIVATION_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "CORE_WAKE_01_IDLE_1C_DEEP_SLEEP": {
        "active_core_mask": [0],
        "smt_enabled": False,
        "igpu_awake": False,
        "npu_awake": False,
        "target_pl1_watts": 9.0,
        "popis_sk": "1 aktívne fyzické jadro, ostatné jadrá v hlbokom spánku C6/C8 pre minimálny odber.",
    },
    "CORE_WAKE_02_UMA_THERMAL_SHIELD_2C": {
        "active_core_mask": [0, 1],
        "smt_enabled": False,
        "igpu_awake": False,
        "npu_awake": False,
        "target_pl1_watts": 15.0,
        "popis_sk": "2 fyzické jadrá, okamžité odpojenie iGPU zbernice pri UMA >= 0.78 alebo teplote >= 85 C.",
    },
    "CORE_WAKE_03_REALTIME_4P_ZERO_JITTER": {
        "active_core_mask": [0, 1, 2, 3],
        "smt_enabled": False,
        "igpu_awake": False,
        "npu_awake": False,
        "target_pl1_watts": 22.0,
        "popis_sk": "4 fyzické P-jadrá bez SMT súrodencov — eliminuje 2.45x chvostové kmitanie latencie pri 120 Hz.",
    },
    "CORE_WAKE_04_HETERO_3P_1VSYNC_IRIS_XE": {
        "active_core_mask": [0, 1, 2, 3],
        "smt_enabled": False,
        "igpu_awake": True,
        "npu_awake": False,
        "target_pl1_watts": 28.0,
        "popis_sk": "3 fyzické jadrá logika + 1 jadro 133 ns BAR0/VSync + prebudených 768 prúdových jednotiek Iris Xe.",
    },
    "CORE_WAKE_05_HYBRID_GEN12_PCORE_ECORE_NPU": {
        "active_core_mask": [0, 1, 2, 3],
        "smt_enabled": False,
        "igpu_awake": True,
        "npu_awake": True,
        "target_pl1_watts": 20.0,
        "popis_sk": "Hybridný dohľad pre Intel Gen12+ a Core Ultra: odozva na P-jadrách, telemetria na E-jadrách a NPU.",
    },
    "CORE_WAKE_06_BATCH_BURST_8T_ALL_THREADS": {
        "active_core_mask": [0, 1, 2, 3, 4, 5, 6, 7],
        "smt_enabled": True,
        "igpu_awake": True,
        "npu_awake": False,
        "target_pl1_watts": 28.0,
        "popis_sk": "Plné prebudenie všetkých 8 logických vlákien a 4 OpenVINO prúdov pre hromadné dávkové výpočty.",
    },
}


GOVERNED_SCRIPT_REGISTRY: Dict[str, Dict[str, str]] = {
    "realtime_120hz": {
        "script": "projects/trigonometric_hybrid_platform_framework/universal_hardware_mmio_aperture_kernel.py",
        "openvino_preset": "OV_MT_01_REALTIME_120HZ_LOW_JITTER",
        "core_strategy": "CORE_WAKE_03_REALTIME_4P_ZERO_JITTER",
    },
    "coprocessor_visual_ai": {
        "script": "projects/trigonometric_hybrid_platform_framework/hybrid_rust_and_intel_xe_cuda_streams.py",
        "openvino_preset": "OV_MT_03_HETERO_IRIS_XE_SINGLE_STREAM_PLUS_3T_CPU",
        "core_strategy": "CORE_WAKE_04_HETERO_3P_1VSYNC_IRIS_XE",
    },
    "batch_benchmark": {
        "script": "scripts/run_stock_vs_krystal_benchmarks.py",
        "openvino_preset": "OV_MT_02_BATCH_THROUGHPUT_MAX_STREAMS",
        "core_strategy": "CORE_WAKE_06_BATCH_BURST_8T_ALL_THREADS",
    },
    "hybrid_background_telemetry": {
        "script": "projects/trigonometric_hybrid_platform_framework/monetization_corporate_and_openvino_addon_kernel.py",
        "openvino_preset": "OV_MT_04_HYBRID_GEN12_PLUS_ECORE_TELEMETRY",
        "core_strategy": "CORE_WAKE_05_HYBRID_GEN12_PCORE_ECORE_NPU",
    },
    "idle_eco": {
        "script": "projects/trigonometric_hybrid_platform_framework/priority_tabular_synthetic_logger.py",
        "openvino_preset": "OV_MT_01_REALTIME_120HZ_LOW_JITTER",
        "core_strategy": "CORE_WAKE_01_IDLE_1C_DEEP_SLEEP",
    },
    "uma_thermal_emergency": {
        "script": "projects/trigonometric_hybrid_platform_framework/krystal_uma_cuda_vulkan_library.py",
        "openvino_preset": "OV_MT_05_UMA_SATURATED_THERMAL_ECO_SHIELD",
        "core_strategy": "CORE_WAKE_02_UMA_THERMAL_SHIELD_2C",
    },
}


class IntegratedEfficiencyOverseer:
    """Integrovaný dozorný systém efektivity bota: sleduje UMA tlak, teplotu, zaťaženie CPU a typ úlohy
    a deterministicky volí stratégiu zapínania jadier, OpenVINO preset aj cieľový skript.
    """

    def __init__(self) -> None:
        if VITAL_MAX_HP != 6:
            raise ContractViolation(f"[ERR_OVERSEER_HP] Očakávané VITAL_MAX_HP == 6, zistené {VITAL_MAX_HP}")
        self._active_mode: str = "realtime_120hz"
        self._dwell_counter: int = 0
        self._decisions_count: int = 0

    def evaluate_and_select(
        self,
        workload_intent: str = "coprocessor_visual_ai",
        uma_utilization: float = 0.36,
        junction_temp_c: float = 61.0,
        cpu_load_pct: float = 42.0,
        vital_hp: int = VITAL_MAX_HP,
    ) -> Dict[str, Any]:
        if vital_hp != VITAL_MAX_HP:
            raise ContractViolation("[ERR_OVERSEER_DELTA_S_ZERO] Neplatné VITAL_MAX_HP — zmena stavu zamietnutá")
        if not (0.0 <= uma_utilization <= 1.0):
            raise ContractViolation("[ERR_OVERSEER_UMA] Hodnota UMA mimo rozsahu 0..1")

        # Okamžitý ochranný zostup pri saturácii UMA zbernice alebo vysokej teplote (Princíp P8)
        if uma_utilization >= UMA_SATURATION_THRESHOLD or junction_temp_c >= 85.0:
            resolved_mode = "uma_thermal_emergency"
            self._dwell_counter = 0
            self._active_mode = resolved_mode
        elif cpu_load_pct < 8.0 and workload_intent == "idle_eco":
            resolved_mode = "idle_eco"
            self._active_mode = resolved_mode
        else:
            candidate = workload_intent if workload_intent in GOVERNED_SCRIPT_REGISTRY else "realtime_120hz"
            if self._active_mode == "uma_thermal_emergency" and self._dwell_counter < 12:
                self._dwell_counter += 1
                resolved_mode = "uma_thermal_emergency"
            else:
                self._dwell_counter = 0
                self._active_mode = candidate
                resolved_mode = candidate

        route = GOVERNED_SCRIPT_REGISTRY[resolved_mode]
        core_strat = CORE_ACTIVATION_STRATEGIES[route["core_strategy"]]
        self._decisions_count += 1

        # Výpočet indexu energeticko-časovej efektivity (užitočný výkon / wattový strop)
        watts = float(core_strat["target_pl1_watts"])
        speedup_factor = 4.41 if core_strat["igpu_awake"] else (2.79 if len(core_strat["active_core_mask"]) == 4 else 1.50)
        efficiency_score = round((speedup_factor * 100.0) / watts, 2)

        return {
            "vital_max_hp": VITAL_MAX_HP,
            "decision_sequence": self._decisions_count,
            "resolved_mode": resolved_mode,
            "selected_script": route["script"],
            "selected_openvino_preset": route["openvino_preset"],
            "selected_core_activation_strategy": route["core_strategy"],
            "core_activation_details": core_strat,
            "efficiency_overseer_index": efficiency_score,
            "hysteresis_dwell_step": self._dwell_counter,
        }

    def format_slovak_summary(self, arg: str = "") -> str:
        intent = arg.strip().lower() if arg.strip() else "coprocessor_visual_ai"
        verdict = self.evaluate_and_select(workload_intent=intent)
        det = verdict["core_activation_details"]
        return (
            f"Integrovaný dozor efektivity (VITAL_MAX_HP == {VITAL_MAX_HP})\n"
            f"- Zvolený režim: {verdict['resolved_mode']}\n"
            f"- Vybraný skript: {verdict['selected_script']}\n"
            f"- OpenVINO preset: {verdict['selected_openvino_preset']}\n"
            f"- Stratégia zapínania jadier: {verdict['selected_core_activation_strategy']} "
            f"(aktívne jadrá={det['active_core_mask']}, iGPU={det['igpu_awake']}, PL1={det['target_pl1_watts']} W)\n"
            f"- Index efektivity: {verdict['efficiency_overseer_index']}"
        )


INTEGRATED_EFFICIENCY_OVERSEER = IntegratedEfficiencyOverseer()
