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


class SovereignResourceArbitrationEngine:
    """Arbitrážny rozhodovací systém rozdeľujúci vzácnu šírku pásma UMA (< 0.78), tepelný rozpočet (W),
    fyzické CPU jadrá a koprocesorové sloty medzi súperiace úlohy podľa medznej užitočnosti a tieňovej ceny.
    """

    TIER_WEIGHTS: Dict[str, float] = {"P0": 100.0, "P1": 10.0, "P2": 1.0}

    def __init__(self, overseer: IntegratedEfficiencyOverseer | None = None) -> None:
        if VITAL_MAX_HP != 6:
            raise ContractViolation("[ERR_ARB_HP] VITAL_MAX_HP musí byť 6")
        self.overseer = overseer or IntegratedEfficiencyOverseer()

    def clear_arbitration_round(
        self,
        bids: List[Dict[str, Any]] | None = None,
        current_uma: float = 0.34,
        junction_temp_c: float = 61.0,
        power_cap_watts: float = 28.0,
    ) -> Dict[str, Any]:
        if not (0.0 <= current_uma <= 1.0):
            raise ContractViolation("[ERR_ARB_UMA] Neplatný stav UMA zbernice")

        default_bids = [
            {"agent_id": "UI_120HZ_SCANOUT_GUARD", "tier": "P0", "utility": 95.0, "uma_delta": 0.12, "watts": 4.5, "preferred_unit": "CPU_VSYNC_CORE"},
            {"agent_id": "XERJ_AST_AND_OPEN_VINO_AI", "tier": "P1", "utility": 82.0, "uma_delta": 0.18, "watts": 9.0, "preferred_unit": "IRIS_XE_768_SP"},
            {"agent_id": "SLM_FLASH_ATTENTION_PREFILL", "tier": "P1", "utility": 74.0, "uma_delta": 0.11, "watts": 6.5, "preferred_unit": "CPU_4P_AVX512"},
            {"agent_id": "P2_OFFLINE_LOG_COMPRESSION", "tier": "P2", "utility": 25.0, "uma_delta": 0.16, "watts": 7.5, "preferred_unit": "ECORE_BACKGROUND"},
        ]
        active_bids = bids if bids is not None else default_bids

        uma_headroom = max(0.0, UMA_SATURATION_THRESHOLD - current_uma)
        watt_headroom = max(0.0, power_cap_watts - (8.0 if junction_temp_c >= 85.0 else 0.0))

        # Tieňová cena zdrojov stúpa nelineárne pri zmenšujúcej sa rezerve zbernice a výkonu
        shadow_price_uma = 1.0 / max(uma_headroom, 0.05)
        shadow_price_watt = 1.0 / max(watt_headroom, 1.0)

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for b in active_bids:
            tier = str(b.get("tier", "P2"))
            w_tier = self.TIER_WEIGHTS.get(tier, 1.0)
            cost = shadow_price_uma * float(b["uma_delta"]) + shadow_price_watt * float(b["watts"]) + 1e-6
            arb_score = (float(b["utility"]) * w_tier) / cost
            scored.append((round(arb_score, 3), b))

        scored.sort(key=lambda item: item[0], reverse=True)

        accepted: List[Dict[str, Any]] = []
        deferred_shadow: List[Dict[str, Any]] = []
        rem_uma = uma_headroom
        rem_watts = watt_headroom

        for score, b in scored:
            u_req = float(b["uma_delta"])
            w_req = float(b["watts"])
            if u_req <= rem_uma + 1e-9 and w_req <= rem_watts + 1e-9:
                rem_uma = round(rem_uma - u_req, 4)
                rem_watts = round(rem_watts - w_req, 3)
                accepted.append({**b, "arbitration_score": score, "decision": "GRANTED_IMMEDIATE"})
            else:
                deferred_shadow.append({**b, "arbitration_score": score, "decision": "DEFERRED_TO_VBLANK_SHADOW_DELTA_S_ZERO"})

        final_uma = round(current_uma + (uma_headroom - rem_uma), 4)
        gov = self.overseer.evaluate_and_select(
            workload_intent="coprocessor_visual_ai" if any(x["preferred_unit"] == "IRIS_XE_768_SP" for x in accepted) else "realtime_120hz",
            uma_utilization=final_uma,
            junction_temp_c=junction_temp_c,
        )

        return {
            "vital_max_hp": VITAL_MAX_HP,
            "shadow_prices": {"lambda_uma": round(shadow_price_uma, 3), "lambda_watt": round(shadow_price_watt, 3)},
            "final_committed_uma": final_uma,
            "remaining_watt_headroom": rem_watts,
            "accepted_agents": accepted,
            "deferred_agents": deferred_shadow,
            "governed_execution_plan": gov,
        }

    def format_slovak_arbitration(self, arg: str = "") -> str:
        uma = 0.70 if "tlak" in arg.lower() or "high" in arg.lower() else 0.34
        res = self.clear_arbitration_round(current_uma=uma)
        granted = ", ".join(f"{a['agent_id']} ({a['arbitration_score']})" for a in res["accepted_agents"])
        deferred = ", ".join(a["agent_id"] for a in res["deferred_agents"]) or "žiadne"
        return (
            f"Arbitrážne rozhodnutie zdrojov (VITAL_MAX_HP == {VITAL_MAX_HP})\n"
            f"- Tieňová cena UMA: {res['shadow_prices']['lambda_uma']} | Konečné UMA: {res['final_committed_uma']}\n"
            f"- Schválené úlohy: {granted}\n"
            f"- Odložené do VBlank tieňa: {deferred}\n"
            f"- Zvolená stratégia jadier: {res['governed_execution_plan']['selected_core_activation_strategy']}"
        )


INTEGRATED_EFFICIENCY_OVERSEER = IntegratedEfficiencyOverseer()
SOVEREIGN_ARBITRATION_ENGINE = SovereignResourceArbitrationEngine(INTEGRATED_EFFICIENCY_OVERSEER)


class BotOsAndGpuOpportunityGenerator:
    """Proaktívny vyhľadávač a návrhár nových optimalizačných príležitostí pre operačný systém
    a grafický ovládač, prepojený s arbitrážnym rozhodovaním (`EXPLORE -> ARBITRATE -> VERIFY`).
    """

    def __init__(self, arbitrator: SovereignResourceArbitrationEngine | None = None) -> None:
        if VITAL_MAX_HP != 6:
            raise ContractViolation("[ERR_OPP_HP] VITAL_MAX_HP musí byť 6")
        self.arbitrator = arbitrator or SOVEREIGN_ARBITRATION_ENGINE

    def discover_and_propose_opportunities(self) -> Dict[str, Any]:
        import psutil
        from projects.trigonometric_hybrid_platform_framework.combat_turn_planner import (
            DEFAULT_CACHE_DIR,
            get_cached_hw_profile,
        )
        from projects.trigonometric_hybrid_platform_framework.intel_accelerator_and_power_thread_governor import (
            get_shared_ov_core,
        )

        vm = psutil.virtual_memory()
        proc = psutil.Process()
        hw = get_cached_hw_profile()
        _, core, avail_devs = get_shared_ov_core(DEFAULT_CACHE_DIR)

        avail_ram_gb = round(vm.available / (1024.0 ** 3), 2)
        try:
            cur_aff = list(proc.cpu_affinity())
        except Exception:
            cur_aff = list(range(hw.logical_cores))
        warm_blobs = len(list(DEFAULT_CACHE_DIR.glob("*"))) if DEFAULT_CACHE_DIR.exists() else 0

        usm_device_mb = 0.0
        if "GPU" in avail_devs:
            try:
                stats = core.get_property("GPU", "GPU_MEMORY_STATISTICS")
                if isinstance(stats, dict):
                    usm_device_mb = round(float(stats.get("usm_device", 0.0)) / (1024.0 * 1024.0), 2)
            except Exception:
                usm_device_mb = 0.0

        opportunities: List[Dict[str, Any]] = []

        # 1. Príležitosť OS: Zúženie SMT afinity na fyzické jadrá pre horúce vlákna
        if len(cur_aff) > hw.physical_cores:
            opportunities.append({
                "opportunity_id": "OPP_OS_01_PCORE_AFFINITY_CONSOLIDATION",
                "target_plane": "OPERAČNÝ_SYSTÉM_PLÁNOVAČ",
                "trigger_observed": f"Proces je rozptýlený na {len(cur_aff)} logických vláknach namiesto {hw.physical_cores} fyzických jadier",
                "proposed_action_sk": f"Zoskupiť kritické 120 Hz vlákna na fyzické jadrá [0..{hw.physical_cores - 1}] a eliminovať 2,45x SMT kmitanie.",
                "tier": "P1",
                "utility": 86.0,
                "uma_delta": 0.03,
                "watts": 2.0,
                "preferred_unit": "CPU_4P_AVX512",
            })

        # 2. Príležitosť OS: Ochrana pred swapovaním pri nízkej voľnej RAM
        if avail_ram_gb < 4.5:
            opportunities.append({
                "opportunity_id": "OPP_OS_02_LOW_RAM_COMPACT_ARENA_GUARD",
                "target_plane": "OPERAČNÝ_SYSTÉM_VIRTUÁLNA_PAMÄŤ",
                "trigger_observed": f"Voľná systémová RAM je iba {avail_ram_gb} GB (< 4,5 GB prah)",
                "proposed_action_sk": "Aktivovať kompaktnú 64 MB zero-copy arénu a 40,37x kompresiu logov na zabránenie stránkovania na NVMe disk.",
                "tier": "P0",
                "utility": 94.0,
                "uma_delta": 0.02,
                "watts": 1.0,
                "preferred_unit": "CPU_VSYNC_CORE",
            })
        else:
            opportunities.append({
                "opportunity_id": "OPP_OS_02_PREFAULT_LARGE_UMA_WINDOW",
                "target_plane": "OPERAČNÝ_SYSTÉM_VIRTUÁLNA_PAMÄŤ",
                "trigger_observed": f"Dostatočná voľná RAM ({avail_ram_gb} GB)",
                "proposed_action_sk": "Pred-mapovať 256 MB zarovnané okno pre nulové výpadky stránok počas behu.",
                "tier": "P1",
                "utility": 72.0,
                "uma_delta": 0.06,
                "watts": 1.5,
                "preferred_unit": "CPU_4P_AVX512",
            })

        # 3. Príležitosť Grafického Ovládača: Eliminácia interných WDDM/DRM bounce-buffer kópií
        opportunities.append({
            "opportunity_id": "OPP_GPU_01_ZERO_BOUNCE_64KB_USM_BRIDGE",
            "target_plane": "GRAFICKÝ_OVLÁDAČ_PAMÄŤ",
            "trigger_observed": f"Aktívna GPU USM pamäť ovládača: {usm_device_mb} MB, predhriate bloby: {warm_blobs}",
            "proposed_action_sk": "Zarovnať všetky zdieľané buffery na 64 B cache riadok a 64 KB D3D12/Vulkan hranicu, aby ovládač preskočil skryté kopírovanie.",
            "tier": "P1",
            "utility": 89.0,
            "uma_delta": 0.08,
            "watts": 3.5,
            "preferred_unit": "IRIS_XE_768_SP",
        })

        # 4. Príležitosť Grafického Ovládača: Predhriatie shader/OpenVINO blobov a ochrana pred TDR
        opportunities.append({
            "opportunity_id": "OPP_GPU_02_SHADER_CACHE_AND_SLM_K256_TILING",
            "target_plane": "GRAFICKÝ_OVLÁDAČ_KOMPILÁTOR_A_FRONTY",
            "trigger_observed": f"Podskupinový strop K < 449 a 64 KB SLM na každom zo 6 Subslice blokov",
            "proposed_action_sk": "Rozdeliť dlhé redukcie na K_tile = 256 v 64 KB SLM (nulové rozliatie 128 GRF registrov) a udržiavať predhriatu cache blobov.",
            "tier": "P1",
            "utility": 84.0,
            "uma_delta": 0.07,
            "watts": 3.0,
            "preferred_unit": "IRIS_XE_768_SP",
        })

        # 5. Príležitosť OS + Ovládača: Presun pomalej telemetrie do VBlank tieňa (245,97x úspora)
        opportunities.append({
            "opportunity_id": "OPP_OS_GPU_03_VBLANK_SHADOW_TELEMETRY_COALESCING",
            "target_plane": "KOOPERÁCIA_OS_A_GRAFICKÉHO_OVLÁDAČA",
            "trigger_observed": "Priamy dotaz do GPU ovládača trvá ~32,7 us oproti ~133 ns v zrkadle BAR0",
            "proposed_action_sk": "Čítať horúce metriky výhradne zo 133 ns `mmap` BAR0 registra a dotazy na ovládač spájať iba raz za 100 ms vo VBlank tieni.",
            "tier": "P0",
            "utility": 92.0,
            "uma_delta": 0.01,
            "watts": 0.5,
            "preferred_unit": "CPU_VSYNC_CORE",
        })

        arb_result = self.arbitrator.clear_arbitration_round(
            bids=opportunities,
            current_uma=min(0.72, round((1.0 - (vm.available / max(vm.total, 1))) * 0.55, 4)),
        )

        return {
            "vital_max_hp": VITAL_MAX_HP,
            "live_host_snapshot": {
                "available_ram_gb": avail_ram_gb,
                "cpu_affinity_threads": len(cur_aff),
                "warm_driver_blobs": warm_blobs,
                "gpu_usm_device_mb": usm_device_mb,
            },
            "discovered_opportunities_count": len(opportunities),
            "opportunities": opportunities,
            "arbitrated_execution": arb_result,
        }

    def format_slovak_opportunities(self, _arg: str = "") -> str:
        res = self.discover_and_propose_opportunities()
        snap = res["live_host_snapshot"]
        lines = [
            f"Nájdené príležitosti pre OS a grafický ovládač (voľná RAM={snap['available_ram_gb']} GB, GPU USM={snap['gpu_usm_device_mb']} MB):"
        ]
        for opp in res["opportunities"]:
            lines.append(f"- [{opp['target_plane']}] {opp['opportunity_id']}: {opp['proposed_action_sk']}")
        lines.append(f"Schválené arbitrážou okamžite: {len(res['arbitrated_execution']['accepted_agents'])}/{res['discovered_opportunities_count']}")
        return "\n".join(lines)


BOT_OPPORTUNITY_GENERATOR = BotOsAndGpuOpportunityGenerator(SOVEREIGN_ARBITRATION_ENGINE)
