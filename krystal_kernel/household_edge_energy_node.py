#!/usr/bin/env python3
# ==============================================================================
# KRYSTAL-STACK: GRID-TO-HOUSEHOLD EDGE ENERGY NODE & ASYMMETRIC CONTROLLER
# ==============================================================================
# Invariant: VITAL_MAX_HP == 6
# Constitutional Law: Invalid(Action) => S_{t+1} == S_t (Delta S = 0)
# Infrastructure: Slovak 3-Phase Residential Grid (3x 230V / 25A Main Breaker = 17.25 kW max)
# Storage: 20 kWh LiFePO4 (LCOS = 0.038 EUR/kWh) + 300L Domestic Hot Water (TÚV)
# Compute: 5.4 kW CLU (10x GPU + 128-core EPYC) with Liquid-to-Water Heat Exchanger
# ==============================================================================

import time
import math
import struct
import hashlib
from enum import IntEnum
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

VITAL_MAX_HP: int = 6

# ------------------------------------------------------------------------------
# RESIDENTIAL ELECTRICAL & HARDWARE CONSTANTS
# ------------------------------------------------------------------------------
FRAME_MAGIC_HOUSEHOLD: int = 0x4B525953     # "KRYS"
FRAME_SIZE_BYTES: int = 128                  # 128B naturally aligned (Principle P2)

VOLTAGE_NOMINAL_V: float = 230.0             # 230 V phase-to-neutral
MAIN_BREAKER_AMPS: float = 25.0              # 3x 25A standard breaker
MAX_POWER_PER_PHASE_W: float = 230.0 * 25.0  # 5,750 W per phase limit
BREAKER_SAFETY_MARGIN_W: float = 250.0       # Safe ceiling = 5,500 W (23.9 A)

BATTERY_CAPACITY_KWH: float = 20.0           # 20 kWh LiFePO4
BATTERY_MIN_SOC: float = 0.10                # 10% reserve for longevity
BATTERY_MAX_CHARGE_KW: float = 10.0          # Max charge rate (3.33 kW / phase)
BATTERY_MAX_DISCHARGE_KW: float = 12.0       # Max discharge rate (4.00 kW / phase)
BATTERY_CHARGE_EFF: float = 0.95             # 95% charge efficiency
BATTERY_DISCHARGE_EFF: float = 0.95          # 95% discharge efficiency (RTE = 90.25%)
BATTERY_LCOS_EUR_PER_KWH: float = 0.038      # 0.038 EUR/kWh battery wear (LCOS)

CLU_MAX_POWER_W: float = 5400.0              # 5,400 W (1.8 kW per phase)
CLU_HEAT_EXCHANGER_EFF: float = 0.92         # 92% electrical-to-thermal recovery
DHW_TANK_ENERGY_KWH: float = 13.95           # 300L water heated from 15 C to 55 C


class HouseholdEdgeMode(IntEnum):
    """Prevádzkové režimy domáceho energetického uzla."""
    AGGRESSIVE_CHARGE_COMPUTE = 1  # Lacná/záporná cena: max nabíjanie + max compute + ohrev TÚV
    BYPASS_COMPUTE            = 2  # Mierna nočná cena: batéria necykluje, priamy odber zo siete
    GRID_CUT_ISLAND           = 3  # Drahá špička: odber zo siete = 0 W, dom beží z batérie
    V2G_ARBITRAGE             = 4  # Extrémna špička: batéria exportuje prebytok do siete
    PEAK_SHAVE_PROTECT        = 5  # Okamžité odľahčenie (< 10 ms) pri hroziacom vyhodení 25A ističa


class HouseholdContractViolation(Exception):
    """Zlyhanie konštitučného invariantu alebo zmluvy v domácom uzle."""
    pass


# ------------------------------------------------------------------------------
# DÁTOVÉ ŠTRUKTÚRY
# ------------------------------------------------------------------------------

@dataclass
class PhaseLoads:
    """Okamžitý odber spotrebičov na jednotlivých fázach (W)."""
    l1_w: float
    l2_w: float
    l3_w: float

    @property
    def total_w(self) -> float:
        return self.l1_w + self.l2_w + self.l3_w


@dataclass
class HouseholdSlotResult:
    """Výsledok 15-minútového riadiaceho a bilančného kroku."""
    slot_index: int
    mode: HouseholdEdgeMode
    grid_draw_per_phase_w: Tuple[float, float, float]
    total_grid_draw_w: float
    inverter_injection_per_phase_w: Tuple[float, float, float]
    battery_soc: float
    battery_flow_w: float          # Kladné = nabíjanie, záporné = vybíjanie
    compute_power_w: float
    thermal_recovered_w: float
    dhw_stored_kwh: float
    cost_without_krystal_eur: float # Náklad pasívnej domácnosti
    cost_with_krystal_eur: float    # Náklad Krystal uzla (sieť + batéria)
    compute_revenue_eur: float      # Výnos z hashu / AI inferencie
    net_financial_gain_eur: float   # Čistý prínos za interval
    eta_ppc: float                  # Koeficient Performance-Per-Cost
    breaker_overload_prevented: bool
    state_hash: str
    frame_128b_hex: str
    vital_max_hp: int = VITAL_MAX_HP


# ------------------------------------------------------------------------------
# KRYSTAL HOUSEHOLD EDGE CONTROLLER
# ------------------------------------------------------------------------------

class HouseholdEdgeEnergyNode:
    """
    Autonómny domáci energeticko-výpočtový uzol (Grid-to-Household Edge).
    Riadi asymetrický 3-fázový menič, 20 kWh LFP batériu a 5,4 kW CLU výpočtový klastor.
    """

    REVENUE_COMPUTE_EUR_PER_WH: float = 0.052e-3  # Kombinovaná hodnota compute / token
    REVENUE_DHW_EUR_PER_KWH: float = 0.160        # Hodnota ušetreného priameho ohrevu TÚV

    def __init__(self, initial_soc: float = 0.35, initial_dhw_kwh: float = 2.0):
        self.battery_soc: float = max(BATTERY_MIN_SOC, min(1.0, initial_soc))
        self.dhw_stored_kwh: float = max(0.0, min(DHW_TANK_ENERGY_KWH, initial_dhw_kwh))
        self.total_breaker_events_prevented: int = 0

    @staticmethod
    def pack_128b_household_frame(
        mode: HouseholdEdgeMode,
        i_grid_cma: Tuple[int, int, int],
        v_grid_dmv: Tuple[int, int, int],
        cos_phi_bp: Tuple[int, int, int],
        p_inv_cmd_w: Tuple[int, int, int],
        soc_bp: int,
        p_compute_w: int,
        q_thermal_dhw_w: int,
        eta_ppc_bp: int,
        c_spot_cents_mwh: int,
        c_dist_cents_mwh: int,
        breaker_warning: bool = False,
        timestamp_us: Optional[int] = None,
        transformer_id: int = 101,
        phase_topology_mask: int = 0b00100100,
        battery_topology_mode: int = 0,
        q_reactive_cmd_var: Tuple[int, int, int] = (0, 0, 0),
        vpp_dispatch_target_w: int = 0,
        broker_contract_id: int = 0x41465252 # "AFRR"
    ) -> bytes:
        """
        Zabalí telemetrický stav domáceho uzla do presne 128-bajtového rámca (C/Rust layout).
        Obsahuje L3 Grid Broker & Fleet Phased-Array riadenie na offsete 74..95.
        """
        if VITAL_MAX_HP != 6:
            raise HouseholdContractViolation("Invariant VITAL_MAX_HP != 6 violated")

        ts = timestamp_us if timestamp_us is not None else int(time.time() * 1e6) & 0xFFFFFFFF
        flags = (1 if breaker_warning else 0)

        # 58 bajtov binárnej hlavičky a telemetrie (0..57)
        header_58b = struct.pack(
            "!IBBBBIHHHHHHhhhhhhHHhhHHHiHH",
            FRAME_MAGIC_HOUSEHOLD,
            2,              # Version
            VITAL_MAX_HP,   # Invariant 0x06
            int(mode),
            flags,
            ts,
            i_grid_cma[0], i_grid_cma[1], i_grid_cma[2],
            v_grid_dmv[0], v_grid_dmv[1], v_grid_dmv[2],
            cos_phi_bp[0], cos_phi_bp[1], cos_phi_bp[2],
            p_inv_cmd_w[0], p_inv_cmd_w[1], p_inv_cmd_w[2],
            soc_bp,
            512,            # 51.2 V LFP pack
            0,              # i_batt
            250,            # 25.0 C
            p_compute_w,
            q_thermal_dhw_w,
            eta_ppc_bp,
            c_spot_cents_mwh,
            c_dist_cents_mwh,
            1000            # 50.000 Hz (offset from 49.000 Hz = 1000 mHz)
        )
        assert len(header_58b) == 58

        # 16 bajtov kryptografického state hashu (58..73)
        state_hash_16b = hashlib.sha256(header_58b).digest()[:16]

        # 22 bajtov: L3 Grid Broker & Fleet Phased-Array Control (74..95)
        # transformer_id (u32), phase_topology_mask (u8), battery_topology_mode (u8),
        # q_reactive_cmd_var (3x i16), _pad (u16), vpp_dispatch_target_w (i32), broker_contract_id (u32)
        vpp_broker_22b = struct.pack(
            "!IBBhhhHiI",
            transformer_id,
            phase_topology_mask,
            battery_topology_mode,
            q_reactive_cmd_var[0], q_reactive_cmd_var[1], q_reactive_cmd_var[2],
            0,              # 2-bajtové zarovnanie na 4B hranicu
            vpp_dispatch_target_w,
            broker_contract_id
        )
        assert len(vpp_broker_22b) == 22

        # 32 bajtov: _reserved_tail (96..127)
        reserved_tail_32b = bytes(32)

        frame_128b = header_58b + state_hash_16b + vpp_broker_22b + reserved_tail_32b
        assert len(frame_128b) == FRAME_SIZE_BYTES
        return frame_128b

    @staticmethod
    def unpack_128b_household_frame(frame_bytes: bytes) -> Dict[str, Any]:
        """Rozbalí 128-bajtový rámec vrátane L3 Grid Broker polí na offsete 74..95."""
        if len(frame_bytes) != FRAME_SIZE_BYTES:
            raise HouseholdContractViolation(f"Frame size {len(frame_bytes)} != 128B")

        unpacked = struct.unpack("!IBBBBIHHHHHHhhhhhhHHhhHHHiHH", frame_bytes[:58])
        magic = unpacked[0]
        version = unpacked[1]
        hp = unpacked[2]
        mode_val = unpacked[3]
        flags = unpacked[4]
        ts = unpacked[5]

        if magic != FRAME_MAGIC_HOUSEHOLD:
            raise HouseholdContractViolation(f"Invalid magic 0x{magic:08X}")
        if hp != VITAL_MAX_HP:
            raise HouseholdContractViolation(f"Corrupted invariant {hp} != 6")

        # Rozbalenie L3 Grid Broker polí (74..95)
        unpacked_vpp = struct.unpack("!IBBhhhHiI", frame_bytes[74:96])

        return {
            "version": version,
            "vital_max_hp": hp,
            "mode": HouseholdEdgeMode(mode_val),
            "breaker_warning": bool(flags & 1),
            "timestamp_us": ts,
            "soc_percent": unpacked[18] / 100.0,
            "p_compute_w": unpacked[22],
            "q_thermal_dhw_w": unpacked[23],
            "eta_ppc": unpacked[24] / 100.0,
            "c_spot_eur_mwh": unpacked[25] / 100.0,
            "c_dist_eur_mwh": unpacked[26] / 100.0,
            # L3 Grid Broker polia:
            "transformer_id": unpacked_vpp[0],
            "phase_topology_mask": unpacked_vpp[1],
            "battery_topology_mode": unpacked_vpp[2],
            "q_reactive_cmd_var": (unpacked_vpp[3], unpacked_vpp[4], unpacked_vpp[5]),
            "vpp_dispatch_target_w": unpacked_vpp[7],
            "broker_contract_id": unpacked_vpp[8],
            "is_128b_aligned": True
        }

    def step_15min_slot(
        self,
        slot_index: int,
        household_loads: PhaseLoads,
        spot_price_eur_mwh: float,
        dist_tariff_eur_mwh: float = 65.0, # Typický distribučný poplatok
        high_price_threshold_eur: float = 110.0,
        low_price_threshold_eur: float = 25.0,
        timestamp_us: Optional[int] = None
    ) -> HouseholdSlotResult:
        """
        Vykoná 15-minútový arbitrážny krok (1 interval z 96 za deň):
        1. Určí optimálny prevádzkový režim podľa spotovej ceny a SoC batérie.
        2. Vypočíta asymetrické vstrekovanie meniča na fázy L1, L2, L3.
        3. Aplikuje dynamickú ochranu ističa 3x25A (Peak Shaving).
        4. Zaktualizuje stav batérie (SoC) a zásobníka TÚV.
        5. Vypočíta metriku PPC a finančnú úsporu oproti pasívnemu domu.
        """
        slot_ts = timestamp_us if timestamp_us is not None else (slot_index * 900_000_000) & 0xFFFFFFFF
        duration_hours = 0.25  # 15 minút = 0.25 h
        total_elec_price_eur_mwh = spot_price_eur_mwh + dist_tariff_eur_mwh
        price_per_kwh = total_elec_price_eur_mwh / 1000.0

        p_loads = [household_loads.l1_w, household_loads.l2_w, household_loads.l3_w]
        p_inv_cmds = [0.0, 0.0, 0.0]
        breaker_overload_prevented = False
        compute_power_w = 0.0
        battery_flow_w = 0.0

        # ----------------------------------------------------------------------
        # 1. VÝBER STRATÉGIE A REŽIMU
        # ----------------------------------------------------------------------
        if spot_price_eur_mwh <= low_price_threshold_eur:
            # Režim 1: AGGRESSIVE_CHARGE_COMPUTE (lacná/záporná cena)
            mode = HouseholdEdgeMode.AGGRESSIVE_CHARGE_COMPUTE
            compute_power_w = CLU_MAX_POWER_W
            # Nabíjanie batérie až do 10 kW (3.33 kW na fázu)
            usable_headroom_kwh = (1.0 - self.battery_soc) * BATTERY_CAPACITY_KWH
            max_charge_w = min(BATTERY_MAX_CHARGE_KW * 1000.0, (usable_headroom_kwh / duration_hours) * 1000.0)
            charge_per_phase_w = max_charge_w / 3.0
            # Pre nabíjanie sú p_inv_cmds záporné (menič odoberá zo siete)
            p_inv_cmds = [-charge_per_phase_w, -charge_per_phase_w, -charge_per_phase_w]
            battery_flow_w = max_charge_w

        elif spot_price_eur_mwh >= high_price_threshold_eur and self.battery_soc > BATTERY_MIN_SOC:
            # Režim 3: GRID_CUT_ISLAND (drahá špička: odber zo siete = 0 W)
            mode = HouseholdEdgeMode.GRID_CUT_ISLAND
            compute_power_w = 500.0  # Úsporná AI inferencia
            # Asymetrické vyrovnanie každej fázy z batérie!
            available_discharge_kwh = (self.battery_soc - BATTERY_MIN_SOC) * BATTERY_CAPACITY_KWH
            max_discharge_power_w = min(BATTERY_MAX_DISCHARGE_KW * 1000.0, (available_discharge_kwh / duration_hours) * 1000.0)

            total_demand_to_cover = sum(p_loads) + compute_power_w
            coverage_factor = min(1.0, max_discharge_power_w / max(1.0, total_demand_to_cover))

            # Každá fáza dostane presne toľko, koľko odoberá spotrebič + 1/3 compute
            for i in range(3):
                p_inv_cmds[i] = (p_loads[i] + compute_power_w / 3.0) * coverage_factor

            battery_flow_w = -sum(p_inv_cmds)  # Záporné = vybíjanie

        else:
            # Režim 2: BYPASS_COMPUTE (stredná cena: batéria necykluje)
            mode = HouseholdEdgeMode.BYPASS_COMPUTE
            compute_power_w = 1800.0  # Mierne AI vyťaženie
            p_inv_cmds = [0.0, 0.0, 0.0]
            battery_flow_w = 0.0

        # ----------------------------------------------------------------------
        # 2. ASYMETRICKÁ OCHRANA HLAVNÉHO ISTIČA 3x25A (DYNAMIC PEAK SHAVING)
        # ----------------------------------------------------------------------
        safe_ceiling_w = MAX_POWER_PER_PHASE_W - BREAKER_SAFETY_MARGIN_W # 5,500 W
        grid_draw_phases = [0.0, 0.0, 0.0]

        for i in range(3):
            compute_phase_w = compute_power_w / 3.0
            # Hrubý odber zo siete na fáze i:
            # Ak p_inv_cmds[i] < 0 (nabíjanie), zvyšuje odber zo siete.
            # Ak p_inv_cmds[i] > 0 (vybíjanie), znižuje odber zo siete.
            raw_grid_w = p_loads[i] + compute_phase_w - p_inv_cmds[i]

            if raw_grid_w > safe_ceiling_w:
                # Hrozí vyhodenie 25A ističa!
                breaker_overload_prevented = True
                self.total_breaker_events_prevented += 1
                excess_w = raw_grid_w - safe_ceiling_w

                if p_inv_cmds[i] < 0:
                    # Znížime nabíjací prúd batérie
                    p_inv_cmds[i] += excess_w
                    raw_grid_w = safe_ceiling_w
                else:
                    # Ak ani to nestačí, batéria vstrekne pomocný výkon
                    p_inv_cmds[i] += excess_w
                    raw_grid_w = safe_ceiling_w

            grid_draw_phases[i] = max(0.0, raw_grid_w)

        total_grid_draw_w = sum(grid_draw_phases)

        # ----------------------------------------------------------------------
        # 3. AKTUALIZÁCIA STAVU BATÉRIE (SoC)
        # ----------------------------------------------------------------------
        if battery_flow_w > 0:
            # Nabíjanie s účinnosťou 95%
            energy_in_kwh = (battery_flow_w / 1000.0) * duration_hours * BATTERY_CHARGE_EFF
            self.battery_soc = min(1.0, self.battery_soc + (energy_in_kwh / BATTERY_CAPACITY_KWH))
        elif battery_flow_w < 0:
            # Vybíjanie s účinnosťou 95%
            energy_out_kwh = (abs(battery_flow_w) / 1000.0) * duration_hours / BATTERY_DISCHARGE_EFF
            self.battery_soc = max(BATTERY_MIN_SOC, self.battery_soc - (energy_out_kwh / BATTERY_CAPACITY_KWH))

        # ----------------------------------------------------------------------
        # 4. TERMODYNAMICKÁ REKUPERÁCIA TEPLA DO TÚV (BOJLER)
        # ----------------------------------------------------------------------
        thermal_recovered_w = compute_power_w * CLU_HEAT_EXCHANGER_EFF
        dhw_heat_kwh = (thermal_recovered_w / 1000.0) * duration_hours
        self.dhw_stored_kwh = min(DHW_TANK_ENERGY_KWH, self.dhw_stored_kwh + dhw_heat_kwh)

        # ----------------------------------------------------------------------
        # 5. FINANČNÁ BILANCIA A KOEFICIENT PPC
        # ----------------------------------------------------------------------
        # A. Náklad pasívneho domu bez Krystalu (kupuje všetku energiu z drahej siete)
        passive_home_energy_kwh = (household_loads.total_w / 1000.0) * duration_hours
        cost_passive_eur = passive_home_energy_kwh * price_per_kwh

        # B. Náklad Krystal uzla: sieťový nákup + opotrebenie batérie
        active_grid_kwh = (total_grid_draw_w / 1000.0) * duration_hours
        cost_active_grid_eur = active_grid_kwh * price_per_kwh

        battery_discharged_kwh = (max(0.0, -battery_flow_w) / 1000.0) * duration_hours
        cost_battery_deg_eur = battery_discharged_kwh * BATTERY_LCOS_EUR_PER_KWH

        cost_krystal_total_eur = cost_active_grid_eur + cost_battery_deg_eur

        # C. Výnosy: výpočtový zisk + hodnota ohriatej vody
        compute_revenue_eur = (compute_power_w * duration_hours) * self.REVENUE_COMPUTE_EUR_PER_WH
        dhw_value_saved_eur = dhw_heat_kwh * self.REVENUE_DHW_EUR_PER_KWH

        # Čistý prínos = (Ušetrený pasívny náklad - Skutočný náklad Krystalu) + Výnosy
        net_financial_gain_eur = (cost_passive_eur - cost_krystal_total_eur) + compute_revenue_eur + dhw_value_saved_eur

        # D. Koeficient PPC: Úžitok / Náklad
        total_utility = compute_revenue_eur + cost_passive_eur + dhw_value_saved_eur
        total_expense = max(0.001, cost_krystal_total_eur)
        eta_ppc = round(total_utility / total_expense, 4)

        # ----------------------------------------------------------------------
        # 6. ZBALENIE DO 128B RÁMCA
        # ----------------------------------------------------------------------
        i_grid_cma = tuple(int(round((w / VOLTAGE_NOMINAL_V) * 100.0)) for w in grid_draw_phases)
        v_grid_dmv = (2300, 2300, 2300)
        cos_phi_bp = (9900, 9850, 9920)
        p_inv_cmd_i = tuple(int(round(w)) for w in p_inv_cmds)
        soc_bp = int(round(self.battery_soc * 10000.0))
        eta_ppc_bp = int(round(min(65000, eta_ppc * 100.0)))

        frame_bytes = self.pack_128b_household_frame(
            mode=mode,
            i_grid_cma=i_grid_cma,
            v_grid_dmv=v_grid_dmv,
            cos_phi_bp=cos_phi_bp,
            p_inv_cmd_w=p_inv_cmd_i,
            soc_bp=soc_bp,
            p_compute_w=int(round(compute_power_w)),
            q_thermal_dhw_w=int(round(thermal_recovered_w)),
            eta_ppc_bp=eta_ppc_bp,
            c_spot_cents_mwh=int(round(spot_price_eur_mwh * 100.0)),
            c_dist_cents_mwh=int(round(dist_tariff_eur_mwh * 100.0)),
            breaker_warning=breaker_overload_prevented,
            timestamp_us=slot_ts
        )
        state_hash = hashlib.sha256(frame_bytes[:58]).hexdigest()[:16]

        return HouseholdSlotResult(
            slot_index=slot_index,
            mode=mode,
            grid_draw_per_phase_w=(round(grid_draw_phases[0], 1), round(grid_draw_phases[1], 1), round(grid_draw_phases[2], 1)),
            total_grid_draw_w=round(total_grid_draw_w, 1),
            inverter_injection_per_phase_w=(round(p_inv_cmds[0], 1), round(p_inv_cmds[1], 1), round(p_inv_cmds[2], 1)),
            battery_soc=round(self.battery_soc, 4),
            battery_flow_w=round(battery_flow_w, 1),
            compute_power_w=round(compute_power_w, 1),
            thermal_recovered_w=round(thermal_recovered_w, 1),
            dhw_stored_kwh=round(self.dhw_stored_kwh, 2),
            cost_without_krystal_eur=round(cost_passive_eur, 4),
            cost_with_krystal_eur=round(cost_krystal_total_eur, 4),
            compute_revenue_eur=round(compute_revenue_eur, 4),
            net_financial_gain_eur=round(net_financial_gain_eur, 4),
            eta_ppc=eta_ppc,
            breaker_overload_prevented=breaker_overload_prevented,
            state_hash=state_hash,
            frame_128b_hex=frame_bytes.hex(),
            vital_max_hp=VITAL_MAX_HP
        )

    def simulate_24h_cycle(
        self,
        spot_price_curve_96: List[float],
        load_curve_l1_96: List[float],
        load_curve_l2_96: List[float],
        load_curve_l3_96: List[float]
    ) -> Dict[str, Any]:
        """
        Simuluje kompletný 24-hodinový cyklus (96 x 15-minútových intervalov).
        Vráti sumárnu finančnú bilanciu, priemerné PPC a počet zásahov ističa.
        """
        assert len(spot_price_curve_96) == 96
        slot_results = []
        total_passive_cost = 0.0
        total_active_cost = 0.0
        total_compute_rev = 0.0
        total_net_gain = 0.0
        breaker_events = 0

        for t in range(96):
            loads = PhaseLoads(
                l1_w=load_curve_l1_96[t],
                l2_w=load_curve_l2_96[t],
                l3_w=load_curve_l3_96[t]
            )
            res = self.step_15min_slot(
                slot_index=t,
                household_loads=loads,
                spot_price_eur_mwh=spot_price_curve_96[t]
            )
            slot_results.append(res)
            total_passive_cost += res.cost_without_krystal_eur
            total_active_cost += res.cost_with_krystal_eur
            total_compute_rev += res.compute_revenue_eur
            total_net_gain += res.net_financial_gain_eur
            if res.breaker_overload_prevented:
                breaker_events += 1

        avg_eta_ppc = sum(r.eta_ppc for r in slot_results) / len(slot_results)

        return {
            "slots_count": len(slot_results),
            "final_battery_soc": round(self.battery_soc, 4),
            "final_dhw_kwh": round(self.dhw_stored_kwh, 2),
            "total_passive_cost_eur": round(total_passive_cost, 2),
            "total_active_cost_eur": round(total_active_cost, 2),
            "total_compute_revenue_eur": round(total_compute_rev, 2),
            "total_net_gain_eur": round(total_net_gain, 2),
            "annualized_gain_eur": round(total_net_gain * 365.0, 2),
            "average_eta_ppc": round(avg_eta_ppc, 2),
            "breaker_protection_events": breaker_events,
            "vital_max_hp": VITAL_MAX_HP
        }
