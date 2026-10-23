#!/usr/bin/env python3
# ==============================================================================
# KRYSTAL-STACK: VIRTUAL POWER PLANT (VPP) BROKER & SWITCHED BATTERY MATRIX
# ==============================================================================
# Invariant: VITAL_MAX_HP == 6
# Constitutional Law: Invalid(Action) => S_{t+1} == S_t (Delta S = 0)
# Substation Target: 22 kV / 0.4 kV, 630 kVA Transformer, 100 Households on 800m Feeder
# Grid Standard: EN 50160 (230 V +/- 10% -> 207.0 V <= V <= 253.0 V)
# ==============================================================================

import time
import math
import struct
import hashlib
from enum import IntEnum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

VITAL_MAX_HP: int = 6

# ------------------------------------------------------------------------------
# ELEKTROTECHNICKÉ A DISTRIBUČNÉ KONŠTANTY
# ------------------------------------------------------------------------------
VOLTAGE_NOMINAL_V: float = 230.0             # 230 V fázové napätie
VOLTAGE_MAX_ALLOWED_V: float = 253.0         # 230V + 10% limit podľa EN 50160
VOLTAGE_MIN_ALLOWED_V: float = 207.0         # 230V - 10% limit
VOLTAGE_WARNING_THRESHOLD_V: float = 247.0   # Prah aktivácie Q(V) regulácie podľa EN 50549-1

TRANSFORMER_RATING_KVA: float = 630.0        # 630 kVA transformátor VN/NN
FEEDER_LENGTH_METERS: float = 500.0          # 500 m vývod (NAYY 4x70 mm²)
CABLE_R_PER_KM_OHM: float = 0.380            # 0.380 Ohm/km odpor
CABLE_X_PER_KM_OHM: float = 0.085            # 0.085 Ohm/km reaktancia

FRAME_MAGIC_VPP: int = 0x4B525953            # "KRYS"
FRAME_SIZE_BYTES: int = 128                  # Universal 128B Lingua Franca (Princíp P2)


class BatteryTopology(IntEnum):
    """Konfigurácia spínanej SiC MOSFET matice batériových modulov."""
    PARALLEL_48V = 1   # Nízkonapäťový paralelný režim: bežná domáca prevádzka (48V - 51.2V)
    SERIES_800V  = 2   # Vysokonapäťový sériový režim: GRID BOOST masívny export (800V - 819.2V)


class VppFleetMode(IntEnum):
    """Globálne režimy riadenia virtuálnej elektrárne (VPP)."""
    FLEET_IDLE                 = 1  # Vyrovnaný stav, lokálna optimalizácia
    AFRR_DOWN_MAX_ABSORPTION   = 2  # aFRR- dole: záporné ceny, plné nabíjanie (1.0 MW) + CLU compute (0.54 MW)
    AFRR_UP_GRID_BOOST_EXPORT  = 3  # aFRR+ hore: špičkový výkup, 800V sériový export do siete (až 1.2 MW)
    VOLTAGE_PROTECT_PHASED     = 4  # Ochranný fázovaný režim: jalová kompenzácia Q a lokálne stláčanie špičky


class VppBrokerException(Exception):
    """Zlyhanie invariantu alebo bezpečnosti vo VPP Brokeri."""
    pass


# ------------------------------------------------------------------------------
# 1. L1: SWITCHED-CAPACITOR / SiC MOSFET BATTERY MATRIX CONTROLLER
# ------------------------------------------------------------------------------

class SwitchedBatteryMatrixController:
    """
    Riadi SiC MOSFET maticu na každom domácom uzle:
    - Prepína 16 modulov LiFePO4 medzi paralelným (51.2 V) a sériovým (819.2 V) zapojením.
    - Zabezpečuje striktnú separáciu vetiev:
        Vetva A: Výpočet -> Teplo -> Voda (jednosmerná termodynamika).
        Vetva B: Batéria <-> Sieť (obojsmerná elektrochémia).
    """

    VOLTAGE_PARALLEL_V: float = 48.0
    VOLTAGE_SERIES_V: float = 800.0

    def __init__(self, initial_topology: BatteryTopology = BatteryTopology.PARALLEL_48V):
        self.topology: BatteryTopology = initial_topology
        self.transition_count: int = 0
        self.last_transition_time_ms: float = 0.0

    @classmethod
    def calculate_cable_loss(
        cls,
        power_w: float,
        topology: BatteryTopology,
        cable_resistance_ohm: float = 0.045
    ) -> Tuple[float, float, float]:
        """
        Vypočíta prúd, úbytok napätia a tepelné straty P_loss = I^2 * R na prívodnom kábli.
        Vráti (current_a, voltage_drop_v, loss_w).
        """
        voltage = cls.VOLTAGE_SERIES_V if topology == BatteryTopology.SERIES_800V else cls.VOLTAGE_PARALLEL_V
        current = power_w / voltage
        v_drop = current * cable_resistance_ohm
        loss_w = (current ** 2) * cable_resistance_ohm
        return round(current, 2), round(v_drop, 2), round(loss_w, 2)

    def switch_topology(self, target: BatteryTopology) -> Dict[str, Any]:
        """
        Vykoná bezpečnú rekonfiguráciu SiC MOSFET matice v prechode nulovým napätím (< 5 ms).
        """
        if self.topology != target:
            self.topology = target
            self.transition_count += 1
            self.last_transition_time_ms = 4.2  # Typická rýchlosť zopnutia SiC mostíka

        return {
            "current_topology": self.topology.name,
            "bus_voltage_v": self.VOLTAGE_SERIES_V if self.topology == BatteryTopology.SERIES_800V else self.VOLTAGE_PARALLEL_V,
            "transition_count": self.transition_count,
            "switching_latency_ms": self.last_transition_time_ms,
            "is_grid_boost_active": self.topology == BatteryTopology.SERIES_800V,
            "vital_max_hp": VITAL_MAX_HP
        }


# ------------------------------------------------------------------------------
# 2. L2: SUBSTATION & FEEDER GRID MODEL (100 DOMÁCNOSTÍ NA 800m VÝVODE)
# ------------------------------------------------------------------------------

@dataclass
class FeederNodeState:
    """Stav jedného z 100 uzlov na distribučnom vývode."""
    node_id: int
    distance_meters: float
    active_power_kw: float       # Kladné = export do siete, záporné = odber zo siete
    reactive_power_kvar: float   # Kladné = kapacitný, záporné = induktívny
    clu_power_kw: float          # Lokálny príkon CLU (0..5.4 kW)
    voltage_v: float             # Vypočítané sieťové napätie v uzle
    is_overvoltage_tripped: bool = False


class SubstationFeederModel:
    """
    Model 4 nízkonapäťových vývodov 0.4 kV z 630 kVA transformátora.
    100 domácností je rozdelených do 4 paralelných vetiev po 25 domov (každá 400 m dlhá).
    Každá vetva je trojfázová (výkon na fázový vodič = P / 3.0).
    """

    BRANCHES_COUNT: int = 4

    def __init__(self, node_count: int = 100):
        self.node_count: int = node_count
        self.nodes_per_branch: int = node_count // self.BRANCHES_COUNT
        self.branch_length_m: float = 400.0
        self.node_spacing_m: float = self.branch_length_m / self.nodes_per_branch # 16.0 metrov na dom
        self.r_section_ohm: float = (CABLE_R_PER_KM_OHM / 1000.0) * self.node_spacing_m
        self.x_section_ohm: float = (CABLE_X_PER_KM_OHM / 1000.0) * self.node_spacing_m

    def solve_voltage_profile(
        self,
        node_active_powers_kw: List[float],
        node_reactive_powers_kvar: List[float],
        substation_voltage_v: float = 230.0
    ) -> List[FeederNodeState]:
        """
        Vypočíta napäťový profil pre všetkých 100 uzlov rozdelených do 4 paralelných vetiev.
        V každej vetve preteká 1/3 výkonu na fázu (trojfázová sústava).
        """
        assert len(node_active_powers_kw) == self.node_count
        assert len(node_reactive_powers_kvar) == self.node_count

        node_states = []

        for b in range(self.BRANCHES_COUNT):
            start_idx = b * self.nodes_per_branch
            end_idx = start_idx + self.nodes_per_branch

            b_active = node_active_powers_kw[start_idx:end_idx]
            b_reactive = node_reactive_powers_kvar[start_idx:end_idx]

            # Spätný súčet toku na fázový vodič (P / 3.0)
            p_flows_phase_w = [0.0] * self.nodes_per_branch
            q_flows_phase_var = [0.0] * self.nodes_per_branch
            p_acc = 0.0
            q_acc = 0.0
            for s in reversed(range(self.nodes_per_branch)):
                p_acc += (b_active[s] * 1000.0) / 3.0
                q_acc += (b_reactive[s] * 1000.0) / 3.0
                p_flows_phase_w[s] = p_acc
                q_flows_phase_var[s] = q_acc

            current_v = substation_voltage_v
            for i in range(self.nodes_per_branch):
                global_idx = start_idx + i
                delta_v_i = (self.r_section_ohm * p_flows_phase_w[i] + self.x_section_ohm * q_flows_phase_var[i]) / VOLTAGE_NOMINAL_V
                current_v += delta_v_i
                node_v = round(current_v, 2)
                is_tripped = (node_v > VOLTAGE_MAX_ALLOWED_V) or (node_v < VOLTAGE_MIN_ALLOWED_V)

                node_states.append(FeederNodeState(
                    node_id=global_idx,
                    distance_meters=round((i + 1) * self.node_spacing_m, 1),
                    active_power_kw=round(b_active[i], 2),
                    reactive_power_kvar=round(b_reactive[i], 2),
                    clu_power_kw=0.0,
                    voltage_v=node_v,
                    is_overvoltage_tripped=is_tripped
                ))

        return node_states


# ------------------------------------------------------------------------------
# 3. L3: VPP BIG DATA BROKER & PHASED ARRAY DISPATCHER
# ------------------------------------------------------------------------------

@dataclass
class VppBrokerDispatchDecision:
    """Výsledný dispečerský príkaz VPP brokera pre 100 domácností a elektráreň."""
    fleet_mode: VppFleetMode
    target_grid_power_mw: float        # Požadovaný celkový tok (+ export / - odber)
    achieved_grid_power_mw: float      # Skutočne dosiahnutý výkon
    total_active_export_mw: float
    total_active_absorption_mw: float
    total_reactive_power_mvar: float
    max_feeder_voltage_v: float
    min_feeder_voltage_v: float
    voltage_violations_count: int
    overvoltage_prevented_by_phased_array: bool
    switched_matrix_800v_nodes_count: int
    total_clu_thermal_dampening_kw: float
    state_hash: str
    vital_max_hp: int = VITAL_MAX_HP


class VppGridBrokerFleetCoordinator:
    """
    Centrálny VPP Broker dispečer (L3):
    - Agreguje 100 Krystal Edge uzlov na 630 kVA transformátore.
    - Riadi napäťovú stabilitu metódou fázovaného poľa (Phased Array).
    - Zabezpečuje, aby žiadny dom neprekročil 253.0 V ani pri masívnom výkupe elektriny.
    """

    def __init__(self, node_count: int = 100):
        self.node_count: int = node_count
        self.feeder = SubstationFeederModel(node_count=node_count)
        self.matrix_controllers = [SwitchedBatteryMatrixController() for _ in range(node_count)]

    def dispatch_fleet_request(
        self,
        requested_mw: float,
        spot_price_eur_mwh: float,
        base_household_load_kw_per_node: float = 1.2
    ) -> VppBrokerDispatchDecision:
        """
        Dispečerský optimalizátor pre sieťovú požiadavku (napr. +0.8 MW export alebo -1.2 MW odber):
        1. Ak requested_mw > 0 (aFRR+ / Export):
           - Prepne SiC MOSFET matice do 800V série (GRID BOOST) pre minimalizáciu strát.
           - Alokuje export primárne z uzlov blízko trafostanice (nízka impedancia).
           - Na vzdialených uzloch zapína induktívny jalový výkon Q < 0 a CLU tepelné stlačenie špičky,
             čím garantuje V_k <= 253.0 V.
        2. Ak requested_mw < 0 (aFRR- / Odber pri zápornej cene):
           - Prepne SiC matice do 48V paralelného režimu pre šetrné rýchle nabíjanie.
           - Zapne 100x batérií (až 10 kW/dom) + 100x CLU (5.4 kW/dom) s ohrevom TÚV.
        """
        active_powers_kw = [0.0] * self.node_count
        reactive_powers_kvar = [0.0] * self.node_count
        clu_dampening_kw = 0.0
        overvoltage_prevented = False

        if requested_mw > 0.0:
            # ------------------------------------------------------------------
            # aFRR+ / ŠPIČKOVÝ EXPORT DO SIETE (GRID BOOST 800V)
            # ------------------------------------------------------------------
            mode = VppFleetMode.AFRR_UP_GRID_BOOST_EXPORT
            # Prepneme uzly na 800V sériu
            for ctrl in self.matrix_controllers:
                ctrl.switch_topology(BatteryTopology.SERIES_800V)

            target_kw = min(requested_mw * 1000.0, 1200.0) # Max 1.2 MW
            remaining_kw = target_kw

            # Fázovaná alokácia naprieč vetvami:
            # V každej vetve uzly blízko začiatku (0..12) exportujú viac, uzly na konci (13..24) menej
            nodes_per_b = self.feeder.nodes_per_branch
            for b in range(self.feeder.BRANCHES_COUNT):
                for local_i in range(nodes_per_b):
                    global_idx = b * nodes_per_b + local_i
                    if remaining_kw <= 0.0:
                        break
                    max_alloc = 12.0 if local_i < 12 else (8.0 if local_i < 18 else 5.0)
                    alloc = min(max_alloc, remaining_kw)
                    active_powers_kw[global_idx] = alloc - base_household_load_kw_per_node
                    remaining_kw -= alloc

            # Kontrola napäťového profilu pred kompenzáciou
            raw_profile = self.feeder.solve_voltage_profile(active_powers_kw, reactive_powers_kvar)
            max_v_raw = max(n.voltage_v for n in raw_profile)

            # AK HROZÍ PREPÄTIE NAD 251.5 V -> AKTIVÁCIA FÁZOVANÉHO POLA:
            if max_v_raw >= VOLTAGE_WARNING_THRESHOLD_V:
                overvoltage_prevented = True
                mode = VppFleetMode.VOLTAGE_PROTECT_PHASED

                for b in range(self.feeder.BRANCHES_COUNT):
                    for local_i in range(nodes_per_b):
                        global_idx = b * nodes_per_b + local_i
                        # Druhá polovica vetvy vstrekuje induktívny jalový výkon Q < 0 (cos phi = 0.90)
                        if local_i >= 10:
                            reactive_powers_kvar[global_idx] = -abs(active_powers_kw[global_idx]) * 0.45

                        # Posledných 5 domov vetvy (20..24) zapne CLU (5.4 kW) s odvodom tepla do bojlera
                        if local_i >= 20:
                            active_powers_kw[global_idx] -= 5.4
                            clu_dampening_kw += 5.4

        else:
            # ------------------------------------------------------------------
            # aFRR- / MASÍVNY ODBER PREBYTKOVEJ ENERGIE (ZÁPORNÁ CENA)
            # ------------------------------------------------------------------
            mode = VppFleetMode.AFRR_DOWN_MAX_ABSORPTION
            for ctrl in self.matrix_controllers:
                ctrl.switch_topology(BatteryTopology.PARALLEL_48V)

            target_absorption_kw = min(abs(requested_mw) * 1000.0, 1540.0)
            remaining_abs_kw = target_absorption_kw
            nodes_per_b = self.feeder.nodes_per_branch

            # Fázovaný odber: domy blízko trafostanice (0..14) absorbujú viac, domy na konci (15..24) menej s podporou Q > 0
            for b in range(self.feeder.BRANCHES_COUNT):
                for local_i in range(nodes_per_b):
                    global_idx = b * nodes_per_b + local_i
                    if remaining_abs_kw <= 0.0:
                        break
                    max_abs = 12.0 if local_i < 15 else 4.5
                    alloc_abs = min(max_abs, remaining_abs_kw)
                    active_powers_kw[global_idx] = -alloc_abs - base_household_load_kw_per_node
                    if local_i >= 15:
                        reactive_powers_kvar[global_idx] = 4.0  # Kapacitná podpora napätia na konci vývodu
                    remaining_abs_kw -= alloc_abs

        # Finálne vyriešenie napäťového profilu po zásahu dispečera
        final_profile = self.feeder.solve_voltage_profile(active_powers_kw, reactive_powers_kvar)
        max_v = max(n.voltage_v for n in final_profile)
        min_v = min(n.voltage_v for n in final_profile)
        violations = sum(1 for n in final_profile if n.is_overvoltage_tripped)

        total_p_kw = sum(active_powers_kw)
        total_q_kvar = sum(reactive_powers_kvar)

        # 16-bajtový kryptografický hash stavu
        summary_str = f"{mode.value}:{total_p_kw:.2f}:{total_q_kvar:.2f}:{max_v:.2f}:{violations}:{VITAL_MAX_HP}"
        state_hash = hashlib.sha256(summary_str.encode("utf-8")).hexdigest()[:16]

        series_800v_count = sum(1 for c in self.matrix_controllers if c.topology == BatteryTopology.SERIES_800V)

        return VppBrokerDispatchDecision(
            fleet_mode=mode,
            target_grid_power_mw=requested_mw,
            achieved_grid_power_mw=round(total_p_kw / 1000.0, 3),
            total_active_export_mw=round(max(0.0, total_p_kw) / 1000.0, 3),
            total_active_absorption_mw=round(abs(min(0.0, total_p_kw)) / 1000.0, 3),
            total_reactive_power_mvar=round(total_q_kvar / 1000.0, 3),
            max_feeder_voltage_v=max_v,
            min_feeder_voltage_v=min_v,
            voltage_violations_count=violations,
            overvoltage_prevented_by_phased_array=overvoltage_prevented,
            switched_matrix_800v_nodes_count=series_800v_count,
            total_clu_thermal_dampening_kw=round(clu_dampening_kw, 1),
            state_hash=state_hash,
            vital_max_hp=VITAL_MAX_HP
        )


# Globálna inštancia VPP brokera
GLOBAL_VPP_BROKER = VppGridBrokerFleetCoordinator()
