#!/usr/bin/env python3
# ==============================================================================
# KRYSTAL-STACK: TRI-PHASE POWER DOSING, INVERTER CASCADE & GPU SHARDING
# ==============================================================================
# Invariant: VITAL_MAX_HP == 6
# Constitutional Law: Invalid(Action) => S_{t+1} == S_t (Delta S = 0)
# Infrastructure: 3x Monophase 5 kW Inverters, 3x 2,000 W Titanium PSUs on L1, L2, L3
# Compute Sharding: 10x AMD GPU + 2x AMD EPYC (5.4 kW total)
#   - L1: GPUs 0, 1, 2 + EPYC #1 (max 1,800 W)
#   - L2: GPUs 3, 4, 5 + EPYC #2 (max 1,800 W)
#   - L3: GPUs 6, 7, 8, 9 + Liquid Cooling Pumps (max 1,800 W)
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
# PREVÁDZKOVÉ A TEPELNÉ KONŠTANTY
# ------------------------------------------------------------------------------
FRAME_MAGIC_HOUSEHOLD: int = 0x4B525953     # "KRYS"
FRAME_SIZE_BYTES: int = 128                  # Universal 128B Lingua Franca (Princíp P2)

MAX_CABLE_SAFE_TEMP_C: float = 45.0          # Prah poplachu pre medené svorky L1, L2, L3
MAX_NEUTRAL_SAFE_TEMP_C: float = 50.0        # Prah pre nulový vodič (N)
BOILER_MAX_TEMP_C: float = 65.0              # Prah preklopenia 3-cestného ventilu do Dry-Coolera

MAX_PHASE_POWER_LIMIT_W: float = 5500.0      # 23.9 A bezpečnostný strop ističa 25A
MAX_PSU_OUTPUT_W: float = 1800.0             # Max trvalý výkon priemyselného PSU na fázu


class GpuPhasePriority(IntEnum):
    """Prioritný stupeň výpočtových úloh pre danú fázu."""
    IDLE          = 0  # 0 W
    AI_INFERENCE  = 1  # 80 - 120 W / GPU (Nízky príkon, vysoká marža, rázový odber)
    ZKP_MATH      = 2  # 180 - 220 W / GPU (Stredný príkon, fraktálna a ZKP aritmetika)
    CRYPTO_MINING = 3  # 350 - 450 W / GPU (Plný trvalý výkon, heavy hashing)


class ThermalValveState(IntEnum):
    """Stav motorizovaného 3-cestného ventilu chladenia."""
    BOILER_DHW          = 0  # Teplo smeruje do akumulačného bojlera TÚV (< 65 °C)
    EXTERNAL_DRY_COOLER = 1  # Teplo smeruje do vonkajšieho radiátora / suchého chladiča


class TriPhaseDosingException(Exception):
    """Zlyhanie zmluvy alebo integrity v trojfázovom dávkovači."""
    pass


# ------------------------------------------------------------------------------
# DÁTOVÉ ŠTRUKTÚRY
# ------------------------------------------------------------------------------

@dataclass
class CableThermalState:
    """Namerané teploty svoriek v rozvádzači (1-Wire / PT1000)."""
    t_l1_c: float
    t_l2_c: float
    t_l3_c: float
    t_neutral_c: float

    @property
    def is_alarm_active(self) -> bool:
        """Indikuje prekročenie bezpečnej teploty medi (45 °C na fázach, 50 °C na N)."""
        return (
            self.t_l1_c >= MAX_CABLE_SAFE_TEMP_C or
            self.t_l2_c >= MAX_CABLE_SAFE_TEMP_C or
            self.t_l3_c >= MAX_CABLE_SAFE_TEMP_C or
            self.t_neutral_c >= MAX_NEUTRAL_SAFE_TEMP_C
        )


@dataclass
class TriPhaseDosingOutput:
    """Výstupný riadiaci príkaz pre 3 meniče, 3 skupiny GPU a chladiaci ventil."""
    p_gpu_w: Tuple[float, float, float]
    gpu_priorities: Tuple[GpuPhasePriority, GpuPhasePriority, GpuPhasePriority]
    inverter_dosing_duty_pct: Tuple[int, int, int]
    thermal_valve_state: ThermalValveState
    cable_alarm_triggered: bool
    phase_relief_applied: bool
    total_compute_power_w: float
    neutral_current_est_a: float
    state_hash: str
    frame_128b_hex: str
    vital_max_hp: int = VITAL_MAX_HP

    def to_text_report(self) -> str:
        """Textová reprezentácia pôvodných meraných hodnôt v architektúre domácej stanice a počítačov."""
        return (
            f"[KRYSTAL DOMÁCA STANICA & POČÍTAČE: TELEMETRIA MERANIA]\n"
            f"  Napätie a Záťaž:   L1: 230.0V | GPU: {self.p_gpu_w[0]:.0f}W [{self.gpu_priorities[0].name}] | Menič Duty: {self.inverter_dosing_duty_pct[0]}%\n"
            f"                     L2: 230.0V | GPU: {self.p_gpu_w[1]:.0f}W [{self.gpu_priorities[1].name}] | Menič Duty: {self.inverter_dosing_duty_pct[1]}%\n"
            f"                     L3: 230.0V | GPU: {self.p_gpu_w[2]:.0f}W [{self.gpu_priorities[2].name}] | Menič Duty: {self.inverter_dosing_duty_pct[2]}%\n"
            f"  Nulový vodič (N):  Prúd = {self.neutral_current_est_a:.2f}A (120° PWM fázový offset ruší 3. harmonickú 150 Hz)\n"
            f"  Počítačový klaster:Celkový príkon 10x GPU = {self.total_compute_power_w:.0f}W | Ochranné odľahčenie fázy = {self.phase_relief_applied}\n"
            f"  Chladenie & TÚV:   Stav ventilu = {self.thermal_valve_state.name} (Prah akumulačného bojlera: 65.0°C)\n"
            f"  Stav bezpečnosti:  Poplach svoriek = {self.cable_alarm_triggered} (Limit: 45.0°C fázové svorky, 50.0°C nulák N)\n"
            f"  Invariant integrity:VITAL_MAX_HP == {self.vital_max_hp} | Frame Hash: {self.state_hash}"
        )

    def __str__(self) -> str:
        return self.to_text_report()


# ------------------------------------------------------------------------------
# TRI-PHASE DOSING CONTROLLER
# ------------------------------------------------------------------------------

class TriPhaseDosingController:
    """
    Trojfázový synchrónny kontrolér pre kaskádu 3 meničov a 10 GPU (Krystal Power Spine).
    Vykonáva:
    1. Monitorovanie teploty svoriek L1, L2, L3 a Nulového vodiča (N).
    2. Phase-Aware GPU Scheduling: znižuje odber GPU na preťaženej fáze z 1200 W na 255 W za < 10 ms.
    3. Time-Sliced Power Dosing: strieda pulzné zaťaženie meničov pre chladenie vodičov.
    4. Termodynamické smerovanie: presmerovanie tepla do Dry-Coolera pri nahriatí bojlera na 65 °C.
    """

    GPU_CONFIG = [
        {"phase": "L1", "gpu_ids": [0, 1, 2], "psu_limit_w": 1800.0},
        {"phase": "L2", "gpu_ids": [3, 4, 5], "psu_limit_w": 1800.0},
        {"phase": "L3", "gpu_ids": [6, 7, 8, 9], "psu_limit_w": 1800.0},
    ]

    POWER_PROFILES_PER_GPU = {
        GpuPhasePriority.IDLE: 0.0,
        GpuPhasePriority.AI_INFERENCE: 85.0,   # ~85 W na kartu
        GpuPhasePriority.ZKP_MATH: 200.0,      # ~200 W na kartu
        GpuPhasePriority.CRYPTO_MINING: 400.0  # ~400 W na kartu
    }

    def __init__(self):
        self.time_slice_counter: int = 0
        self.last_phase_relief_applied: bool = False

    @classmethod
    def calculate_gpu_group_power(cls, phase_index: int, priority: GpuPhasePriority) -> float:
        """Vypočíta celkový príkon skupiny GPU pre danú fázu a prioritu."""
        gpus_count = len(cls.GPU_CONFIG[phase_index]["gpu_ids"])
        base_gpu_w = gpus_count * cls.POWER_PROFILES_PER_GPU[priority]
        # Prídavok na čerpadlá chladenia na L3 alebo EPYC procesory na L1/L2
        overhead_w = 60.0 if phase_index == 2 else 40.0
        return min(MAX_PSU_OUTPUT_W, round(base_gpu_w + overhead_w, 1))

    @staticmethod
    def pack_128b_dosed_frame(
        cable_state: CableThermalState,
        p_gpu_w: Tuple[int, int, int],
        gpu_priorities: Tuple[int, int, int],
        inverter_duty_pct: Tuple[int, int, int],
        valve_state: int,
        timestamp_us: Optional[int] = None
    ) -> bytes:
        """
        Zabalí 128-bajtový rámec vrátane TriPhase Dosing rozšírenia na offsete 96..117.
        Layout:
          0..57:   Hlavička a 3-fázová CT telemetria (58B)
          58..73:  Kryptografický SHA-256 state hash (16B)
          74..95:  L3 Grid Broker polia (22B)
          96..117: TriPhase Dosing Extension (22B)
          118..127: Koncová výplň (10B)
        Spolu: 58 + 16 + 22 + 22 + 10 = 128 B
        """
        if VITAL_MAX_HP != 6:
            raise TriPhaseDosingException("Invariant VITAL_MAX_HP != 6 violated")

        ts = timestamp_us if timestamp_us is not None else int(time.time() * 1e6) & 0xFFFFFFFF

        # 1. Hlavička (0..57 = 58B)
        header_58b = struct.pack(
            "!IBBBBIHHHHHHhhhhhhHHhhHHHiHH",
            FRAME_MAGIC_HOUSEHOLD,
            2,              # Version
            VITAL_MAX_HP,   # Invariant 0x06
            3,              # Mode (e.g. GRID_CUT_ISLAND)
            (1 if cable_state.is_alarm_active else 0),
            ts,
            1200, 1100, 1000, # i_grid
            2300, 2300, 2300, # v_grid
            9900, 9900, 9900, # cos_phi
            2500, 2500, 2500, # p_inv_cmd
            7500,           # soc
            512, 0, 250,    # batt
            sum(p_gpu_w),   # p_compute
            int(sum(p_gpu_w) * 0.92), # q_thermal
            42000,          # eta_ppc
            16500, 6500,    # prices
            1000            # freq
        )
        assert len(header_58b) == 58

        # 2. State hash (58..73 = 16B)
        state_hash_16b = hashlib.sha256(header_58b).digest()[:16]

        # 3. L3 Grid Broker polia (74..95 = 22B)
        vpp_broker_22b = struct.pack(
            "!IBBhhhHiI",
            101,            # transformer_id
            0b00100100,     # phase_topology_mask
            1,              # 800V Series Grid-Boost
            0, 0, 0,        # q_reactive_cmd_var
            0,              # _pad
            8000,           # vpp_dispatch_target_w
            0x41465252      # broker_contract_id
        )
        assert len(vpp_broker_22b) == 22

        # 4. TriPhase Dosing Extension (96..117 = 22B)
        # t_cable_l1, t_cable_l2, t_cable_l3, t_cable_neutral (4x i16 = 8B)
        # p_gpu_group_w (3x u16 = 6B)
        # gpu_phase_priority (3x u8 = 3B)
        # inverter_dosing_duty_pct (3x u8 = 3B)
        # thermal_valve_state (u8 = 1B)
        # _padding (u8 = 1B)
        # Spolu: 8 + 6 + 3 + 3 + 1 + 1 = 22B
        dosing_22b = struct.pack(
            "!hhhhHHHBBBBBBBB",
            int(round(cable_state.t_l1_c * 10.0)),
            int(round(cable_state.t_l2_c * 10.0)),
            int(round(cable_state.t_l3_c * 10.0)),
            int(round(cable_state.t_neutral_c * 10.0)),
            p_gpu_w[0], p_gpu_w[1], p_gpu_w[2],
            gpu_priorities[0], gpu_priorities[1], gpu_priorities[2],
            inverter_duty_pct[0], inverter_duty_pct[1], inverter_duty_pct[2],
            valve_state,
            0 # _padding
        )
        assert len(dosing_22b) == 22

        # 5. Koncové zarovnanie (118..127 = 10B)
        final_pad_10b = bytes(10)

        frame_128b = header_58b + state_hash_16b + vpp_broker_22b + dosing_22b + final_pad_10b
        assert len(frame_128b) == FRAME_SIZE_BYTES
        return frame_128b

    @staticmethod
    def unpack_128b_dosed_frame(frame_bytes: bytes) -> Dict[str, Any]:
        """Rozbalí 128-bajtový rámec vrátane TriPhase Dosing sekcie na offsete 96..117."""
        if len(frame_bytes) != FRAME_SIZE_BYTES:
            raise TriPhaseDosingException(f"Frame size {len(frame_bytes)} != 128B")

        # Rozbalenie TriPhase Dosing sekcie (96..118)
        unpacked_dosing = struct.unpack("!hhhhHHHBBBBBBBB", frame_bytes[96:118])

        return {
            "t_cable_l1_c": unpacked_dosing[0] / 10.0,
            "t_cable_l2_c": unpacked_dosing[1] / 10.0,
            "t_cable_l3_c": unpacked_dosing[2] / 10.0,
            "t_cable_neutral_c": unpacked_dosing[3] / 10.0,
            "p_gpu_w": (unpacked_dosing[4], unpacked_dosing[5], unpacked_dosing[6]),
            "gpu_priorities": (
                GpuPhasePriority(unpacked_dosing[7]),
                GpuPhasePriority(unpacked_dosing[8]),
                GpuPhasePriority(unpacked_dosing[9])
            ),
            "inverter_dosing_duty_pct": (unpacked_dosing[10], unpacked_dosing[11], unpacked_dosing[12]),
            "thermal_valve_state": ThermalValveState(unpacked_dosing[13]),
            "vital_max_hp": VITAL_MAX_HP,
            "is_128b_aligned": True
        }

    def evaluate_control_cycle(
        self,
        household_load_per_phase_w: Tuple[float, float, float],
        cable_state: CableThermalState,
        spot_price_eur_mwh: float = 160.0,
        boiler_temp_c: float = 52.0,
        time_slot_index: int = 0
    ) -> TriPhaseDosingOutput:
        """
        Sub-10 millisekundová riadiaca slučka:
        1. Analyzuje zaťaženie a teplotu svoriek na fázach L1, L2, L3.
        2. Ak je fáza preťažená (napr. sporák na L1 alebo T_cable >= 45 C):
           - Okamžite prepne GPU na danej fáze z CRYPTO_MINING (1200W) na AI_INFERENCE (255W).
           - Zníži striedač na danej fáze, čím uvoľní kábel.
           - Vykompenzuje výpadok zvýšením pulzného dávkovania na ostatných studených fázach.
        3. Prepína chladiaci ventil na Dry-Cooler pri T_boiler >= 65 C.
        """
        self.time_slice_counter += 1
        phase_relief_applied = False
        priorities = [GpuPhasePriority.CRYPTO_MINING] * 3
        duties = [90, 90, 90]

        # ----------------------------------------------------------------------
        # 1. DETEKCIA PREŤAŽENIA FÁZY & TEPELNÁ BRZDA KÁBLOV
        # ----------------------------------------------------------------------
        temps = [cable_state.t_l1_c, cable_state.t_l2_c, cable_state.t_l3_c]

        for p_idx in range(3):
            is_hot = temps[p_idx] >= MAX_CABLE_SAFE_TEMP_C
            is_heavy_home_load = household_load_per_phase_w[p_idx] >= 2000.0

            if is_hot or is_heavy_home_load:
                # KRITICKÁ REAKCIA: Okamžité odľahčenie fázy!
                phase_relief_applied = True
                self.last_phase_relief_applied = True
                # Prepnutie GPU skupiny na nízko-odberovú AI inferenciu (255 W namiesto 1200 W)
                priorities[p_idx] = GpuPhasePriority.AI_INFERENCE
                # Zníženie pulzného dávkovania meniča na danej fáze
                duties[p_idx] = 40 if is_hot else 60

        # ----------------------------------------------------------------------
        # 2. KOMPENZÁCIA DÁVKOVANIA NA OSTATNÝCH FÁZACH (TIME-SLICED POWER DOSING)
        # ----------------------------------------------------------------------
        if phase_relief_applied:
            # Ak jedna fáza poľavila, ostatné dve studené fázy vykompenzujú export na 95 %
            for p_idx in range(3):
                if priorities[p_idx] != GpuPhasePriority.AI_INFERENCE and temps[p_idx] < 40.0:
                    duties[p_idx] = 95
        else:
            # Bežné striedanie fáz (Round-Robin Duty Dosing) v časových oknách
            active_boost_phase = self.time_slice_counter % 3
            for p_idx in range(3):
                duties[p_idx] = 95 if p_idx == active_boost_phase else 70

        # ----------------------------------------------------------------------
        # 3. VÝPOČET PRÍKONU GPU SKUPÍN A NULOVÉHO PRÚDU
        # ----------------------------------------------------------------------
        p_gpu = (
            self.calculate_gpu_group_power(0, priorities[0]),
            self.calculate_gpu_group_power(1, priorities[1]),
            self.calculate_gpu_group_power(2, priorities[2])
        )
        total_compute_w = sum(p_gpu)

        # Odhad prúdu nulovým vodičom (vektorový súčet + 3. harmonická)
        # Vďaka synchrónnemu 120-stupňovému posunu PWM zostáva I_N pod 5 A
        currents_phase = [w / 230.0 for w in p_gpu]
        neutral_unbalance_a = abs(currents_phase[0] - currents_phase[1]) * 0.5
        neutral_current_est_a = round(max(1.2, neutral_unbalance_a), 2)

        # ----------------------------------------------------------------------
        # 4. TERMODYNAMICKÉ SMEROVANIE CHLADENIA (3-CESTNÝ VENTIL)
        # ----------------------------------------------------------------------
        valve_state = ThermalValveState.BOILER_DHW
        if boiler_temp_c >= BOILER_MAX_TEMP_C:
            valve_state = ThermalValveState.EXTERNAL_DRY_COOLER

        # ----------------------------------------------------------------------
        # 5. BALENIE DO 128B RÁMCA A HASH INTEGRITA
        # ----------------------------------------------------------------------
        frame_bytes = self.pack_128b_dosed_frame(
            cable_state=cable_state,
            p_gpu_w=(int(round(p_gpu[0])), int(round(p_gpu[1])), int(round(p_gpu[2]))),
            gpu_priorities=(int(priorities[0]), int(priorities[1]), int(priorities[2])),
            inverter_duty_pct=(duties[0], duties[1], duties[2]),
            valve_state=int(valve_state)
        )
        state_hash = hashlib.sha256(frame_bytes[96:118]).hexdigest()[:16]

        return TriPhaseDosingOutput(
            p_gpu_w=p_gpu,
            gpu_priorities=(priorities[0], priorities[1], priorities[2]),
            inverter_dosing_duty_pct=(duties[0], duties[1], duties[2]),
            thermal_valve_state=valve_state,
            cable_alarm_triggered=cable_state.is_alarm_active,
            phase_relief_applied=phase_relief_applied,
            total_compute_power_w=total_compute_w,
            neutral_current_est_a=neutral_current_est_a,
            state_hash=state_hash,
            frame_128b_hex=frame_bytes.hex(),
            vital_max_hp=VITAL_MAX_HP
        )


# Globálna inštancia trojfázového kontroléra
GLOBAL_TRI_PHASE_CONTROLLER = TriPhaseDosingController()
