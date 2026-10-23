#!/usr/bin/env python3
# ==============================================================================
# KRYSTAL-STACK: PREDICTIVE GRID-AWARE ENERGY ARBITRAGE ENGINE (MOONSHOT C)
# ==============================================================================
# Invariant: VITAL_MAX_HP == 6
# Constitutional Law: Invalid(Action) => S_{t+1} == S_t (Delta S = 0)
# Target Hardware: 10x AMD GPU + 128-core AMD EPYC (500 W - 5,400 W dynamic range)
# Grid Target: SEPS / OKTE Spot Market (50.000 Hz, FCR Primary Regulation)
# ==============================================================================

import time
import math
import struct
import hashlib
from enum import IntEnum
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

from krystal_kernel.community_showcase_integrator import (
    XdpNetworkDescriptorBridge,
    MultiModelChatInferenceAdapter,
    VITAL_MAX_HP
)

# ------------------------------------------------------------------------------
# GRID & HARDWARE CONSTANTS
# ------------------------------------------------------------------------------
NOMINAL_GRID_FREQ_HZ: float = 50.000
FREQ_DEADBAND_HZ: float = 0.020           # +/- 20 mHz ENTSO-E deadband
FREQ_FULL_ACTIVATION_HZ: float = 0.200     # +/- 200 mHz full FCR activation
CLUSTER_MIN_POWER_WATTS: float = 500.0     # Idle / minimal single-stream AI
CLUSTER_MAX_POWER_WATTS: float = 5400.0    # 10x GPU + 128-core EPYC cluster peak
THERMAL_THROTTLE_TEMP_C: float = 82.0      # Junction thermal throttling threshold
THERMAL_CLAMP_POWER_WATTS: float = 2800.0  # Safe thermal dissipation ceiling
FRAME_MAGIC_ENERGY: int = 0x4B525953       # "KRYS" magic in big-endian uint32
FRAME_SIZE_BYTES: int = 128                # Universal 128B Lingua Franca (Principle P2)


class WorkloadMode(IntEnum):
    """Diskrétne prevádzkové režimy arbitrážneho enginu."""
    CRYPTO_MINING              = 1  # 100% hash compute, maximalizácia odberu pri lacnej/zápornej energii
    HIGH_MARGIN_AI_INFERENCE   = 2  # Prioritizácia LLM inferencie, minimalizácia energetických nákladov
    DYNAMIC_HYBRID_ARBITRAGE   = 3  # Kvadratická optimalizácia pomeru Mining vs. AI
    EMERGENCY_LOAD_SHED        = 4  # Okamžité odľahčenie siete pri kritickej podfrekvencii (< 49.80 Hz)


class EnergyArbitrageException(Exception):
    """Zlyhanie invariantu alebo zmluvnej integrity v energetickej arbitráži."""
    pass


# ------------------------------------------------------------------------------
# DÁTOVÉ ŠTRUKTÚRY TELEMETRIE A ROZHODNUTIA
# ------------------------------------------------------------------------------

@dataclass
class GridTelemetry:
    """Stav synchrónnej siete a spotového trhu s elektrinou."""
    frequency_hz: float
    spot_price_eur_mwh: float
    timestamp_us: int = 0

    @property
    def frequency_dev_hz(self) -> float:
        """Odchýlka od 50.000 Hz."""
        return round(self.frequency_hz - NOMINAL_GRID_FREQ_HZ, 4)


@dataclass
class ArbitrageDecision:
    """Výstupné rozhodnutie riadiaceho enginu s plnou telemetriou a hash integritou."""
    workload_mode: WorkloadMode
    allocated_power_watts: float
    mining_ratio_alpha: float
    ai_inference_ratio: float
    fcr_response_kw: float
    expected_profit_eur_per_hour: float
    grid_cost_eur_per_hour: float
    fcr_revenue_eur_per_hour: float
    thermal_throttled: bool
    state_hash: str
    frame_128b_hex: str
    vital_max_hp: int = VITAL_MAX_HP


# ------------------------------------------------------------------------------
# PREDICTIVE GRID-AWARE ENERGY ARBITRAGE ENGINE
# ------------------------------------------------------------------------------

class PredictiveEnergyArbitrageEngine:
    """
    Prediktívny energetický arbitrážny engine (Moonshot C).
    Autonómne moduluje príkon klastra medzi 500 W a 5 400 W na základe spotových cien OKTE,
    frekvenčnej stability siete (FCR) a termálnej telemetrie kremíka.
    """

    # Ekonomické koeficienty prepočítané na watt-hodinu
    REVENUE_MINE_EUR_PER_WH: float = 0.045e-3   # ~0.045 EUR/kWh hodnota vyťaženého hashu
    REVENUE_AI_MAX_EUR_PER_HOUR: float = 4.59   # Max hodnota pri plnom vyťažení AI inference
    FCR_CAPACITY_FEE_EUR_PER_KW_H: float = 0.018 # Poplatok za rezervovanú flexibilitu

    @staticmethod
    def pack_128b_energy_frame(
        grid_freq_hz: float,
        spot_price_eur_mwh: float,
        target_power_w: float,
        mode: WorkloadMode,
        alpha_ratio: float,
        timestamp_us: Optional[int] = None
    ) -> bytes:
        """
        Zabalí telemetrický a arbitrážny stav do presne 128-bajtového rámca (Princíp P2).
        Layout:
          [0..3]   Magic: 0x4B525953
          [4]      Version: 2
          [5]      VitalMaxHp: 6 (Invariant)
          [6..7]   Mode (uint16)
          [8..11]  Timestamp Delta (uint32)
          [12..15] Grid Frequency (uint32 in millihertz, e.g. 50020 for 50.020 Hz)
          [16..17] Frequency Dev (int16 in millihertz, e.g. +20)
          [18..21] Spot Price (int32 in cents/MWh, e.g. 4500 for 45.00 EUR)
          [22..23] Target Power (uint16 in Watts, 500..5400)
          [24..25] Alpha Ratio (uint16 in basis points 0..10000)
          [26..127] Zero padding (102 bytes)
        """
        if VITAL_MAX_HP != 6:
            raise EnergyArbitrageException("Constitutional invariant VITAL_MAX_HP != 6 violated")

        ts = timestamp_us if timestamp_us is not None else int(time.time() * 1e6) & 0xFFFFFFFF
        freq_mhz = int(round(grid_freq_hz * 1000.0))
        dev_mhz = int(round((grid_freq_hz - NOMINAL_GRID_FREQ_HZ) * 1000.0))
        price_cents = int(round(spot_price_eur_mwh * 100.0))
        power_w = int(round(target_power_w))
        alpha_bp = int(round(alpha_ratio * 10000.0))

        # 26 bajtov binárnej hlavičky
        header = struct.pack(
            "!IBBHIIhiHH",
            FRAME_MAGIC_ENERGY,
            2,              # Version
            VITAL_MAX_HP,   # Invariant 0x06
            int(mode),
            ts,
            freq_mhz,
            dev_mhz,
            price_cents,
            power_w,
            alpha_bp
        )
        # Zarovnanie na presne 128 bajtov
        frame_128b = header.ljust(FRAME_SIZE_BYTES, b"\x00")
        assert len(frame_128b) == FRAME_SIZE_BYTES
        return frame_128b

    @staticmethod
    def unpack_128b_energy_frame(frame_bytes: bytes) -> Dict[str, Any]:
        """Rozbalí 128-bajtový rámec a overí integritu magic kódu a invariantu."""
        if len(frame_bytes) != FRAME_SIZE_BYTES:
            raise EnergyArbitrageException(f"Frame size {len(frame_bytes)} != 128B")

        magic, version, hp, mode_val, ts, freq_mhz, dev_mhz, price_cents, power_w, alpha_bp = struct.unpack(
            "!IBBHIIhiHH",
            frame_bytes[:26]
        )

        if magic != FRAME_MAGIC_ENERGY:
            raise EnergyArbitrageException(f"Invalid magic 0x{magic:08X}")
        if hp != VITAL_MAX_HP:
            raise EnergyArbitrageException(f"Corrupted invariant {hp} != 6")

        return {
            "version": version,
            "vital_max_hp": hp,
            "workload_mode": WorkloadMode(mode_val),
            "timestamp_us": ts,
            "grid_freq_hz": freq_mhz / 1000.0,
            "frequency_dev_hz": dev_mhz / 1000.0,
            "spot_price_eur_mwh": price_cents / 100.0,
            "target_power_watts": float(power_w),
            "alpha_ratio": alpha_bp / 10000.0,
            "is_128b_aligned": True
        }

    @classmethod
    def solve_arbitrage(
        cls,
        telemetry: GridTelemetry,
        silicon_junction_temp_c: float = 58.0,
        ai_queue_depth: int = 16
    ) -> ArbitrageDecision:
        """
        Kvadratický riešič energetickej arbitráže:
        1. Kontrola havarijnej podfrekvencie (< 49.80 Hz) -> EMERGENCY_LOAD_SHED.
        2. Analýza cenového prahu OKTE a FCR požiadaviek siete.
        3. Výpočet optimálneho príkonu P* v rozsahu [500 W, 5400 W] a pomeru alpha.
        4. Ochranná brzda kremíka (Physical-First Actuation): clamp pri Tj >= 82 C.
        5. Vygenerovanie deterministického 128B rámca a stavového hashu.
        """
        freq = telemetry.frequency_hz
        dev = telemetry.frequency_dev_hz
        spot = telemetry.spot_price_eur_mwh

        # ----------------------------------------------------------------------
        # 1. POHOTOVOSTNÁ DETEKCIA PODFREKVENCIE (SEVERITY GUARD)
        # ----------------------------------------------------------------------
        if freq < 49.800:
            # Kritická podfrekvencia: sieť kolabuje, okamžité zhodenie na technologické minimum
            mode = WorkloadMode.EMERGENCY_LOAD_SHED
            power = CLUSTER_MIN_POWER_WATTS
            alpha = 0.0
            fcr_response_kw = -(CLUSTER_MAX_POWER_WATTS - CLUSTER_MIN_POWER_WATTS) / 1000.0
        # ----------------------------------------------------------------------
        # 2. DISKRÉTNE REŽIMY PODĽA TRHU A SIETE
        # ----------------------------------------------------------------------
        elif spot <= 20.0 or dev >= 0.050:
            # Lacná/záporná elektrina ALEBO prebytok energie v sieti (nadfrekvencia)
            # Maximálny odber (Crypto Mining / Hash Workload)
            mode = WorkloadMode.CRYPTO_MINING
            alpha = 1.0
            # Ak je cena záporná, modulujeme až na absolútne maximum 5400 W
            if spot < 0.0:
                power = CLUSTER_MAX_POWER_WATTS
            else:
                power = min(CLUSTER_MAX_POWER_WATTS, 4500.0 + (dev / FREQ_FULL_ACTIVATION_HZ) * 900.0)
            fcr_response_kw = (power - CLUSTER_MIN_POWER_WATTS) / 1000.0

        elif spot >= 110.0 or dev <= -0.050:
            # Drahá elektrina ALEBO deficit energie v sieti (podfrekvencia)
            # Prioritizácia vysoko maržovej AI inferencie s útlmom ťažby
            mode = WorkloadMode.HIGH_MARGIN_AI_INFERENCE
            alpha = 0.0
            # Škálovanie príkonu podľa hĺbky AI fronty (500 W až 1800 W)
            power = min(1800.0, CLUSTER_MIN_POWER_WATTS + (min(ai_queue_depth, 64) / 64.0) * 1300.0)
            fcr_response_kw = -(CLUSTER_MAX_POWER_WATTS - power) / 1000.0

        else:
            # ------------------------------------------------------------------
            # 3. KONTINUÁLNA KVADRATICKÁ HYBRIDNÁ ARBITRÁŽ
            # ------------------------------------------------------------------
            mode = WorkloadMode.DYNAMIC_HYBRID_ARBITRAGE
            # Pomer alpha klesá lineárne s rastúcou cenou elektriny od 20 EUR do 110 EUR
            alpha = max(0.0, min(1.0, (110.0 - spot) / (110.0 - 20.0)))
            # Kvadratická reakcia na spotovú cenu v zmesi
            power_spread = CLUSTER_MAX_POWER_WATTS - CLUSTER_MIN_POWER_WATTS
            power = CLUSTER_MIN_POWER_WATTS + (alpha ** 1.5) * power_spread
            fcr_response_kw = (dev / FREQ_FULL_ACTIVATION_HZ) * (power_spread / 1000.0)

        # ----------------------------------------------------------------------
        # 4. FYZIKÁLNE PREDCHÁDZANIE PREHRIATIA (PRINCÍP P7)
        # ----------------------------------------------------------------------
        thermal_throttled = False
        if silicon_junction_temp_c >= THERMAL_THROTTLE_TEMP_C:
            if power > THERMAL_CLAMP_POWER_WATTS:
                power = THERMAL_CLAMP_POWER_WATTS
                thermal_throttled = True

        # Striktné ohraničenie [500 W, 5400 W]
        power = max(CLUSTER_MIN_POWER_WATTS, min(CLUSTER_MAX_POWER_WATTS, power))
        alpha = max(0.0, min(1.0, alpha))
        ai_ratio = round(1.0 - alpha, 4)

        # ----------------------------------------------------------------------
        # 5. EKONOMICKÉ BILANCOVANIE (EUR / HODINA)
        # ----------------------------------------------------------------------
        # Cena energie zo siete (pri zápornej cene je grid_cost záporný = čistý príjem!)
        grid_cost_eur_h = power * (spot / 1e6)

        # Zisk z miningu: r_mine * W * 1h
        mine_rev_eur_h = alpha * (power * cls.REVENUE_MINE_EUR_PER_WH)

        # Zisk z AI inferencie: nelineárna krivka saturácie
        ai_power = (1.0 - alpha) * power
        ai_rev_eur_h = (1.0 - alpha) * cls.REVENUE_AI_MAX_EUR_PER_HOUR * (
            1.0 - math.exp(-max(0.0, ai_power - CLUSTER_MIN_POWER_WATTS) / 1200.0)
        )

        # FCR regulačný bonus za odchýlku mimo pásma necitlivosti
        if abs(dev) > FREQ_DEADBAND_HZ:
            fcr_rev_eur_h = abs(dev / FREQ_FULL_ACTIVATION_HZ) * cls.FCR_CAPACITY_FEE_EUR_PER_KW_H * (power / 1000.0)
        else:
            fcr_rev_eur_h = 0.0

        net_profit_eur_h = round(mine_rev_eur_h + ai_rev_eur_h + fcr_rev_eur_h - grid_cost_eur_h, 4)

        # ----------------------------------------------------------------------
        # 6. ZBALENIE DO 128B RÁMCA A HASH INTEGRITA
        # ----------------------------------------------------------------------
        frame_bytes = cls.pack_128b_energy_frame(
            grid_freq_hz=freq,
            spot_price_eur_mwh=spot,
            target_power_w=power,
            mode=mode,
            alpha_ratio=alpha,
            timestamp_us=telemetry.timestamp_us
        )
        state_hash = hashlib.sha256(frame_bytes[:26]).hexdigest()[:16]

        return ArbitrageDecision(
            workload_mode=mode,
            allocated_power_watts=round(power, 2),
            mining_ratio_alpha=round(alpha, 4),
            ai_inference_ratio=ai_ratio,
            fcr_response_kw=round(fcr_response_kw, 3),
            expected_profit_eur_per_hour=net_profit_eur_h,
            grid_cost_eur_per_hour=round(grid_cost_eur_h, 4),
            fcr_revenue_eur_per_hour=round(fcr_rev_eur_h, 4),
            thermal_throttled=thermal_throttled,
            state_hash=state_hash,
            frame_128b_hex=frame_bytes.hex(),
            vital_max_hp=VITAL_MAX_HP
        )

    # --------------------------------------------------------------------------
    # PREPOJENIE NA XDP A MULTI-MODEL SHOWCASE ADAPTÉRY
    # --------------------------------------------------------------------------

    @classmethod
    def ingest_xdp_energy_telemetry_packet(
        cls,
        src_ip: str,
        grid_freq_hz: float,
        spot_price_eur_mwh: float,
        silicon_junction_temp_c: float = 58.0
    ) -> Dict[str, Any]:
        """
        Prijme sieťovú telemetriu cez XDP deskriptor (Princíp P1),
        vyrieši energetickú arbitráž a vráti zjednotený stavový kontrakt.
        """
        # 1. Zabalenie do XDP sieťového deskriptora
        xdp_desc = XdpNetworkDescriptorBridge.pack_xdp_packet_descriptor(
            src_ip=src_ip,
            dst_ip="10.0.0.1",
            src_port=9999,
            dst_port=8080,
            protocol=17, # UDP pre nízku latenciu
            packet_len=FRAME_SIZE_BYTES
        )

        # 2. Vyriešenie arbitráže
        telemetry = GridTelemetry(
            frequency_hz=grid_freq_hz,
            spot_price_eur_mwh=spot_price_eur_mwh
        )
        decision = cls.solve_arbitrage(telemetry, silicon_junction_temp_c=silicon_junction_temp_c)

        # 3. Dispečing AI inferencie ak je alokovaný výkon pre AI
        ai_dispatch_summary = None
        if decision.ai_inference_ratio > 0.05:
            ai_dispatch_summary = MultiModelChatInferenceAdapter.dispatch_parallel_comparison(
                prompt=f"Arbitrage AI dispatch at {decision.allocated_power_watts}W, mode={decision.workload_mode.name}"
            )

        return {
            "xdp_descriptor": xdp_desc,
            "arbitrage_decision": asdict(decision),
            "ai_dispatch": ai_dispatch_summary,
            "vital_max_hp": VITAL_MAX_HP,
            "is_contract_valid": True
        }


# Globálna inštancia enginu
GLOBAL_ENERGY_ARBITRAGE_ENGINE = PredictiveEnergyArbitrageEngine()
