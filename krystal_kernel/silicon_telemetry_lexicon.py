#!/usr/bin/env python3
# ==============================================================================
# KRYSTAL-STACK: SILICON TELEMETRY, HEXADECIMAL EVENT CODE LATTICE &
# OPENVINO DYNAMIC POLICY ARBITER (MODULE H-07)
# ==============================================================================
# Core Invariant: VITAL_MAX_HP == 6
# Hardware Target: Intel Core i5-1135G7 (AVX-512) + Intel Iris Xe (80 EUs Gen12)
# ==============================================================================

import struct
import time
import math
from enum import IntEnum
from typing import Dict, Any, List, Optional, Tuple, Union

VITAL_MAX_HP: int = 6


class ContractViolation(Exception):
    """Constitutional contract invariant violation in silicon telemetry."""
    pass


# ==============================================================================
# 1. HEXADECIMAL TELEMETRY EVENT CODE LATTICE
# ==============================================================================

class TelemetryEventCode(IntEnum):
    """
    Kompaktná 1-bajtová hexadecimálna mriežka telemetrických stavov procesora,
    zbernice a inferenčného runtimeu.
    """
    # 0x10 - 0x1F: Termálne a napájacie stavy kremíka (Silicon Thermal & Power)
    SILICON_NOMINAL_ACTIVE    = 0x10  # Tj <= 65.0 C, plné frekvencie
    SILICON_THERMAL_ELEVATED  = 0x11  # 65.0 C < Tj <= 82.0 C, stredná záťaž
    SILICON_THERMAL_THROTTLE  = 0x12  # Tj > 82.0 C, PL1/PL2 clamp / frekvencia znížená

    # 0x20 - 0x2F: Impedancia a saturácia pamäťovej zbernice UMA (Bus Contention)
    UMA_CONTENTION_LOW        = 0x20  # tau_cache_transit < 2.5 µs, čistá UMA zbernica
    UMA_CONTENTION_HIGH       = 0x21  # tau_cache_transit >= 5.0 µs, saturácia zbernice

    # 0x30 - 0x3F: Šumové prahy a kvalita signálu napájania (Noise Floor / Decibels)
    NOISE_FLOOR_NOMINAL_DB    = 0x30  # Return Loss RL >= 25.0 dB, čisté napájanie
    NOISE_FLOOR_DEGRADED_DB   = 0x31  # Return Loss RL < 18.0 dB, zvlnenie a odraz

    # 0x40 - 0x4F: Fázy inferencie OpenVINO (OpenVINO Inference Phases)
    OPENVINO_GPU_PREFILL      = 0x40  # Iris Xe GPU 80 EUs vykonáva prefill
    OPENVINO_CPU_DECODE       = 0x41  # CPU AVX-512 vykonáva autoregresívny decode

    # 0x50 - 0x5F: Hranica odovzdania tenzorov (Boundary Handoff & Synchronization)
    HANDOFF_SYNC_WAIT         = 0x50  # Čakanie na zeEventHostSynchronize
    HANDOFF_COMPLETE          = 0x51  # Úspešný prechod rámca cez zdieľanú pamäť

    # 0xF0 - 0xFF: Riadiace a konštitučné udalosti (Control & Invariants)
    POLICY_STRATEGY_SHIFT     = 0xFE  # Zmena politiky plánovača (HYBRID <-> GPU_ONLY)
    CONSTITUTIONAL_HEARTBEAT  = 0xFF  # Potvrdenie invariantu VITAL_MAX_HP == 6


# ==============================================================================
# 2. TYPOVANÉ STAVOVÉ TRIEDY (EVENT STRUCTS)
# ==============================================================================

class BaseTelemetryEvent:
    """Základná trieda telemetrickej udalosti s 8-bajtovým binárnym kódovaním."""
    def __init__(self, code: TelemetryEventCode, raw_value: int = 0, timestamp_us: Optional[int] = None, **_kwargs):
        self.code = code
        self.raw_value = raw_value & 0xFFFF
        self.timestamp_us = timestamp_us if timestamp_us is not None else int(time.perf_counter() * 1_000_000) & 0xFFFFFFFF
        self.invariant = VITAL_MAX_HP

        if self.invariant != 6:
            raise ContractViolation(f"Constitutional invariant violated: {self.invariant} != 6")

    def pack(self) -> bytes:
        """
        Zbalenie do kompaktného 8-bajtového pulzného rámca:
        <I (4B uint32 timestamp_us) + B (1B event_code) + B (1B invariant 0x06) + H (2B uint16 raw_value)
        """
        return struct.pack("<IBBH", self.timestamp_us, int(self.code), self.invariant, self.raw_value)

    @classmethod
    def unpack(cls, buffer: bytes) -> "BaseTelemetryEvent":
        """Rozbalenie 8-bajtového pulzného rámca s overením invariantu."""
        if len(buffer) < 8:
            raise ValueError(f"Buffer must be at least 8 bytes, got {len(buffer)}")
        timestamp_us, code_val, invariant, raw_val = struct.unpack("<IBBH", buffer[:8])
        if invariant != VITAL_MAX_HP:
            raise ContractViolation(f"Malformed frame invariant: {invariant} != {VITAL_MAX_HP}")
        
        try:
            code = TelemetryEventCode(code_val)
        except ValueError:
            code = TelemetryEventCode.CONSTITUTIONAL_HEARTBEAT

        klass = EVENT_CLASS_MAP.get(code, BaseTelemetryEvent)
        return klass(raw_value=raw_val, timestamp_us=timestamp_us)

    @property
    def formatted_measurement(self) -> str:
        """Pôvodná nameraná fyzikálna hodnota v architektúre počítača."""
        return f"raw={self.raw_value}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hex_code": f"0x{self.code:02X}",
            "name": self.code.name,
            "raw_value": self.raw_value,
            "formatted_measurement": self.formatted_measurement,
            "timestamp_us": self.timestamp_us,
            "vital_max_hp": self.invariant
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [0x{self.code:02X} {self.code.name}] measurement='{self.formatted_measurement}' (raw={self.raw_value}) t={self.timestamp_us}µs>"


class SiliconNominalActiveEvent(BaseTelemetryEvent):
    def __init__(self, temp_c: Optional[float] = None, raw_value: Optional[int] = None, **kwargs):
        val = raw_value if raw_value is not None else int(round((temp_c if temp_c is not None else 55.0) * 100))
        super().__init__(TelemetryEventCode.SILICON_NOMINAL_ACTIVE, raw_value=val, **kwargs)

    @property
    def temperature_c(self) -> float:
        return self.raw_value / 100.0

    @property
    def formatted_measurement(self) -> str:
        return f"{self.temperature_c:.2f}°C"


class SiliconThermalElevatedEvent(BaseTelemetryEvent):
    def __init__(self, temp_c: Optional[float] = None, raw_value: Optional[int] = None, **kwargs):
        val = raw_value if raw_value is not None else int(round((temp_c if temp_c is not None else 72.0) * 100))
        super().__init__(TelemetryEventCode.SILICON_THERMAL_ELEVATED, raw_value=val, **kwargs)

    @property
    def temperature_c(self) -> float:
        return self.raw_value / 100.0

    @property
    def formatted_measurement(self) -> str:
        return f"{self.temperature_c:.2f}°C (ELEVATED)"


class SiliconThermalThrottleEvent(BaseTelemetryEvent):
    def __init__(self, temp_c: Optional[float] = None, raw_value: Optional[int] = None, **kwargs):
        val = raw_value if raw_value is not None else int(round((temp_c if temp_c is not None else 88.0) * 100))
        super().__init__(TelemetryEventCode.SILICON_THERMAL_THROTTLE, raw_value=val, **kwargs)

    @property
    def temperature_c(self) -> float:
        return self.raw_value / 100.0

    @property
    def formatted_measurement(self) -> str:
        return f"{self.temperature_c:.2f}°C (THROTTLE)"


class UmaContentionLowEvent(BaseTelemetryEvent):
    def __init__(self, latency_us: Optional[float] = None, raw_value: Optional[int] = None, **kwargs):
        val = raw_value if raw_value is not None else int(round((latency_us if latency_us is not None else 1.2) * 100))
        super().__init__(TelemetryEventCode.UMA_CONTENTION_LOW, raw_value=val, **kwargs)

    @property
    def latency_us(self) -> float:
        return self.raw_value / 100.0

    @property
    def formatted_measurement(self) -> str:
        return f"{self.latency_us:.2f}µs"


class UmaContentionHighEvent(BaseTelemetryEvent):
    def __init__(self, latency_us: Optional[float] = None, raw_value: Optional[int] = None, **kwargs):
        val = raw_value if raw_value is not None else int(round((latency_us if latency_us is not None else 6.8) * 100))
        super().__init__(TelemetryEventCode.UMA_CONTENTION_HIGH, raw_value=val, **kwargs)

    @property
    def latency_us(self) -> float:
        return self.raw_value / 100.0

    @property
    def formatted_measurement(self) -> str:
        return f"{self.latency_us:.2f}µs (SATURATED)"


class NoiseFloorNominalEvent(BaseTelemetryEvent):
    def __init__(self, return_loss_db: Optional[float] = None, raw_value: Optional[int] = None, **kwargs):
        val = raw_value if raw_value is not None else int(round((return_loss_db if return_loss_db is not None else 28.5) * 10))
        super().__init__(TelemetryEventCode.NOISE_FLOOR_NOMINAL_DB, raw_value=val, **kwargs)

    @property
    def return_loss_db(self) -> float:
        return self.raw_value / 10.0

    @property
    def formatted_measurement(self) -> str:
        return f"{self.return_loss_db:.1f}dB"


class NoiseFloorDegradedEvent(BaseTelemetryEvent):
    def __init__(self, return_loss_db: Optional[float] = None, raw_value: Optional[int] = None, **kwargs):
        val = raw_value if raw_value is not None else int(round((return_loss_db if return_loss_db is not None else 14.2) * 10))
        super().__init__(TelemetryEventCode.NOISE_FLOOR_DEGRADED_DB, raw_value=val, **kwargs)

    @property
    def return_loss_db(self) -> float:
        return self.raw_value / 10.0

    @property
    def formatted_measurement(self) -> str:
        return f"{self.return_loss_db:.1f}dB (DEGRADED)"


class OpenVinoGpuPrefillEvent(BaseTelemetryEvent):
    def __init__(self, prompt_tokens: Optional[int] = None, raw_value: Optional[int] = None, **kwargs):
        val = raw_value if raw_value is not None else (prompt_tokens if prompt_tokens is not None else 32)
        super().__init__(TelemetryEventCode.OPENVINO_GPU_PREFILL, raw_value=val, **kwargs)

    @property
    def prompt_tokens(self) -> int:
        return self.raw_value

    @property
    def formatted_measurement(self) -> str:
        return f"{self.prompt_tokens} tokens"


class OpenVinoCpuDecodeEvent(BaseTelemetryEvent):
    def __init__(self, decode_step: Optional[int] = None, raw_value: Optional[int] = None, **kwargs):
        val = raw_value if raw_value is not None else (decode_step if decode_step is not None else 1)
        super().__init__(TelemetryEventCode.OPENVINO_CPU_DECODE, raw_value=val, **kwargs)

    @property
    def decode_step(self) -> int:
        return self.raw_value

    @property
    def formatted_measurement(self) -> str:
        return f"Step {self.decode_step}"


class HandoffSyncWaitEvent(BaseTelemetryEvent):
    def __init__(self, wait_us: Optional[float] = None, raw_value: Optional[int] = None, **kwargs):
        val = raw_value if raw_value is not None else int(round((wait_us if wait_us is not None else 2.4) * 100))
        super().__init__(TelemetryEventCode.HANDOFF_SYNC_WAIT, raw_value=val, **kwargs)

    @property
    def wait_us(self) -> float:
        return self.raw_value / 100.0

    @property
    def formatted_measurement(self) -> str:
        return f"{self.wait_us:.2f}µs"


class HandoffCompleteEvent(BaseTelemetryEvent):
    def __init__(self, handoff_us: Optional[float] = None, raw_value: Optional[int] = None, **kwargs):
        val = raw_value if raw_value is not None else int(round((handoff_us if handoff_us is not None else 11.9) * 100))
        super().__init__(TelemetryEventCode.HANDOFF_COMPLETE, raw_value=val, **kwargs)

    @property
    def handoff_us(self) -> float:
        return self.raw_value / 100.0

    @property
    def formatted_measurement(self) -> str:
        return f"{self.handoff_us:.2f}µs"


class PolicyStrategyShiftEvent(BaseTelemetryEvent):
    def __init__(self, shift_id: int = 1, raw_value: Optional[int] = None, **kwargs):
        val = raw_value if raw_value is not None else shift_id
        super().__init__(TelemetryEventCode.POLICY_STRATEGY_SHIFT, raw_value=val, **kwargs)

    @property
    def formatted_measurement(self) -> str:
        return f"Strategy #{self.raw_value}"


class ConstitutionalHeartbeatEvent(BaseTelemetryEvent):
    def __init__(self, raw_value: Optional[int] = None, **kwargs):
        val = raw_value if raw_value is not None else VITAL_MAX_HP
        super().__init__(TelemetryEventCode.CONSTITUTIONAL_HEARTBEAT, raw_value=val, **kwargs)

    @property
    def formatted_measurement(self) -> str:
        return "VITAL_HP=6"


EVENT_CLASS_MAP = {
    TelemetryEventCode.SILICON_NOMINAL_ACTIVE: SiliconNominalActiveEvent,
    TelemetryEventCode.SILICON_THERMAL_ELEVATED: SiliconThermalElevatedEvent,
    TelemetryEventCode.SILICON_THERMAL_THROTTLE: SiliconThermalThrottleEvent,
    TelemetryEventCode.UMA_CONTENTION_LOW: UmaContentionLowEvent,
    TelemetryEventCode.UMA_CONTENTION_HIGH: UmaContentionHighEvent,
    TelemetryEventCode.NOISE_FLOOR_NOMINAL_DB: NoiseFloorNominalEvent,
    TelemetryEventCode.NOISE_FLOOR_DEGRADED_DB: NoiseFloorDegradedEvent,
    TelemetryEventCode.OPENVINO_GPU_PREFILL: OpenVinoGpuPrefillEvent,
    TelemetryEventCode.OPENVINO_CPU_DECODE: OpenVinoCpuDecodeEvent,
    TelemetryEventCode.HANDOFF_SYNC_WAIT: HandoffSyncWaitEvent,
    TelemetryEventCode.HANDOFF_COMPLETE: HandoffCompleteEvent,
    TelemetryEventCode.POLICY_STRATEGY_SHIFT: PolicyStrategyShiftEvent,
    TelemetryEventCode.CONSTITUTIONAL_HEARTBEAT: ConstitutionalHeartbeatEvent,
}


# ==============================================================================
# 3. PULZOVÝ SEKVENČNÝ ZÁZNAMNÍK (PULSE SEQUENCE LOGGER)
# ==============================================================================

class PulseSequenceLogger:
    """
    Vysoko-priepustný kruhový záznamník pulzných rámcov.
    Uchováva telemetriu v pevnom 8-bajtovom formáte bez pamäťových alokácií.
    """
    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        # Predalokovaný bajtový buffer pre capacity * 8 bajtov
        self._buffer = bytearray(capacity * 8)
        self._write_idx = 0
        self._total_events = 0

    def log(self, event: BaseTelemetryEvent) -> bytes:
        """Zapíše telemetrickú udalosť do kruhovej pamäte."""
        frame = event.pack()
        slot = (self._write_idx % self.capacity) * 8
        self._buffer[slot:slot + 8] = frame
        self._write_idx += 1
        self._total_events += 1
        return frame

    def read_all(self) -> List[BaseTelemetryEvent]:
        """Prečíta všetky aktuálne platné udalosti z kruhového buffera v chronologickom poradí."""
        count = min(self._total_events, self.capacity)
        start_idx = max(0, self._write_idx - count)
        events = []
        for i in range(start_idx, self._write_idx):
            slot = (i % self.capacity) * 8
            raw_frame = bytes(self._buffer[slot:slot + 8])
            events.append(BaseTelemetryEvent.unpack(raw_frame))
        return events

    def to_pulse_diagram(self, max_items: int = 32) -> str:
        """Vygeneruje kompaktný textový sekvenčný diagram pulzov (Morse/Timeline)."""
        events = self.read_all()[-max_items:]
        if not events:
            return "[PULSE_LOGGER_EMPTY]"

        lines = ["[PULSE_SEQUENCE_TIMELINE]"]
        for ev in events:
            # Vizuálny pulzný marker podľa kategórie udalosti
            if ev.code in (TelemetryEventCode.SILICON_THERMAL_THROTTLE, TelemetryEventCode.UMA_CONTENTION_HIGH):
                pulse = "▲ [HIGH_STRESS]"
            elif ev.code in (TelemetryEventCode.OPENVINO_GPU_PREFILL, TelemetryEventCode.OPENVINO_CPU_DECODE):
                pulse = "◆ [COMPUTE]"
            elif ev.code in (TelemetryEventCode.HANDOFF_SYNC_WAIT, TelemetryEventCode.HANDOFF_COMPLETE):
                pulse = "● [HANDOFF]"
            else:
                pulse = "▪ [NOMINAL]"

            lines.append(f"  +{ev.timestamp_us:08d}µs | 0x{ev.code:02X} {ev.code.name:<25} | {ev.formatted_measurement:<18} | {pulse}")
        return "\n".join(lines)

    def clear(self):
        """Vyprázdni logger."""
        self._write_idx = 0
        self._total_events = 0


# ==============================================================================
# 4. OPENVINO DYNAMIC SILICON POLICY ARBITER
# ==============================================================================

class OpenVinoSiliconPolicyArbiter:
    """
    Arbitruje vykonávacie stratégie (GPU_ONLY, HYBRID_PREFILL_DECODE, CPU_ONLY)
    na základe telemetrických dát kremíka, napäťového šumu v dB a UMA latencie.
    """
    def __init__(self, logger: Optional[PulseSequenceLogger] = None):
        self.logger = logger or PulseSequenceLogger(capacity=2048)
        self.current_policy = "HYBRID_PREFILL_DECODE"
        self.p_star_breakpoint = 3  # Baseline derived in E-06A

    def evaluate_state(
        self,
        temperature_c: float,
        return_loss_db: float,
        cache_transit_latency_us: float,
        prompt_tokens: int,
        generation_tokens: int = 1
    ) -> Dict[str, Any]:
        """
        Vyhodnotí prevádzkový stav kremíka a vydá arbitrovanú politiku.
        """
        # 1. Zaznamenanie termálnej telemetrie
        if temperature_c > 82.0:
            t_ev = SiliconThermalThrottleEvent(temp_c=temperature_c)
            is_throttled = True
        elif temperature_c > 65.0:
            t_ev = SiliconThermalElevatedEvent(temp_c=temperature_c)
            is_throttled = False
        else:
            t_ev = SiliconNominalActiveEvent(temp_c=temperature_c)
            is_throttled = False
        self.logger.log(t_ev)

        # 2. Zaznamenanie UMA zbernicovej impedancie
        if cache_transit_latency_us >= 4.5:
            uma_ev = UmaContentionHighEvent(latency_us=cache_transit_latency_us)
            high_contention = True
        else:
            uma_ev = UmaContentionLowEvent(latency_us=cache_transit_latency_us)
            high_contention = False
        self.logger.log(uma_ev)

        # 3. Zaznamenanie šumového prahu / stability napájania
        if return_loss_db < 18.0:
            noise_ev = NoiseFloorDegradedEvent(return_loss_db=return_loss_db)
            degraded_noise = True
        else:
            noise_ev = NoiseFloorNominalEvent(return_loss_db=return_loss_db)
            degraded_noise = False
        self.logger.log(noise_ev)

        # 4. Arbitračné pravidlá (Policy Arbitration)
        previous_policy = self.current_policy

        if is_throttled:
            # CPU je priškrtené termálnym limitom -> GPU_ONLY
            selected_policy = "GPU_ONLY"
            reason = "SILICON_THERMAL_THROTTLE (Tj > 82°C)"
        elif high_contention and prompt_tokens >= 64:
            # UMA ring bus je saturovaný a prompt je dlhý (E-06A Inversion Guard) -> GPU_ONLY
            selected_policy = "GPU_ONLY"
            reason = "UMA_RING_CONTENTION_SATURATION (tau >= 4.5µs, P >= 64)"
        elif degraded_noise:
            # Vysoké zvlnenie / šum -> konzervatívny jednodoménový režim
            selected_policy = "DAMPENED_BATCH_POLICY"
            reason = "NOISE_FLOOR_DEGRADED (RL < 18.0 dB)"
        elif prompt_tokens < self.p_star_breakpoint:
            # Krátky prompt pod P* -> čisté GPU_ONLY
            selected_policy = "GPU_ONLY"
            reason = f"PROMPT_BELOW_P_STAR (P={prompt_tokens} < P*={self.p_star_breakpoint})"
        else:
            # Nominálny optimálny hybridný režim
            selected_policy = "HYBRID_PREFILL_DECODE"
            reason = "OPTIMAL_PHASE_SPLIT (GPU Prefill -> CPU Decode)"

        if selected_policy != previous_policy:
            self.current_policy = selected_policy
            shift_ev = PolicyStrategyShiftEvent(shift_id=0x01)
            self.logger.log(shift_ev)

        # Periodický konštitučný heartbeat
        self.logger.log(ConstitutionalHeartbeatEvent())

        return {
            "selected_policy": selected_policy,
            "previous_policy": previous_policy,
            "policy_shifted": selected_policy != previous_policy,
            "arbitration_reason": reason,
            "telemetry_snapshot": {
                "temperature_c": temperature_c,
                "is_throttled": is_throttled,
                "return_loss_db": return_loss_db,
                "degraded_noise": degraded_noise,
                "cache_transit_latency_us": cache_transit_latency_us,
                "high_contention": high_contention,
                "prompt_tokens": prompt_tokens,
                "generation_tokens": generation_tokens
            },
            "vital_max_hp": VITAL_MAX_HP
        }


# Globálna inštancia
GLOBAL_SILICON_LOGGER = PulseSequenceLogger(capacity=4096)
GLOBAL_SILICON_ARBITER = OpenVinoSiliconPolicyArbiter(logger=GLOBAL_SILICON_LOGGER)
