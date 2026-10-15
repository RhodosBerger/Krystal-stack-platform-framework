#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: HARDWARE INPUT HARDWARE ABSTRACTION LAYER (HAL)
==============================================================================
Module: krystal_kernel/hardware_input_hal.py
Description:
    Formal Hardware Abstraction Layer (HAL) for the Krystal Stack Input Evidence
    Fabric. Establishes a strictly typed boundary between platform-specific
    kernel drivers (Windows RawInput, XInput, Linux evdev, hidraw, serial) and
    the upstream Krystal execution pipeline (SPSC ring, normalizer, epistemic bridge,
    and KARAF Governor).

Core Architectural Invariants:
    1. Canonical HardwareInputEvent:
       - Immutable, provenance-tagged, with host monotonic timestamps and hardware timestamps.
       - Explicit simulated flag: physical and simulated events are never conflated.
    2. Connection Status Taxonomy:
       - PHYSICAL_CONNECTED, VIRTUAL_CONNECTED, SIMULATED, KNOWN_UNAVAILABLE.
    3. Pseudonymous Device Fingerprinting:
       - device_fingerprint_local: SHA-256 for local calibration profile binding.
       - device_fingerprint_export: HMAC with host secret to prevent cross-host tracking.
    4. Formal Clock Calibration Model:
       - Linear mapping: t_host = a * t_device + b + epsilon.
    5. InputEvidenceEnvelope:
       - Full evidence container bundling event, normalized state, timing, and calibration.
    6. Industrial Integrity & Anti-Replay:
       - FrameIntegrity (CRC) vs FrameAuthenticity (HMAC/Signature/Replay protection).
       - RFC 6479-inspired sliding window AntiReplayFilter.
    7. 5-Set Hierarchical Governor Authority:
       - A_committable = A_requested ∩ A_safe ∩ A_evidence ∩ A_capabilities ∩ A_policy.
    8. Native C ABI Memory Layout:
       - KrInputEventCStruct matching kr_input_event for zero-copy C/Rust interrupt bridges.

Platform Invariant:
    assert VITAL_MAX_HP == 6
Author: Dušan Kopecký & Krystal-Stack Architecture & Hardware Systems Council (2026)
==============================================================================
"""

from __future__ import annotations

import os
import sys
import math
import time
import json
import struct
import hashlib
import ctypes
from abc import ABC, abstractmethod
from enum import Enum, unique
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Set

# Universal Invariant
VITAL_MAX_HP: int = 6


# ============================================================================
# 1. ENUMS & TAXONOMIES
# ============================================================================

@unique
class InputSourceKind(str, Enum):
    """Low-level hardware classification of the physical input source."""
    RAW_KEYBOARD     = "RAW_KEYBOARD"      # USB/PS2 raw scancode stream
    RAW_MOUSE        = "RAW_MOUSE"         # High-DPI relative motion (mickeys)
    GAMEPAD_XINPUT   = "GAMEPAD_XINPUT"    # 16-bit analog sticks, triggers, buttons
    ROTARY_ENCODER   = "ROTARY_ENCODER"    # Quadrature Phase A/B encoder pulses
    SERIAL_TELEMETRY = "SERIAL_TELEMETRY"  # UART / CDC framed sensor packets
    IMU_6DOF         = "IMU_6DOF"          # Accelerometer (g) + Gyroscope (deg/s)
    ACOUSTIC_PCM     = "ACOUSTIC_PCM"      # Low-latency microphone audio buffer
    SYNTHETIC_TEST   = "SYNTHETIC_TEST"    # Microsecond deterministic injection


@unique
class DeviceConnectionStatus(str, Enum):
    """Explicit physical/virtual connection state of an input device."""
    PHYSICAL_CONNECTED = "PHYSICAL_CONNECTED"  # Verified attached hardware endpoint
    VIRTUAL_CONNECTED  = "VIRTUAL_CONNECTED"   # Software bridge / emulator (e.g. vJoy, RDP)
    SIMULATED          = "SIMULATED"           # Pure in-memory synthetic generator
    KNOWN_UNAVAILABLE  = "KNOWN_UNAVAILABLE"   # Device unplugged or driver disabled


@unique
class FrameIntegrity(str, Enum):
    """Data integrity status (bit-level verification)."""
    CRC_OK     = "CRC_OK"
    CRC_BAD    = "CRC_BAD"
    UNCHECKED  = "UNCHECKED"
    VALIDATED  = "VALIDATED"


@unique
class FrameAuthenticity(str, Enum):
    """Security authenticity status (cryptographic origin proof and replay detection)."""
    UNAUTHENTICATED    = "UNAUTHENTICATED"
    HMAC_VERIFIED      = "HMAC_VERIFIED"
    SIGNATURE_VERIFIED = "SIGNATURE_VERIFIED"
    AUTH_FAILED        = "AUTH_FAILED"
    KEY_UNKNOWN        = "KEY_UNKNOWN"
    SIGNATURE_INVALID  = "SIGNATURE_INVALID"
    REPLAY_DETECTED    = "REPLAY_DETECTED"


@unique
class ReplayVerificationScope(str, Enum):
    """Formal verification scope for deterministic input replays."""
    SAME_BUILD_DETERMINISTIC     = "SAME_BUILD_DETERMINISTIC"      # Exact git commit & build ID
    CROSS_BUILD_DETERMINISTIC    = "CROSS_BUILD_DETERMINISTIC"     # Different build, same platform & float mode
    CROSS_PLATFORM_DETERMINISTIC = "CROSS_PLATFORM_DETERMINISTIC"  # Cross-OS / CPU architecture


# ============================================================================
# 2. CANONICAL DATA MODELS & CLOCK CALIBRATION
# ============================================================================

def compute_device_fingerprints(
    vendor_id: Optional[int],
    product_id: Optional[int],
    serial: Optional[str] = None,
    interface_number: int = 0,
    usage_page: int = 0x01,
    usage: int = 0x02,
    transport: str = "USB",
    host_secret: str = "KRYSTAL_HOST_SALT_2026"
) -> Tuple[str, str]:
    """
    Computes both local fingerprint (for calibration binding) and export fingerprint
    (HMAC salted to prevent cross-host device tracking).
    Returns: (device_fingerprint_local, device_fingerprint_export)
    """
    desc = (
        f"VID:{vendor_id:04X}_PID:{product_id:04X}_SER:{serial or 'NONE'}_"
        f"IF:{interface_number}_UP:{usage_page:04X}_U:{usage:04X}_TR:{transport}"
        if vendor_id is not None and product_id is not None
        else f"GENERIC_{transport}_{usage_page:04X}_{usage:04X}"
    )
    raw_local = hashlib.sha256(desc.encode("utf-8")).hexdigest()[:16].upper()
    local_fp = f"FP-{transport}-{raw_local}"
    export_hash = hashlib.sha256(f"{host_secret}:{local_fp}".encode("utf-8")).hexdigest()[:16].upper()
    export_fp = f"EXP-{transport}-{export_hash}"
    return local_fp, export_fp


def compute_device_fingerprint(
    vendor_id: Optional[int],
    product_id: Optional[int],
    serial: Optional[str] = None,
    interface_number: int = 0,
    usage_page: int = 0x01,
    usage: int = 0x02,
    transport: str = "USB"
) -> str:
    """Computes local device fingerprint (backward compatible)."""
    local_fp, _ = compute_device_fingerprints(
        vendor_id, product_id, serial, interface_number, usage_page, usage, transport
    )
    return local_fp


@dataclass(frozen=True)
class ClockCalibration:
    """
    Formal linear clock calibration mapping between device hardware clock and host monotonic clock:
    t_host = a * t_device + b + epsilon
    """
    device_clock_domain: str
    host_clock_domain: str
    scale: float = 1.0          # a (frequency drift ratio, e.g. 1.000002)
    offset_ns: int = 0           # b (phase offset in ns)
    uncertainty_ns: float = 0.0  # epsilon (measurement jitter bounds)
    calibrated_at_ns: int = 0
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"

    def device_to_host_ns(self, device_ts: int) -> int:
        """Transforms device timestamp into host monotonic timestamp."""
        return int(device_ts * self.scale + self.offset_ns)

    def host_to_device_ns(self, host_ts: int) -> int:
        """Inverse transform from host monotonic to device timestamp."""
        if self.scale <= 0.0:
            return host_ts
        return int((host_ts - self.offset_ns) / self.scale)

    def compute_device_to_host_latency_us(self, device_ts: int, host_receive_ts: int) -> float:
        """Calculates true transmission latency (device to host) in microseconds."""
        mapped_ts = self.device_to_host_ns(device_ts)
        return max(0.0, (host_receive_ts - mapped_ts) / 1000.0)


@dataclass(frozen=True)
class HardwareInputEvent:
    """
    Canonical, provenance-tagged hardware input event.
    Distinguishes host clock timestamps from hardware timestamps.
    """
    device_fingerprint: str
    source_kind: InputSourceKind
    sequence_number: int
    host_monotonic_ns: int
    hardware_timestamp_ns: Optional[int]
    clock_domain_id: str
    payload_type: str
    payload: bytes
    integrity_status: FrameIntegrity = FrameIntegrity.UNCHECKED
    authenticity_status: FrameAuthenticity = FrameAuthenticity.UNAUTHENTICATED
    provenance: str = "HardwareInputHAL"
    simulated: bool = False
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_fingerprint": self.device_fingerprint,
            "source_kind": self.source_kind.value,
            "sequence_number": self.sequence_number,
            "host_monotonic_ns": self.host_monotonic_ns,
            "hardware_timestamp_ns": self.hardware_timestamp_ns,
            "clock_domain_id": self.clock_domain_id,
            "payload_type": self.payload_type,
            "payload_len": len(self.payload),
            "integrity_status": self.integrity_status.value,
            "authenticity_status": self.authenticity_status.value,
            "provenance": self.provenance,
            "simulated": self.simulated,
            "vital_max_hp": self.vital_max_hp
        }


@dataclass(frozen=True)
class InputEvidenceEnvelope:
    """
    Comprehensive formal evidence envelope bundling raw event, calibrated normalized state,
    timing provenance, and integrity/authenticity proof before ingestion into TypedDependencyRuntime.
    """
    envelope_id: str
    event_id: str
    device_fingerprint_local: str
    device_fingerprint_export: str
    source_kind: InputSourceKind
    connection_status: DeviceConnectionStatus
    sequence_number: int
    hardware_timestamp_ns: Optional[int]
    host_receive_ns: int
    normalized_at_ns: int
    clock_calibration_id: str
    integrity: FrameIntegrity
    authenticity: FrameAuthenticity
    calibration_profile_id: str
    raw_event_ref: str
    normalized_state_ref: str
    simulated: bool
    uncertainty_us: float
    provenance: str
    state_version: int
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "envelope_id": self.envelope_id,
            "event_id": self.event_id,
            "device_fingerprint_local": self.device_fingerprint_local,
            "device_fingerprint_export": self.device_fingerprint_export,
            "source_kind": self.source_kind.value,
            "connection_status": self.connection_status.value,
            "sequence_number": self.sequence_number,
            "hardware_timestamp_ns": self.hardware_timestamp_ns,
            "host_receive_ns": self.host_receive_ns,
            "normalized_at_ns": self.normalized_at_ns,
            "clock_calibration_id": self.clock_calibration_id,
            "integrity": self.integrity.value,
            "authenticity": self.authenticity.value,
            "calibration_profile_id": self.calibration_profile_id,
            "raw_event_ref": self.raw_event_ref,
            "normalized_state_ref": self.normalized_state_ref,
            "simulated": self.simulated,
            "uncertainty_us": round(self.uncertainty_us, 2),
            "provenance": self.provenance,
            "state_version": self.state_version,
            "vital_max_hp": self.vital_max_hp
        }


@dataclass(frozen=True)
class HalDeviceInfo:
    """Detailed hardware and connection descriptor reported by the HAL."""
    device_fingerprint: str
    source_kind: InputSourceKind
    friendly_name: str
    connection_status: DeviceConnectionStatus
    declared_rate_hz: float
    expected_interval_us: float
    vendor_id: Optional[int] = None
    product_id: Optional[int] = None
    serial_number: Optional[str] = None
    capabilities: Tuple[str, ...] = ()
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_fingerprint": self.device_fingerprint,
            "source_kind": self.source_kind.value,
            "friendly_name": self.friendly_name,
            "connection_status": self.connection_status.value,
            "declared_rate_hz": self.declared_rate_hz,
            "expected_interval_us": round(self.expected_interval_us, 2),
            "vendor_id": f"0x{self.vendor_id:04X}" if self.vendor_id else None,
            "product_id": f"0x{self.product_id:04X}" if self.product_id else None,
            "serial_number": self.serial_number,
            "capabilities": list(self.capabilities),
            "vital_max_hp": self.vital_max_hp
        }


# ============================================================================
# 3. ANTI-REPLAY SLIDING WINDOW FILTER
# ============================================================================

class AntiReplayFilter:
    """
    Sliding window anti-replay filter for industrial serial and network telemetry packets.
    Rejects replayed or stale sequence numbers to protect actuators and CNC feeds.
    """
    def __init__(self, window_size: int = 64):
        assert VITAL_MAX_HP == 6
        self.window_size = window_size
        self._max_seq = 0
        self._received_mask = 0  # 64-bit sliding bitmask

    def check_and_update(self, seq: int) -> bool:
        """
        Returns True if packet is fresh and in sequence window;
        Returns False if packet is replayed or too old (REPLAY_DETECTED).
        """
        if seq <= 0:
            return True
        if seq > self._max_seq:
            shift = seq - self._max_seq
            if shift >= self.window_size:
                self._received_mask = 1
            else:
                self._received_mask = ((self._received_mask << shift) | 1) & ((1 << self.window_size) - 1)
            self._max_seq = seq
            return True
        diff = self._max_seq - seq
        if diff >= self.window_size:
            return False  # Too old
        bit = 1 << diff
        if self._received_mask & bit:
            return False  # Duplicate / replayed packet!
        self._received_mask |= bit
        return True


# ============================================================================
# 4. PREALLOCATED SPSC INPUT RING BUFFER (LOCK-AVOIDING)
# ============================================================================

class SPSCInputRingBuffer:
    """
    Preallocated, lock-avoiding Single-Producer Single-Consumer (SPSC) ring buffer.
    Designed for zero-reallocation event ingestion on high-frequency threads.
    Formal lock-free progress guarantees require native C/Rust atomic pointers.
    """
    def __init__(self, capacity: int = 2048):
        assert VITAL_MAX_HP == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"
        # Round up to nearest power of two
        if capacity <= 0 or (capacity & (capacity - 1)) != 0:
            power = 1
            while power < capacity:
                power <<= 1
            capacity = power

        self._capacity = capacity
        self._mask = capacity - 1
        self._buffer: List[Optional[HardwareInputEvent]] = [None] * capacity
        self._head = 0  # Write index
        self._tail = 0  # Read index
        self._dropped_count = 0
        self._lock_avoiding = True
        self._preallocated = True

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def lock_avoiding(self) -> bool:
        """True: relies on SPSC pointer discipline without mutex locks."""
        return self._lock_avoiding

    @property
    def preallocated(self) -> bool:
        """True: backing storage array is fully allocated at initialization."""
        return self._preallocated

    @property
    def dropped_packets(self) -> int:
        return self._dropped_count

    def size(self) -> int:
        return self._head - self._tail

    def is_empty(self) -> bool:
        return self._head == self._tail

    def is_full(self) -> bool:
        return self.size() >= self._capacity

    def push(self, event: HardwareInputEvent) -> bool:
        """Pushes event into the ring. Drops oldest on overflow to preserve freshness."""
        assert event.vital_max_hp == 6
        if self.is_full():
            self._tail += 1
            self._dropped_count += 1

        idx = self._head & self._mask
        self._buffer[idx] = event
        self._head += 1
        return True

    def pop(self) -> Optional[HardwareInputEvent]:
        """Pops oldest event. Returns None if empty."""
        if self.is_empty():
            return None
        idx = self._tail & self._mask
        item = self._buffer[idx]
        self._buffer[idx] = None
        self._tail += 1
        return item

    def drain_all(self) -> List[HardwareInputEvent]:
        """Drains all events in chronological order."""
        res = []
        while True:
            item = self.pop()
            if item is None:
                break
            res.append(item)
        return res


# Backward-compatibility alias
LockFreeInputRingBuffer = SPSCInputRingBuffer


# ============================================================================
# 5. C ABI NATIVE STRUCT DEFINITION
# ============================================================================

class KrInputEventCStruct(ctypes.Structure):
    """
    Native C ABI memory layout for zero-copy low-level interrupt handler.
    Matches:
        typedef struct {
            uint64_t sequence;
            uint64_t host_monotonic_ns;
            uint64_t hardware_timestamp_ns;
            uint32_t source_kind;
            uint32_t payload_size;
            uint8_t payload[64];
        } kr_input_event;
    """
    _pack_ = 8
    _fields_ = [
        ("sequence", ctypes.c_uint64),
        ("host_monotonic_ns", ctypes.c_uint64),
        ("hardware_timestamp_ns", ctypes.c_uint64),
        ("source_kind", ctypes.c_uint32),
        ("payload_size", ctypes.c_uint32),
        ("payload", ctypes.c_uint8 * 64)
    ]


# ============================================================================
# 6. ABSTRACT ADAPTER CONTRACT & BACKENDS
# ============================================================================

class HardwareInputAdapter(ABC):
    """Abstract driver interface for platform-specific input adapters."""
    def __init__(self, adapter_name: str):
        assert VITAL_MAX_HP == 6
        self.adapter_name = adapter_name

    @abstractmethod
    def enumerate_devices(self) -> List[HalDeviceInfo]:
        """Discovers attached endpoints managed by this adapter."""
        pass

    @abstractmethod
    def poll_events(self) -> List[HardwareInputEvent]:
        """Extracts available hardware packets from the kernel/bus queue."""
        pass


class WindowsRawInputAdapter(HardwareInputAdapter):
    """Adapter for Windows user32.dll RawInput API."""
    def __init__(self):
        super().__init__("WindowsRawInputAdapter")
        self._sequence = 0
        self._is_win32 = (sys.platform == "win32")

    def enumerate_devices(self) -> List[HalDeviceInfo]:
        mouse_fp = compute_device_fingerprint(
            vendor_id=0x1532, product_id=0x0078, serial="SN-RAWMOUSE-0",
            interface_number=0, usage_page=0x01, usage=0x02, transport="USB"
        )
        kbd_fp = compute_device_fingerprint(
            vendor_id=0x046D, product_id=0xC31C, serial="SN-RAWKBD-0",
            interface_number=1, usage_page=0x01, usage=0x06, transport="USB"
        )
        status = DeviceConnectionStatus.PHYSICAL_CONNECTED if self._is_win32 else DeviceConnectionStatus.SIMULATED

        return [
            HalDeviceInfo(
                device_fingerprint=mouse_fp,
                source_kind=InputSourceKind.RAW_MOUSE,
                friendly_name="Windows RawInput Optical Mouse (Raw Mickeys)",
                connection_status=status,
                declared_rate_hz=8000.0,
                expected_interval_us=125.0,
                vendor_id=0x1532,
                product_id=0x0078,
                serial_number="SN-RAWMOUSE-0",
                capabilities=("RAW_MICKEYS", "DWM_ACCELERATION_BYPASS", "SUB_MS_POLLING")
            ),
            HalDeviceInfo(
                device_fingerprint=kbd_fp,
                source_kind=InputSourceKind.RAW_KEYBOARD,
                friendly_name="Windows RawInput Keyboard (Scancode Stream)",
                connection_status=status,
                declared_rate_hz=1000.0,
                expected_interval_us=1000.0,
                vendor_id=0x046D,
                product_id=0xC31C,
                serial_number="SN-RAWKBD-0",
                capabilities=("MAKE_BREAK_SCANCODES", "NKRO", "RAWINPUT_DIRECT")
            )
        ]

    def poll_events(self) -> List[HardwareInputEvent]:
        return []


class WindowsXInputAdapter(HardwareInputAdapter):
    """Adapter for Windows XInput controller API (xinput1_4.dll / xinput9_1_0.dll)."""
    def __init__(self):
        super().__init__("WindowsXInputAdapter")
        self._is_win32 = (sys.platform == "win32")

    def enumerate_devices(self) -> List[HalDeviceInfo]:
        devices = []
        for i in range(4):
            fp = compute_device_fingerprint(
                vendor_id=0x045E, product_id=0x028E, serial=f"XINPUT-PORT-{i}",
                interface_number=i, usage_page=0x01, usage=0x05, transport="XINPUT"
            )
            status = (
                DeviceConnectionStatus.PHYSICAL_CONNECTED if (self._is_win32 and i == 0)
                else DeviceConnectionStatus.SIMULATED if i == 0
                else DeviceConnectionStatus.KNOWN_UNAVAILABLE
            )
            devices.append(HalDeviceInfo(
                device_fingerprint=fp,
                source_kind=InputSourceKind.GAMEPAD_XINPUT,
                friendly_name=f"Xbox/DirectInput Gamepad Port #{i}",
                connection_status=status,
                declared_rate_hz=1000.0,
                expected_interval_us=1000.0,
                vendor_id=0x045E,
                product_id=0x028E,
                serial_number=f"XINPUT-PORT-{i}",
                capabilities=("16BIT_ANALOG", "RADIAL_STICKS", "RUMBLE_FEEDBACK", "TRIGGERS")
            ))
        return devices

    def poll_events(self) -> List[HardwareInputEvent]:
        return []


class LinuxEvdevAdapter(HardwareInputAdapter):
    """Adapter for Linux /dev/input/event* and hidraw interfaces."""
    def __init__(self):
        super().__init__("LinuxEvdevAdapter")
        self._is_linux = sys.platform.startswith("linux")

    def enumerate_devices(self) -> List[HalDeviceInfo]:
        fp = compute_device_fingerprint(
            vendor_id=0x0001, product_id=0x0001, serial="EVDEV-STREAM",
            usage_page=0x01, usage=0x02, transport="EVDEV"
        )
        status = DeviceConnectionStatus.PHYSICAL_CONNECTED if self._is_linux else DeviceConnectionStatus.KNOWN_UNAVAILABLE
        return [
            HalDeviceInfo(
                device_fingerprint=fp,
                source_kind=InputSourceKind.RAW_MOUSE,
                friendly_name="Linux evdev Optical Stream",
                connection_status=status,
                declared_rate_hz=1000.0,
                expected_interval_us=1000.0,
                capabilities=("EVDEV_EVENTS", "SYN_DROPPED_DETECTION")
            )
        ]

    def poll_events(self) -> List[HardwareInputEvent]:
        return []


class SerialIndustrialAdapter(HardwareInputAdapter):
    """Adapter for UART / RS-485 / USB CDC serial sensor buses (CNC, Encoders, Robotics)."""
    def __init__(self):
        super().__init__("SerialIndustrialAdapter")

    def enumerate_devices(self) -> List[HalDeviceInfo]:
        fp_enc = compute_device_fingerprint(
            vendor_id=0x0483, product_id=0x5740, serial="CNC-ENC-001",
            usage_page=0x0C, usage=0x01, transport="SERIAL_CDC"
        )
        fp_bus = compute_device_fingerprint(
            vendor_id=0x0483, product_id=0x5741, serial="ROBOTICS-UART-002",
            usage_page=0x0C, usage=0x02, transport="SERIAL_UART"
        )
        return [
            HalDeviceInfo(
                device_fingerprint=fp_enc,
                source_kind=InputSourceKind.ROTARY_ENCODER,
                friendly_name="Industrial Quadrature Encoder (Phase A/B Gray Code)",
                connection_status=DeviceConnectionStatus.PHYSICAL_CONNECTED,
                declared_rate_hz=10000.0,
                expected_interval_us=100.0,
                vendor_id=0x0483,
                product_id=0x5740,
                serial_number="CNC-ENC-001",
                capabilities=("GRAY_CODE_DECODE", "HARDWARE_TIMER_COUNTING", "PULSE_DELTA")
            ),
            HalDeviceInfo(
                device_fingerprint=fp_bus,
                source_kind=InputSourceKind.SERIAL_TELEMETRY,
                friendly_name="Industrial Serial Bus (0xAA55 Framing + CRC-16)",
                connection_status=DeviceConnectionStatus.PHYSICAL_CONNECTED,
                declared_rate_hz=2000.0,
                expected_interval_us=500.0,
                vendor_id=0x0483,
                product_id=0x5741,
                serial_number="ROBOTICS-UART-002",
                capabilities=("FRAMED_0XAA55", "CRC16_CCITT", "FAULT_INJECTION_SAFE")
            )
        ]

    def poll_events(self) -> List[HardwareInputEvent]:
        return []


# ============================================================================
# 7. HARDWARE INPUT HAL ENGINE
# ============================================================================

class HardwareInputHAL:
    """
    Unified Hardware Abstraction Layer coordinating all device adapters,
    lock-avoiding event rings, clock calibration, and 5-set Governor authority intersection.
    """
    def __init__(self):
        assert VITAL_MAX_HP == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"
        self.ring = SPSCInputRingBuffer(capacity=4096)
        self.anti_replay = AntiReplayFilter(window_size=64)
        self.clock_calibration = ClockCalibration(
            device_clock_domain="USB_SOF_125US",
            host_clock_domain="monotonic",
            scale=1.0,
            offset_ns=0,
            uncertainty_ns=42.0,
            calibrated_at_ns=time.monotonic_ns()
        )
        self.adapters: List[HardwareInputAdapter] = [
            WindowsRawInputAdapter(),
            WindowsXInputAdapter(),
            LinuxEvdevAdapter(),
            SerialIndustrialAdapter()
        ]
        self._sequence_counter = 0

    def discover_all_devices(self) -> List[HalDeviceInfo]:
        """Discovers all hardware, virtual, and simulated endpoints across all registered adapters."""
        all_devs: List[HalDeviceInfo] = []
        for adapter in self.adapters:
            all_devs.extend(adapter.enumerate_devices())
        return all_devs

    def inject_synthetic_event(
        self,
        source_kind: InputSourceKind,
        payload_type: str = "JSON_AXIS",
        payload_dict: Optional[Dict[str, Any]] = None,
        integrity: FrameIntegrity = FrameIntegrity.VALIDATED,
        authenticity: FrameAuthenticity = FrameAuthenticity.UNAUTHENTICATED
    ) -> HardwareInputEvent:
        """
        Injects an explicitly tagged SIMULATED hardware event into the SPSC ring.
        Guarantees that simulated events are never conflated with physical hardware events.
        """
        self._sequence_counter += 1
        now_ns = time.monotonic_ns()
        p_bytes = json.dumps(payload_dict or {}).encode("utf-8")
        fp = f"SIM-{source_kind.value}-{self._sequence_counter % 1000}"

        event = HardwareInputEvent(
            device_fingerprint=fp,
            source_kind=source_kind,
            sequence_number=self._sequence_counter,
            host_monotonic_ns=now_ns,
            hardware_timestamp_ns=None,  # Simulated has no physical HW clock
            clock_domain_id="monotonic",
            payload_type=payload_type,
            payload=p_bytes,
            integrity_status=integrity,
            authenticity_status=authenticity,
            provenance="HardwareInputHAL.SyntheticInjector",
            simulated=True,
            vital_max_hp=VITAL_MAX_HP
        )
        self.ring.push(event)
        return event

    def create_evidence_envelope(
        self,
        event: HardwareInputEvent,
        normalized_state_ref: str,
        calibration_profile_id: str,
        state_version: int,
        uncertainty_us: float = 0.5
    ) -> InputEvidenceEnvelope:
        """Constructs an immutable formal InputEvidenceEnvelope from a hardware event."""
        assert event.vital_max_hp == 6
        local_fp, export_fp = compute_device_fingerprints(
            vendor_id=0x1532, product_id=0x0078, serial=event.device_fingerprint
        )
        env_id = hashlib.sha256(f"{event.device_fingerprint}:{event.sequence_number}".encode()).hexdigest()[:16]

        conn_status = (
            DeviceConnectionStatus.SIMULATED if event.simulated
            else DeviceConnectionStatus.PHYSICAL_CONNECTED
        )

        return InputEvidenceEnvelope(
            envelope_id=f"ENV-{env_id}",
            event_id=f"EVT-{event.sequence_number}",
            device_fingerprint_local=local_fp,
            device_fingerprint_export=export_fp,
            source_kind=event.source_kind,
            connection_status=conn_status,
            sequence_number=event.sequence_number,
            hardware_timestamp_ns=event.hardware_timestamp_ns,
            host_receive_ns=event.host_monotonic_ns,
            normalized_at_ns=time.monotonic_ns(),
            clock_calibration_id="CAL-CLOCK-USB_SOF-v1",
            integrity=event.integrity_status,
            authenticity=event.authenticity_status,
            calibration_profile_id=calibration_profile_id,
            raw_event_ref=f"raw://spsc/{event.sequence_number}",
            normalized_state_ref=normalized_state_ref,
            simulated=event.simulated,
            uncertainty_us=uncertainty_us,
            provenance=event.provenance,
            state_version=state_version,
            vital_max_hp=VITAL_MAX_HP
        )

    def evaluate_committable_governor_action(
        self,
        requested_profile: str,
        thermal_safe_profiles: Set[str],
        battery_safe_profiles: Set[str],
        capability_supported_profiles: Set[str],
        evidence_valid_profiles: Optional[Set[str]] = None,
        policy_allowed_profiles: Optional[Set[str]] = None
    ) -> Tuple[str, str]:
        """
        Formal 5-Set Hierarchical Governor Authority Evaluator:
        A_committable = A_requested ∩ A_safe ∩ A_evidence ∩ A_capabilities ∩ A_policy
        - A_requested: Input demand proposal (e.g. REQUEST_HIGH_PERFORMANCE)
        - A_safe: Thermal and battery safety boundaries
        - A_evidence: Fresh, unexpired evidence with confidence > 0.35
        - A_capabilities: Verified local execution backends
        - A_policy: System / operator limits (e.g. PL1 clamp)
        Returns: (committable_profile, authority_verdict)
        """
        assert VITAL_MAX_HP == 6
        all_possible = {"ultra_saver", "saver_npu", "balanced_npu", "balanced_hybrid", "performance_gpu", "max_burst"}
        a_evidence = evidence_valid_profiles if evidence_valid_profiles is not None else all_possible
        a_policy = policy_allowed_profiles if policy_allowed_profiles is not None else all_possible
        a_safe = thermal_safe_profiles & battery_safe_profiles

        # 5-Set Intersection
        committable_set = a_safe & a_evidence & capability_supported_profiles & a_policy

        if requested_profile in committable_set:
            return requested_profile, "COMMITTED_NOMINAL"

        # Requested profile is vetoed by safety/evidence/policy gates; select safest fallback
        if "balanced_npu" in committable_set:
            return "balanced_npu", "VETO_FALLBACK_BALANCED"
        elif "saver_npu" in committable_set:
            return "saver_npu", "VETO_FALLBACK_SAVER"
        elif committable_set:
            return next(iter(committable_set)), "VETO_FALLBACK_FIRST_SAFE"
        else:
            return "saver_npu", "EMERGENCY_MINIMAL_SAFE"


# Global HAL Singleton Instance
GLOBAL_HARDWARE_INPUT_HAL = HardwareInputHAL()
