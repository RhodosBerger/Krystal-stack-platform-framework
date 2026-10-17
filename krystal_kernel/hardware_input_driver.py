#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: HARDWARE INPUT DEVICE DRIVER & TELEMETRY SOURCE SUBSYSTEM
==============================================================================
Module: krystal_kernel/hardware_input_driver.py
Description:
    Production-grade low-level hardware input device driver and physical telemetry
    source framework for Krystal Stack and KARAF (Krystal Adaptive Runtime Algorithm Fabric).

Hardware-Level Device Sources:
    1. Windows RawInput API (user32.dll):
       - Accurate capture path:
         Physical Device -> USB/xHCI -> Windows Kernel USB+HID Stack -> Raw Input Subsystem
         -> WM_INPUT / GetRawInputData -> Krystal Input Adapter -> SPSC Ring.
       - Bypasses high-level pointer acceleration curves, ballistics, and GUI message coalescing.
       - (Note: DWM is part of the render/presentation/scanout path, not the input capture pipeline.)
    2. Windows XInput API (xinput1_4.dll / xinput9_1_0.dll):
       - Direct polling of 16-bit analog thumbsticks, analog trigger potentiometers/Hall sensors,
         and digital button masks with sub-millisecond QPC timestamps.
       - Bidirectional haptic feedback / rumble motor actuation.
    3. Quadrature Rotary / Linear Encoders (Industrial & Robotics Telemetry):
       - 2-channel Phase A / Phase B Gray code decoding.
       - Pulse counting, angular velocity (RPM), cumulative angular position.
    4. Serial Telemetry Stream Driver (UART / USB CDC Virtual COM):
       - Framed binary packet parsing (Header, Length, Payload, CRC-16 CCITT).
       - Explicit separation between FrameIntegrity (CRC) and FrameAuthenticity (HMAC/Signature).
    5. Acoustic / Voice-to-Action PCM Telemetry:
       - Low-latency PCM audio ring buffer monitoring (RMS amplitude, VAD activity).

Decomposed End-to-End Latency Model:
    T_input-to-photon = T_input + T_simulation + T_render + T_display
      - T_input:      device -> kernel -> RawInput -> Krystal SPSC Ring
      - T_simulation: Krystal input dequeue -> state update & logic
      - T_render:     state -> GPU dispatch -> swapchain present
      - T_display:    present -> DirectFlip/scanout -> physical display photons

Key Architectural Capabilities:
    - SPSCInputRingBuffer: Preallocated, lock-avoiding single-producer single-consumer ring.
    - Deadzone & Anti-Jitter Filtering: Radial deadzone for thumbsticks, EWMA smoothing.
    - Deterministic Input Frame Journaling: Produces canonical hex hashes and
      (StateVersion, TurnVersion, InputSequence, StateHash) tuples for rollback and replays.
    - Epistemic Integration: Bridges input cadence, jitter, and user activity into
      TypedDependencyRuntime (OBS_INPUT_POLL_RATE_HZ, OBS_INPUT_JITTER_US, OBS_INPUT_ACTIVITY).
    - Hierarchical Governor Contract:
      Input activity score is strictly an advisory request (REQUEST_HIGH_PERFORMANCE).
      Safety gates (thermal veto, battery critical, capability unavailable) have absolute veto:
      A_committable = A_input-request ∩ A_thermal-safe ∩ A_battery-safe ∩ A_capability-supported.

Constitutional Invariant:
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
from ctypes import wintypes
from enum import Enum, unique
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Callable, Set

# Platform Invariant
VITAL_MAX_HP: int = 6


# ============================================================================
# 1. ENUMS & HARDWARE SOURCE TAXONOMIES
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
class UserActivityState(str, Enum):
    """Derived user interaction state driving Governor power-state transitions."""
    IDLE           = "IDLE"           # No input activity (> 10s), eligible for saver_npu
    LOW_INTENSITY  = "LOW_INTENSITY"  # Sporadic interaction, balanced_npu
    ACTIVE         = "ACTIVE"         # Continuous navigation, balanced_hybrid
    TACTICAL_BURST = "TACTICAL_BURST" # High-frequency twitch reflexes, performance_gpu


# ============================================================================
# 2. C-ABI WINDOWS API STRUCTS & TYPEDEFS
# ============================================================================

# RawInput Constants (winuser.h)
RIDEV_INPUTSINK       = 0x00000100
RIDEV_DEVNOTIFY       = 0x00002000
RID_INPUT             = 0x10000003
RIM_TYPEMOUSE         = 0
RIM_TYPEKEYBOARD      = 1
RIM_TYPEHID           = 2

# XInput Constants (xinput.h)
XINPUT_GAMEPAD_DPAD_UP        = 0x0001
XINPUT_GAMEPAD_DPAD_DOWN      = 0x0002
XINPUT_GAMEPAD_DPAD_LEFT      = 0x0004
XINPUT_GAMEPAD_DPAD_RIGHT     = 0x0008
XINPUT_GAMEPAD_START          = 0x0010
XINPUT_GAMEPAD_BACK           = 0x0020
XINPUT_GAMEPAD_LEFT_THUMB     = 0x0040
XINPUT_GAMEPAD_RIGHT_THUMB    = 0x0080
XINPUT_GAMEPAD_LEFT_SHOULDER  = 0x0100
XINPUT_GAMEPAD_RIGHT_SHOULDER = 0x0200
XINPUT_GAMEPAD_A              = 0x1000
XINPUT_GAMEPAD_B              = 0x2000
XINPUT_GAMEPAD_X              = 0x4000
XINPUT_GAMEPAD_Y              = 0x8000

# Thumbstick Deadzones (Standard Microsoft specs)
XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE  = 7849
XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE = 8689
XINPUT_GAMEPAD_TRIGGER_THRESHOLD    = 30


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class RAWMOUSE(ctypes.Structure):
    class _DUMMYUNION(ctypes.Union):
        class _DUMMYSTRUCT(ctypes.Structure):
            _fields_ = [
                ("usButtonFlags", wintypes.USHORT),
                ("usButtonData", wintypes.USHORT),
            ]
        _fields_ = [
            ("ulButtons", wintypes.ULONG),
            ("_struct", _DUMMYSTRUCT),
        ]
        _anonymous_ = ("_struct",)

    _fields_ = [
        ("usFlags", wintypes.USHORT),
        ("_union", _DUMMYUNION),
        ("ulRawButtons", wintypes.ULONG),
        ("lLastX", wintypes.LONG),
        ("lLastY", wintypes.LONG),
        ("ulExtraInformation", wintypes.ULONG),
    ]
    _anonymous_ = ("_union",)


class RAWKEYBOARD(ctypes.Structure):
    _fields_ = [
        ("MakeCode", wintypes.USHORT),
        ("Flags", wintypes.USHORT),
        ("Reserved", wintypes.USHORT),
        ("VKey", wintypes.USHORT),
        ("Message", wintypes.UINT),
        ("ExtraInformation", wintypes.ULONG),
    ]


class RAWHID(ctypes.Structure):
    _fields_ = [
        ("dwSizeHid", wintypes.DWORD),
        ("dwCount", wintypes.DWORD),
        ("bRawData", wintypes.BYTE * 1),
    ]


class RAWINPUT(ctypes.Structure):
    class _DUMMYUNION(ctypes.Union):
        _fields_ = [
            ("mouse", RAWMOUSE),
            ("keyboard", RAWKEYBOARD),
            ("hid", RAWHID),
        ]
    _fields_ = [
        ("header", RAWINPUTHEADER),
        ("data", _DUMMYUNION),
    ]


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]


class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ("wButtons", wintypes.WORD),
        ("bLeftTrigger", wintypes.BYTE),
        ("bRightTrigger", wintypes.BYTE),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]


class XINPUT_STATE(ctypes.Structure):
    _fields_ = [
        ("dwPacketNumber", wintypes.DWORD),
        ("Gamepad", XINPUT_GAMEPAD),
    ]


class XINPUT_VIBRATION(ctypes.Structure):
    _fields_ = [
        ("wLeftMotorSpeed", wintypes.WORD),
        ("wRightMotorSpeed", wintypes.WORD),
    ]


# ============================================================================
# 3. DATA MODELS
# ============================================================================

@dataclass(frozen=True)
class RawHardwarePacket:
    """Exact, unparsed byte-level packet captured from the physical bus."""
    device_id: str
    source_kind: InputSourceKind
    hw_timestamp_ns: int
    sequence_id: int
    raw_payload: bytes
    flags: int = 0
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"


@dataclass
class NormalizedInputState:
    """
    Parsed, normalized, and jitter-filtered input state representation.
    Axes are strictly mapped to [-1.0, 1.0] (or [0.0, 1.0] for triggers).
    """
    device_id: str
    source_kind: InputSourceKind
    timestamp_ns: int
    sequence_id: int
    axes: Dict[str, float] = field(default_factory=dict)
    buttons: int = 0
    relative_motion: Tuple[int, int] = (0, 0)
    deadzone_applied: bool = False
    jitter_us: float = 0.0
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"

    def is_button_pressed(self, button_mask: int) -> bool:
        return bool(self.buttons & button_mask)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "source_kind": self.source_kind.value,
            "timestamp_ns": self.timestamp_ns,
            "sequence_id": self.sequence_id,
            "axes": {k: round(v, 4) for k, v in self.axes.items()},
            "buttons": self.buttons,
            "relative_motion": list(self.relative_motion),
            "deadzone_applied": self.deadzone_applied,
            "jitter_us": round(self.jitter_us, 2),
            "vital_max_hp": self.vital_max_hp
        }


@dataclass(frozen=True)
class InputReplayFrame:
    """
    Deterministic replay frame tuple as required by Krystal multiplayer and
    state reconciliation: (StateVersion, TurnVersion, InputSequence, StateHash).
    """
    state_version: int
    turn_version: int
    input_sequence: Tuple[int, ...]
    random_seed: int
    ruleset_version: str
    state_hash: str
    timestamp_ns: int
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_version": self.state_version,
            "turn_version": self.turn_version,
            "input_sequence": list(self.input_sequence),
            "random_seed": self.random_seed,
            "ruleset_version": self.ruleset_version,
            "state_hash": self.state_hash,
            "timestamp_ns": self.timestamp_ns,
            "vital_max_hp": self.vital_max_hp
        }


# ============================================================================
# 4. PREALLOCATED SPSC INPUT RING BUFFER (LOCK-AVOIDING)
# ============================================================================

class SPSCInputRingBuffer:
    """
    Preallocated, lock-avoiding Single-Producer Single-Consumer (SPSC) ring buffer
    for ingesting hardware input packets without GC reallocation in the fast path.
    Formal lock-free progress guarantees require native C/Rust atomic head/tail pointers.
    """
    def __init__(self, capacity: int = 1024):
        assert VITAL_MAX_HP == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"
        # Capacity must be a power of two for bitwise wrap-around
        if capacity <= 0 or (capacity & (capacity - 1)) != 0:
            power = 1
            while power < capacity:
                power <<= 1
            capacity = power

        self._capacity = capacity
        self._mask = capacity - 1
        self._buffer: List[Optional[RawHardwarePacket]] = [None] * capacity
        self._head = 0  # Write index
        self._tail = 0  # Read index
        self._dropped_packets = 0
        self._lock_avoiding = True
        self._preallocated = True

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def lock_avoiding(self) -> bool:
        """True: relies on single-producer/single-consumer pointer discipline."""
        return self._lock_avoiding

    @property
    def preallocated(self) -> bool:
        """True: array buffer is preallocated at construction."""
        return self._preallocated

    @property
    def dropped_packets(self) -> int:
        return self._dropped_packets

    def size(self) -> int:
        return self._head - self._tail

    def is_empty(self) -> bool:
        return self._head == self._tail

    def is_full(self) -> bool:
        return self.size() >= self._capacity

    def push(self, packet: RawHardwarePacket) -> bool:
        """Push packet into buffer. Drops oldest packet if full to preserve freshness."""
        assert packet.vital_max_hp == 6
        if self.is_full():
            # Drop oldest packet to prevent latency lag buildup
            self._tail += 1
            self._dropped_packets += 1

        idx = self._head & self._mask
        self._buffer[idx] = packet
        self._head += 1
        return True

    def pop(self) -> Optional[RawHardwarePacket]:
        """Pop packet from buffer. Returns None if empty."""
        if self.is_empty():
            return None
        idx = self._tail & self._mask
        item = self._buffer[idx]
        self._buffer[idx] = None
        self._tail += 1
        return item

    def drain_all(self) -> List[RawHardwarePacket]:
        """Drain all pending packets in chronological order."""
        packets = []
        while True:
            pkt = self.pop()
            if pkt is None:
                break
            packets.append(pkt)
        return packets


# Backward-compatibility alias
LockFreeInputRingBuffer = SPSCInputRingBuffer


# ============================================================================
# 5. DEADZONE & ANTI-JITTER SIGNAL PROCESSOR
# ============================================================================

class DeadzoneAndJitterFilter:
    """
    Applies mathematically rigorous radial and axial deadzones and EWMA smoothing
    to eliminate electrical noise and potentiometer drift from analog HW inputs.
    """
    def __init__(
        self,
        radial_deadzone: float = 0.15,
        axial_deadzone: float = 0.08,
        smoothing_alpha: float = 0.85
    ):
        assert VITAL_MAX_HP == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"
        self.radial_deadzone = radial_deadzone
        self.axial_deadzone = axial_deadzone
        self.smoothing_alpha = smoothing_alpha
        self._smoothed_axes: Dict[str, float] = {}
        self._last_timestamp_ns = 0
        self._ewma_jitter_us = 0.0

    def filter_radial_stick(self, x: float, y: float) -> Tuple[float, float, bool]:
        """
        Radial deadzone: preserves circular throw and continuous angle direction
        while eliminating center sensor wobble.
        """
        magnitude = math.hypot(x, y)
        if magnitude <= self.radial_deadzone:
            return 0.0, 0.0, True

        # Normalized rescaling from deadzone threshold to 1.0
        normalized_mag = min(1.0, (magnitude - self.radial_deadzone) / (1.0 - self.radial_deadzone))
        scale = normalized_mag / magnitude
        return x * scale, y * scale, False

    def filter_axis(self, axis_name: str, value: float) -> float:
        """Axial deadzone and EWMA low-pass filter."""
        if abs(value) <= self.axial_deadzone:
            clean_val = 0.0
        else:
            sign = 1.0 if value > 0 else -1.0
            clean_val = sign * ((abs(value) - self.axial_deadzone) / (1.0 - self.axial_deadzone))

        # EWMA filter: y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
        prev = self._smoothed_axes.get(axis_name, clean_val)
        filtered = self.smoothing_alpha * clean_val + (1.0 - self.smoothing_alpha) * prev
        self._smoothed_axes[axis_name] = filtered
        return filtered

    def update_jitter(self, current_ns: int, expected_interval_us: float) -> float:
        """Computes hardware inter-arrival jitter in microseconds."""
        if self._last_timestamp_ns == 0:
            self._last_timestamp_ns = current_ns
            return 0.0

        delta_us = (current_ns - self._last_timestamp_ns) / 1000.0
        self._last_timestamp_ns = current_ns
        jitter = abs(delta_us - expected_interval_us)
        self._ewma_jitter_us = 0.9 * self._ewma_jitter_us + 0.1 * jitter
        return self._ewma_jitter_us


# ============================================================================
# 6. CRC-16 CCITT & PACKET DECODER FOR SERIAL TELEMETRY
# ============================================================================

def compute_crc16_ccitt(data: bytes) -> int:
    """Computes CRC-16 CCITT (polynomial 0x1021, init 0xFFFF) for telemetry framing."""
    crc = 0xFFFF
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


class SerialTelemetryDecoder:
    """
    Decodes framed hardware telemetry streams (UART, RS-485, USB CDC)
    from microcontrollers, quadrature encoders, and CNC feed heads.
    Frame: [0xAA, 0x55] (Preamble) + [1B Length] + [1B SourceType] + [Payload] + [2B CRC16]
    """
    PREAMBLE = b"\xAA\x55"

    def __init__(self):
        assert VITAL_MAX_HP == 6
        self._rx_buffer = bytearray()

    def feed_bytes(self, chunk: bytes) -> List[Tuple[int, bytes]]:
        """Ingests raw serial bytes and yields verified (source_type, payload) frames."""
        self._rx_buffer.extend(chunk)
        frames = []

        while len(self._rx_buffer) >= 6:
            # Find preamble
            idx = self._rx_buffer.find(self.PREAMBLE)
            if idx == -1:
                # Retain last byte in case preamble is split across chunks
                self._rx_buffer = self._rx_buffer[-1:]
                break
            if idx > 0:
                del self._rx_buffer[:idx]

            if len(self._rx_buffer) < 6:
                break

            payload_len = self._rx_buffer[2]
            total_frame_len = 2 + 1 + 1 + payload_len + 2  # Preamble(2) + Len(1) + Type(1) + Payload + CRC(2)
            if len(self._rx_buffer) < total_frame_len:
                # Incomplete frame, await more bytes
                break

            frame_data = bytes(self._rx_buffer[:total_frame_len])
            source_type = frame_data[3]
            payload = frame_data[4:4 + payload_len]
            expected_crc = struct.unpack(">H", frame_data[4 + payload_len:total_frame_len])[0]
            computed_crc = compute_crc16_ccitt(frame_data[2:4 + payload_len])

            if expected_crc == computed_crc:
                frames.append((source_type, payload))
                del self._rx_buffer[:total_frame_len]
            else:
                # Corrupted frame, skip preamble to resynchronize
                del self._rx_buffer[:2]

        return frames

    @classmethod
    def encode_frame(cls, source_type: int, payload: bytes) -> bytes:
        """Encodes an outgoing telemetry frame with CRC-16 protection."""
        length = len(payload)
        header_and_body = struct.pack("!BB", length, source_type) + payload
        crc = compute_crc16_ccitt(header_and_body)
        return cls.PREAMBLE + header_and_body + struct.pack(">H", crc)


# ============================================================================
# 7. DETERMINISTIC INPUT JOURNAL (REPLAY & MULTIPLAYER RECONCILIATION)
# ============================================================================

class DeterministicInputJournal:
    """
    Maintains the authoritative input sequence journal:
    (StateVersion, TurnVersion, InputSequence, StateHash).
    Allows client-side prediction, rollback verification, and desync origin analysis.
    """
    def __init__(self, ruleset_version: str = "v1.0.0", max_history: int = 1024):
        assert VITAL_MAX_HP == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"
        self.ruleset_version = ruleset_version
        self.max_history = max_history
        self._journal: Dict[int, InputReplayFrame] = {}  # state_version -> InputReplayFrame
        self._current_state_version = 0

    def record_frame(
        self,
        turn_version: int,
        input_sequence: List[int],
        random_seed: int,
        timestamp_ns: int
    ) -> InputReplayFrame:
        """Records an authoritative input frame and calculates its SHA-256 state hash."""
        self._current_state_version += 1
        seq_tuple = tuple(input_sequence)

        # Canonical deterministic hash payload
        hasher = hashlib.sha256()
        hasher.update(struct.pack("<QQq", self._current_state_version, turn_version, random_seed))
        for val in seq_tuple:
            hasher.update(struct.pack("<q", val))
        hasher.update(self.ruleset_version.encode("utf-8"))
        state_hash = hasher.hexdigest()

        frame = InputReplayFrame(
            state_version=self._current_state_version,
            turn_version=turn_version,
            input_sequence=seq_tuple,
            random_seed=random_seed,
            ruleset_version=self.ruleset_version,
            state_hash=state_hash,
            timestamp_ns=timestamp_ns,
            vital_max_hp=VITAL_MAX_HP
        )

        self._journal[self._current_state_version] = frame
        if len(self._journal) > self.max_history:
            oldest_key = min(self._journal.keys())
            del self._journal[oldest_key]

        return frame

    def verify_prediction(
        self,
        predicted_version: int,
        predicted_hash: str
    ) -> Tuple[bool, Optional[int]]:
        """
        Verifies client prediction against authoritative journal.
        Returns (is_valid, desync_origin_version).
        """
        frame = self._journal.get(predicted_version)
        if frame is None:
            return False, predicted_version
        if frame.state_hash == predicted_hash:
            return True, None
        return False, predicted_version


# ============================================================================
# 8. HARDWARE INPUT DRIVER CORE
# ============================================================================

class HardwareInputDriver:
    """
    Unified Hardware Input Device Driver Engine.
    Handles device enumeration, raw interrupt/polling ingress, packet normalization,
    ring buffering, and epistemic telemetry synthesis.
    """
    def __init__(self, mock_mode: bool = False):
        assert VITAL_MAX_HP == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"
        self.mock_mode = mock_mode or (sys.platform != "win32")
        self.ring_buffer = LockFreeInputRingBuffer(capacity=2048)
        self.filter = DeadzoneAndJitterFilter()
        self.journal = DeterministicInputJournal()
        self.serial_decoder = SerialTelemetryDecoder()

        # Telemetry metrics
        self._sequence_counter = 0
        self._last_event_time_ns = time.monotonic_ns()
        self._active_device_ids: List[str] = []
        self._native_rawinput_active = False
        self._native_xinput_active = False

        # Windows DLL handles
        self._user32 = None
        self._kernel32 = None
        self._xinput = None
        self._qpc_frequency = 10_000_000  # Default 10 MHz fallback

        self._init_native_bindings()

    def _init_native_bindings(self) -> None:
        """Initializes ctypes bindings to Windows user32, kernel32, and xinput."""
        if self.mock_mode:
            self._active_device_ids = ["MOCK_KBD_0", "MOCK_MOU_0", "MOCK_PAD_0"]
            return

        try:
            self._user32 = ctypes.windll.user32
            self._kernel32 = ctypes.windll.kernel32

            # Query QPC frequency
            freq = ctypes.c_int64()
            if self._kernel32.QueryPerformanceFrequency(ctypes.byref(freq)):
                self._qpc_frequency = freq.value

            # Attempt XInput DLL load (1.4 on Win8+, 9.1.0 fallback)
            for dll_name in ("xinput1_4.dll", "xinput9_1_0.dll"):
                try:
                    self._xinput = ctypes.windll.LoadLibrary(dll_name)
                    self._native_xinput_active = True
                    break
                except Exception:
                    continue

            self._native_rawinput_active = (self._user32 is not None)
            self._active_device_ids = ["WIN_RAW_KBD", "WIN_RAW_MOU"]
            if self._native_xinput_active:
                self._active_device_ids.append("WIN_XINPUT_PAD_0")

        except Exception:
            # Fallback to mock mode gracefully
            self.mock_mode = True
            self._active_device_ids = ["FALLBACK_KBD", "FALLBACK_MOU"]

    def read_monotonic_ns(self) -> int:
        """Returns nanoseconds from QPC (on Windows) or time.monotonic_ns()."""
        if not self.mock_mode and self._kernel32 is not None:
            count = ctypes.c_int64()
            if self._kernel32.QueryPerformanceCounter(ctypes.byref(count)):
                return int((count.value * 1_000_000_000) // self._qpc_frequency)
        return time.monotonic_ns()

    def register_raw_input_devices(self, hwnd_target: int = 0) -> bool:
        """
        Registers raw mouse and raw keyboard endpoints via RegisterRawInputDevices.
        Bypasses DWM composition and delivers WM_INPUT directly to target or sink.
        """
        if self.mock_mode or self._user32 is None:
            return True

        try:
            raw_devices = (RAWINPUTDEVICE * 2)()
            # Generic Desktop Keyboard: UsagePage 0x01, Usage 0x06
            raw_devices[0].usUsagePage = 0x01
            raw_devices[0].usUsage = 0x06
            raw_devices[0].dwFlags = RIDEV_INPUTSINK if hwnd_target else 0
            raw_devices[0].hwndTarget = hwnd_target

            # Generic Desktop Mouse: UsagePage 0x01, Usage 0x02
            raw_devices[1].usUsagePage = 0x01
            raw_devices[1].usUsage = 0x02
            raw_devices[1].dwFlags = RIDEV_INPUTSINK if hwnd_target else 0
            raw_devices[1].hwndTarget = hwnd_target

            result = self._user32.RegisterRawInputDevices(
                ctypes.byref(raw_devices),
                2,
                ctypes.sizeof(RAWINPUTDEVICE)
            )
            return bool(result)
        except Exception:
            return False

    def poll_xinput_gamepad(self, user_index: int = 0) -> Optional[NormalizedInputState]:
        """Polls native XInput gamepad state and returns normalized input state."""
        self._sequence_counter += 1
        now_ns = self.read_monotonic_ns()

        if self.mock_mode or not self._native_xinput_active or self._xinput is None:
            # Synthetic gamepad sample for testing/CI
            return NormalizedInputState(
                device_id=f"PAD_{user_index}",
                source_kind=InputSourceKind.GAMEPAD_XINPUT,
                timestamp_ns=now_ns,
                sequence_id=self._sequence_counter,
                axes={"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0, "lt": 0.0, "rt": 0.0},
                buttons=0,
                deadzone_applied=False,
                jitter_us=0.0,
                vital_max_hp=VITAL_MAX_HP
            )

        state = XINPUT_STATE()
        res = self._xinput.XInputGetState(user_index, ctypes.byref(state))
        if res != 0:
            return None  # Controller disconnected

        gp = state.Gamepad
        # Normalize thumbsticks (-32768..32767 -> -1.0..1.0)
        raw_lx = gp.sThumbLX / 32767.0
        raw_ly = gp.sThumbLY / 32767.0
        raw_rx = gp.sThumbRX / 32767.0
        raw_ry = gp.sThumbRY / 32767.0

        # Apply radial deadzone to left stick
        flx, fly, left_dead = self.filter.filter_radial_stick(raw_lx, raw_ly)
        frx, fry, right_dead = self.filter.filter_radial_stick(raw_rx, raw_ry)

        # Normalize triggers (0..255 -> 0.0..1.0)
        lt = gp.bLeftTrigger / 255.0
        rt = gp.bRightTrigger / 255.0
        if gp.bLeftTrigger < XINPUT_GAMEPAD_TRIGGER_THRESHOLD:
            lt = 0.0
        if gp.bRightTrigger < XINPUT_GAMEPAD_TRIGGER_THRESHOLD:
            rt = 0.0

        jitter = self.filter.update_jitter(now_ns, expected_interval_us=1000.0)
        self._last_event_time_ns = now_ns

        norm_state = NormalizedInputState(
            device_id=f"XINPUT_PAD_{user_index}",
            source_kind=InputSourceKind.GAMEPAD_XINPUT,
            timestamp_ns=now_ns,
            sequence_id=self._sequence_counter,
            axes={
                "lx": round(flx, 4),
                "ly": round(fly, 4),
                "rx": round(frx, 4),
                "ry": round(fry, 4),
                "lt": round(lt, 4),
                "rt": round(rt, 4)
            },
            buttons=gp.wButtons,
            deadzone_applied=(left_dead or right_dead),
            jitter_us=jitter,
            vital_max_hp=VITAL_MAX_HP
        )

        # Enqueue into lock-free ring buffer
        raw_payload = struct.pack(
            "<HBBhhhh",
            gp.wButtons, gp.bLeftTrigger, gp.bRightTrigger,
            gp.sThumbLX, gp.sThumbLY, gp.sThumbRX, gp.sThumbRY
        )
        self.ring_buffer.push(RawHardwarePacket(
            device_id=f"XINPUT_PAD_{user_index}",
            source_kind=InputSourceKind.GAMEPAD_XINPUT,
            hw_timestamp_ns=now_ns,
            sequence_id=self._sequence_counter,
            raw_payload=raw_payload,
            vital_max_hp=VITAL_MAX_HP
        ))

        return norm_state

    def set_gamepad_vibration(
        self,
        user_index: int = 0,
        left_motor: float = 0.0,
        right_motor: float = 0.0
    ) -> bool:
        """Actuates controller haptic rumble motors (0.0 to 1.0)."""
        if self.mock_mode or not self._native_xinput_active or self._xinput is None:
            return True
        vib = XINPUT_VIBRATION()
        vib.wLeftMotorSpeed = int(max(0.0, min(1.0, left_motor)) * 65535)
        vib.wRightMotorSpeed = int(max(0.0, min(1.0, right_motor)) * 65535)
        return self._xinput.XInputSetState(user_index, ctypes.byref(vib)) == 0

    def inject_synthetic_event(
        self,
        source_kind: InputSourceKind,
        axes: Optional[Dict[str, float]] = None,
        buttons: int = 0,
        relative_motion: Tuple[int, int] = (0, 0)
    ) -> NormalizedInputState:
        """
        Microsecond-precision synthetic event injection for unit tests,
        stress testing, and deterministic regression benchmarks.
        """
        self._sequence_counter += 1
        now_ns = self.read_monotonic_ns()
        self._last_event_time_ns = now_ns

        applied_axes = axes.copy() if axes else {}
        deadzone_flag = False
        if "lx" in applied_axes and "ly" in applied_axes:
            flx, fly, deadzone_flag = self.filter.filter_radial_stick(
                applied_axes["lx"], applied_axes["ly"]
            )
            applied_axes["lx"] = flx
            applied_axes["ly"] = fly

        norm_state = NormalizedInputState(
            device_id=f"SYNTH_{source_kind.value}",
            source_kind=source_kind,
            timestamp_ns=now_ns,
            sequence_id=self._sequence_counter,
            axes=applied_axes,
            buttons=buttons,
            relative_motion=relative_motion,
            deadzone_applied=deadzone_flag,
            jitter_us=0.0,
            vital_max_hp=VITAL_MAX_HP
        )

        raw_bytes = json.dumps(norm_state.to_dict()).encode("utf-8")
        self.ring_buffer.push(RawHardwarePacket(
            device_id=f"SYNTH_{source_kind.value}",
            source_kind=source_kind,
            hw_timestamp_ns=now_ns,
            sequence_id=self._sequence_counter,
            raw_payload=raw_bytes,
            vital_max_hp=VITAL_MAX_HP
        ))

        return norm_state

    def decode_rotary_encoder_pulse(
        self,
        prev_gray: int,
        curr_gray: int
    ) -> int:
        """
        Decodes 2-bit quadrature Gray code from Phase A and Phase B.
        States: 00 (0), 01 (1), 11 (3), 10 (2).
        Returns +1 (CW), -1 (CCW), or 0 (no movement / invalid).
        """
        # Lookup table for quadrature transitions
        # index: (prev << 2) | curr
        QUADRATURE_LUT = [
             0,  1, -1,  0,  # prev = 0 (00): 00->00, 00->01(+1), 00->10(-1), 00->11(err)
            -1,  0,  0,  1,  # prev = 1 (01): 01->00(-1), 01->01, 01->10(err), 01->11(+1)
             1,  0,  0, -1,  # prev = 2 (10): 10->00(+1), 10->01(err), 10->10, 10->11(-1)
             0, -1,  1,  0   # prev = 3 (11): 11->00(err), 11->01(-1), 11->10(+1), 11->11
        ]
        key = ((prev_gray & 0x03) << 2) | (curr_gray & 0x03)
        return QUADRATURE_LUT[key]

    def derive_user_activity(self) -> UserActivityState:
        """Derives macroscopic user activity state to guide Governor power profiles."""
        now_ns = self.read_monotonic_ns()
        idle_duration_sec = (now_ns - self._last_event_time_ns) / 1e9

        if idle_duration_sec > 10.0:
            return UserActivityState.IDLE
        elif idle_duration_sec > 3.0:
            return UserActivityState.LOW_INTENSITY
        elif self.ring_buffer.size() > 20:
            return UserActivityState.TACTICAL_BURST
        else:
            return UserActivityState.ACTIVE

    def evaluate_governor_demand(
        self,
        thermal_safe_profiles: Set[str],
        battery_safe_profiles: Set[str],
        capability_supported_profiles: Set[str]
    ) -> Tuple[str, str]:
        """
        Hierarchical Governor Authority Contract:
        Input activity generates an advisory request, which is intersected with
        thermal, battery, and capability constraints:
        A_committable = A_input-request ∩ A_thermal-safe ∩ A_battery-safe ∩ A_capability-supported
        Thermal veto, battery critical, and capability absence have absolute veto authority.
        """
        assert VITAL_MAX_HP == 6
        act = self.derive_user_activity()
        advisory_map = {
            UserActivityState.IDLE: "saver_npu",
            UserActivityState.LOW_INTENSITY: "balanced_npu",
            UserActivityState.ACTIVE: "balanced_hybrid",
            UserActivityState.TACTICAL_BURST: "performance_gpu"
        }
        requested = advisory_map.get(act, "balanced_hybrid")
        committable = thermal_safe_profiles & battery_safe_profiles & capability_supported_profiles

        if requested in committable:
            return requested, f"COMMITTED_ADVISORY_{act.value}"
        elif "balanced_npu" in committable:
            return "balanced_npu", f"SAFETY_VETO_FALLBACK_BALANCED (Requested {requested})"
        elif "saver_npu" in committable:
            return "saver_npu", f"SAFETY_VETO_FALLBACK_SAVER (Requested {requested})"
        elif committable:
            return next(iter(committable)), f"SAFETY_VETO_FIRST_COMMITTED (Requested {requested})"
        else:
            return "saver_npu", "EMERGENCY_MINIMAL_SAFE"


# ============================================================================
# 9. EPISTEMIC INPUT BRIDGE (INTEGRATION WITH TYPED DEPENDENCY RUNTIME)
# ============================================================================

class EpistemicInputBridge:
    """
    Ingests live hardware input telemetry into TypedDependencyRuntime as
    strictly typed DependencyValue instances with monotonic timestamps,
    units, and validity half-lives.
    """
    def __init__(self, driver: HardwareInputDriver):
        assert VITAL_MAX_HP == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"
        self.driver = driver
        self._sample_timestamps: List[int] = []

    def record_heartbeat(self, now_ns: int) -> None:
        """Tracks rolling window of timestamps to compute true polling frequency."""
        self._sample_timestamps.append(now_ns)
        # Keep 1 second window
        cutoff = now_ns - 1_000_000_000
        self._sample_timestamps = [ts for ts in self._sample_timestamps if ts >= cutoff]

    def compute_poll_rate_hz(self) -> float:
        """Calculates effective input polling rate in Hz over last 1.0s window."""
        count = len(self._sample_timestamps)
        if count < 2:
            return 60.0  # Default nominal fallback
        duration_s = (self._sample_timestamps[-1] - self._sample_timestamps[0]) / 1e9
        if duration_s <= 0.0:
            return 60.0
        return float(count / duration_s)

    def generate_dependency_values(self) -> List[Any]:
        """
        Synthesizes typed DependencyValues for KARAF and TypedDependencyRuntime.
        Emits:
          - OBS_INPUT_POLL_RATE_HZ: Sensor sampling rate (unit: "Ratio")
          - OBS_INPUT_JITTER_US: Hardware timing jitter (unit: "us")
          - OBS_INPUT_ACTIVITY_SCORE: 0.0 (Idle) to 3.0 (Burst) (unit: "Ratio")
        """
        # Late import to prevent circular dependency
        from krystal_kernel.verification.typed_dependency_runtime import (
            DependencyValue, EpistemicState
        )

        now_ns = self.driver.read_monotonic_ns()
        self.record_heartbeat(now_ns)

        poll_rate = self.compute_poll_rate_hz()
        jitter_us = self.driver.filter._ewma_jitter_us
        activity = self.driver.derive_user_activity()

        activity_map = {
            UserActivityState.IDLE: 0.0,
            UserActivityState.LOW_INTENSITY: 1.0,
            UserActivityState.ACTIVE: 2.0,
            UserActivityState.TACTICAL_BURST: 3.0,
        }
        activity_score = activity_map.get(activity, 1.0)
        validity_ttl_ns = 500_000_000  # 500 ms validity

        dep_poll_rate = DependencyValue(
            id="OBS_INPUT_POLL_RATE_HZ",
            value=float(poll_rate),
            unit="Ratio",
            epistemic_state=EpistemicState.OBS,
            observed_at_ns=now_ns,
            valid_until_ns=now_ns + validity_ttl_ns,
            confidence=0.99,
            provenance="HardwareInputDriver.QPC",
            clock_domain_id="monotonic"
        )

        dep_jitter = DependencyValue(
            id="OBS_INPUT_JITTER_US",
            value=float(jitter_us),
            unit="us",
            epistemic_state=EpistemicState.OBS,
            observed_at_ns=now_ns,
            valid_until_ns=now_ns + validity_ttl_ns,
            confidence=0.95,
            provenance="DeadzoneAndJitterFilter.EWMA",
            clock_domain_id="monotonic"
        )

        dep_activity = DependencyValue(
            id="OBS_INPUT_ACTIVITY_SCORE",
            value=float(activity_score),
            unit="Ratio",
            epistemic_state=EpistemicState.DER,
            observed_at_ns=now_ns,
            valid_until_ns=now_ns + validity_ttl_ns,
            confidence=1.0,
            provenance="HardwareInputDriver.derive_user_activity",
            parent_ids=("OBS_INPUT_POLL_RATE_HZ", "OBS_INPUT_JITTER_US"),
            clock_domain_id="monotonic"
        )

        return [dep_poll_rate, dep_jitter, dep_activity]


# Global Singleton Instance
GLOBAL_HARDWARE_INPUT_DRIVER = HardwareInputDriver()
