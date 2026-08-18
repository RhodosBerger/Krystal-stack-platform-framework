"""
Guardian Hooks - Python Bridge to C Runtime

IPC glue for Python Guardian to drive C validators/runtimes
and receive telemetry. Integrates with thread_boost_layer and rpg_craft_system.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import IntEnum
from threading import Thread, Event

from .shared_memory_ipc import (
    IPCBridge, TelemetryPacket, SignalPacket,
    create_ipc_server, create_ipc_client
)
from .hybrid_event_pipeline import TelemetrySnapshot, PresetType

logger = logging.getLogger(__name__)


class BoostAction(IntEnum):
    """Actions matching C GAMESA_ACTION_* defines."""
    BOOST = 0x01
    THROTTLE = 0x02
    MIGRATE = 0x03
    IDLE = 0x04


class CorePolicy(IntEnum):
    """P/E core distribution policies."""
    ALL_P = 0       # All performance cores
    ALL_E = 1       # All efficiency cores
    BALANCED = 2    # Mix of P and E
    ADAPTIVE = 3    # Dynamic based on load


@dataclass
class ZoneState:
    """Mirror of C gamesa_zone_t."""
    zone_id: int
    grid_x: int
    grid_y: int
    grid_z: int
    p_core_mask: int
    e_core_mask: int
    gpu_block_id: int
    gpu_vram_offset: int
    gpu_vram_size: int
    signal_strength: float
    active: bool
    last_update_ns: int


@dataclass
class BoostConfig:
    """Boost configuration to send to C runtime."""
    clock_multiplier: float = 1.0
    voltage_offset_mv: int = 0
    power_limit_percent: float = 100.0
    thermal_limit_c: float = 95.0
    core_policy: CorePolicy = CorePolicy.ADAPTIVE
    gpu_clock_offset_mhz: int = 0
    vram_clock_offset_mhz: int = 0


@dataclass
class CraftPreset:
    """Crafted preset from RPG system."""
    preset_id: int
    name: str
    boost_config: BoostConfig
    target_zones: List[int]
    priority: int
    duration_ms: int
    cooldown_ms: int


class GuardianBridge:
    """
    Main bridge between Python Guardian and C runtime.

    Responsibilities:
    - Send boost/throttle signals to zones
    - Receive and process telemetry
    - Apply crafted presets
    - Manage zone lifecycle
    """

    def __init__(self, is_server: bool = True):
        self.is_server = is_server
        self.ipc: Optional[IPCBridge] = None

        # State tracking
        self.zones: Dict[int, ZoneState] = {}
        self.active_presets: Dict[int, CraftPreset] = {}
        self.telemetry_history: List[TelemetrySnapshot] = []

        # Callbacks
        self._on_telemetry: Optional[Callable[[TelemetrySnapshot], None]] = None
        self._on_zone_update: Optional[Callable[[ZoneState], None]] = None

        # Background processing
        self._running = False
        self._stop_event = Event()
        self._thread: Optional[Thread] = None

    def connect(self):
        """Connect to C runtime via IPC."""
        try:
            if self.is_server:
                self.ipc = create_ipc_server()
            else:
                self.ipc = create_ipc_client()
            logger.info(f"Guardian bridge connected (server={self.is_server})")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from C runtime."""
        self.stop_polling()
        if self.ipc:
            self.ipc.close()
            self.ipc = None

    # Zone Management

    def register_zone(self, zone: ZoneState):
        """Register zone state (called when C creates zone)."""
        self.zones[zone.zone_id] = zone
        if self._on_zone_update:
            self._on_zone_update(zone)

    def update_zone_signal(self, zone_id: int, strength: float) -> bool:
        """Update zone signal strength."""
        if not self.ipc:
            return False

        success = self.ipc.send_signal(
            zone_id=zone_id,
            strength=strength,
            action=BoostAction.BOOST if strength > 0.5 else BoostAction.THROTTLE,
        )

        if success and zone_id in self.zones:
            self.zones[zone_id].signal_strength = strength

        return success

    def migrate_zone(self, zone_id: int, target_cores: int) -> bool:
        """Migrate zone to different cores."""
        if not self.ipc:
            return False

        return self.ipc.send_signal(
            zone_id=zone_id,
            strength=1.0,
            action=BoostAction.MIGRATE,
            target=float(target_cores),
        )

    def idle_zone(self, zone_id: int) -> bool:
        """Set zone to idle state."""
        if not self.ipc:
            return False

        return self.ipc.send_signal(
            zone_id=zone_id,
            strength=0.0,
            action=BoostAction.IDLE,
        )

    # Preset Application

    def apply_preset(self, preset: CraftPreset) -> bool:
        """Apply crafted preset to zones."""
        if not self.ipc:
            return False

        success = True
        for zone_id in preset.target_zones:
            # Send boost signal with preset priority
            signal_sent = self.ipc.send_signal(
                zone_id=zone_id,
                strength=preset.priority / 100.0,
                action=BoostAction.BOOST,
                target=preset.boost_config.clock_multiplier,
            )
            success = success and signal_sent

        if success:
            self.active_presets[preset.preset_id] = preset

        return success

    def remove_preset(self, preset_id: int) -> bool:
        """Remove active preset."""
        if preset_id not in self.active_presets:
            return False

        preset = self.active_presets[preset_id]
        for zone_id in preset.target_zones:
            self.update_zone_signal(zone_id, 0.5)  # Reset to neutral

        del self.active_presets[preset_id]
        return True

    # Telemetry Processing

    def process_telemetry(self, packet: TelemetryPacket) -> TelemetrySnapshot:
        """Convert IPC packet to TelemetrySnapshot."""
        snapshot = TelemetrySnapshot(
            timestamp=packet.timestamp_ns / 1e9,
            cpu_util=packet.cpu_util,
            gpu_util=packet.gpu_util,
            memory_util=packet.memory_util,
            temp_cpu=packet.temp_cpu,
            temp_gpu=packet.temp_gpu,
            frametime_ms=packet.frametime_ms,
            power_draw=packet.power_draw,
            zone_id=0,
            pe_core_mask=packet.pe_mask,
        )

        self.telemetry_history.append(snapshot)
        if len(self.telemetry_history) > 1000:
            self.telemetry_history = self.telemetry_history[-500:]

        if self._on_telemetry:
            self._on_telemetry(snapshot)

        return snapshot

    def poll_telemetry(self) -> List[TelemetrySnapshot]:
        """Poll all available telemetry."""
        if not self.ipc:
            return []

        packets = self.ipc.receive_all_telemetry()
        return [self.process_telemetry(p) for p in packets]

    # Background Processing

    def start_polling(self, interval_ms: int = 16):
        """Start background telemetry polling."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()

        def poll_loop():
            while not self._stop_event.is_set():
                self.poll_telemetry()
                time.sleep(interval_ms / 1000.0)

        self._thread = Thread(target=poll_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started telemetry polling ({interval_ms}ms interval)")

    def stop_polling(self):
        """Stop background polling."""
        if not self._running:
            return

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._running = False
        logger.info("Stopped telemetry polling")

    # Callbacks

    def on_telemetry(self, callback: Callable[[TelemetrySnapshot], None]):
        """Register telemetry callback."""
        self._on_telemetry = callback

    def on_zone_update(self, callback: Callable[[ZoneState], None]):
        """Register zone update callback."""
        self._on_zone_update = callback

    # Statistics

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "connected": self.ipc is not None,
            "polling": self._running,
            "zones": len(self.zones),
            "active_presets": len(self.active_presets),
            "telemetry_samples": len(self.telemetry_history),
            "ipc_stats": self.ipc.get_stats() if self.ipc else None,
        }

    def get_zone_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all zones."""
        return [
            {
                "zone_id": z.zone_id,
                "grid": (z.grid_x, z.grid_y, z.grid_z),
                "signal": z.signal_strength,
                "active": z.active,
                "gpu_vram_mb": z.gpu_vram_size / (1024 * 1024),
            }
            for z in self.zones.values()
        ]


class ValidatorBridge:
    """Bridge to C binary validators."""

    def __init__(self, validator_path: str = "/opt/gamesa/bin"):
        self.validator_path = validator_path
        self.validators: Dict[str, str] = {}

    def register_validator(self, name: str, binary: str):
        """Register C validator binary."""
        import os
        full_path = os.path.join(self.validator_path, binary)
        if os.path.exists(full_path):
            self.validators[name] = full_path
            logger.info(f"Registered validator: {name} -> {full_path}")
        else:
            logger.warning(f"Validator not found: {full_path}")

    def validate(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run validator on data."""
        if name not in self.validators:
            return {"valid": False, "error": f"Unknown validator: {name}"}

        import subprocess
        try:
            result = subprocess.run(
                [self.validators[name]],
                input=json.dumps(data),
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            return {
                "valid": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}


# Factory functions
def create_guardian_bridge(server: bool = True) -> GuardianBridge:
    """Create guardian bridge."""
    bridge = GuardianBridge(is_server=server)
    bridge.connect()
    return bridge


def create_validator_bridge(path: str = "/opt/gamesa/bin") -> ValidatorBridge:
    """Create validator bridge."""
    return ValidatorBridge(validator_path=path)
