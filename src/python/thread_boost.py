"""
Thread Boost Layer

Bridges world/grid systems with task planner's external memory and RPG Craft boost maps.

Features:
- Thread boost zones with grid coverage
- P/E core affinity masks
- Pre-allocated memory blocks
- Signal strength based elevation
- Auto-inhibition of weak flows
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum, auto
from collections import deque
import time


# ============================================================
# ENUMS
# ============================================================

class CoreType(Enum):
    """CPU core types."""
    P_CORE = auto()  # Performance core
    E_CORE = auto()  # Efficiency core


class ZoneStatus(Enum):
    """Thread boost zone status."""
    ACTIVE = auto()
    ELEVATED = auto()
    BACKGROUND = auto()
    INHIBITED = auto()


class FlowPriority(Enum):
    """Data flow priority levels."""
    REALTIME = 4
    HIGH = 3
    NORMAL = 2
    LOW = 1
    BACKGROUND = 0


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class CoreMask:
    """CPU core affinity mask."""
    p_cores: Set[int] = field(default_factory=lambda: {0, 1, 2, 3})
    e_cores: Set[int] = field(default_factory=lambda: {4, 5, 6, 7})

    def get_mask(self, prefer_p: bool = True) -> Set[int]:
        if prefer_p:
            return self.p_cores | self.e_cores
        return self.e_cores | self.p_cores

    def p_only(self) -> Set[int]:
        return self.p_cores

    def e_only(self) -> Set[int]:
        return self.e_cores


@dataclass
class VoltageCurve:
    """Voltage/frequency curve for boost."""
    base_freq_mhz: int = 2400
    boost_freq_mhz: int = 4800
    base_voltage_mv: int = 900
    boost_voltage_mv: int = 1200

    def get_voltage_for_freq(self, freq_mhz: int) -> int:
        ratio = (freq_mhz - self.base_freq_mhz) / (self.boost_freq_mhz - self.base_freq_mhz)
        ratio = max(0, min(1, ratio))
        return int(self.base_voltage_mv + ratio * (self.boost_voltage_mv - self.base_voltage_mv))


@dataclass
class MemoryBlock:
    """Pre-allocated memory block."""
    block_id: str
    size_bytes: int
    numa_node: int = 0
    hugepages: bool = False
    locked: bool = False  # mlock'd


@dataclass
class ThreadBoostZone:
    """A thread boost zone for high-value data streams."""
    zone_id: str
    name: str

    # Grid coverage
    grid_x: Tuple[int, int] = (0, 1)  # (start, end)
    grid_y: Tuple[int, int] = (0, 1)
    grid_z: Tuple[int, int] = (0, 1)

    # Core affinity
    core_mask: CoreMask = field(default_factory=CoreMask)
    prefer_p_cores: bool = True

    # Memory
    memory_blocks: List[MemoryBlock] = field(default_factory=list)
    total_memory_mb: int = 64

    # State
    status: ZoneStatus = ZoneStatus.ACTIVE
    signal_strength: float = 0.5
    priority: FlowPriority = FlowPriority.NORMAL

    # Metrics
    throughput_mbps: float = 0.0
    latency_us: float = 100.0
    utilization: float = 0.0


@dataclass
class BoostPreset:
    """RPG Craft style boost preset."""
    preset_id: str
    name: str
    voltage_curve: VoltageCurve
    core_mask: CoreMask
    power_limit_w: int = 65
    thermal_limit_c: int = 85

    def to_dict(self) -> Dict:
        return {
            "preset_id": self.preset_id,
            "name": self.name,
            "freq_range": [self.voltage_curve.base_freq_mhz, self.voltage_curve.boost_freq_mhz],
            "voltage_range": [self.voltage_curve.base_voltage_mv, self.voltage_curve.boost_voltage_mv],
            "p_cores": list(self.core_mask.p_cores),
            "e_cores": list(self.core_mask.e_cores),
            "power_limit": self.power_limit_w,
            "thermal_limit": self.thermal_limit_c
        }


# ============================================================
# RPG CRAFT BOOST MAPS
# ============================================================

class RPGCraftBoostMaps:
    """RPG-style boost configuration maps."""

    PRESETS = {
        "LEGENDARY_BURST": BoostPreset(
            preset_id="LEGEND",
            name="Legendary Burst",
            voltage_curve=VoltageCurve(2400, 5200, 900, 1350),
            core_mask=CoreMask({0, 1, 2, 3}, {4, 5, 6, 7}),
            power_limit_w=125,
            thermal_limit_c=95
        ),
        "EPIC_PERFORMANCE": BoostPreset(
            preset_id="EPIC",
            name="Epic Performance",
            voltage_curve=VoltageCurve(2400, 4800, 900, 1250),
            core_mask=CoreMask({0, 1, 2, 3}, {4, 5, 6, 7}),
            power_limit_w=95,
            thermal_limit_c=90
        ),
        "RARE_BALANCED": BoostPreset(
            preset_id="RARE",
            name="Rare Balanced",
            voltage_curve=VoltageCurve(2400, 4400, 900, 1150),
            core_mask=CoreMask({0, 1}, {4, 5, 6, 7}),
            power_limit_w=65,
            thermal_limit_c=85
        ),
        "COMMON_EFFICIENT": BoostPreset(
            preset_id="COMMON",
            name="Common Efficient",
            voltage_curve=VoltageCurve(2000, 3600, 850, 1050),
            core_mask=CoreMask({0}, {4, 5, 6, 7}),
            power_limit_w=45,
            thermal_limit_c=80
        ),
        "BACKGROUND_SILENT": BoostPreset(
            preset_id="BG",
            name="Background Silent",
            voltage_curve=VoltageCurve(1600, 2400, 800, 950),
            core_mask=CoreMask(set(), {4, 5, 6, 7}),
            power_limit_w=25,
            thermal_limit_c=70
        )
    }

    @classmethod
    def get_for_priority(cls, priority: FlowPriority) -> BoostPreset:
        mapping = {
            FlowPriority.REALTIME: "LEGENDARY_BURST",
            FlowPriority.HIGH: "EPIC_PERFORMANCE",
            FlowPriority.NORMAL: "RARE_BALANCED",
            FlowPriority.LOW: "COMMON_EFFICIENT",
            FlowPriority.BACKGROUND: "BACKGROUND_SILENT"
        }
        return cls.PRESETS[mapping[priority]]


# ============================================================
# EXTERNAL MEMORY EMULATION (1GB)
# ============================================================

class ExternalMemoryEmulator:
    """
    Emulates 1GB external memory for task planner.

    Uses memory-mapped regions with NUMA awareness.
    """

    def __init__(self, size_mb: int = 1024):
        self.total_size_mb = size_mb
        self.allocated_blocks: Dict[str, MemoryBlock] = {}
        self.free_size_mb = size_mb

    def allocate(self, block_id: str, size_mb: int,
                 numa_node: int = 0, hugepages: bool = False) -> Optional[MemoryBlock]:
        """Allocate memory block."""
        if size_mb > self.free_size_mb:
            return None

        block = MemoryBlock(
            block_id=block_id,
            size_bytes=size_mb * 1024 * 1024,
            numa_node=numa_node,
            hugepages=hugepages
        )
        self.allocated_blocks[block_id] = block
        self.free_size_mb -= size_mb
        return block

    def free(self, block_id: str) -> bool:
        """Free memory block."""
        if block_id not in self.allocated_blocks:
            return False

        block = self.allocated_blocks.pop(block_id)
        self.free_size_mb += block.size_bytes // (1024 * 1024)
        return True

    def get_stats(self) -> Dict:
        return {
            "total_mb": self.total_size_mb,
            "free_mb": self.free_size_mb,
            "used_mb": self.total_size_mb - self.free_size_mb,
            "blocks": len(self.allocated_blocks)
        }


# ============================================================
# THREAD BOOST LAYER
# ============================================================

class ThreadBoostLayer:
    """
    Main Thread Boost Layer.

    Manages boost zones, core affinity, memory allocation,
    and signal-strength-based flow elevation.
    """

    def __init__(self):
        self.zones: Dict[str, ThreadBoostZone] = {}
        self.external_memory = ExternalMemoryEmulator(1024)  # 1GB
        self.boost_maps = RPGCraftBoostMaps()

        # Telemetry
        self.telemetry_history: deque = deque(maxlen=1000)
        self.zone_counter = 0

        # Thresholds
        self.elevation_threshold = 0.7
        self.inhibition_threshold = 0.2

    def create_zone(self, name: str, grid_coverage: Tuple[Tuple[int, int], ...],
                    memory_mb: int = 64, priority: FlowPriority = FlowPriority.NORMAL) -> ThreadBoostZone:
        """Create a new thread boost zone."""
        self.zone_counter += 1
        zone_id = f"ZONE_{self.zone_counter:04d}"

        # Allocate memory
        memory_block = self.external_memory.allocate(
            f"MEM_{zone_id}",
            memory_mb,
            hugepages=(memory_mb >= 64)
        )

        # Get core mask from priority
        preset = self.boost_maps.get_for_priority(priority)

        zone = ThreadBoostZone(
            zone_id=zone_id,
            name=name,
            grid_x=grid_coverage[0] if len(grid_coverage) > 0 else (0, 1),
            grid_y=grid_coverage[1] if len(grid_coverage) > 1 else (0, 1),
            grid_z=grid_coverage[2] if len(grid_coverage) > 2 else (0, 1),
            core_mask=preset.core_mask,
            prefer_p_cores=(priority.value >= FlowPriority.HIGH.value),
            memory_blocks=[memory_block] if memory_block else [],
            total_memory_mb=memory_mb,
            priority=priority
        )

        self.zones[zone_id] = zone
        return zone

    def update_signal(self, zone_id: str, signal_strength: float,
                      throughput_mbps: float = None, latency_us: float = None):
        """Update zone signal strength and metrics."""
        if zone_id not in self.zones:
            return

        zone = self.zones[zone_id]
        zone.signal_strength = signal_strength

        if throughput_mbps is not None:
            zone.throughput_mbps = throughput_mbps
        if latency_us is not None:
            zone.latency_us = latency_us

        # Auto-adjust status based on signal
        self._evaluate_zone(zone)

    def _evaluate_zone(self, zone: ThreadBoostZone):
        """Evaluate and adjust zone status."""
        old_status = zone.status

        if zone.signal_strength >= self.elevation_threshold:
            zone.status = ZoneStatus.ELEVATED
            zone.priority = FlowPriority.HIGH
            zone.prefer_p_cores = True
        elif zone.signal_strength >= 0.4:
            zone.status = ZoneStatus.ACTIVE
            zone.priority = FlowPriority.NORMAL
        elif zone.signal_strength >= self.inhibition_threshold:
            zone.status = ZoneStatus.BACKGROUND
            zone.priority = FlowPriority.LOW
            zone.prefer_p_cores = False
        else:
            zone.status = ZoneStatus.INHIBITED
            zone.priority = FlowPriority.BACKGROUND
            zone.prefer_p_cores = False

        # Update core mask based on new priority
        preset = self.boost_maps.get_for_priority(zone.priority)
        zone.core_mask = preset.core_mask

        if old_status != zone.status:
            self._log_transition(zone, old_status)

    def _log_transition(self, zone: ThreadBoostZone, old_status: ZoneStatus):
        """Log zone status transition."""
        self.telemetry_history.append({
            "timestamp": time.time(),
            "zone_id": zone.zone_id,
            "transition": f"{old_status.name} -> {zone.status.name}",
            "signal": zone.signal_strength
        })

    def get_boost_preset(self, zone_id: str) -> Optional[BoostPreset]:
        """Get current boost preset for zone."""
        if zone_id not in self.zones:
            return None
        zone = self.zones[zone_id]
        return self.boost_maps.get_for_priority(zone.priority)

    def tick(self) -> Dict:
        """Periodic update tick."""
        active = sum(1 for z in self.zones.values() if z.status == ZoneStatus.ACTIVE)
        elevated = sum(1 for z in self.zones.values() if z.status == ZoneStatus.ELEVATED)
        inhibited = sum(1 for z in self.zones.values() if z.status == ZoneStatus.INHIBITED)

        return {
            "total_zones": len(self.zones),
            "active": active,
            "elevated": elevated,
            "inhibited": inhibited,
            "memory": self.external_memory.get_stats()
        }

    def generate_report(self) -> str:
        """Generate telemetry report."""
        lines = ["=== THREAD BOOST LAYER REPORT ===\n"]

        for zone in self.zones.values():
            preset = self.get_boost_preset(zone.zone_id)
            lines.append(f"Zone: {zone.name} ({zone.zone_id})")
            lines.append(f"  Status: {zone.status.name}")
            lines.append(f"  Signal: {zone.signal_strength:.2f}")
            lines.append(f"  Priority: {zone.priority.name}")
            lines.append(f"  Cores: P={list(zone.core_mask.p_cores)} E={list(zone.core_mask.e_cores)}")
            lines.append(f"  Memory: {zone.total_memory_mb}MB")
            if preset:
                lines.append(f"  Boost: {preset.name} ({preset.voltage_curve.boost_freq_mhz}MHz)")
            lines.append("")

        lines.append(f"External Memory: {self.external_memory.get_stats()}")
        return "\n".join(lines)

    def destroy_zone(self, zone_id: str) -> bool:
        """Destroy a zone and free resources."""
        if zone_id not in self.zones:
            return False

        zone = self.zones.pop(zone_id)
        for block in zone.memory_blocks:
            self.external_memory.free(block.block_id)

        return True


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate Thread Boost Layer."""
    print("=" * 60)
    print("THREAD BOOST LAYER")
    print("=" * 60)

    layer = ThreadBoostLayer()

    # Create zones for different workloads
    print("\nCreating boost zones...")

    gpu_zone = layer.create_zone(
        "GPU_RENDER",
        grid_coverage=((0, 4), (0, 16), (0, 32)),
        memory_mb=256,
        priority=FlowPriority.HIGH
    )
    print(f"Created: {gpu_zone.name} ({gpu_zone.zone_id})")

    ai_zone = layer.create_zone(
        "AI_INFERENCE",
        grid_coverage=((0, 2), (0, 8), (16, 32)),
        memory_mb=128,
        priority=FlowPriority.REALTIME
    )
    print(f"Created: {ai_zone.name} ({ai_zone.zone_id})")

    io_zone = layer.create_zone(
        "IO_BUFFER",
        grid_coverage=((4, 7), (0, 16), (0, 8)),
        memory_mb=64,
        priority=FlowPriority.LOW
    )
    print(f"Created: {io_zone.name} ({io_zone.zone_id})")

    # Update signals
    print("\nUpdating signal strengths...")
    layer.update_signal(gpu_zone.zone_id, 0.85, throughput_mbps=1500)
    layer.update_signal(ai_zone.zone_id, 0.95, latency_us=50)
    layer.update_signal(io_zone.zone_id, 0.15, throughput_mbps=200)

    # Tick
    tick_result = layer.tick()
    print(f"\nTick: {tick_result}")

    # Report
    print("\n" + layer.generate_report())

    # Show boost presets
    print("Boost Presets by Zone:")
    for zone_id, zone in layer.zones.items():
        preset = layer.get_boost_preset(zone_id)
        if preset:
            print(f"  {zone.name}: {preset.name}")
            print(f"    Freq: {preset.voltage_curve.base_freq_mhz}-{preset.voltage_curve.boost_freq_mhz}MHz")
            print(f"    Power: {preset.power_limit_w}W")


if __name__ == "__main__":
    demo()
