"""
MAVB - Memory-Augmented Virtual Bus

3D Memory Fabric for Cross-Forex resource trading.

Axes:
- X (Width): Memory locality tiers (L1/L2/L3, LLC, VRAM, swap)
- Y (Height): Temporal slots per 16ms market frame
- Z (Depth): Compute intensity / Hex depth

Agents trade voxels for precise memory/bandwidth/latency allocation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
from collections import deque
import time
import json
import random


# ============================================================
# ENUMS & TYPES
# ============================================================

class MemoryTier(Enum):
    """X-axis: Memory locality tiers."""
    L1_CACHE = 0
    L2_CACHE = 1
    L3_CACHE = 2
    LLC = 3
    DRAM = 4
    VRAM = 5
    SWAP = 6


class ResourceType(Enum):
    """Tradeable resources in MAVB."""
    COMPUTE_CREDITS = auto()
    THERMAL_HEADROOM = auto()
    LATENCY_BUDGET = auto()
    PREFETCH_WINDOW = auto()
    VRAM_PREFETCH = auto()
    CACHE_LINE = auto()
    DMA_SLOT = auto()
    BANDWIDTH_GBPS = auto()


class TradeStatus(Enum):
    """Status of MAVB trades."""
    PENDING = auto()
    APPROVED = auto()
    REJECTED = auto()
    EXPIRED = auto()
    CONSUMED = auto()


# ============================================================
# VOXEL CELL
# ============================================================

@dataclass
class VoxelCell:
    """Single cell in the 3D voxel fabric."""
    x: int  # Memory tier
    y: int  # Temporal slot
    z: int  # Compute depth

    # Capacity
    capacity_bytes: int = 1024 * 1024  # 1MB default
    bandwidth_gbps: float = 10.0
    latency_ns: float = 100.0
    temperature_c: float = 50.0

    # State
    committed_bytes: int = 0
    reserved_bytes: int = 0
    owner_agent: Optional[str] = None

    # Metrics
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    congestion_score: float = 0.0

    def available_bytes(self) -> int:
        return self.capacity_bytes - self.committed_bytes - self.reserved_bytes

    def utilization(self) -> float:
        return (self.committed_bytes + self.reserved_bytes) / self.capacity_bytes

    def is_hot(self, threshold: float = 75.0) -> bool:
        return self.temperature_c > threshold

    def to_dict(self) -> Dict:
        return {
            "coords": [self.x, self.y, self.z],
            "capacity": self.capacity_bytes,
            "committed": self.committed_bytes,
            "reserved": self.reserved_bytes,
            "utilization": self.utilization(),
            "temperature": self.temperature_c,
            "congestion": self.congestion_score
        }


# ============================================================
# MAVB TRADE
# ============================================================

@dataclass
class MAVBTrade:
    """Trade request for voxel resources."""
    trade_id: str
    source_agent: str
    voxel_coords: Tuple[int, int, int]  # (x, y, z)
    resource: ResourceType
    bytes_requested: int
    duration_ms: int

    # Bid
    latency_gain_ms: float = 0.0
    thermal_cost_c: float = 0.0

    # State
    status: TradeStatus = TradeStatus.PENDING
    timestamp: float = field(default_factory=time.time)
    permit_id: Optional[str] = None
    expiry: Optional[float] = None

    def to_json(self) -> str:
        return json.dumps({
            "type": "MAVB_TRADE",
            "trade_id": self.trade_id,
            "source": self.source_agent,
            "voxel": {"x": self.voxel_coords[0], "y": self.voxel_coords[1], "z": self.voxel_coords[2]},
            "request": {
                "resource": self.resource.name,
                "bytes": self.bytes_requested,
                "duration_ms": self.duration_ms
            },
            "bid": {
                "latency_gain_ms": self.latency_gain_ms,
                "thermal_cost_c": self.thermal_cost_c
            },
            "status": self.status.name
        })


@dataclass
class VoxelPermit:
    """Permit granted for voxel access."""
    permit_id: str
    trade_id: str
    voxel_coords: Tuple[int, int, int]
    bytes_granted: int
    valid_from: float
    valid_until: float
    constraints: Dict[str, float] = field(default_factory=dict)


# ============================================================
# PHYSICAL CAPTURE LAYER (PCL)
# ============================================================

class PhysicalCaptureLayer:
    """
    Hooks system metrics and converts to voxel metadata.

    Sources: /sys/class/drm/**, perf_event_open, NUMA metrics
    """

    def __init__(self):
        self.metrics: Dict[str, float] = {}
        self.last_capture: float = 0

    def capture(self) -> Dict[str, float]:
        """Capture current physical state (simulated)."""
        self.last_capture = time.time()

        # Simulated hardware metrics
        self.metrics = {
            "cpu_bandwidth_gbps": 40 + random.gauss(0, 5),
            "gpu_bandwidth_gbps": 200 + random.gauss(0, 20),
            "l3_latency_ns": 40 + random.gauss(0, 5),
            "dram_latency_ns": 80 + random.gauss(0, 10),
            "vram_latency_ns": 150 + random.gauss(0, 20),
            "cpu_temp_c": 55 + random.gauss(0, 5),
            "gpu_temp_c": 65 + random.gauss(0, 8),
            "numa_node0_free_mb": 8000 + random.gauss(0, 500),
            "numa_node1_free_mb": 8000 + random.gauss(0, 500),
        }
        return self.metrics

    def get_voxel_metadata(self, x: int, y: int, z: int) -> Dict:
        """Get metadata for specific voxel based on physical state."""
        tier = MemoryTier(min(x, 6))

        # Bandwidth by tier
        bandwidth_map = {
            MemoryTier.L1_CACHE: 500,
            MemoryTier.L2_CACHE: 200,
            MemoryTier.L3_CACHE: 100,
            MemoryTier.LLC: 80,
            MemoryTier.DRAM: 40,
            MemoryTier.VRAM: 200,
            MemoryTier.SWAP: 0.5,
        }

        # Latency by tier
        latency_map = {
            MemoryTier.L1_CACHE: 1,
            MemoryTier.L2_CACHE: 4,
            MemoryTier.L3_CACHE: 15,
            MemoryTier.LLC: 30,
            MemoryTier.DRAM: 80,
            MemoryTier.VRAM: 150,
            MemoryTier.SWAP: 10000,
        }

        # Temperature increases with depth (compute intensity)
        base_temp = self.metrics.get("cpu_temp_c", 55) if x < 5 else self.metrics.get("gpu_temp_c", 65)
        temp = base_temp + z * 0.5

        return {
            "bandwidth_gbps": bandwidth_map.get(tier, 10),
            "latency_ns": latency_map.get(tier, 100),
            "temperature_c": temp
        }


# ============================================================
# VIRTUAL ALLOCATION LAYER (VAL)
# ============================================================

class VirtualAllocationLayer:
    """
    Manages 3D voxel tensor with reservations and commitments.

    Implements optimistic reservations with Guardian veto hooks.
    """

    def __init__(self, dims: Tuple[int, int, int] = (7, 16, 32)):
        """
        Args:
            dims: (x_tiers, y_slots, z_depth) = (7 memory tiers, 16ms slots, 32 hex depths)
        """
        self.dims = dims
        self.voxels: List[List[List[VoxelCell]]] = []
        self._initialize_fabric()

        self.reservations: Dict[str, VoxelPermit] = {}
        self.pending_trades: Dict[str, MAVBTrade] = {}

    def _initialize_fabric(self):
        """Initialize 3D voxel tensor."""
        x_size, y_size, z_size = self.dims

        # Capacity by tier (bytes)
        tier_capacity = {
            0: 32 * 1024,        # L1: 32KB
            1: 256 * 1024,       # L2: 256KB
            2: 8 * 1024 * 1024,  # L3: 8MB
            3: 16 * 1024 * 1024, # LLC: 16MB
            4: 1024 * 1024 * 1024,  # DRAM: 1GB per slot
            5: 256 * 1024 * 1024,   # VRAM: 256MB per slot
            6: 4 * 1024 * 1024 * 1024,  # Swap: 4GB
        }

        self.voxels = []
        for x in range(x_size):
            tier = []
            for y in range(y_size):
                slot = []
                for z in range(z_size):
                    cell = VoxelCell(
                        x=x, y=y, z=z,
                        capacity_bytes=tier_capacity.get(x, 1024 * 1024)
                    )
                    slot.append(cell)
                tier.append(slot)
            self.voxels.append(tier)

    def get_voxel(self, x: int, y: int, z: int) -> Optional[VoxelCell]:
        """Get voxel at coordinates."""
        try:
            return self.voxels[x][y][z]
        except IndexError:
            return None

    def reserve(self, trade: MAVBTrade) -> Optional[VoxelPermit]:
        """Attempt optimistic reservation."""
        x, y, z = trade.voxel_coords
        voxel = self.get_voxel(x, y, z)

        if not voxel:
            trade.status = TradeStatus.REJECTED
            return None

        if voxel.available_bytes() < trade.bytes_requested:
            trade.status = TradeStatus.REJECTED
            return None

        # Optimistic reservation
        voxel.reserved_bytes += trade.bytes_requested
        voxel.access_count += 1
        voxel.last_access = time.time()

        permit = VoxelPermit(
            permit_id=f"PERMIT_{trade.trade_id}",
            trade_id=trade.trade_id,
            voxel_coords=trade.voxel_coords,
            bytes_granted=trade.bytes_requested,
            valid_from=time.time(),
            valid_until=time.time() + trade.duration_ms / 1000,
            constraints={"max_thermal": 80.0}
        )

        trade.permit_id = permit.permit_id
        trade.expiry = permit.valid_until
        self.reservations[permit.permit_id] = permit

        return permit

    def commit(self, permit_id: str) -> bool:
        """Convert reservation to commitment."""
        permit = self.reservations.get(permit_id)
        if not permit:
            return False

        x, y, z = permit.voxel_coords
        voxel = self.get_voxel(x, y, z)
        if not voxel:
            return False

        voxel.reserved_bytes -= permit.bytes_granted
        voxel.committed_bytes += permit.bytes_granted

        return True

    def release(self, permit_id: str) -> bool:
        """Release a reservation or commitment."""
        permit = self.reservations.pop(permit_id, None)
        if not permit:
            return False

        x, y, z = permit.voxel_coords
        voxel = self.get_voxel(x, y, z)
        if voxel:
            # Try reserved first, then committed
            if voxel.reserved_bytes >= permit.bytes_granted:
                voxel.reserved_bytes -= permit.bytes_granted
            else:
                voxel.committed_bytes -= permit.bytes_granted

        return True

    def cleanup_expired(self) -> int:
        """Release expired reservations. Returns count released."""
        now = time.time()
        expired = [pid for pid, p in self.reservations.items() if p.valid_until < now]
        for pid in expired:
            self.release(pid)
        return len(expired)

    def get_heatmap(self) -> List[Dict]:
        """Generate heatmap snapshot for debugging."""
        heatmap = []
        for x in range(self.dims[0]):
            for y in range(self.dims[1]):
                for z in range(self.dims[2]):
                    voxel = self.voxels[x][y][z]
                    if voxel.utilization() > 0.1 or voxel.is_hot():
                        heatmap.append(voxel.to_dict())
        return heatmap


# ============================================================
# ACTION RELAY LAYER (ARL)
# ============================================================

class ActionRelayLayer:
    """
    Emits MAVB_TRADE payloads and maps to kernel actions.

    Actions: cache prefetch, DMA priming, VRAM residency hints
    """

    def __init__(self):
        self.pending_actions: deque = deque(maxlen=1000)
        self.completed_actions: deque = deque(maxlen=1000)

    def emit_trade(self, trade: MAVBTrade) -> Dict:
        """Emit trade payload."""
        payload = json.loads(trade.to_json())
        self.pending_actions.append(payload)
        return payload

    def map_to_action(self, permit: VoxelPermit) -> Dict:
        """Map permit to kernel action."""
        x, y, z = permit.voxel_coords
        tier = MemoryTier(min(x, 6))

        action = {
            "type": "KERNEL_ACTION",
            "permit_id": permit.permit_id,
            "voxel": permit.voxel_coords,
            "timestamp": time.time()
        }

        if tier in (MemoryTier.L1_CACHE, MemoryTier.L2_CACHE, MemoryTier.L3_CACHE):
            action["action"] = "CACHE_PREFETCH"
            action["params"] = {"lines": permit.bytes_granted // 64}
        elif tier == MemoryTier.VRAM:
            action["action"] = "VRAM_RESIDENCY"
            action["params"] = {"bytes": permit.bytes_granted, "priority": z}
        elif tier == MemoryTier.DRAM:
            action["action"] = "DMA_PRIME"
            action["params"] = {"buffer_size": permit.bytes_granted}
        else:
            action["action"] = "ALLOCATE"
            action["params"] = {"bytes": permit.bytes_granted}

        self.completed_actions.append(action)
        return action


# ============================================================
# GUARDIAN ARBITER
# ============================================================

class GuardianArbiter:
    """
    Arbitrates voxel trades with safety constraints.

    - Thermal fuse
    - Cross-axis contention detection
    - Fallback suggestions
    """

    def __init__(self, val: VirtualAllocationLayer, pcl: PhysicalCaptureLayer):
        self.val = val
        self.pcl = pcl
        self.thermal_threshold = 80.0
        self.contention_threshold = 0.85

    def evaluate_trade(self, trade: MAVBTrade) -> Tuple[bool, Dict]:
        """
        Evaluate trade request.

        Returns:
            (approved, directive)
        """
        x, y, z = trade.voxel_coords
        voxel = self.val.get_voxel(x, y, z)

        if not voxel:
            return False, {"reason": "invalid_voxel", "redirect_voxel": None}

        # Thermal check
        meta = self.pcl.get_voxel_metadata(x, y, z)
        if meta["temperature_c"] > self.thermal_threshold:
            # Find cooler alternative
            redirect = self._find_cool_voxel(x, y, trade.bytes_requested)
            return False, {
                "reason": "thermal_saturation",
                "temperature": meta["temperature_c"],
                "redirect_voxel": redirect
            }

        # Contention check
        if voxel.utilization() > self.contention_threshold:
            redirect = self._find_available_voxel(x, y, trade.bytes_requested)
            return False, {
                "reason": "contention_exceeded",
                "utilization": voxel.utilization(),
                "redirect_voxel": redirect
            }

        # Capacity check
        if voxel.available_bytes() < trade.bytes_requested:
            return False, {
                "reason": "insufficient_capacity",
                "available": voxel.available_bytes(),
                "requested": trade.bytes_requested
            }

        # Approved
        return True, {"reason": "approved"}

    def _find_cool_voxel(self, x: int, y: int, min_bytes: int) -> Optional[Tuple[int, int, int]]:
        """Find a cooler voxel in same tier/slot."""
        for z in range(self.val.dims[2]):
            voxel = self.val.get_voxel(x, y, z)
            meta = self.pcl.get_voxel_metadata(x, y, z)
            if voxel and meta["temperature_c"] < self.thermal_threshold * 0.9:
                if voxel.available_bytes() >= min_bytes:
                    return (x, y, z)
        return None

    def _find_available_voxel(self, x: int, y: int, min_bytes: int) -> Optional[Tuple[int, int, int]]:
        """Find available voxel in same tier/slot."""
        for z in range(self.val.dims[2]):
            voxel = self.val.get_voxel(x, y, z)
            if voxel and voxel.utilization() < 0.5 and voxel.available_bytes() >= min_bytes:
                return (x, y, z)
        return None

    def generate_directive(self, trade: MAVBTrade, permit: Optional[VoxelPermit],
                          evaluation: Dict) -> Dict:
        """Generate Guardian directive."""
        return {
            "type": "DIRECTIVE",
            "trade_id": trade.trade_id,
            "status": trade.status.name,
            "voxel_permit": permit.permit_id if permit else None,
            "params": evaluation
        }


# ============================================================
# MAVB BUS (MAIN INTERFACE)
# ============================================================

class MAVBus:
    """
    Memory-Augmented Virtual Bus - Main Interface.

    Provides unified 3D memory fabric for Cross-Forex agents.
    """

    def __init__(self, dims: Tuple[int, int, int] = (7, 16, 32)):
        self.pcl = PhysicalCaptureLayer()
        self.val = VirtualAllocationLayer(dims)
        self.arl = ActionRelayLayer()
        self.guardian = GuardianArbiter(self.val, self.pcl)

        self.trade_counter = 0
        self.trade_history: deque = deque(maxlen=10000)
        self.audit_log: deque = deque(maxlen=10000)

    def submit_trade(self, agent: str, voxel: Tuple[int, int, int],
                     resource: ResourceType, bytes_req: int, duration_ms: int,
                     latency_gain: float = 0, thermal_cost: float = 0) -> Dict:
        """
        Submit a voxel trade request.

        Args:
            agent: Source agent ID
            voxel: (x, y, z) coordinates
            resource: Resource type
            bytes_req: Bytes requested
            duration_ms: Duration in milliseconds
            latency_gain: Expected latency improvement
            thermal_cost: Expected thermal cost

        Returns:
            Directive with permit or rejection
        """
        self.trade_counter += 1
        trade_id = f"TRADE_{self.trade_counter:06d}"

        trade = MAVBTrade(
            trade_id=trade_id,
            source_agent=agent,
            voxel_coords=voxel,
            resource=resource,
            bytes_requested=bytes_req,
            duration_ms=duration_ms,
            latency_gain_ms=latency_gain,
            thermal_cost_c=thermal_cost
        )

        # Capture current physical state
        self.pcl.capture()

        # Guardian evaluation
        approved, evaluation = self.guardian.evaluate_trade(trade)

        permit = None
        if approved:
            permit = self.val.reserve(trade)
            if permit:
                trade.status = TradeStatus.APPROVED
                # Emit action
                self.arl.map_to_action(permit)
            else:
                trade.status = TradeStatus.REJECTED
                evaluation["reason"] = "reservation_failed"
        else:
            trade.status = TradeStatus.REJECTED

        # Emit trade payload
        self.arl.emit_trade(trade)

        # Audit log
        self._audit(trade, permit, evaluation)

        # History
        self.trade_history.append(trade)

        return self.guardian.generate_directive(trade, permit, evaluation)

    def consume(self, permit_id: str) -> bool:
        """Mark permit as consumed (commit the allocation)."""
        success = self.val.commit(permit_id)
        if success:
            self._audit_simple(f"CONSUMED: {permit_id}")
        return success

    def release(self, permit_id: str) -> bool:
        """Release a permit."""
        success = self.val.release(permit_id)
        if success:
            self._audit_simple(f"RELEASED: {permit_id}")
        return success

    def tick(self) -> Dict:
        """Periodic maintenance tick."""
        expired = self.val.cleanup_expired()
        self.pcl.capture()

        # Update voxel temperatures from PCL
        for x in range(self.val.dims[0]):
            for y in range(self.val.dims[1]):
                for z in range(self.val.dims[2]):
                    voxel = self.val.voxels[x][y][z]
                    meta = self.pcl.get_voxel_metadata(x, y, z)
                    voxel.temperature_c = meta["temperature_c"]
                    voxel.bandwidth_gbps = meta["bandwidth_gbps"]
                    voxel.latency_ns = meta["latency_ns"]

        return {
            "expired_released": expired,
            "active_reservations": len(self.val.reservations),
            "total_trades": self.trade_counter
        }

    def get_heatmap(self) -> Dict:
        """Get current heatmap for market ticker."""
        return {
            "type": "MARKET_TICKER",
            "mavb_heatmap": self.val.get_heatmap(),
            "timestamp": time.time()
        }

    def get_voxel_info(self, x: int, y: int, z: int) -> Optional[Dict]:
        """Get detailed info for specific voxel."""
        voxel = self.val.get_voxel(x, y, z)
        if not voxel:
            return None
        meta = self.pcl.get_voxel_metadata(x, y, z)
        info = voxel.to_dict()
        info.update(meta)
        return info

    def _audit(self, trade: MAVBTrade, permit: Optional[VoxelPermit], evaluation: Dict):
        """Log to audit trail."""
        self.audit_log.append({
            "timestamp": time.time(),
            "trade_id": trade.trade_id,
            "voxel": trade.voxel_coords,
            "status": trade.status.name,
            "permit_id": permit.permit_id if permit else None,
            "evaluation": evaluation
        })

    def _audit_simple(self, message: str):
        self.audit_log.append({"timestamp": time.time(), "message": message})

    def stats(self) -> Dict:
        """Get bus statistics."""
        return {
            "total_trades": self.trade_counter,
            "active_reservations": len(self.val.reservations),
            "pending_actions": len(self.arl.pending_actions),
            "completed_actions": len(self.arl.completed_actions),
            "audit_entries": len(self.audit_log),
            "dims": self.val.dims
        }


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate MAVB."""
    print("=" * 60)
    print("MAVB - Memory-Augmented Virtual Bus")
    print("=" * 60)

    bus = MAVBus()

    print("\nSubmitting trades from various agents...\n")

    # Agent 1: GPU wants VRAM prefetch
    result = bus.submit_trade(
        agent="AGENT_IRIS_XE",
        voxel=(5, 4, 18),  # VRAM, slot 4, depth 18
        resource=ResourceType.VRAM_PREFETCH,
        bytes_req=8 * 1024 * 1024,  # 8MB
        duration_ms=12,
        latency_gain=1.8,
        thermal_cost=0.6
    )
    print(f"GPU Trade: {result['status']}")
    if result.get('voxel_permit'):
        print(f"  Permit: {result['voxel_permit']}")

    # Agent 2: CPU wants cache prefetch
    result = bus.submit_trade(
        agent="AGENT_CPU_CORE",
        voxel=(2, 5, 10),  # L3, slot 5, depth 10
        resource=ResourceType.CACHE_LINE,
        bytes_req=256 * 1024,  # 256KB
        duration_ms=16,
        latency_gain=0.5,
        thermal_cost=0.1
    )
    print(f"CPU Trade: {result['status']}")

    # Agent 3: Memory controller wants DMA
    result = bus.submit_trade(
        agent="AGENT_MEM_CTRL",
        voxel=(4, 8, 5),  # DRAM, slot 8, depth 5
        resource=ResourceType.DMA_SLOT,
        bytes_req=64 * 1024 * 1024,  # 64MB
        duration_ms=32,
        latency_gain=2.0,
        thermal_cost=0.3
    )
    print(f"MEM Trade: {result['status']}")

    # Tick to update
    tick_result = bus.tick()
    print(f"\nTick: {tick_result}")

    # Stats
    print(f"\nBus Stats: {bus.stats()}")

    # Heatmap sample
    heatmap = bus.get_heatmap()
    print(f"\nHeatmap entries: {len(heatmap['mavb_heatmap'])}")

    # Voxel info
    info = bus.get_voxel_info(5, 4, 18)
    if info:
        print(f"\nVoxel (5,4,18) info:")
        for k, v in info.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    demo()
