"""
GAMESA TPU Memory Management System

Advanced memory management for TPU accelerators that integrates with the
3D grid memory system and cross-forex trading. This system manages
on-chip TPU memory, host memory transfers, and memory coherence across
TPU accelerators.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum, auto
from collections import defaultdict, deque
import time
import threading
import uuid
from datetime import datetime
import struct

from . import (
    # Core GAMESA components
    ResourceType, Priority, AllocationRequest, Allocation,
    Effect, Capability, create_guardian_checker,
    Contract, create_guardian_validator,
    TelemetrySnapshot, Signal, SignalKind, Domain,
    Runtime, RuntimeVar, RuntimeFunc
)
from .tpu_bridge import TPUBoostBridge, TPUPreset
from .memory_coherence_protocol import MemoryCoherenceProtocol, CoherenceState
from .cross_forex_memory_trading import CrossForexManager
from .platform_hal import BaseHAL, HALFactory
from .mavb import MemoryGrid3D  # 3D Grid Memory System


# ============================================================
# ENUMS
# ============================================================

class TPURegion(Enum):
    """TPU memory regions."""
    ON_CHIP_SRAM = "on_chip_sram"      # Fast on-chip memory
    HOST_MEMORY = "host_memory"        # System RAM
    COHERENT_MEMORY = "coherent_memory" # Coherent shared memory
    TENSOR_MEMORY = "tensor_memory"    # Specialized tensor storage


class TPUTransferType(Enum):
    """Types of memory transfers."""
    HOST_TO_TPU = "host_to_tpu"
    TPU_TO_HOST = "tpu_to_host"
    TPU_TO_TPU = "tpu_to_tpu"
    INTRA_TPU = "intra_tpu"


class TPUAccessPattern(Enum):
    """Access patterns for TPU memory."""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    STRIDED = "strided"
    TILE_BASED = "tile_based"
    PREDICTABLE = "predictable"


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class TPUAddress:
    """Represents a TPU memory address."""
    region: TPURegion
    offset: int  # Offset within region
    size: int    # Size of allocation
    tpu_id: Optional[int] = None
    coherence_state: CoherenceState = CoherenceState.INVALID


@dataclass
class TPUMemoryRequest:
    """Request for TPU memory allocation."""
    request_id: str
    agent_id: str
    region: TPURegion
    size: int  # Size in bytes
    access_pattern: TPUAccessPattern = TPUAccessPattern.RANDOM
    priority: int = Priority.NORMAL.value
    duration_ms: int = 10000  # 10 seconds default
    bid_credits: float = 10.0
    tpu_preference: Optional[int] = None
    temporal_locality: int = 0  # How many frames this data is needed
    spatial_locality: int = 0   # Related to which other data


@dataclass
class TPUMemoryAllocation:
    """Granted TPU memory allocation."""
    allocation_id: str
    request_id: str
    agent_id: str
    address: TPUAddress
    granted_size: int
    granted_at: float
    expires_at: float
    status: str = "active"
    trading_cost: float = 0.0
    transfer_cost: float = 0.0
    access_history: List[Tuple[float, int]] = field(default_factory=list)  # (timestamp, access_count)


@dataclass
class TPUMemoryMetrics:
    """Metrics for TPU memory system."""
    allocated_bytes: int = 0
    available_bytes: int = 0
    peak_utilization: float = 0.0
    avg_access_latency: float = 0.0  # in microseconds
    transfer_bandwidth_mb: float = 0.0
    coherence_hits: int = 0
    coherence_misses: int = 0
    cross_forex_volume: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TPUMemoryContext:
    """Context for TPU memory allocation decisions."""
    access_pattern: TPUAccessPattern = TPUAccessPattern.RANDOM
    performance_critical: bool = False
    temporal_locality: int = 1
    spatial_locality: int = 1
    tensor_size_hint: Optional[Tuple[int, int, int]] = None  # (height, width, channels)
    transfer_frequency: str = "low"  # low, medium, high


# ============================================================
# TPU MEMORY MANAGER
# ============================================================

class TPUMemoryManager:
    """
    Manages memory allocation and transfers for TPU accelerators.
    Integrates with 3D grid memory system and cross-forex trading.
    """

    def __init__(self, coherence_manager: MemoryCoherenceProtocol,
                 cross_forex_manager: CrossForexManager,
                 hal: BaseHAL):
        self.coherence_manager = coherence_manager
        self.cross_forex_manager = cross_forex_manager
        self.hal = hal

        # Memory pools for different TPU regions
        self.memory_pools: Dict[TPURegion, Dict] = {
            TPURegion.ON_CHIP_SRAM: {
                'total_size': self._get_on_chip_memory_size(),
                'allocated': 0,
                'allocations': {},  # allocation_id -> address
                'free_blocks': []   # List of (start, end, size)
            },
            TPURegion.HOST_MEMORY: {
                'total_size': self._get_host_memory_size(),
                'allocated': 0,
                'allocations': {},
                'free_blocks': []
            },
            TPURegion.COHERENT_MEMORY: {
                'total_size': self._get_coherent_memory_size(),
                'allocated': 0,
                'allocations': {},
                'free_blocks': []
            }
        }

        # Track memory allocations
        self.active_allocations: Dict[str, TPUMemoryAllocation] = {}
        self.allocation_history = deque(maxlen=10000)

        # Performance metrics
        self.metrics = TPUMemoryMetrics()
        self.access_times = deque(maxlen=1000)
        self.transfer_times = deque(maxlen=1000)

        # Lock for thread safety
        self.lock = threading.RLock()

        # Initialize memory pools
        self._initialize_memory_pools()

    def _get_on_chip_memory_size(self) -> int:
        """Get size of on-chip TPU memory."""
        # Default to a reasonable size, but should be detected from hardware
        return 8 * 1024 * 1024  # 8MB

    def _get_host_memory_size(self) -> int:
        """Get size of available host memory for TPU."""
        # Use a portion of system memory
        import psutil
        total_memory = psutil.virtual_memory().total
        return min(int(total_memory * 0.1), 2 * 1024 * 1024 * 1024)  # 10% or 2GB, whichever is smaller

    def _get_coherent_memory_size(self) -> int:
        """Get size of coherent memory."""
        return 512 * 1024 * 1024  # 512MB

    def _initialize_memory_pools(self):
        """Initialize memory pools with free blocks."""
        for region, pool in self.memory_pools.items():
            pool['free_blocks'].append((0, pool['total_size'], pool['total_size']))

    def request_memory(self, request: TPUMemoryRequest) -> Optional[TPUMemoryAllocation]:
        """Request memory allocation from TPU memory pools."""
        with self.lock:
            print(f"Processing TPU memory request: {request.request_id}")

            # Validate request
            if not self._validate_request(request):
                print(f"Invalid request: {request.request_id}")
                return None

            # Determine optimal region based on request characteristics
            target_region = self._determine_optimal_region(request)

            # Allocate memory in the target region
            allocation = self._allocate_in_region(target_region, request)

            if allocation:
                # Update metrics
                self.metrics.allocated_bytes += allocation.granted_size
                self.active_allocations[allocation.allocation_id] = allocation
                self.allocation_history.append(allocation)

                print(f"Memory allocation successful: {allocation.allocation_id}")
                return allocation

            print(f"Memory allocation failed for request {request.request_id}")
            return None

    def _validate_request(self, request: TPUMemoryRequest) -> bool:
        """Validate memory request."""
        if request.size <= 0:
            return False

        if request.size > self.memory_pools[request.region]['total_size']:
            return False

        if request.priority < 1 or request.priority > 10:
            return False

        return True

    def _determine_optimal_region(self, request: TPUMemoryRequest) -> TPURegion:
        """Determine optimal memory region for the request."""
        # High priority and small size requests go to on-chip SRAM
        if (request.priority >= 8 and request.size <= 1024 * 1024 and  # <= 1MB
            self._would_fit_in_on_chip(request.size)):
            return TPURegion.ON_CHIP_SRAM

        # Performance critical requests with predictable access patterns
        if (request.access_pattern in [TPUAccessPattern.SEQUENTIAL, TPUAccessPattern.PREDICTABLE] and
            request.size <= 4 * 1024 * 1024):  # <= 4MB
            return TPURegion.ON_CHIP_SRAM

        # Everything else goes to host memory
        return TPURegion.HOST_MEMORY

    def _would_fit_in_on_chip(self, size: int) -> bool:
        """Check if size would fit in on-chip memory."""
        pool = self.memory_pools[TPURegion.ON_CHIP_SRAM]
        available = pool['total_size'] - pool['allocated']
        return size <= available

    def _allocate_in_region(self, region: TPURegion, request: TPUMemoryRequest) -> Optional[TPUMemoryAllocation]:
        """Allocate memory in a specific region."""
        pool = self.memory_pools[region]

        # Find a suitable free block (first-fit algorithm)
        allocated_block = None
        for i, (start, end, size) in enumerate(pool['free_blocks']):
            if size >= request.size:
                # Allocate this block
                allocated_start = start
                allocated_size = request.size
                allocated_end = start + request.size

                # Update the free block list
                new_free_blocks = []
                if start < allocated_start:
                    new_free_blocks.append((start, allocated_start, allocated_start - start))
                if allocated_end < end:
                    new_free_blocks.append((allocated_end, end, end - allocated_end))

                # Replace the current free block with new ones
                pool['free_blocks'] = pool['free_blocks'][:i] + new_free_blocks + pool['free_blocks'][i+1:]

                # Update allocated bytes
                pool['allocated'] += allocated_size

                # Create address and allocation
                address = TPUAddress(
                    region=region,
                    offset=allocated_start,
                    size=allocated_size,
                    tpu_id=request.tpu_preference
                )

                allocation = TPUMemoryAllocation(
                    allocation_id=f"TPU_MEM_{uuid.uuid4().hex[:8]}",
                    request_id=request.request_id,
                    agent_id=request.agent_id,
                    address=address,
                    granted_size=allocated_size,
                    granted_at=time.time(),
                    expires_at=time.time() + (request.duration_ms / 1000.0),
                    trading_cost=request.bid_credits
                )

                # Register allocation
                pool['allocations'][allocation.allocation_id] = address

                # Initialize coherence state
                self.coherence_manager.initialize_address(address.offset, address.size)

                return allocation

        return None

    def free_allocation(self, allocation_id: str) -> bool:
        """Free a TPU memory allocation."""
        with self.lock:
            if allocation_id not in self.active_allocations:
                return False

            allocation = self.active_allocations[allocation_id]
            address = allocation.address

            # Find the appropriate pool
            pool = self.memory_pools[address.region]

            # Add the freed block back to free blocks (with coalescing)
            freed_start = address.offset
            freed_end = address.offset + address.size
            freed_size = address.size

            # Find where to insert the freed block
            inserted = False
            for i, (start, end, size) in enumerate(pool['free_blocks']):
                if freed_end <= start:
                    # Insert before this block
                    pool['free_blocks'].insert(i, (freed_start, freed_end, freed_size))
                    inserted = True
                    break
                elif freed_start >= end:
                    # Continue looking
                    continue
                else:
                    # This should not happen if allocation was valid
                    return False

            if not inserted:
                # Add at the end
                pool['free_blocks'].append((freed_start, freed_end, freed_size))

            # Coalesce adjacent free blocks
            pool['free_blocks'] = self._coalesce_free_blocks(pool['free_blocks'])

            # Update allocated bytes
            pool['allocated'] -= freed_size

            # Remove from active allocations
            del self.active_allocations[allocation_id]

            # Clean up coherence state
            self.coherence_manager.invalidate_address(address.offset, address.size)

            # Update metrics
            self.metrics.allocated_bytes -= freed_size

            return True

    def _coalesce_free_blocks(self, free_blocks: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """Coalesce adjacent free memory blocks."""
        if not free_blocks:
            return []

        # Sort by start address
        sorted_blocks = sorted(free_blocks)

        coalesced = [sorted_blocks[0]]
        for current_start, current_end, current_size in sorted_blocks[1:]:
            last_start, last_end, last_size = coalesced[-1]

            if last_end == current_start:
                # Adjacent blocks - coalesce them
                coalesced[-1] = (last_start, current_end, last_size + current_size)
            else:
                # Non-adjacent - add as new block
                coalesced.append((current_start, current_end, current_size))

        return coalesced

    def access_memory(self, allocation_id: str, offset: int, size: int) -> bool:
        """Simulate memory access for performance tracking."""
        with self.lock:
            if allocation_id not in self.active_allocations:
                return False

            allocation = self.active_allocations[allocation_id]
            address = allocation.address

            # Validate access bounds
            if offset + size > address.size:
                return False

            # Track access for performance analysis
            access_time = time.time()
            allocation.access_history.append((access_time, size))

            # Update access metrics
            self.access_times.append(access_time)

            # Update coherence
            success = self.coherence_manager.access_memory(
                address.offset + offset, size, CoherenceState.SHARED
            )

            return success

    def transfer_memory(self, src_allocation: str, dest_allocation: str,
                        size: int, transfer_type: TPUTransferType) -> bool:
        """Transfer data between memory allocations."""
        with self.lock:
            if src_allocation not in self.active_allocations:
                return False
            if dest_allocation not in self.active_allocations:
                return False

            start_time = time.time()

            # Simulate transfer - in a real implementation, this would copy data
            # between memory locations based on transfer type
            transfer_success = True

            # Calculate transfer time based on size and bandwidth
            estimated_time = size / (self.metrics.transfer_bandwidth_mb * 1024 * 1024)  # seconds
            actual_time = time.time() - start_time

            self.transfer_times.append(actual_time)

            # Update metrics
            if transfer_success:
                # Update coherence state for the transferred data
                src_addr = self.active_allocations[src_allocation].address
                dest_addr = self.active_allocations[dest_allocation].address

                self.coherence_manager.transfer_memory(
                    src_addr.offset, dest_addr.offset, size
                )

            return transfer_success

    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        with self.lock:
            status = {}
            for region, pool in self.memory_pools.items():
                total = pool['total_size']
                allocated = pool['allocated']
                available = total - allocated
                utilization = (allocated / total) * 100 if total > 0 else 0

                status[region.value] = {
                    'total_bytes': total,
                    'allocated_bytes': allocated,
                    'available_bytes': available,
                    'utilization_percent': utilization,
                    'free_blocks': len(pool['free_blocks']),
                    'allocations': len(pool['allocations'])
                }

            # Calculate avg metrics
            avg_access_time = sum(self.access_times) / len(self.access_times) if self.access_times else 0
            avg_transfer_time = sum(self.transfer_times) / len(self.transfer_times) if self.transfer_times else 0

            # Update metrics
            self.metrics.avg_access_latency = avg_access_time * 1_000_000  # Convert to microseconds
            self.metrics.available_bytes = sum(s['available_bytes'] for s in status.values())

            return {
                'memory_pools': status,
                'metrics': {
                    'allocated_bytes': self.metrics.allocated_bytes,
                    'available_bytes': self.metrics.available_bytes,
                    'peak_utilization': self.metrics.peak_utilization,
                    'avg_access_latency_us': self.metrics.avg_access_latency,
                    'transfer_bandwidth_mb': self.metrics.transfer_bandwidth_mb,
                    'active_allocations': len(self.active_allocations),
                    'coherence_hits': self.metrics.coherence_hits,
                    'coherence_misses': self.metrics.coherence_misses
                },
                'timestamp': time.time()
            }

    def optimize_memory_layout(self, context: TPUMemoryContext) -> bool:
        """Optimize memory layout based on access patterns."""
        with self.lock:
            # Analyze current allocations and access patterns
            # This would involve moving frequently accessed data to faster memory
            # and organizing data for optimal cache usage

            # For now, just update layout statistics
            print(f"Optimizing memory layout for {context.access_pattern.value} pattern")

            # Would implement algorithms like:
            # - Temporal proximity: place short-lived objects together
            # - Spatial proximity: place related data together
            # - Prefetching: predict upcoming accesses and load data
            # - Compression: compress less-frequently accessed data

            return True


# ============================================================
# INTEGRATION WITH 3D GRID SYSTEM
# ============================================================

class TPU3DGridMemoryAdapter:
    """
    Adapter to integrate TPU memory management with 3D grid system.
    Maps TPU memory allocations to 3D grid coordinates.
    """

    def __init__(self, memory_manager: TPUMemoryManager, grid_3d: MemoryGrid3D):
        self.memory_manager = memory_manager
        self.grid_3d = grid_3d
        self.allocation_mapping: Dict[str, Tuple[int, int, int]] = {}  # allocation_id -> (x, y, z)

    def allocate_3d(self, size: int, compute_intensity: int = 0,
                    temporal_slot: int = 0, memory_tier: int = 0) -> Optional[TPUMemoryAllocation]:
        """Allocate memory using 3D grid coordinate system."""
        # Determine optimal TPU region based on grid location
        if memory_tier <= 2:  # On-chip memory tiers
            region = TPURegion.ON_CHIP_SRAM
        else:
            region = TPURegion.HOST_MEMORY

        # Create memory request with grid context
        request = TPUMemoryRequest(
            request_id=f"GRID_{uuid.uuid4().hex[:8]}",
            agent_id="GRID_ALLOCATOR",
            region=region,
            size=size,
            priority=7,
            temporal_locality=temporal_slot,
            spatial_locality=compute_intensity
        )

        # Request allocation
        allocation = self.memory_manager.request_memory(request)

        if allocation:
            # Map to 3D coordinates
            grid_x = memory_tier  # Memory tier
            grid_y = temporal_slot  # Temporal slot
            grid_z = compute_intensity  # Compute intensity

            self.allocation_mapping[allocation.allocation_id] = (grid_x, grid_y, grid_z)

            # Register in 3D grid system
            self.grid_3d.set_signal_strength(grid_x, grid_y, grid_z, 0.5)  # Medium signal

        return allocation

    def get_3d_coordinates(self, allocation_id: str) -> Optional[Tuple[int, int, int]]:
        """Get 3D coordinates for an allocation."""
        return self.allocation_mapping.get(allocation_id)

    def optimize_grid_layout(self) -> bool:
        """Optimize memory layout based on 3D grid proximity."""
        # Analyze 3D grid to optimize memory placement
        # Move related data to nearby memory locations
        # This would implement spatial locality optimization

        return True


# ============================================================
# DEMO
# ============================================================

def demo_tpu_memory_management():
    """Demonstrate TPU memory management system."""
    print("=== GAMESA TPU Memory Management Demo ===\n")

    # Create dependencies
    from .memory_coherence_protocol import MemoryCoherenceProtocol
    from .cross_forex_memory_trading import CrossForexManager
    from .platform_hal import HALFactory
    from .mavb import MemoryGrid3D

    coherence_manager = MemoryCoherenceProtocol()
    cross_forex_manager = CrossForexManager()
    hal = HALFactory.create()
    grid_3d = MemoryGrid3D()  # Use default 16x16x16 grid

    # Initialize TPU memory manager
    memory_manager = TPUMemoryManager(coherence_manager, cross_forex_manager, hal)
    print("TPU Memory Manager initialized")

    # Initialize 3D grid adapter
    grid_adapter = TPU3DGridMemoryAdapter(memory_manager, grid_3d)
    print("3D Grid Memory Adapter initialized")

    # Create memory requests
    requests = [
        TPUMemoryRequest(
            request_id=f"REQ_{i:03d}",
            agent_id="TEST_APP",
            region=TPURegion.HOST_MEMORY,
            size=1024 * 1024 * 2,  # 2MB
            priority=8 if i % 2 == 0 else 5,  # Alternating high/medium priority
            access_pattern=TPUAccessPattern.SEQUENTIAL if i % 3 == 0 else TPUAccessPattern.RANDOM
        )
        for i in range(5)
    ]

    # Process memory requests
    allocations = []
    print("\nProcessing memory allocation requests...")
    for req in requests:
        alloc = memory_manager.request_memory(req)
        if alloc:
            allocations.append(alloc)
            print(f"  Allocated {alloc.granted_size:,} bytes -> {alloc.allocation_id}")
        else:
            print(f"  FAILED to allocate for {req.request_id}")

    print(f"\nTotal allocations: {len(allocations)}")

    # Test memory access
    if allocations:
        print(f"\nTesting memory access for first allocation...")
        success = memory_manager.access_memory(allocations[0].allocation_id, 0, 1024)
        print(f"  Access success: {success}")

    # Test 3D grid allocation
    print(f"\nTesting 3D grid memory allocation...")
    grid_alloc = grid_adapter.allocate_3d(
        size=512 * 1024,  # 512KB
        compute_intensity=5,
        temporal_slot=10,
        memory_tier=1  # On-chip tier
    )
    if grid_alloc:
        coords = grid_adapter.get_3d_coordinates(grid_alloc.allocation_id)
        print(f"  3D Grid allocation: {coords} -> {grid_alloc.allocation_id}")

    # Show memory status
    print(f"\nMemory Status:")
    status = memory_manager.get_memory_status()
    for region, info in status['memory_pools'].items():
        print(f"  {region}: {info['utilization_percent']:.1f}% used "
              f"({info['allocated_bytes']:,}/{info['total_bytes']:,} bytes)")

    print(f"\nMetrics:")
    metrics = status['metrics']
    print(f"  Active allocations: {metrics['active_allocations']}")
    print(f"  Avg access latency: {metrics['avg_access_latency_us']:.2f} μs")

    # Test memory freeing
    if allocations:
        print(f"\nFreeing first allocation...")
        success = memory_manager.free_allocation(allocations[0].allocation_id)
        print(f"  Free success: {success}")

        # Check status after freeing
        status_after = memory_manager.get_memory_status()
        print(f"  Active allocations after free: {status_after['metrics']['active_allocations']}")

    print(f"\nTPU Memory Management demo completed successfully!")


if __name__ == "__main__":
    demo_tpu_memory_management()