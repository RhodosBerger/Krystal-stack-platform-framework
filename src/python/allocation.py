"""
Resource Allocation Schema - Best Practices Implementation

Implements allocation strategies for CPU, GPU, memory, and budget distribution
following established patterns: pool allocation, slab allocation, buddy system
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import heapq


class AllocationStrategy(Enum):
    """Allocation strategy types."""
    FIRST_FIT = "first_fit"      # Allocate first available block
    BEST_FIT = "best_fit"        # Allocate smallest sufficient block
    WORST_FIT = "worst_fit"      # Allocate largest available block
    POOL = "pool"                # Fixed-size block allocation
    SLAB = "slab"                # Object-cached allocation
    BUDDY = "buddy"              # Power-of-two block splitting


class ResourceType(Enum):
    """Resource type being allocated."""
    CPU_CORES = "cpu_cores"
    CPU_TIME = "cpu_time"
    GPU_COMPUTE = "gpu_compute"
    GPU_MEMORY = "gpu_memory"
    SYSTEM_MEMORY = "system_memory"
    BANDWIDTH = "bandwidth"
    POWER_BUDGET = "power_budget"
    THERMAL_HEADROOM = "thermal_headroom"


class Priority(Enum):
    """Allocation priority levels."""
    CRITICAL = 0    # System-critical, cannot be preempted
    HIGH = 1        # High priority, preempts normal
    NORMAL = 2      # Default priority
    LOW = 3         # Background tasks
    BEST_EFFORT = 4 # Allocate if available


@dataclass
class AllocationConstraints:
    """Constraints for allocation."""
    min_amount: Optional[int] = None
    max_amount: Optional[int] = None
    alignment: Optional[int] = None
    affinity: Optional[List[int]] = None        # Preferred cores/nodes
    anti_affinity: Optional[List[str]] = None   # Don't colocate with these
    timeout_ms: Optional[int] = None
    preemptible: bool = True


@dataclass
class AllocationRequest:
    """Allocation request."""
    id: str
    resource_type: ResourceType
    amount: int
    priority: Priority = Priority.NORMAL
    constraints: AllocationConstraints = field(default_factory=AllocationConstraints)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Allocation:
    """Allocation result."""
    id: str
    request_id: str
    resource_type: ResourceType
    amount: int
    block_id: int
    offset: int
    timestamp: float
    expires_at: Optional[float] = None


@dataclass
class Block:
    """Block in the allocation pool."""
    id: int
    offset: int
    size: int
    allocated: bool = False
    allocation_id: Optional[str] = None
    priority: Priority = Priority.BEST_EFFORT


@dataclass
class PoolStats:
    """Pool statistics."""
    total_capacity: int
    allocated: int
    available: int
    utilization: float
    fragmentation: float
    block_count: int
    free_block_count: int
    allocation_count: int
    largest_free_block: int


class AllocationError(Exception):
    """Allocation error base class."""
    pass


class InsufficientResourcesError(AllocationError):
    pass


class NotFoundError(AllocationError):
    pass


class InvalidConstraintsError(AllocationError):
    pass


class ResourcePool:
    """Resource pool for a single resource type."""

    def __init__(
        self,
        resource_type: ResourceType,
        capacity: int,
        strategy: AllocationStrategy = AllocationStrategy.BEST_FIT,
    ):
        self.resource_type = resource_type
        self.total_capacity = capacity
        self.strategy = strategy
        self.blocks: OrderedDict[int, Block] = OrderedDict()  # offset -> block
        self.allocations: Dict[str, Allocation] = {}
        self._next_block_id = 1
        self._lock = threading.RLock()

        # Initialize with single free block
        self.blocks[0] = Block(id=0, offset=0, size=capacity)

    def allocate(self, request: AllocationRequest) -> Allocation:
        """Allocate resources."""
        with self._lock:
            amount = self._apply_constraints(request.amount, request.constraints)

            # Find suitable block based on strategy
            block_offset = self._find_block(amount)
            if block_offset is None:
                raise InsufficientResourcesError(
                    f"Cannot allocate {amount} of {self.resource_type.value}"
                )

            block = self.blocks[block_offset]

            # Split block if necessary
            if block.size > amount:
                self._split_block(block_offset, amount)

            # Update block
            block = self.blocks[block_offset]
            allocation_id = f"alloc-{request.id}-{self._next_block_id}"
            block.allocated = True
            block.allocation_id = allocation_id
            block.priority = request.priority

            # Create allocation record
            now = time.time()
            expires_at = None
            if request.constraints.timeout_ms:
                expires_at = now + request.constraints.timeout_ms / 1000.0

            allocation = Allocation(
                id=allocation_id,
                request_id=request.id,
                resource_type=self.resource_type,
                amount=amount,
                block_id=block.id,
                offset=block_offset,
                timestamp=now,
                expires_at=expires_at,
            )

            self.allocations[allocation_id] = allocation
            self._next_block_id += 1

            return allocation

    def free(self, allocation_id: str) -> None:
        """Free an allocation."""
        with self._lock:
            allocation = self.allocations.pop(allocation_id, None)
            if not allocation:
                raise NotFoundError(f"Allocation {allocation_id} not found")

            block = self.blocks.get(allocation.offset)
            if block:
                block.allocated = False
                block.allocation_id = None
                self._coalesce_blocks(allocation.offset)

    def stats(self) -> PoolStats:
        """Get pool statistics."""
        with self._lock:
            allocated = sum(b.size for b in self.blocks.values() if b.allocated)
            free_blocks = [b for b in self.blocks.values() if not b.allocated]
            largest_free = max((b.size for b in free_blocks), default=0)

            return PoolStats(
                total_capacity=self.total_capacity,
                allocated=allocated,
                available=self.total_capacity - allocated,
                utilization=allocated / self.total_capacity if self.total_capacity > 0 else 0,
                fragmentation=self._calculate_fragmentation(),
                block_count=len(self.blocks),
                free_block_count=len(free_blocks),
                allocation_count=len(self.allocations),
                largest_free_block=largest_free,
            )

    def _apply_constraints(self, amount: int, constraints: AllocationConstraints) -> int:
        """Apply constraints to requested amount."""
        result = amount

        if constraints.min_amount:
            result = max(result, constraints.min_amount)
        if constraints.max_amount:
            result = min(result, constraints.max_amount)
        if constraints.alignment:
            result = ((result + constraints.alignment - 1) // constraints.alignment) * constraints.alignment

        return result

    def _find_block(self, size: int) -> Optional[int]:
        """Find a suitable block based on strategy."""
        free_blocks = [(o, b) for o, b in self.blocks.items() if not b.allocated and b.size >= size]

        if not free_blocks:
            return None

        if self.strategy == AllocationStrategy.FIRST_FIT:
            return free_blocks[0][0]
        elif self.strategy == AllocationStrategy.BEST_FIT:
            return min(free_blocks, key=lambda x: x[1].size)[0]
        elif self.strategy == AllocationStrategy.WORST_FIT:
            return max(free_blocks, key=lambda x: x[1].size)[0]
        elif self.strategy == AllocationStrategy.BUDDY:
            # Round up to power of two
            size = 1 << (size - 1).bit_length()
            suitable = [(o, b) for o, b in free_blocks if b.size >= size]
            return min(suitable, key=lambda x: x[1].size)[0] if suitable else None
        else:
            return free_blocks[0][0]

    def _split_block(self, offset: int, size: int) -> None:
        """Split a block into allocated and free portions."""
        block = self.blocks[offset]
        remaining = block.size - size

        # Resize original block
        block.size = size

        # Create new free block
        new_offset = offset + size
        self.blocks[new_offset] = Block(
            id=self._next_block_id,
            offset=new_offset,
            size=remaining,
        )
        self._next_block_id += 1

        # Re-sort blocks by offset
        self.blocks = OrderedDict(sorted(self.blocks.items()))

    def _coalesce_blocks(self, offset: int) -> None:
        """Coalesce adjacent free blocks."""
        offsets = list(self.blocks.keys())
        idx = offsets.index(offset)

        # Try to merge with next block
        if idx + 1 < len(offsets):
            next_offset = offsets[idx + 1]
            next_block = self.blocks[next_offset]
            current_block = self.blocks[offset]

            if not next_block.allocated and current_block.offset + current_block.size == next_offset:
                current_block.size += next_block.size
                del self.blocks[next_offset]
                offsets = list(self.blocks.keys())
                idx = offsets.index(offset)

        # Try to merge with previous block
        if idx > 0:
            prev_offset = offsets[idx - 1]
            prev_block = self.blocks[prev_offset]
            current_block = self.blocks[offset]

            if not prev_block.allocated and prev_block.offset + prev_block.size == offset:
                prev_block.size += current_block.size
                del self.blocks[offset]

    def _calculate_fragmentation(self) -> float:
        """Calculate memory fragmentation."""
        free_blocks = [b for b in self.blocks.values() if not b.allocated]
        if not free_blocks:
            return 0.0

        total_free = sum(b.size for b in free_blocks)
        largest_free = max(b.size for b in free_blocks)

        if total_free == 0:
            return 0.0

        return 1.0 - (largest_free / total_free)


class Allocator:
    """Multi-resource allocator."""

    def __init__(self):
        self.pools: Dict[ResourceType, ResourcePool] = {}
        self.allocation_log: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    def init_pool(
        self,
        resource_type: ResourceType,
        capacity: int,
        strategy: AllocationStrategy = AllocationStrategy.BEST_FIT,
    ) -> None:
        """Initialize a resource pool."""
        with self._lock:
            self.pools[resource_type] = ResourcePool(resource_type, capacity, strategy)

    def allocate(self, request: AllocationRequest) -> Allocation:
        """Allocate from a specific pool."""
        with self._lock:
            pool = self.pools.get(request.resource_type)
            if not pool:
                raise InvalidConstraintsError(f"No pool for {request.resource_type.value}")

            allocation = pool.allocate(request)

            self.allocation_log.append({
                "timestamp": time.time(),
                "event": "allocated",
                "allocation_id": allocation.id,
                "resource_type": request.resource_type.value,
                "amount": allocation.amount,
            })

            return allocation

    def free(self, resource_type: ResourceType, allocation_id: str) -> None:
        """Free an allocation."""
        with self._lock:
            pool = self.pools.get(resource_type)
            if not pool:
                raise NotFoundError(f"No pool for {resource_type.value}")

            pool.free(allocation_id)

            self.allocation_log.append({
                "timestamp": time.time(),
                "event": "freed",
                "allocation_id": allocation_id,
                "resource_type": resource_type.value,
            })

    def stats(self) -> Dict[ResourceType, PoolStats]:
        """Get stats for all pools."""
        with self._lock:
            return {rt: pool.stats() for rt, pool in self.pools.items()}

    def cleanup_expired(self) -> int:
        """Cleanup expired allocations. Returns count of cleaned allocations."""
        count = 0
        now = time.time()

        with self._lock:
            for resource_type, pool in self.pools.items():
                expired = [
                    alloc_id
                    for alloc_id, alloc in pool.allocations.items()
                    if alloc.expires_at and alloc.expires_at < now
                ]

                for alloc_id in expired:
                    try:
                        pool.free(alloc_id)
                        self.allocation_log.append({
                            "timestamp": now,
                            "event": "expired",
                            "allocation_id": alloc_id,
                            "resource_type": resource_type.value,
                        })
                        count += 1
                    except AllocationError:
                        pass

        return count

    def get_log(self) -> List[Dict[str, Any]]:
        """Get allocation log."""
        return self.allocation_log.copy()


# Convenience functions
def create_default_allocator() -> Allocator:
    """Create an allocator with default pools."""
    allocator = Allocator()
    allocator.init_pool(ResourceType.CPU_CORES, 16, AllocationStrategy.FIRST_FIT)
    allocator.init_pool(ResourceType.CPU_TIME, 100000, AllocationStrategy.BEST_FIT)
    allocator.init_pool(ResourceType.GPU_MEMORY, 8192, AllocationStrategy.BUDDY)
    allocator.init_pool(ResourceType.SYSTEM_MEMORY, 32768, AllocationStrategy.BEST_FIT)
    allocator.init_pool(ResourceType.POWER_BUDGET, 100, AllocationStrategy.FIRST_FIT)
    return allocator


if __name__ == "__main__":
    # Demo
    allocator = create_default_allocator()

    # Allocate CPU cores
    cpu_req = AllocationRequest(
        id="task-1",
        resource_type=ResourceType.CPU_CORES,
        amount=4,
        priority=Priority.HIGH,
    )
    cpu_alloc = allocator.allocate(cpu_req)
    print(f"Allocated CPU: {cpu_alloc}")

    # Allocate memory
    mem_req = AllocationRequest(
        id="task-1",
        resource_type=ResourceType.SYSTEM_MEMORY,
        amount=4096,
        priority=Priority.NORMAL,
        constraints=AllocationConstraints(alignment=64),
    )
    mem_alloc = allocator.allocate(mem_req)
    print(f"Allocated Memory: {mem_alloc}")

    # Print stats
    print("\nPool Statistics:")
    for rt, stats in allocator.stats().items():
        print(f"  {rt.value}: {stats.allocated}/{stats.total_capacity} "
              f"({stats.utilization*100:.1f}% util, {stats.fragmentation*100:.1f}% frag)")

    # Free allocations
    allocator.free(ResourceType.CPU_CORES, cpu_alloc.id)
    allocator.free(ResourceType.SYSTEM_MEMORY, mem_alloc.id)

    print("\nAfter freeing:")
    for rt, stats in allocator.stats().items():
        print(f"  {rt.value}: {stats.allocated}/{stats.total_capacity}")
