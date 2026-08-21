"""
GAMESA Advanced Allocation - Sophisticated Resource Management

Branched allocation systems for efficient yet sophisticated resource management:
1. HierarchicalAllocator - Multi-level resource pools
2. PredictiveAllocator - Anticipate needs before demand
3. ElasticAllocator - Dynamic scaling with backpressure
4. FairShareAllocator - Weighted fair queuing
5. SpeculativeAllocator - Pre-allocate based on patterns
6. CoalescingAllocator - Batch small allocations
"""

import time
import math
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque, defaultdict
from enum import Enum
from abc import ABC, abstractmethod


# =============================================================================
# BASE ALLOCATION TYPES
# =============================================================================

class ResourceType(Enum):
    """Types of allocatable resources."""
    CPU_CYCLES = "cpu_cycles"
    GPU_COMPUTE = "gpu_compute"
    MEMORY_MB = "memory_mb"
    THERMAL_BUDGET = "thermal_budget"
    POWER_WATTS = "power_watts"
    BANDWIDTH_MBPS = "bandwidth_mbps"
    LATENCY_BUDGET_MS = "latency_budget_ms"


class AllocationPriority(Enum):
    """Allocation priority levels."""
    CRITICAL = 0    # Safety-critical, always allocate
    REALTIME = 1    # Real-time requirements
    HIGH = 2        # High priority tasks
    NORMAL = 3      # Normal operations
    LOW = 4         # Background tasks
    IDLE = 5        # Only when excess capacity


@dataclass
class AllocationRequest:
    """Request for resource allocation."""
    request_id: str
    resource_type: ResourceType
    amount: float
    priority: AllocationPriority
    requester: str
    deadline_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Allocation:
    """Granted allocation."""
    allocation_id: str
    request: AllocationRequest
    granted_amount: float
    granted_at: float
    expires_at: Optional[float] = None
    renewable: bool = True


@dataclass
class ResourcePool:
    """Pool of allocatable resources."""
    resource_type: ResourceType
    total_capacity: float
    allocated: float = 0.0
    reserved: float = 0.0  # Reserved for high-priority

    @property
    def available(self) -> float:
        return self.total_capacity - self.allocated - self.reserved

    @property
    def utilization(self) -> float:
        return self.allocated / self.total_capacity if self.total_capacity > 0 else 0


# =============================================================================
# 1. HIERARCHICAL ALLOCATOR
# =============================================================================

class HierarchicalAllocator:
    """
    Multi-level resource pools with overflow handling.

    Levels:
    - L0: Critical reserves (always available)
    - L1: Real-time pool (guaranteed latency)
    - L2: Shared pool (fair distribution)
    - L3: Burst pool (temporary overflow)
    """

    def __init__(self):
        self.levels: Dict[int, Dict[ResourceType, ResourcePool]] = {
            0: {},  # Critical
            1: {},  # Real-time
            2: {},  # Shared
            3: {},  # Burst
        }
        self.allocations: Dict[str, Allocation] = {}
        self.allocation_counter = 0

        # Initialize default pools
        self._init_default_pools()

    def _init_default_pools(self):
        """Initialize default resource pools."""
        defaults = {
            ResourceType.CPU_CYCLES: (10, 30, 50, 10),      # % per level
            ResourceType.GPU_COMPUTE: (5, 25, 60, 10),
            ResourceType.MEMORY_MB: (256, 512, 1024, 256),
            ResourceType.THERMAL_BUDGET: (5, 10, 10, 5),    # °C
            ResourceType.POWER_WATTS: (5, 8, 12, 3),
        }

        for res_type, capacities in defaults.items():
            for level, capacity in enumerate(capacities):
                self.levels[level][res_type] = ResourcePool(
                    resource_type=res_type,
                    total_capacity=capacity,
                )

    def allocate(self, request: AllocationRequest) -> Optional[Allocation]:
        """Allocate from appropriate level based on priority."""
        # Map priority to starting level
        priority_level = {
            AllocationPriority.CRITICAL: 0,
            AllocationPriority.REALTIME: 1,
            AllocationPriority.HIGH: 2,
            AllocationPriority.NORMAL: 2,
            AllocationPriority.LOW: 2,
            AllocationPriority.IDLE: 3,
        }.get(request.priority, 2)

        # Try to allocate from appropriate level, overflow to higher levels
        for level in range(priority_level, 4):
            pool = self.levels[level].get(request.resource_type)
            if pool and pool.available >= request.amount:
                return self._grant_allocation(request, pool, level)

        # Try borrowing from lower priority levels
        for level in range(priority_level - 1, -1, -1):
            pool = self.levels[level].get(request.resource_type)
            if pool and pool.available >= request.amount:
                return self._grant_allocation(request, pool, level)

        return None  # Cannot satisfy

    def _grant_allocation(self, request: AllocationRequest, pool: ResourcePool, level: int) -> Allocation:
        """Grant allocation from pool."""
        self.allocation_counter += 1
        allocation = Allocation(
            allocation_id=f"alloc_{self.allocation_counter}",
            request=request,
            granted_amount=request.amount,
            granted_at=time.time(),
            expires_at=time.time() + 60 if level == 3 else None,  # Burst expires
        )

        pool.allocated += request.amount
        self.allocations[allocation.allocation_id] = allocation

        return allocation

    def release(self, allocation_id: str):
        """Release allocation."""
        if allocation_id in self.allocations:
            alloc = self.allocations[allocation_id]
            # Find and release from pool
            for level in self.levels.values():
                pool = level.get(alloc.request.resource_type)
                if pool:
                    pool.allocated = max(0, pool.allocated - alloc.granted_amount)
                    break
            del self.allocations[allocation_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get allocation statistics."""
        stats = {}
        for level, pools in self.levels.items():
            stats[f"L{level}"] = {
                res.value: {"util": pool.utilization, "avail": pool.available}
                for res, pool in pools.items()
            }
        return stats


# =============================================================================
# 2. PREDICTIVE ALLOCATOR
# =============================================================================

class PredictiveAllocator:
    """
    Anticipate resource needs before demand spikes.

    Uses exponential smoothing + pattern recognition.
    """

    def __init__(self):
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.predictions: Dict[str, float] = {}
        self.pre_allocations: Dict[str, float] = {}

        # Smoothing parameters
        self.alpha = 0.3  # Level smoothing
        self.beta = 0.1   # Trend smoothing
        self.gamma = 0.2  # Seasonal smoothing

        # State
        self.smoothed: Dict[str, float] = {}
        self.trend: Dict[str, float] = {}

    def observe(self, resource_type: str, demand: float):
        """Observe demand for prediction."""
        self.history[resource_type].append({
            "timestamp": time.time(),
            "demand": demand,
        })

        # Update Holt-Winters smoothing
        if resource_type not in self.smoothed:
            self.smoothed[resource_type] = demand
            self.trend[resource_type] = 0
        else:
            prev_smooth = self.smoothed[resource_type]
            self.smoothed[resource_type] = (
                self.alpha * demand +
                (1 - self.alpha) * (prev_smooth + self.trend[resource_type])
            )
            self.trend[resource_type] = (
                self.beta * (self.smoothed[resource_type] - prev_smooth) +
                (1 - self.beta) * self.trend[resource_type]
            )

    def predict(self, resource_type: str, horizon_steps: int = 5) -> float:
        """Predict future demand."""
        if resource_type not in self.smoothed:
            return 0.0

        # Holt-Winters prediction
        prediction = (
            self.smoothed[resource_type] +
            horizon_steps * self.trend[resource_type]
        )

        # Add safety margin based on variance
        history = self.history[resource_type]
        if len(history) > 10:
            demands = [h["demand"] for h in history]
            variance = sum((d - sum(demands)/len(demands))**2 for d in demands) / len(demands)
            std = variance ** 0.5
            prediction += std * 0.5  # Add half std as margin

        self.predictions[resource_type] = max(0, prediction)
        return self.predictions[resource_type]

    def should_pre_allocate(self, resource_type: str, current_allocation: float) -> Tuple[bool, float]:
        """Determine if pre-allocation needed."""
        predicted = self.predict(resource_type)

        if predicted > current_allocation * 1.2:  # 20% above current
            pre_alloc_amount = predicted - current_allocation
            return True, pre_alloc_amount

        return False, 0.0

    def get_all_predictions(self) -> Dict[str, float]:
        """Get predictions for all tracked resources."""
        return {
            res: self.predict(res)
            for res in self.history.keys()
        }


# =============================================================================
# 3. ELASTIC ALLOCATOR
# =============================================================================

class ElasticAllocator:
    """
    Dynamic scaling with backpressure handling.

    Scales allocations up/down based on demand and capacity.
    """

    def __init__(self, min_scale: float = 0.5, max_scale: float = 2.0):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.current_scale: Dict[str, float] = {}
        self.base_allocations: Dict[str, float] = {}
        self.backpressure: Dict[str, float] = {}  # 0-1, 1 = max pressure

        # Scaling parameters
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.scale_step = 0.1

    def set_base(self, resource_type: str, base_amount: float):
        """Set base allocation amount."""
        self.base_allocations[resource_type] = base_amount
        self.current_scale[resource_type] = 1.0
        self.backpressure[resource_type] = 0.0

    def update_pressure(self, resource_type: str, utilization: float, queue_depth: int = 0):
        """Update backpressure based on utilization and queue."""
        # Combine utilization and queue depth into pressure metric
        queue_pressure = min(1.0, queue_depth / 10)
        pressure = 0.7 * utilization + 0.3 * queue_pressure

        # Exponential moving average
        current = self.backpressure.get(resource_type, 0)
        self.backpressure[resource_type] = 0.8 * current + 0.2 * pressure

    def get_elastic_allocation(self, resource_type: str) -> float:
        """Get current elastic allocation amount."""
        base = self.base_allocations.get(resource_type, 0)
        scale = self.current_scale.get(resource_type, 1.0)
        return base * scale

    def adjust_scale(self, resource_type: str) -> float:
        """Adjust scale based on backpressure."""
        pressure = self.backpressure.get(resource_type, 0)
        current = self.current_scale.get(resource_type, 1.0)

        if pressure > self.scale_up_threshold:
            # Scale up
            new_scale = min(self.max_scale, current + self.scale_step)
        elif pressure < self.scale_down_threshold:
            # Scale down
            new_scale = max(self.min_scale, current - self.scale_step)
        else:
            new_scale = current

        self.current_scale[resource_type] = new_scale
        return new_scale

    def apply_backpressure(self, resource_type: str) -> Dict[str, Any]:
        """Apply backpressure controls."""
        pressure = self.backpressure.get(resource_type, 0)

        actions = {
            "pressure": pressure,
            "scale": self.current_scale.get(resource_type, 1.0),
        }

        if pressure > 0.9:
            actions["action"] = "reject_new"
            actions["throttle"] = 0.5
        elif pressure > 0.7:
            actions["action"] = "queue_new"
            actions["throttle"] = 0.8
        else:
            actions["action"] = "accept"
            actions["throttle"] = 1.0

        return actions


# =============================================================================
# 4. FAIR SHARE ALLOCATOR
# =============================================================================

@dataclass
class FairShareClient:
    """Client with fair share allocation."""
    client_id: str
    weight: float = 1.0
    usage: float = 0.0
    allocation: float = 0.0
    deficit: float = 0.0


class FairShareAllocator:
    """
    Weighted fair queuing for multi-tenant allocation.

    Implements max-min fairness with weights.
    """

    def __init__(self, total_capacity: float):
        self.total_capacity = total_capacity
        self.clients: Dict[str, FairShareClient] = {}
        self.allocation_round = 0

    def register_client(self, client_id: str, weight: float = 1.0):
        """Register client for fair share."""
        self.clients[client_id] = FairShareClient(
            client_id=client_id,
            weight=weight,
        )

    def compute_fair_shares(self) -> Dict[str, float]:
        """Compute fair share for each client."""
        if not self.clients:
            return {}

        total_weight = sum(c.weight for c in self.clients.values())
        shares = {}

        for client_id, client in self.clients.items():
            # Base fair share
            fair_share = (client.weight / total_weight) * self.total_capacity

            # Adjust for deficit (clients who got less before get more now)
            adjusted_share = fair_share + client.deficit * 0.1

            shares[client_id] = min(adjusted_share, self.total_capacity * 0.5)  # Cap at 50%
            client.allocation = shares[client_id]

        return shares

    def request(self, client_id: str, amount: float) -> float:
        """Request allocation, returns granted amount."""
        if client_id not in self.clients:
            self.register_client(client_id)

        client = self.clients[client_id]
        fair_share = self.compute_fair_shares().get(client_id, 0)

        # Grant up to fair share
        granted = min(amount, fair_share - client.usage)
        granted = max(0, granted)

        client.usage += granted

        # Track deficit
        if granted < amount:
            client.deficit += amount - granted

        return granted

    def release(self, client_id: str, amount: float):
        """Release allocation."""
        if client_id in self.clients:
            self.clients[client_id].usage = max(0, self.clients[client_id].usage - amount)

    def reset_round(self):
        """Reset for new allocation round."""
        self.allocation_round += 1
        for client in self.clients.values():
            # Decay deficit
            client.deficit *= 0.9
            client.usage = 0


# =============================================================================
# 5. SPECULATIVE ALLOCATOR
# =============================================================================

class SpeculativeAllocator:
    """
    Pre-allocate based on observed patterns.

    Learns allocation patterns and speculatively reserves resources.
    """

    def __init__(self):
        self.patterns: Dict[str, List[Dict]] = defaultdict(list)
        self.speculative_reserves: Dict[str, float] = {}
        self.hit_rate: Dict[str, float] = {}
        self.pattern_confidence: Dict[str, float] = {}

    def record_pattern(self, context: str, resource_type: str, amount: float):
        """Record allocation pattern for context."""
        self.patterns[context].append({
            "timestamp": time.time(),
            "resource": resource_type,
            "amount": amount,
        })

        # Keep only recent patterns
        if len(self.patterns[context]) > 100:
            self.patterns[context] = self.patterns[context][-100:]

    def predict_for_context(self, context: str) -> Dict[str, float]:
        """Predict allocations needed for context."""
        if context not in self.patterns or len(self.patterns[context]) < 5:
            return {}

        # Aggregate by resource type
        resource_amounts: Dict[str, List[float]] = defaultdict(list)
        for pattern in self.patterns[context]:
            resource_amounts[pattern["resource"]].append(pattern["amount"])

        predictions = {}
        for resource, amounts in resource_amounts.items():
            # Use 75th percentile as prediction
            sorted_amounts = sorted(amounts)
            idx = int(len(sorted_amounts) * 0.75)
            predictions[resource] = sorted_amounts[idx]

            # Compute confidence
            mean = sum(amounts) / len(amounts)
            variance = sum((a - mean)**2 for a in amounts) / len(amounts)
            cv = (variance ** 0.5) / mean if mean > 0 else 1
            self.pattern_confidence[f"{context}:{resource}"] = max(0, 1 - cv)

        return predictions

    def speculate(self, context: str) -> Dict[str, float]:
        """Speculatively reserve resources for context."""
        predictions = self.predict_for_context(context)

        for resource, amount in predictions.items():
            confidence = self.pattern_confidence.get(f"{context}:{resource}", 0.5)
            # Only speculate if confident
            if confidence > 0.6:
                self.speculative_reserves[f"{context}:{resource}"] = amount * confidence

        return self.speculative_reserves

    def claim_speculative(self, context: str, resource: str) -> Optional[float]:
        """Claim speculative reservation."""
        key = f"{context}:{resource}"
        if key in self.speculative_reserves:
            amount = self.speculative_reserves.pop(key)
            # Record hit
            self.hit_rate[key] = self.hit_rate.get(key, 0.5) * 0.9 + 0.1
            return amount
        else:
            # Record miss
            self.hit_rate.setdefault(key, 0.5)
            self.hit_rate[key] = self.hit_rate[key] * 0.9
            return None


# =============================================================================
# 6. COALESCING ALLOCATOR
# =============================================================================

@dataclass
class PendingAllocation:
    """Pending allocation waiting to be coalesced."""
    request: AllocationRequest
    queued_at: float


class CoalescingAllocator:
    """
    Batch small allocations for efficiency.

    Reduces allocation overhead by coalescing similar requests.
    """

    def __init__(self, batch_size: int = 10, batch_timeout_ms: float = 50):
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.pending: Dict[ResourceType, List[PendingAllocation]] = defaultdict(list)
        self.allocation_counter = 0

    def queue(self, request: AllocationRequest) -> str:
        """Queue request for batching."""
        pending = PendingAllocation(request=request, queued_at=time.time())
        self.pending[request.resource_type].append(pending)

        # Check if should flush
        if self._should_flush(request.resource_type):
            return self._flush(request.resource_type)

        return "queued"

    def _should_flush(self, resource_type: ResourceType) -> bool:
        """Check if batch should be flushed."""
        pending = self.pending[resource_type]

        if len(pending) >= self.batch_size:
            return True

        if pending:
            oldest = pending[0].queued_at
            if (time.time() - oldest) * 1000 >= self.batch_timeout_ms:
                return True

        return False

    def _flush(self, resource_type: ResourceType) -> str:
        """Flush and coalesce pending allocations."""
        pending = self.pending[resource_type]
        if not pending:
            return "empty"

        # Coalesce by priority
        by_priority: Dict[AllocationPriority, List[PendingAllocation]] = defaultdict(list)
        for p in pending:
            by_priority[p.request.priority].append(p)

        # Allocate coalesced batches
        results = []
        for priority in sorted(by_priority.keys(), key=lambda x: x.value):
            batch = by_priority[priority]
            total_amount = sum(p.request.amount for p in batch)

            self.allocation_counter += 1
            results.append({
                "batch_id": f"batch_{self.allocation_counter}",
                "priority": priority.name,
                "count": len(batch),
                "total_amount": total_amount,
            })

        # Clear pending
        self.pending[resource_type] = []

        return f"flushed:{len(results)}_batches"

    def flush_all(self) -> Dict[ResourceType, str]:
        """Flush all pending batches."""
        results = {}
        for resource_type in list(self.pending.keys()):
            results[resource_type] = self._flush(resource_type)
        return results


# =============================================================================
# UNIFIED ALLOCATION SYSTEM
# =============================================================================

class UnifiedAllocationSystem:
    """
    Combines all allocators into sophisticated allocation system.
    """

    def __init__(self):
        self.hierarchical = HierarchicalAllocator()
        self.predictive = PredictiveAllocator()
        self.elastic = ElasticAllocator()
        self.fair_share = FairShareAllocator(total_capacity=100)
        self.speculative = SpeculativeAllocator()
        self.coalescing = CoalescingAllocator()

        # Initialize elastic bases
        for res in ResourceType:
            self.elastic.set_base(res.value, 10)

    def allocate(self, request: AllocationRequest, context: str = "default") -> Dict[str, Any]:
        """
        Sophisticated allocation combining all strategies.
        """
        result = {
            "request_id": request.request_id,
            "resource": request.resource_type.value,
            "requested": request.amount,
        }

        # 1. Check speculative reserves first
        speculative = self.speculative.claim_speculative(context, request.resource_type.value)
        if speculative and speculative >= request.amount:
            result["allocation"] = speculative
            result["source"] = "speculative"
            return result

        # 2. Try fair share for normal priority
        if request.priority in (AllocationPriority.NORMAL, AllocationPriority.LOW):
            granted = self.fair_share.request(request.requester, request.amount)
            if granted >= request.amount * 0.8:
                result["allocation"] = granted
                result["source"] = "fair_share"
                return result

        # 3. Check elastic scaling
        elastic_amount = self.elastic.get_elastic_allocation(request.resource_type.value)
        backpressure = self.elastic.apply_backpressure(request.resource_type.value)

        if backpressure["action"] == "reject_new" and request.priority.value > 1:
            result["allocation"] = 0
            result["source"] = "rejected_backpressure"
            result["backpressure"] = backpressure
            return result

        # 4. Hierarchical allocation
        allocation = self.hierarchical.allocate(request)
        if allocation:
            result["allocation"] = allocation.granted_amount
            result["allocation_id"] = allocation.allocation_id
            result["source"] = "hierarchical"

            # Record for prediction
            self.predictive.observe(request.resource_type.value, request.amount)
            self.speculative.record_pattern(context, request.resource_type.value, request.amount)

            return result

        # 5. Queue for coalescing if small request
        if request.amount < 5:
            self.coalescing.queue(request)
            result["allocation"] = 0
            result["source"] = "queued_coalesce"
            return result

        result["allocation"] = 0
        result["source"] = "failed"
        return result

    def update_telemetry(self, telemetry: Dict[str, float]):
        """Update allocators with telemetry."""
        # Update elastic backpressure
        for res in ResourceType:
            key = res.value
            util = telemetry.get(f"{key}_util", 0.5)
            self.elastic.update_pressure(key, util)
            self.elastic.adjust_scale(key)

        # Update predictions
        for key, value in telemetry.items():
            if "util" in key or "usage" in key:
                self.predictive.observe(key, value)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive allocation status."""
        return {
            "hierarchical": self.hierarchical.get_stats(),
            "predictions": self.predictive.get_all_predictions(),
            "elastic_scales": dict(self.elastic.current_scale),
            "backpressure": dict(self.elastic.backpressure),
            "fair_share_clients": len(self.fair_share.clients),
            "speculative_reserves": len(self.speculative.speculative_reserves),
        }


def create_allocation_system() -> UnifiedAllocationSystem:
    """Factory function."""
    return UnifiedAllocationSystem()


if __name__ == "__main__":
    system = UnifiedAllocationSystem()

    print("=== GAMESA Advanced Allocation ===\n")

    # Register fair share clients
    system.fair_share.register_client("game", weight=3.0)
    system.fair_share.register_client("system", weight=1.0)
    system.fair_share.register_client("background", weight=0.5)

    # Simulate allocations
    requests = [
        AllocationRequest("req_1", ResourceType.CPU_CYCLES, 20, AllocationPriority.REALTIME, "game"),
        AllocationRequest("req_2", ResourceType.GPU_COMPUTE, 15, AllocationPriority.HIGH, "game"),
        AllocationRequest("req_3", ResourceType.MEMORY_MB, 256, AllocationPriority.NORMAL, "system"),
        AllocationRequest("req_4", ResourceType.THERMAL_BUDGET, 5, AllocationPriority.CRITICAL, "safety"),
    ]

    for req in requests:
        result = system.allocate(req, context="gaming")
        print(f"{req.request_id}: {req.resource_type.value}")
        print(f"  Requested: {req.amount}, Allocated: {result['allocation']}")
        print(f"  Source: {result['source']}")
        print()

    print("Status:")
    status = system.get_status()
    print(f"  Elastic scales: {status['elastic_scales']}")
    print(f"  Fair share clients: {status['fair_share_clients']}")
