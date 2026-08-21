"""
GAMESA Memory Manager - Swap and Memory Tiering System

Implements:
- Memory tier classification (Hot/Warm/Cold/Frozen)
- Predictive prefetch engine
- Adaptive zRAM compression
- Cross-Forex memory trading
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum, auto
from collections import deque
import time
import threading
import heapq


# ============================================================
# ENUMS & DATA STRUCTURES
# ============================================================

class MemoryTier(Enum):
    """Memory tier classification."""
    HOT = 0       # L1/L2 cache, <1ms access
    WARM = 1      # L3/LLC, <10ms
    COLD = 2      # RAM, <100ms
    FROZEN = 3    # Swap candidate
    SWAPPED = 4   # On disk/compressed
    EVICTED = 5   # Removed


class SwapTier(Enum):
    """Swap storage tiers."""
    ZRAM_FAST = 0     # LZ4 compressed RAM
    ZRAM_RATIO = 1    # ZSTD compressed RAM
    OPTANE = 2        # Intel Optane / fast NVMe
    NVME_FAST = 3     # Fast NVMe SSD
    NVME_STANDARD = 4 # Standard SSD
    HDD = 5           # Hard disk (last resort)
    NETWORK = 6       # Network swap


class CompressionAlgo(Enum):
    """Compression algorithms."""
    NONE = "none"
    LZ4 = "lz4"
    LZ4HC = "lz4hc"
    ZSTD = "zstd"
    ZSTD_FAST = "zstd_fast"


@dataclass
class MemoryRegion:
    """Tracked memory region."""
    region_id: str
    address: int
    size: int
    tier: MemoryTier = MemoryTier.COLD
    owner: str = ""
    priority: int = 5
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    compressible: bool = True
    swappable: bool = True
    compressed_size: int = 0
    compression_algo: CompressionAlgo = CompressionAlgo.NONE

    @property
    def compression_ratio(self) -> float:
        if self.compressed_size > 0:
            return self.size / self.compressed_size
        return 1.0

    @property
    def age(self) -> float:
        return time.time() - self.last_access


@dataclass
class MemoryStats:
    """Memory statistics."""
    total_mb: float = 0.0
    used_mb: float = 0.0
    free_mb: float = 0.0
    cached_mb: float = 0.0
    swap_used_mb: float = 0.0
    swap_free_mb: float = 0.0
    zram_used_mb: float = 0.0
    zram_ratio: float = 1.0
    page_faults: int = 0
    swap_ins: int = 0
    swap_outs: int = 0


# ============================================================
# TIER MANAGER
# ============================================================

class TierManager:
    """
    Manages memory tier classification and migration.

    Implements Hot → Warm → Cold → Frozen → Swapped lifecycle.
    """

    # Tier timeout thresholds (seconds)
    TIER_TIMEOUTS = {
        MemoryTier.HOT: 0.1,      # 100ms
        MemoryTier.WARM: 1.0,     # 1s
        MemoryTier.COLD: 10.0,    # 10s
        MemoryTier.FROZEN: 60.0,  # 60s
    }

    # Access frequency thresholds for promotion
    PROMOTION_THRESHOLDS = {
        MemoryTier.WARM: 10.0,    # 10 access/s to promote to HOT
        MemoryTier.COLD: 2.0,     # 2 access/s to promote to WARM
        MemoryTier.FROZEN: 0.5,   # 0.5 access/s to promote to COLD
        MemoryTier.SWAPPED: 0.1,  # Any access promotes from SWAPPED
    }

    def __init__(self):
        self._regions: Dict[str, MemoryRegion] = {}
        self._tier_lists: Dict[MemoryTier, List[str]] = {t: [] for t in MemoryTier}
        self._lock = threading.RLock()
        self._stats = {
            "promotions": 0,
            "demotions": 0,
            "migrations": 0
        }

    def register_region(self, region: MemoryRegion):
        """Register a memory region for tracking."""
        with self._lock:
            self._regions[region.region_id] = region
            self._tier_lists[region.tier].append(region.region_id)

    def unregister_region(self, region_id: str):
        """Unregister a memory region."""
        with self._lock:
            if region_id in self._regions:
                region = self._regions[region_id]
                self._tier_lists[region.tier].remove(region_id)
                del self._regions[region_id]

    def record_access(self, region_id: str):
        """Record an access to a region."""
        with self._lock:
            region = self._regions.get(region_id)
            if region:
                region.last_access = time.time()
                region.access_count += 1
                self._check_promotion(region)

    def _check_promotion(self, region: MemoryRegion):
        """Check if region should be promoted."""
        if region.tier == MemoryTier.HOT:
            return  # Already at top

        # Calculate access frequency
        age = max(0.001, region.age)
        frequency = region.access_count / age

        target_tier = region.tier
        threshold = self.PROMOTION_THRESHOLDS.get(region.tier, 0)

        if frequency > threshold:
            # Promote one tier
            tier_order = list(MemoryTier)
            current_idx = tier_order.index(region.tier)
            if current_idx > 0:
                target_tier = tier_order[current_idx - 1]

        if target_tier != region.tier:
            self._migrate_region(region, target_tier)
            self._stats["promotions"] += 1

    def tick(self) -> Dict:
        """Periodic tier maintenance."""
        demotions = []
        now = time.time()

        with self._lock:
            for region_id, region in self._regions.items():
                if region.tier == MemoryTier.SWAPPED:
                    continue

                timeout = self.TIER_TIMEOUTS.get(region.tier, 10.0)
                if region.age > timeout:
                    demotions.append(region)

        # Process demotions
        for region in demotions:
            self._demote_region(region)
            self._stats["demotions"] += 1

        return {
            "demotions": len(demotions),
            "stats": self._stats.copy()
        }

    def _demote_region(self, region: MemoryRegion):
        """Demote region to lower tier."""
        tier_order = list(MemoryTier)
        current_idx = tier_order.index(region.tier)

        if current_idx < len(tier_order) - 1:
            target_tier = tier_order[current_idx + 1]
            self._migrate_region(region, target_tier)

    def _migrate_region(self, region: MemoryRegion, target_tier: MemoryTier):
        """Migrate region to new tier."""
        with self._lock:
            old_tier = region.tier
            self._tier_lists[old_tier].remove(region.region_id)
            region.tier = target_tier
            self._tier_lists[target_tier].append(region.region_id)
            self._stats["migrations"] += 1

    def get_tier_stats(self) -> Dict[str, int]:
        """Get region counts per tier."""
        with self._lock:
            return {t.name: len(self._tier_lists[t]) for t in MemoryTier}

    def get_swap_candidates(self, target_count: int) -> List[MemoryRegion]:
        """Get regions suitable for swap-out."""
        candidates = []

        with self._lock:
            # Prioritize FROZEN tier
            for region_id in self._tier_lists[MemoryTier.FROZEN]:
                region = self._regions.get(region_id)
                if region and region.swappable:
                    candidates.append(region)
                    if len(candidates) >= target_count:
                        break

            # If not enough, check COLD tier
            if len(candidates) < target_count:
                for region_id in self._tier_lists[MemoryTier.COLD]:
                    region = self._regions.get(region_id)
                    if region and region.swappable:
                        candidates.append(region)
                        if len(candidates) >= target_count:
                            break

        # Sort by age (oldest first)
        candidates.sort(key=lambda r: -r.age)
        return candidates[:target_count]


# ============================================================
# PREDICTIVE PREFETCH
# ============================================================

class AccessPattern:
    """Detected access pattern."""
    SEQUENTIAL = "sequential"
    STRIDED = "strided"
    RANDOM = "random"
    TEMPORAL = "temporal"


class PrefetchPredictor:
    """
    Predicts future memory accesses for prefetching.

    Detects patterns:
    - Sequential access (addr, addr+size, addr+2*size)
    - Strided access (regular intervals)
    - Temporal patterns (frame-based)
    """

    def __init__(self, history_size: int = 1000):
        self._history: deque = deque(maxlen=history_size)
        self._stride_detector = StrideDetector()
        self._temporal_buffer: Dict[str, List[int]] = {}
        self.confidence = 0.0

    def observe(self, address: int, context: str = ""):
        """Record a memory access."""
        self._history.append({
            "address": address,
            "time": time.time(),
            "context": context
        })
        self._stride_detector.update(address)

        # Track per-context accesses
        if context:
            if context not in self._temporal_buffer:
                self._temporal_buffer[context] = []
            self._temporal_buffer[context].append(address)

    def predict_next(self, count: int = 8) -> List[int]:
        """Predict next likely memory accesses."""
        predictions = []

        # Try stride prediction
        stride_pred = self._stride_detector.predict(count)
        if stride_pred and self._stride_detector.confidence > 0.7:
            predictions.extend(stride_pred)
            self.confidence = self._stride_detector.confidence

        # Try sequential prediction
        if len(self._history) >= 2:
            last = self._history[-1]["address"]
            prev = self._history[-2]["address"]
            diff = last - prev

            if 0 < diff < 4096:  # Likely sequential
                for i in range(1, count + 1):
                    predictions.append(last + diff * i)

        self.confidence = min(1.0, len(set(predictions)) / count) if count > 0 else 0

        return list(set(predictions))[:count]

    def get_pattern(self) -> str:
        """Identify current access pattern."""
        if self._stride_detector.confidence > 0.8:
            if self._stride_detector.stride == 1:
                return AccessPattern.SEQUENTIAL
            return AccessPattern.STRIDED
        return AccessPattern.RANDOM


class StrideDetector:
    """Detects strided access patterns."""

    def __init__(self, window: int = 10):
        self._addresses: deque = deque(maxlen=window)
        self._strides: deque = deque(maxlen=window - 1)
        self.stride = 0
        self.confidence = 0.0

    def update(self, address: int):
        """Update with new address."""
        if self._addresses:
            stride = address - self._addresses[-1]
            self._strides.append(stride)
            self._detect_stride()

        self._addresses.append(address)

    def _detect_stride(self):
        """Detect common stride."""
        if len(self._strides) < 3:
            self.confidence = 0.0
            return

        # Find most common stride
        stride_counts: Dict[int, int] = {}
        for s in self._strides:
            stride_counts[s] = stride_counts.get(s, 0) + 1

        if stride_counts:
            best_stride = max(stride_counts, key=stride_counts.get)
            self.stride = best_stride
            self.confidence = stride_counts[best_stride] / len(self._strides)

    def predict(self, count: int) -> List[int]:
        """Predict next addresses based on stride."""
        if self.confidence < 0.5 or not self._addresses:
            return []

        last = self._addresses[-1]
        return [last + self.stride * i for i in range(1, count + 1)]


# ============================================================
# ADAPTIVE COMPRESSION (zRAM)
# ============================================================

class AdaptiveCompressor:
    """
    Adaptive compression with algorithm selection.

    Selects algorithm based on:
    - Data compressibility
    - Urgency (memory pressure)
    - CPU availability
    """

    ALGO_PROFILES = {
        CompressionAlgo.LZ4: {"speed": 10, "ratio": 2.0, "cpu": 1},
        CompressionAlgo.LZ4HC: {"speed": 5, "ratio": 2.5, "cpu": 2},
        CompressionAlgo.ZSTD_FAST: {"speed": 7, "ratio": 2.8, "cpu": 2},
        CompressionAlgo.ZSTD: {"speed": 3, "ratio": 3.5, "cpu": 4},
    }

    def __init__(self):
        self._stats: Dict[CompressionAlgo, Dict] = {
            algo: {"count": 0, "total_ratio": 0.0, "total_time": 0.0}
            for algo in CompressionAlgo if algo != CompressionAlgo.NONE
        }

    def select_algorithm(self, estimated_ratio: float,
                        urgency: float,
                        cpu_available: float) -> CompressionAlgo:
        """Select optimal compression algorithm."""

        # High urgency = fast algorithm
        if urgency > 0.8:
            return CompressionAlgo.LZ4

        # Low CPU = fast algorithm
        if cpu_available < 0.3:
            return CompressionAlgo.LZ4

        # High compressibility = better ratio algorithm
        if estimated_ratio > 3.0 and urgency < 0.4:
            return CompressionAlgo.ZSTD

        if estimated_ratio > 2.5 and urgency < 0.6:
            return CompressionAlgo.ZSTD_FAST

        if estimated_ratio > 2.0:
            return CompressionAlgo.LZ4HC

        return CompressionAlgo.LZ4

    def compress(self, data: bytes, urgency: float = 0.5,
                cpu_available: float = 0.5) -> Tuple[bytes, CompressionAlgo, float]:
        """
        Compress data with adaptive algorithm selection.

        Returns: (compressed_data, algorithm_used, ratio)
        """
        # Estimate compressibility from sample
        sample_ratio = self._estimate_ratio(data[:1024] if len(data) > 1024 else data)

        algo = self.select_algorithm(sample_ratio, urgency, cpu_available)

        # Simulate compression (in real impl, use actual compression)
        start = time.time()
        compressed = self._do_compress(data, algo)
        elapsed = time.time() - start

        ratio = len(data) / len(compressed) if compressed else 1.0

        # Update stats
        self._stats[algo]["count"] += 1
        self._stats[algo]["total_ratio"] += ratio
        self._stats[algo]["total_time"] += elapsed

        return compressed, algo, ratio

    def _estimate_ratio(self, sample: bytes) -> float:
        """Estimate compression ratio from sample."""
        if not sample:
            return 1.0

        # Simple entropy estimation
        byte_counts = [0] * 256
        for b in sample:
            byte_counts[b] += 1

        entropy = 0.0
        for count in byte_counts:
            if count > 0:
                p = count / len(sample)
                entropy -= p * (p.bit_length() - 1) if p > 0 else 0

        # Lower entropy = higher compressibility
        return max(1.0, 8.0 / max(entropy, 0.1))

    def _do_compress(self, data: bytes, algo: CompressionAlgo) -> bytes:
        """Perform compression (simulated)."""
        # In real implementation, use actual compression libraries
        profile = self.ALGO_PROFILES.get(algo, {"ratio": 1.5})
        simulated_size = int(len(data) / profile["ratio"])
        return data[:simulated_size]  # Simulated

    def get_stats(self) -> Dict:
        """Get compression statistics."""
        return {
            algo.value: {
                "count": stats["count"],
                "avg_ratio": stats["total_ratio"] / max(1, stats["count"]),
                "avg_time_ms": stats["total_time"] * 1000 / max(1, stats["count"])
            }
            for algo, stats in self._stats.items()
        }


# ============================================================
# SWAP MANAGER
# ============================================================

class SwapManager:
    """
    Manages swap operations across multiple tiers.

    Integrates with Cross-Forex for trading-based decisions.
    """

    def __init__(self, tier_manager: TierManager):
        self.tier_manager = tier_manager
        self.compressor = AdaptiveCompressor()

        # Swap queues per tier
        self._swap_queues: Dict[SwapTier, List] = {t: [] for t in SwapTier}

        # Swap statistics
        self._stats = MemoryStats()
        self._lock = threading.Lock()

        # Configuration
        self.target_free_percent = 15.0
        self.zram_max_percent = 25.0

    def tick(self, memory_pressure: float, thermal_headroom: float) -> Dict:
        """Periodic swap management tick."""
        results = {
            "swapped_out": 0,
            "swapped_in": 0,
            "compressed": 0
        }

        # Calculate target
        pressure_factor = 1.0 + max(0, (memory_pressure - 0.7) * 2)

        # Thermal-aware: more aggressive when hot
        if thermal_headroom < 10:
            pressure_factor *= 1.5

        if memory_pressure > 0.7:
            # Need to swap out
            target_pages = int(pressure_factor * 10)
            candidates = self.tier_manager.get_swap_candidates(target_pages)

            for region in candidates:
                self._swap_out(region, memory_pressure)
                results["swapped_out"] += 1

        # Process swap-in requests (from access patterns)
        results["swapped_in"] = self._process_swap_ins()

        return results

    def _swap_out(self, region: MemoryRegion, urgency: float):
        """Swap out a memory region."""
        # Select swap tier
        swap_tier = self._select_swap_tier(region)

        if swap_tier in (SwapTier.ZRAM_FAST, SwapTier.ZRAM_RATIO):
            # Compress to zRAM
            data = self._read_region(region)  # Simulated
            compressed, algo, ratio = self.compressor.compress(
                data, urgency=urgency
            )
            region.compressed_size = len(compressed)
            region.compression_algo = algo

        # Update tier
        region.tier = MemoryTier.SWAPPED
        self._swap_queues[swap_tier].append(region.region_id)

        self._stats.swap_outs += 1

    def _select_swap_tier(self, region: MemoryRegion) -> SwapTier:
        """Select appropriate swap tier for region."""
        # Check compressibility
        if region.compressible:
            estimated_ratio = 2.5  # Would estimate from data
            if estimated_ratio > 2.5:
                return SwapTier.ZRAM_RATIO
            return SwapTier.ZRAM_FAST

        # Check access frequency
        if region.access_count > 10:
            return SwapTier.OPTANE  # Warm data to fast storage

        return SwapTier.NVME_STANDARD

    def _process_swap_ins(self) -> int:
        """Process pending swap-in requests."""
        # In real impl, would check for accessed swapped pages
        return 0

    def _read_region(self, region: MemoryRegion) -> bytes:
        """Read region data (simulated)."""
        return b'\x00' * region.size

    def get_stats(self) -> MemoryStats:
        """Get swap statistics."""
        return self._stats


# ============================================================
# MEMORY BROKER (Cross-Forex Integration)
# ============================================================

class MemoryBroker:
    """
    Central memory broker for Cross-Forex trading.

    Manages memory as tradeable commodities.
    """

    def __init__(self):
        self.tier_manager = TierManager()
        self.swap_manager = SwapManager(self.tier_manager)
        self.prefetch_predictor = PrefetchPredictor()

        self._commodities = {
            "ram_free_mb": 0.0,
            "zram_used_mb": 0.0,
            "swap_available_mb": 0.0,
            "prefetch_slots": 8,
            "memory_pressure": 0.0
        }

    def update_state(self, memory_stats: Dict):
        """Update memory state from system."""
        self._commodities.update({
            "ram_free_mb": memory_stats.get("free_mb", 0),
            "memory_pressure": memory_stats.get("pressure", 0.5)
        })

    def tick(self, thermal_headroom: float = 20.0) -> Dict:
        """Execute memory management tick."""
        results = {}

        # Tier management
        tier_results = self.tier_manager.tick()
        results["tiers"] = tier_results

        # Swap management
        pressure = self._commodities.get("memory_pressure", 0.5)
        swap_results = self.swap_manager.tick(pressure, thermal_headroom)
        results["swap"] = swap_results

        # Update commodities for Cross-Forex
        results["commodities"] = self.get_commodities()

        return results

    def record_access(self, address: int, region_id: str = "", context: str = ""):
        """Record memory access for tracking and prediction."""
        self.prefetch_predictor.observe(address, context)

        if region_id:
            self.tier_manager.record_access(region_id)

    def get_prefetch_predictions(self, count: int = 8) -> List[int]:
        """Get predicted addresses for prefetch."""
        return self.prefetch_predictor.predict_next(count)

    def get_commodities(self) -> Dict:
        """Get memory commodities for Cross-Forex."""
        tier_stats = self.tier_manager.get_tier_stats()

        return {
            **self._commodities,
            "hot_regions": tier_stats.get("HOT", 0),
            "warm_regions": tier_stats.get("WARM", 0),
            "cold_regions": tier_stats.get("COLD", 0),
            "frozen_regions": tier_stats.get("FROZEN", 0),
            "swapped_regions": tier_stats.get("SWAPPED", 0),
            "prefetch_confidence": self.prefetch_predictor.confidence
        }

    def request_swap_out(self, target_mb: float) -> int:
        """Request swap-out of specified amount."""
        page_size_kb = 4
        target_pages = int(target_mb * 1024 / page_size_kb)

        candidates = self.tier_manager.get_swap_candidates(target_pages)
        swapped = 0

        for region in candidates:
            self.swap_manager._swap_out(region, urgency=0.7)
            swapped += 1

        return swapped

    def request_prefetch(self, addresses: List[int]) -> int:
        """Request prefetch of addresses."""
        # In real impl, would issue actual prefetch instructions
        return len(addresses)


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate memory manager."""
    print("=== GAMESA Memory Manager Demo ===\n")

    broker = MemoryBroker()

    # Register some memory regions
    for i in range(20):
        region = MemoryRegion(
            region_id=f"region_{i}",
            address=0x1000 * i,
            size=4096,
            tier=MemoryTier.COLD,
            owner="test",
            priority=5
        )
        broker.tier_manager.register_region(region)

    print(f"Initial tiers: {broker.tier_manager.get_tier_stats()}")

    # Simulate memory accesses
    print("\nSimulating accesses...")
    for i in range(100):
        # Access pattern: sequential
        addr = 0x1000 * (i % 20)
        broker.record_access(addr, f"region_{i % 20}", "game_loop")

        # Periodic tick
        if i % 10 == 0:
            result = broker.tick(thermal_headroom=15.0)

    print(f"\nFinal tiers: {broker.tier_manager.get_tier_stats()}")

    # Get prefetch predictions
    predictions = broker.get_prefetch_predictions(8)
    print(f"\nPrefetch predictions: {[hex(p) for p in predictions[:4]]}")
    print(f"Prediction confidence: {broker.prefetch_predictor.confidence:.2f}")

    # Get commodities
    print(f"\nMemory commodities: {broker.get_commodities()}")

    # Test compression stats
    print(f"\nCompression stats: {broker.swap_manager.compressor.get_stats()}")


if __name__ == "__main__":
    demo()
