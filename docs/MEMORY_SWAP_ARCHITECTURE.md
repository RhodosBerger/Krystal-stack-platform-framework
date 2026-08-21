# GAMESA Memory & Swap Management Architecture

## Koncept: Memory-Augmented Virtual Bus (MAVB) v2

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GAMESA MEMORY HIERARCHY                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │ L1 Cache │──▶│ L2 Cache │──▶│ L3/LLC   │──▶│ Unified Memory   │ │
│  │  64KB    │   │  1MB     │   │  32MB    │   │  Pool (RAM)      │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────────┘ │
│       │              │              │                   │           │
│       ▼              ▼              ▼                   ▼           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              GAMESA MEMORY BROKER                            │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │   │
│  │  │ Hot     │  │ Warm    │  │ Cold    │  │ Frozen (Swap)   │ │   │
│  │  │ Tier    │  │ Tier    │  │ Tier    │  │ Tier            │ │   │
│  │  │ <1ms    │  │ <10ms   │  │ <100ms  │  │ >100ms          │ │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    SWAP STRATEGIES                           │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │   │
│  │  │ NVMe    │  │ Optane   │  │ zRAM     │  │ Network     │  │   │
│  │  │ Tier    │  │ Tier     │  │ Compress │  │ Swap        │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Memory Tiering System

### 1.1 Tier Definície

```python
class MemoryTier(Enum):
    """Memory tier classification."""
    HOT = 0      # Actively used, in L1/L2 cache
    WARM = 1     # Recently used, in L3/LLC
    COLD = 2     # Infrequently used, in RAM
    FROZEN = 3   # Inactive, candidate for swap
    SWAPPED = 4  # On disk/compressed

@dataclass
class MemoryRegion:
    """Tracked memory region."""
    address: int
    size: int
    tier: MemoryTier
    last_access: float
    access_count: int
    owner: str  # Agent ID
    priority: int
    compressible: bool
    swappable: bool
```

### 1.2 Tier Promotion/Demotion Rules

```
PROMOTION (Cold → Hot):
─────────────────────────────────────────────────────────
Trigger: access_frequency > threshold
         OR predictive_prefetch signal
         OR agent REQUEST_PREFETCH order

Action:  1. Allocate in higher tier
         2. Issue DMA prefetch
         3. Update memory broker state

DEMOTION (Hot → Cold → Frozen):
─────────────────────────────────────────────────────────
Trigger: last_access > timeout[tier]
         OR memory_pressure > threshold
         OR thermal_headroom < limit

Action:  1. Mark for demotion
         2. If compressible: compress in-place (zRAM)
         3. If swappable: queue for swap-out
         4. Release tier allocation
```

### 1.3 Časové Prahy

| Tier | Max Idle Time | Promotion Trigger | Demotion Trigger |
|------|---------------|-------------------|------------------|
| HOT | 100ms | 10+ access/s | <2 access/s |
| WARM | 1s | 2+ access/s | <0.5 access/s |
| COLD | 10s | 0.5+ access/s | <0.1 access/s |
| FROZEN | 60s | Any access | Memory pressure |

---

## 2. Predictive Prefetch Engine

### 2.1 Access Pattern Learning

```python
class AccessPatternPredictor:
    """
    Learns memory access patterns for predictive prefetch.

    Patterns detected:
    - Sequential: addr, addr+stride, addr+2*stride...
    - Strided: Regular interval access
    - Pointer chasing: Follow pointer chains
    - Temporal: Time-based patterns (frame boundaries)
    """

    def __init__(self):
        self.pattern_buffer = RingBuffer(size=1000)
        self.stride_detector = StrideDetector()
        self.markov_chain = MarkovPredictor(order=3)
        self.temporal_model = TemporalModel()

    def observe(self, address: int, timestamp: float, context: str):
        """Record memory access."""
        self.pattern_buffer.append((address, timestamp, context))

        # Update predictors
        self.stride_detector.update(address)
        self.markov_chain.update(address)
        self.temporal_model.update(timestamp, context)

    def predict_next(self, count: int = 8) -> List[int]:
        """Predict next likely accesses."""
        predictions = []

        # Stride prediction
        if self.stride_detector.confidence > 0.8:
            stride = self.stride_detector.detected_stride
            last = self.pattern_buffer.last_address
            predictions.extend([last + stride * i for i in range(1, count+1)])

        # Markov prediction
        markov_pred = self.markov_chain.predict(count)
        predictions.extend(markov_pred)

        return list(set(predictions))[:count]
```

### 2.2 Prefetch Integration s Cross-Forex

```python
class PrefetchTrader(BaseAgent):
    """
    Memory prefetch agent in Cross-Forex market.

    Trades MEMORY_BANDWIDTH commodity for prefetch slots.
    """

    def analyze_market(self, ticker: MarketTicker) -> Optional[TradeOrder]:
        # Get predictions
        predictions = self.predictor.predict_next(8)

        if not predictions:
            return None

        # Check if prefetch is worthwhile
        hex_memory = ticker.commodities.hex_memory
        hex_io = ticker.commodities.hex_io

        # High memory pressure + predictions = prefetch opportunity
        if hex_memory > 0x60 and len(predictions) >= 4:
            return TradeOrder(
                source=self.agent_id,
                action="REQUEST_PREFETCH_BATCH",
                bid=TradeBid(
                    reason="PREDICTIVE_PREFETCH",
                    est_latency_impact=-5.0,  # Expected latency reduction
                    priority=6
                ),
                metadata={
                    "addresses": predictions,
                    "confidence": self.predictor.confidence
                }
            )

        return None
```

---

## 3. Swap Management Strategies

### 3.1 Multi-Tier Swap Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   SWAP TIER HIERARCHY                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Tier 0: zRAM (Compressed RAM)                              │
│  ├── Latency: 1-5ms                                         │
│  ├── Capacity: 25% of RAM                                   │
│  ├── Use case: Highly compressible pages                    │
│  └── Compression: LZ4 (fast) / ZSTD (ratio)                │
│                                                              │
│  Tier 1: Intel Optane / Fast NVMe                           │
│  ├── Latency: 10-50μs                                       │
│  ├── Capacity: 32-128GB                                     │
│  ├── Use case: Warm pages, frequent swap-in                 │
│  └── Wear leveling: GAMESA-aware                            │
│                                                              │
│  Tier 2: Standard NVMe SSD                                  │
│  ├── Latency: 100-500μs                                     │
│  ├── Capacity: 256GB+                                       │
│  ├── Use case: Cold pages, bulk swap                        │
│  └── Write coalescing: Batch writes                         │
│                                                              │
│  Tier 3: Network Swap (Optional)                            │
│  ├── Latency: 1-10ms                                        │
│  ├── Capacity: Unlimited                                    │
│  ├── Use case: Emergency overflow                           │
│  └── Protocol: RDMA / NVMe-oF                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Swap Decision Engine

```python
class SwapDecisionEngine:
    """
    Decides swap tier and timing based on page characteristics.
    """

    def __init__(self):
        self.compressibility_cache = {}
        self.access_history = AccessHistory()

    def select_swap_tier(self, page: MemoryPage) -> SwapTier:
        """Select optimal swap tier for page."""

        # Check compressibility
        compress_ratio = self.estimate_compression(page)

        if compress_ratio > 2.0:
            # Highly compressible - use zRAM
            return SwapTier.ZRAM

        # Check access pattern
        access_freq = self.access_history.get_frequency(page.address)

        if access_freq > 0.1:  # Accessed in last 10s
            # Warm page - use fast storage
            return SwapTier.OPTANE if self.has_optane else SwapTier.NVME_FAST

        if access_freq > 0.01:  # Accessed in last 100s
            return SwapTier.NVME_STANDARD

        # Very cold - can go to slow tier
        return SwapTier.NVME_SLOW

    def estimate_compression(self, page: MemoryPage) -> float:
        """Estimate page compression ratio."""
        # Sample-based estimation
        if page.address in self.compressibility_cache:
            return self.compressibility_cache[page.address]

        # Quick entropy check
        sample = page.read_sample(256)
        entropy = self.calculate_entropy(sample)

        # Low entropy = high compressibility
        ratio = 8.0 / max(entropy, 0.1)
        self.compressibility_cache[page.address] = ratio

        return ratio
```

### 3.3 Proactive Swap-Out

```python
class ProactiveSwapManager:
    """
    Proactively swaps pages before memory pressure hits.

    Integrates with Cross-Forex thermal management.
    """

    def __init__(self, target_free_percent: float = 15.0):
        self.target_free = target_free_percent
        self.swap_queue = PriorityQueue()
        self.decision_engine = SwapDecisionEngine()

    def tick(self, memory_state: MemoryState, thermal: ThermalState):
        """Proactive swap tick."""

        # Calculate pressure
        free_percent = memory_state.free / memory_state.total * 100
        pressure = max(0, self.target_free - free_percent)

        # Thermal-aware: swap more aggressively when hot
        if thermal.headroom < 10:
            pressure *= 1.5  # More aggressive

        if pressure > 0:
            # Find swap candidates
            candidates = self.find_swap_candidates(
                memory_state,
                target_mb=pressure * memory_state.total / 100
            )

            for page in candidates:
                tier = self.decision_engine.select_swap_tier(page)
                self.swap_queue.push(page, tier, priority=-page.last_access)

        # Process swap queue (rate limited)
        self.process_swap_queue(max_pages=100)

    def find_swap_candidates(self, state: MemoryState,
                            target_mb: float) -> List[MemoryPage]:
        """Find pages to swap out."""
        candidates = []
        collected_mb = 0

        # Sort by last access time (LRU-like)
        pages = sorted(state.pages, key=lambda p: p.last_access)

        for page in pages:
            if not page.swappable:
                continue
            if page.tier == MemoryTier.HOT:
                continue  # Don't swap hot pages

            candidates.append(page)
            collected_mb += page.size / 1024 / 1024

            if collected_mb >= target_mb:
                break

        return candidates
```

---

## 4. zRAM Integration

### 4.1 Adaptive Compression

```python
class AdaptiveZRAM:
    """
    Adaptive zRAM with dynamic algorithm selection.
    """

    ALGORITHMS = {
        "lz4": {"speed": 10, "ratio": 2.0},
        "lz4hc": {"speed": 5, "ratio": 2.5},
        "zstd": {"speed": 3, "ratio": 3.5},
        "zstd_fast": {"speed": 7, "ratio": 2.8}
    }

    def __init__(self, max_size_percent: float = 25.0):
        self.max_size_percent = max_size_percent
        self.current_algo = "lz4"
        self.stats = CompressionStats()

    def select_algorithm(self, page: MemoryPage,
                        urgency: float) -> str:
        """Select compression algorithm based on context."""

        if urgency > 0.8:
            # High urgency - use fastest
            return "lz4"

        # Estimate compressibility
        estimated_ratio = self.estimate_ratio(page)

        if estimated_ratio > 3.0:
            # Highly compressible - worth using slower algo
            return "zstd" if urgency < 0.3 else "zstd_fast"

        if estimated_ratio > 2.0:
            return "lz4hc" if urgency < 0.5 else "lz4"

        # Low compressibility - fastest only
        return "lz4"

    def compress_page(self, page: MemoryPage,
                     urgency: float = 0.5) -> CompressedPage:
        """Compress page with adaptive algorithm."""

        algo = self.select_algorithm(page, urgency)

        start = time.time()
        compressed = self.do_compress(page.data, algo)
        elapsed = time.time() - start

        ratio = len(page.data) / len(compressed)

        self.stats.record(algo, ratio, elapsed)

        return CompressedPage(
            original_address=page.address,
            compressed_data=compressed,
            algorithm=algo,
            original_size=len(page.data),
            compressed_size=len(compressed)
        )
```

### 4.2 zRAM Pool Management

```
┌─────────────────────────────────────────────────────────────┐
│                    zRAM POOL STRUCTURE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Fast Pool (LZ4)                                    │    │
│  │  ├── Max: 10% RAM                                   │    │
│  │  ├── Target latency: <1ms                           │    │
│  │  └── Use: Recently demoted pages                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Ratio Pool (ZSTD)                                  │    │
│  │  ├── Max: 15% RAM                                   │    │
│  │  ├── Target ratio: >3x                              │    │
│  │  └── Use: Highly compressible cold pages            │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Migration: Fast Pool ──(age > 30s)──▶ Ratio Pool           │
│             Ratio Pool ──(access)──▶ Fast Pool              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Cross-Forex Memory Market

### 5.1 Memory Commodities

```python
class MemoryCommodity(Enum):
    """Tradeable memory resources."""

    # Capacity commodities
    RAM_MB = "ram_mb"
    ZRAM_MB = "zram_mb"
    SWAP_MB = "swap_mb"

    # Bandwidth commodities
    MEMORY_BW_GBPS = "memory_bw_gbps"
    CACHE_BW_GBPS = "cache_bw_gbps"
    SWAP_IOPS = "swap_iops"

    # Latency commodities
    PREFETCH_SLOTS = "prefetch_slots"
    DMA_CHANNELS = "dma_channels"
```

### 5.2 Memory Trading Agents

```python
class MemoryBrokerAgent(BaseAgent):
    """
    Central memory broker in Cross-Forex market.

    Manages memory allocation, tier migration, and swap.
    """

    def __init__(self, hal: BaseHAL):
        super().__init__("AGENT_MEMORY_BROKER", AgentType.CACHE)
        self.hal = hal
        self.tier_manager = TierManager()
        self.swap_manager = ProactiveSwapManager()
        self.prefetch_engine = PrefetchEngine()

    def analyze_market(self, ticker: MarketTicker) -> Optional[TradeOrder]:
        """Analyze memory market conditions."""

        hex_memory = ticker.commodities.hex_memory
        hex_io = ticker.commodities.hex_io
        thermal = ticker.commodities.thermal_headroom_gpu

        # High memory pressure
        if hex_memory > 0x80:
            # Request swap-out permission
            return TradeOrder(
                source=self.agent_id,
                action="REQUEST_SWAP_OUT",
                bid=TradeBid(
                    reason="MEMORY_PRESSURE_HIGH",
                    est_thermal_saving=2.0,  # Less RAM = less power
                    priority=7
                )
            )

        # High I/O with memory available
        if hex_io > 0x70 and hex_memory < 0x50:
            # Offer to prefetch
            return TradeOrder(
                source=self.agent_id,
                action="OFFER_PREFETCH_CAPACITY",
                bid=TradeBid(
                    reason="IO_BOTTLENECK_DETECTED",
                    est_latency_impact=-3.0,
                    priority=6
                )
            )

        # Thermal pressure - compress more aggressively
        if thermal < 10:
            return TradeOrder(
                source=self.agent_id,
                action="REQUEST_AGGRESSIVE_COMPRESSION",
                bid=TradeBid(
                    reason="THERMAL_PRESSURE",
                    est_thermal_saving=3.0,
                    priority=8
                )
            )

        return None
```

### 5.3 Memory Directives

```python
class MemoryDirectiveExecutor:
    """Executes memory-related Guardian directives."""

    def execute(self, directive: Directive):
        action = directive.params

        if "SWAP_OUT" in directive.permit_id:
            # Execute swap-out
            target_mb = action.get("target_mb", 100)
            self.swap_manager.force_swap(target_mb)

        elif "PREFETCH" in directive.permit_id:
            # Execute prefetch
            addresses = action.get("addresses", [])
            self.prefetch_engine.prefetch_batch(addresses)

        elif "COMPRESSION" in directive.permit_id:
            # Switch compression mode
            mode = action.get("mode", "aggressive")
            self.zram.set_mode(mode)

        elif "TIER_MIGRATE" in directive.permit_id:
            # Migrate pages between tiers
            from_tier = action.get("from_tier")
            to_tier = action.get("to_tier")
            self.tier_manager.migrate(from_tier, to_tier)
```

---

## 6. Platform-Specific Optimizations

### 6.1 Intel

```python
class IntelMemoryOptimizer:
    """Intel-specific memory optimizations."""

    def configure(self):
        # Enable Intel Memory Bandwidth Allocation (MBA)
        self.enable_mba()

        # Configure Optane as swap tier 1
        if self.has_optane:
            self.configure_optane_swap()

        # Use CLFLUSH for precise cache control
        self.prefetch_instruction = "PREFETCHW"
        self.flush_instruction = "CLFLUSHOPT"

    def enable_mba(self):
        """Enable Memory Bandwidth Allocation."""
        # Allocate bandwidth per-agent
        # Gaming agent: 60% bandwidth
        # Background: 30% bandwidth
        # System: 10% bandwidth
        pass
```

### 6.2 AMD

```python
class AMDMemoryOptimizer:
    """AMD-specific memory optimizations."""

    def configure(self):
        # Configure Infinity Cache hints
        self.configure_infinity_cache()

        # CCD-aware NUMA optimization
        self.configure_numa_ccd()

        # Use PREFETCH for cache warming
        self.prefetch_instruction = "PREFETCH"

    def configure_infinity_cache(self):
        """Optimize for Infinity Cache (96MB L3)."""
        # Keep hot game data in Infinity Cache
        # Mark background data as non-cacheable
        pass
```

### 6.3 ARM

```python
class ARMMemoryOptimizer:
    """ARM-specific memory optimizations."""

    def configure(self):
        # Configure big.LITTLE memory affinity
        self.configure_cluster_affinity()

        # Use PRFM for prefetch
        self.prefetch_instruction = "PRFM PLDL1KEEP"

        # Enable MTE (Memory Tagging) if available
        if self.has_mte:
            self.enable_mte()

    def configure_cluster_affinity(self):
        """Keep hot data near big cores."""
        # big cores: Game data
        # LITTLE cores: Background, swap management
        pass
```

---

## 7. Očakávané Výsledky

| Metrika | Bez GAMESA | S GAMESA | Zlepšenie |
|---------|------------|----------|-----------|
| Swap latency (p99) | 50ms | 8ms | **-84%** |
| Page fault rate | 1000/s | 150/s | **-85%** |
| Memory bandwidth util | 45% | 72% | **+60%** |
| Compression ratio | 1.5x | 2.8x | **+87%** |
| Swap write amplification | 3.2x | 1.4x | **-56%** |
| Cold start time | 8s | 2s | **-75%** |

---

## 8. Implementačná Roadmap

```
Phase 1: Memory Tiering
├── Tier classification
├── Basic promotion/demotion
└── Access tracking

Phase 2: Predictive Prefetch
├── Pattern detection
├── Cross-Forex integration
└── Platform-specific prefetch

Phase 3: Advanced Swap
├── Multi-tier swap
├── Adaptive compression
└── Proactive swap-out

Phase 4: Full Integration
├── Guardian memory directives
├── Cross-platform optimization
└── ML-based prediction
```
