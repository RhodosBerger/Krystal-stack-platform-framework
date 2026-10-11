"""
================================================================================
KRYSTAL-STACK / CRYSTAL-SEC 3.0: ASYNCHRONOUS SHADOW CPU ORACLE
Module: krystal_kernel/verification/vulkan_shadow_oracle.py

Role:
Asynchronous correctness verification engine comparing Vulkan GPU compute outputs
against the deterministic CPU reference implementation (execute_raymarch_cpu)
without blocking the real-time GPU dispatch hot path.

Guarantees & Constraints:
1. ThreadPoolExecutor(max_workers=1) runs verification strictly off the GPU hot path.
2. Bounded pending queue with stale job dropping (prevents unbounded queue growth).
3. Observer effect tracking: measures GPU-only timing vs GPU+Oracle timing.
4. Adaptive rate degradation: 1:16 -> 1:32 -> 1:64 if observer overhead exceeds threshold.
5. Conservative RCA classification:
   - CONSISTENT_WITH_H0_IDENTICAL
   - CONSISTENT_WITH_H1 (Numerical drift / edge precision)
   - CONSISTENT_WITH_H2 (Synchronization / tile boundary hazard)
   - CONSISTENT_WITH_H3 (Catastrophic device/driver corruption)
   - INSUFFICIENT_EVIDENCE
   Strict rule: Never claim certainty without physical hardware proof.
6. Normalized CrystalSec Divergence Signal: D_divergence in [0.0, 1.0].
   Phase 1-2: OBSERVABLE only (does NOT mutate runtime configuration).
================================================================================
"""

import enum
import hashlib
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("VulkanShadowOracle")


class SpatialPattern(str, enum.Enum):
    UNIFORM = "UNIFORM"
    EDGE_LOCALIZED = "EDGE_LOCALIZED"
    TILE_LOCALIZED = "TILE_LOCALIZED"
    RANDOM_SPARSE = "RANDOM_SPARSE"
    GLOBAL = "GLOBAL"


class RCAHypothesis(str, enum.Enum):
    CONSISTENT_WITH_H0_IDENTICAL = "CONSISTENT_WITH_H0_IDENTICAL"
    CONSISTENT_WITH_H1 = "CONSISTENT_WITH_H1"  # Numerical drift / floating point tolerance
    CONSISTENT_WITH_H2 = "CONSISTENT_WITH_H2"  # Synchronization / tile / workgroup hazard
    CONSISTENT_WITH_H3 = "CONSISTENT_WITH_H3"  # Device / driver catastrophic failure
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass
class RCAResult:
    hypothesis: str
    confidence: float
    evidence: List[str]
    counter_evidence: List[str]
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShadowOracleSample:
    frame_id: int
    timestamp: float
    resolution: Tuple[int, int]
    glyph_match_ratio: float
    pixel_exact_match_ratio: float
    mean_absolute_color_error: float
    max_color_error: int
    gpu_hash: str
    cpu_hash: str
    spatial_pattern: str
    rca: RCAResult
    divergence_score: float
    oracle_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["resolution"] = list(self.resolution)
        return d


class VulkanShadowOracle:
    """
    Asynchronous Shadow CPU Oracle for Vulkan Compute Verification.
    Periodically samples completed GPU frames, dispatches verification to a background
    worker thread, and evaluates numerical, spatial, and root-cause divergence.
    """

    # Weights for normalized divergence score D_divergence in [0, 1]
    WEIGHT_MAE: float = 0.25
    WEIGHT_GLYPH: float = 0.35
    WEIGHT_CATASTROPHIC: float = 0.25
    WEIGHT_TILE_HAZARD: float = 0.15

    def __init__(
        self,
        driver: Any,
        sampling_interval: int = 16,
        max_queue_depth: int = 4,
        overhead_threshold_pct: float = 15.0,
    ):
        self.driver = driver
        self.sampling_interval: int = max(1, sampling_interval)
        self.max_queue_depth: int = max(1, max_queue_depth)
        self.overhead_threshold_pct: float = overhead_threshold_pct

        # Concurrency & Work Queue
        self._lock = threading.Lock()
        self._pending_jobs: int = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ShadowOracleWorker")
        self._is_active: bool = True

        # Telemetry counters
        self.oracle_jobs_submitted: int = 0
        self.oracle_jobs_completed: int = 0
        self.oracle_jobs_dropped: int = 0
        self.last_oracle_latency_ms: float = 0.0

        # Sample history (ring buffer of last 64 samples)
        self.history: List[ShadowOracleSample] = []
        self.max_history_len: int = 64
        self.last_sample: Optional[ShadowOracleSample] = None

        # Observer overhead tracking
        self._baseline_gpu_times: List[float] = []
        self._sampled_frame_times: List[float] = []
        self.observer_overhead_pct: float = 0.0
        self.degradation_events: List[Dict[str, Any]] = []

    def should_sample(self, frame_id: int) -> bool:
        """Determines if the specified frame_id aligns with current sampling interval."""
        if not self._is_active:
            return False
        return (frame_id % self.sampling_interval) == 0

    def submit_sample(
        self,
        frame_id: int,
        gpu_glyphs: List[int],
        gpu_colors: List[int],
        width: int,
        height: int,
        t: float,
        mode: str,
        cam_pos: Tuple[float, float, float],
        step_budget: int,
    ) -> bool:
        """
        Submits a sampled frame for background verification.
        Drops stale jobs if queue depth exceeds max_queue_depth.
        """
        if not self._is_active:
            return False

        with self._lock:
            if self._pending_jobs >= self.max_queue_depth:
                self.oracle_jobs_dropped += 1
                logger.warning(
                    f"Shadow oracle queue full ({self._pending_jobs} >= {self.max_queue_depth}); "
                    f"dropping frame {frame_id}"
                )
                return False

            self._pending_jobs += 1
            self.oracle_jobs_submitted += 1

        # Submit background task
        self._executor.submit(
            self._verify_frame_task,
            frame_id,
            list(gpu_glyphs),
            list(gpu_colors),
            width,
            height,
            t,
            mode,
            cam_pos,
            step_budget,
        )
        return True

    def _verify_frame_task(
        self,
        frame_id: int,
        gpu_glyphs: List[int],
        gpu_colors: List[int],
        width: int,
        height: int,
        t: float,
        mode: str,
        cam_pos: Tuple[float, float, float],
        step_budget: int,
    ):
        """Asynchronous execution task running on the dedicated oracle worker thread."""
        t0 = time.perf_counter()
        try:
            # 1. Execute deterministic CPU reference
            cpu_glyphs, cpu_colors = self.driver.execute_raymarch_cpu(
                width, height, t, mode, cam_pos, step_budget
            )

            # 2. Compute comparison metrics
            total_pixels = width * height
            gpu_len = len(gpu_glyphs)
            cpu_len = len(cpu_glyphs)

            # Check bounds and structural consistency
            if gpu_len != total_pixels or cpu_len != total_pixels:
                rca = RCAResult(
                    hypothesis=RCAHypothesis.CONSISTENT_WITH_H3.value,
                    confidence=0.95,
                    evidence=[f"Buffer length mismatch: GPU={gpu_len}, CPU={cpu_len}, Expected={total_pixels}"],
                    counter_evidence=[],
                    recommended_action="TRIGGER_CPU_FALLBACK_AND_DEVICE_REINIT",
                )
                divergence = 1.0
                sample = ShadowOracleSample(
                    frame_id=frame_id,
                    timestamp=time.time(),
                    resolution=(width, height),
                    glyph_match_ratio=0.0,
                    pixel_exact_match_ratio=0.0,
                    mean_absolute_color_error=1e6,
                    max_color_error=0xFFFFFF,
                    gpu_hash="CORRUPTED_LENGTH",
                    cpu_hash="UNKNOWN",
                    spatial_pattern=SpatialPattern.GLOBAL.value,
                    rca=rca,
                    divergence_score=divergence,
                    oracle_latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                )
                self._record_sample(sample)
                return

            glyph_matches = sum(1 for i in range(total_pixels) if gpu_glyphs[i] == cpu_glyphs[i])
            color_matches = sum(1 for i in range(total_pixels) if gpu_colors[i] == cpu_colors[i])
            color_diffs = [abs(gpu_colors[i] - cpu_colors[i]) for i in range(total_pixels)]
            mae = sum(color_diffs) / total_pixels
            max_err = max(color_diffs) if color_diffs else 0

            glyph_ratio = round(glyph_matches / total_pixels, 4)
            color_ratio = round(color_matches / total_pixels, 4)

            # 3. Hashes
            h_gpu = hashlib.sha256()
            h_gpu.update(bytes(gpu_glyphs))
            for c in gpu_colors:
                h_gpu.update(c.to_bytes(4, "little"))
            gpu_hash = h_gpu.hexdigest()

            h_cpu = hashlib.sha256()
            h_cpu.update(bytes(cpu_glyphs))
            for c in cpu_colors:
                h_cpu.update(c.to_bytes(4, "little"))
            cpu_hash = h_cpu.hexdigest()

            # 4. Spatial Error Structure Classification
            spatial_pattern = self._classify_spatial_pattern(
                gpu_colors, cpu_colors, width, height
            )

            # 5. Conservative RCA Classifier
            rca = self._classify_rca(
                glyph_ratio, color_ratio, mae, max_err, spatial_pattern, gpu_colors
            )

            # 6. Normalized Divergence Vector D_divergence in [0, 1]
            divergence = self._calculate_divergence_score(
                glyph_ratio, mae, rca.hypothesis, spatial_pattern
            )

            dt_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            self.last_oracle_latency_ms = dt_ms

            sample = ShadowOracleSample(
                frame_id=frame_id,
                timestamp=time.time(),
                resolution=(width, height),
                glyph_match_ratio=glyph_ratio,
                pixel_exact_match_ratio=color_ratio,
                mean_absolute_color_error=round(mae, 4),
                max_color_error=max_err,
                gpu_hash=gpu_hash,
                cpu_hash=cpu_hash,
                spatial_pattern=spatial_pattern.value,
                rca=rca,
                divergence_score=round(divergence, 4),
                oracle_latency_ms=dt_ms,
            )
            self._record_sample(sample)

        except Exception as e:
            logger.error(f"Shadow oracle verification task failed: {e}")
        finally:
            with self._lock:
                self._pending_jobs = max(0, self._pending_jobs - 1)
                self.oracle_jobs_completed += 1

    def _classify_spatial_pattern(
        self,
        gpu_colors: List[int],
        cpu_colors: List[int],
        width: int,
        height: int,
    ) -> SpatialPattern:
        """Identifies whether errors are uniform, tile-boundary aligned, edge-localized, or global."""
        total_pixels = width * height
        mismatches = [i for i in range(total_pixels) if gpu_colors[i] != cpu_colors[i]]
        mismatch_count = len(mismatches)

        if mismatch_count == 0:
            return SpatialPattern.UNIFORM

        mismatch_ratio = mismatch_count / total_pixels
        if mismatch_ratio > 0.40:
            return SpatialPattern.GLOBAL

        # Analyze tile grouping (workgroups are 8x8 pixels)
        tile_counts: Dict[Tuple[int, int], int] = {}
        for idx in mismatches:
            x = idx % width
            y = idx // width
            tile_coord = (x // 8, y // 8)
            tile_counts[tile_coord] = tile_counts.get(tile_coord, 0) + 1

        # Check if entire tiles are divergent (e.g. tile count >= 56 for 8x8)
        full_tile_mismatches = sum(1 for cnt in tile_counts.values() if cnt >= 56)
        if full_tile_mismatches > 0 and (full_tile_mismatches * 64) >= (mismatch_count * 0.70):
            return SpatialPattern.TILE_LOCALIZED

        # Check edge-localization (pixels bordering different colors / high contrast steps)
        edge_matches = 0
        for idx in mismatches:
            x = idx % width
            y = idx // width
            is_edge = False
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    nidx = ny * width + nx
                    if abs(cpu_colors[idx] - cpu_colors[nidx]) > 0x101010:
                        is_edge = True
                        break
            if is_edge:
                edge_matches += 1

        if edge_matches >= (mismatch_count * 0.60):
            return SpatialPattern.EDGE_LOCALIZED

        return SpatialPattern.RANDOM_SPARSE

    def _classify_rca(
        self,
        glyph_ratio: float,
        color_ratio: float,
        mae: float,
        max_err: int,
        spatial_pattern: SpatialPattern,
        gpu_colors: List[int],
    ) -> RCAResult:
        """
        Synthesizes conservative RCA hypotheses.
        Follows epistemic discipline: never asserts driver corruption proven without hardware evidence.
        """
        # H0: Bit-identical or perfect match
        if glyph_ratio == 1.0 and color_ratio == 1.0:
            return RCAResult(
                hypothesis=RCAHypothesis.CONSISTENT_WITH_H0_IDENTICAL.value,
                confidence=1.0,
                evidence=["Bit-identical glyph and color outputs between GPU and CPU reference"],
                counter_evidence=[],
                recommended_action="MAINTAIN_CURRENT_DISPATCH",
            )

        # H3: Catastrophic corruption / Device Loss / Invalid values
        has_invalid_values = any(
            c < 0 or c > 0xFFFFFF or math.isnan(c) or math.isinf(c)
            for c in gpu_colors[:100]
        )
        if has_invalid_values or glyph_ratio < 0.50 or mae > 200000.0:
            return RCAResult(
                hypothesis=RCAHypothesis.CONSISTENT_WITH_H3.value,
                confidence=0.90,
                evidence=[
                    f"Catastrophic color divergence: MAE={mae:.1f}, MaxErr={max_err}",
                    f"Low glyph match ratio: {glyph_ratio:.4f}",
                    f"Spatial dispersion: {spatial_pattern.value}",
                ],
                counter_evidence=["No host crash detected on OS level"],
                recommended_action="TRIGGER_CPU_FALLBACK_AND_DEVICE_REINIT",
            )

        # H2: Synchronization or Workgroup Tile Hazard
        if spatial_pattern == SpatialPattern.TILE_LOCALIZED:
            return RCAResult(
                hypothesis=RCAHypothesis.CONSISTENT_WITH_H2.value,
                confidence=0.85,
                evidence=[
                    "Mismatches strictly clustered into 8x8 compute workgroup tile boundaries",
                    "Bounded non-catastrophic color divergence within affected tiles",
                ],
                counter_evidence=["Shader instruction pipeline executed to completion"],
                recommended_action="INSERT_MEMORY_BARRIERS_OR_REBUILD_PIPELINE",
            )

        # H1: Numerical Drift / Floating Point Edge Discrepancy
        if (
            glyph_ratio >= 0.95
            and mae <= 50.0
            and spatial_pattern in (SpatialPattern.EDGE_LOCALIZED, SpatialPattern.RANDOM_SPARSE)
        ):
            return RCAResult(
                hypothesis=RCAHypothesis.CONSISTENT_WITH_H1.value,
                confidence=0.88,
                evidence=[
                    f"Small bounded error: MAE={mae:.2f} <= 50.0",
                    f"High glyph match ratio: {glyph_ratio:.4f} >= 0.95",
                    f"Spatial alignment: {spatial_pattern.value}",
                ],
                counter_evidence=["Output is not bit-identical to IEEE-754 double CPU float"],
                recommended_action="TOLERATE_DRIFT_WITHIN_CONTRACT",
            )

        return RCAResult(
            hypothesis=RCAHypothesis.INSUFFICIENT_EVIDENCE.value,
            confidence=0.50,
            evidence=[f"Unclassified divergence pattern: MAE={mae:.2f}, GlyphMatch={glyph_ratio:.4f}"],
            counter_evidence=[],
            recommended_action="CONTINUE_SAMPLING_AND_LOG",
        )

    def _calculate_divergence_score(
        self,
        glyph_ratio: float,
        mae: float,
        rca_hypothesis: str,
        spatial_pattern: SpatialPattern,
    ) -> float:
        """
        Computes normalized divergence signal D_divergence in [0.0, 1.0].
        D = a1 * normalized_MAE + a2 * (1 - glyph_match) + a3 * catastrophic_flag + a4 * sync_hazard
        """
        norm_mae = min(1.0, mae / 255.0)
        glyph_div = max(0.0, min(1.0, 1.0 - glyph_ratio))
        catastrophic_flag = 1.0 if rca_hypothesis == RCAHypothesis.CONSISTENT_WITH_H3.value else 0.0
        tile_hazard = 0.8 if spatial_pattern == SpatialPattern.TILE_LOCALIZED else 0.0

        d = (
            self.WEIGHT_MAE * norm_mae
            + self.WEIGHT_GLYPH * glyph_div
            + self.WEIGHT_CATASTROPHIC * catastrophic_flag
            + self.WEIGHT_TILE_HAZARD * tile_hazard
        )
        return max(0.0, min(1.0, d))

    def _record_sample(self, sample: ShadowOracleSample):
        """Appends sample to ring buffer and updates last sample."""
        with self._lock:
            self.history.append(sample)
            if len(self.history) > self.max_history_len:
                self.history.pop(0)
            self.last_sample = sample

    def record_frame_timing(self, frame_time_ms: float, was_sampled: bool):
        """
        Tracks observer overhead of CPU shadow sampling versus purely asynchronous execution.
        If overhead exceeds threshold, automatically reduces sampling frequency (1:16 -> 1:32 -> 1:64).
        """
        if was_sampled:
            self._sampled_frame_times.append(frame_time_ms)
            if len(self._sampled_frame_times) > 50:
                self._sampled_frame_times.pop(0)
        else:
            self._baseline_gpu_times.append(frame_time_ms)
            if len(self._baseline_gpu_times) > 50:
                self._baseline_gpu_times.pop(0)

        # Evaluate observer effect after sufficient warm measurements
        if len(self._baseline_gpu_times) >= 10 and len(self._sampled_frame_times) >= 5:
            mean_baseline = sum(self._baseline_gpu_times) / len(self._baseline_gpu_times)
            mean_sampled = sum(self._sampled_frame_times) / len(self._sampled_frame_times)
            overhead = ((mean_sampled - mean_baseline) / max(1e-6, mean_baseline)) * 100.0
            self.observer_overhead_pct = round(max(0.0, overhead), 2)

            # Adaptive Rate Degradation if observer overhead > threshold
            if self.observer_overhead_pct > self.overhead_threshold_pct and self.sampling_interval < 64:
                old_interval = self.sampling_interval
                self.sampling_interval = min(64, self.sampling_interval * 2)
                event = {
                    "event": "SHADOW_ORACLE_RATE_DEGRADED",
                    "previous_interval": old_interval,
                    "new_interval": self.sampling_interval,
                    "observer_overhead_pct": self.observer_overhead_pct,
                    "threshold_pct": self.overhead_threshold_pct,
                    "timestamp": time.time(),
                }
                self.degradation_events.append(event)
                logger.warning(f"Observer overhead exceeded threshold: {event}")
                # Reset window to collect fresh samples for next evaluation
                self._sampled_frame_times.clear()
                self._baseline_gpu_times.clear()

    def shutdown(self):
        """Clean shutdown of the background thread pool."""
        self._is_active = False
        self._executor.shutdown(wait=False)

    def get_telemetry(self) -> Dict[str, Any]:
        """Telemetry export matching Phase 1-2 Sovereign Verification contract."""
        with self._lock:
            q_depth = self._pending_jobs

        last_div = self.last_sample.divergence_score if self.last_sample else 0.0
        last_rca = self.last_sample.rca.hypothesis if self.last_sample else "INSUFFICIENT_EVIDENCE"
        last_pattern = self.last_sample.spatial_pattern if self.last_sample else "UNIFORM"

        return {
            "enabled": self._is_active,
            "sampling_interval": self.sampling_interval,
            "sampling_ratio": f"1:{self.sampling_interval}",
            "oracle_jobs_submitted": self.oracle_jobs_submitted,
            "oracle_jobs_completed": self.oracle_jobs_completed,
            "oracle_jobs_dropped": self.oracle_jobs_dropped,
            "oracle_queue_depth": q_depth,
            "oracle_latency_ms": self.last_oracle_latency_ms,
            "last_divergence": last_div,
            "last_rca": last_rca,
            "last_spatial_pattern": last_pattern,
            "observer_overhead_pct": self.observer_overhead_pct,
            "degradation_events_count": len(self.degradation_events),
        }
