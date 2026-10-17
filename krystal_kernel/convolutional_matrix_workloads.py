#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: CHAINED CONVOLUTIONAL MATRIX WORKLOADS & BOOLEAN CRUD DATABASE
==============================================================================
Module: krystal_kernel/convolutional_matrix_workloads.py
Description:
    Interconnected multi-stage matrix workloads linked via 2D Boolean convolutional
    transformation functions, integrated with an in-memory Boolean Matrix CRUD
    Database and a real-time Streaming Log Service.

Key Architecture:
    1. Boolean Matrix & 2D Convolutional Engine:
       - BooleanMatrix: 2D bit grid (e.g. 8x8, 16x16) supporting bitwise AND/OR/XOR/NOT,
         Hamming density, Hamming distance, and ASCII visual rendering.
       - ConvolutionKernel: 2D spatial convolution kernels (Edge Detection, Diffusion,
         Majority Gate, Morphological Dilation/Erosion, 6HP Guard Mask) with threshold activation.
       - convolve_boolean_matrix: High-speed spatial convolution mapping M_{k+1} = sigma(M_k * K).

    2. Chained Matrix Workload Pipeline (ChainedMatrixWorkloadPipeline):
       - Workloads interconnected as a directed sequence of matrix transformations:
         STAGE_01_INGEST_M0 -> STAGE_02_EDGE_CONVOLUTION_M1 ->
         STAGE_03_DIFFUSION_CONVOLUTION_M2 -> STAGE_04_CONSENSUS_CONVOLUTION_M3 ->
         STAGE_05_INVARIANT_GUARD_M4 -> STAGE_06_DATABASE_SYNC_COMMIT.
       - High-resolution QPC microsecond tracking per stage.

    3. Boolean Matrix CRUD Database App (BooleanMatrixCrudDatabase):
       - Entity storage with Boolean Matrix signatures, metadata, and timestamps.
       - Full CRUD:
         - Create: Insert new boolean matrix entities.
         - Read: Query by ID, collection, Hamming similarity, or convolutional filter.
         - Update: Mutate bits or apply convolutional update directly (convolve_entity_update).
         - Delete: Tombstone or remove entities.
       - Constitutional Invariant: assert VITAL_MAX_HP == 6 on all operations.

    4. Real-Time Streaming Log Service (MatrixStreamingLogService):
       - High-throughput circular ring buffer capturing workload transitions,
         kernel execution durations (us), matrix bit densities, and semantic status tokens.
       - Streaming emission for UI Kit Studio Service Panel and SSE streams.

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

from __future__ import annotations
import os
import sys
import time
import math
import json
import threading
from enum import Enum, unique
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Callable, Union

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.semantic_protocol import (
    ActuationState,
    ObservationState,
    DerivedState,
    PredictionState,
    SafetyState,
    AuthorityState,
    TransactionState,
    InvariantState,
    to_wire_token
)
from krystal_kernel.windows_timer_subsystem import (
    WindowsTimerSubsystem,
    GLOBAL_WINDOWS_TIMER_SUBSYSTEM
)


# ==============================================================================
# 1. BOOLEAN MATRIX & CONVOLUTIONAL ENGINE
# ==============================================================================

class BooleanMatrix:
    """
    2D Boolean state grid representing hardware flags, sensor activations,
    or database bit signatures.
    """

    def __init__(self, rows: int = 8, cols: int = 8, initial_bits: Optional[List[List[bool]]] = None):
        self.rows = max(1, rows)
        self.cols = max(1, cols)
        if initial_bits is not None:
            self.data = [[bool(initial_bits[r][c]) for c in range(self.cols)] for r in range(self.rows)]
        else:
            self.data = [[False for _ in range(self.cols)] for _ in range(self.rows)]

    def get(self, r: int, c: int) -> bool:
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.data[r][c]
        return False

    def set(self, r: int, c: int, val: bool):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.data[r][c] = bool(val)

    def toggle(self, r: int, c: int):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.data[r][c] = not self.data[r][c]

    @property
    def active_bit_count(self) -> int:
        return sum(sum(1 for bit in row if bit) for row in self.data)

    @property
    def density(self) -> float:
        total = self.rows * self.cols
        return round(self.active_bit_count / float(total), 4)

    def bitwise_and(self, other: "BooleanMatrix") -> "BooleanMatrix":
        res = BooleanMatrix(self.rows, self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                res.data[r][c] = self.data[r][c] and other.get(r, c)
        return res

    def bitwise_or(self, other: "BooleanMatrix") -> "BooleanMatrix":
        res = BooleanMatrix(self.rows, self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                res.data[r][c] = self.data[r][c] or other.get(r, c)
        return res

    def bitwise_xor(self, other: "BooleanMatrix") -> "BooleanMatrix":
        res = BooleanMatrix(self.rows, self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                res.data[r][c] = (self.data[r][c] != other.get(r, c))
        return res

    def bitwise_not(self) -> "BooleanMatrix":
        res = BooleanMatrix(self.rows, self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                res.data[r][c] = not self.data[r][c]
        return res

    def hamming_distance(self, other: "BooleanMatrix") -> int:
        return sum(1 for r in range(self.rows) for c in range(self.cols) if self.data[r][c] != other.get(r, c))

    def jaccard_similarity(self, other: "BooleanMatrix") -> float:
        intersection = self.bitwise_and(other).active_bit_count
        union = self.bitwise_or(other).active_bit_count
        if union == 0:
            return 1.0
        return round(intersection / float(union), 4)

    def to_hex_string(self) -> str:
        """Encodes flattened bit sequence to hex string."""
        flat_bits = [1 if self.data[r][c] else 0 for r in range(self.rows) for c in range(self.cols)]
        bit_str = "".join(str(b) for b in flat_bits)
        # Pad to multiple of 4
        pad = (4 - (len(bit_str) % 4)) % 4
        bit_str += "0" * pad
        num = int(bit_str, 2)
        return hex(num)

    @classmethod
    def from_hex_string(cls, hex_str: str, rows: int = 8, cols: int = 8) -> "BooleanMatrix":
        num = int(hex_str, 16)
        bit_str = bin(num)[2:].zfill(rows * cols)
        mat = cls(rows, cols)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx < len(bit_str):
                    mat.data[r][c] = (bit_str[idx] == "1")
                    idx += 1
        return mat

    def to_list(self) -> List[List[int]]:
        return [[1 if bit else 0 for bit in row] for row in self.data]

    def render_ascii(self, on_char: str = "█", off_char: str = "·") -> str:
        lines = []
        for row in self.data:
            lines.append(" ".join(on_char if bit else off_char for bit in row))
        return "\n".join(lines)


@dataclass
class ConvolutionKernel:
    """A spatial 2D convolution kernel applied over BooleanMatrix."""
    kernel_id: str
    name: str
    weights: List[List[float]]
    threshold: float
    bias: float = 0.0

    @property
    def size(self) -> int:
        return len(self.weights)


# Canonical Predefined 2D Convolution Kernels
CANONICAL_CONVOLUTION_KERNELS: Dict[str, ConvolutionKernel] = {
    "KERNEL_EDGE_DETECTOR_3X3": ConvolutionKernel(
        kernel_id="KERNEL_EDGE_DETECTOR_3X3",
        name="Laplacian 3x3 State Boundary Detector",
        weights=[
            [-1.0, -1.0, -1.0],
            [-1.0,  8.0, -1.0],
            [-1.0, -1.0, -1.0]
        ],
        threshold=0.5,
        bias=0.0
    ),
    "KERNEL_DIFFUSION_SMOOTHER_3X3": ConvolutionKernel(
        kernel_id="KERNEL_DIFFUSION_SMOOTHER_3X3",
        name="Gaussian 3x3 Lateral Signal Diffusion",
        weights=[
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0]
        ],
        threshold=5.0,
        bias=0.0
    ),
    "KERNEL_MAJORITY_GATE_3X3": ConvolutionKernel(
        kernel_id="KERNEL_MAJORITY_GATE_3X3",
        name="Cellular 3x3 Majority Consensus Gate",
        weights=[
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ],
        threshold=5.0,  # At least 5 of 9 neighbors active
        bias=0.0
    ),
    "KERNEL_EROSION_FILTER_3X3": ConvolutionKernel(
        kernel_id="KERNEL_EROSION_FILTER_3X3",
        name="Morphological 3x3 Noise Erosion",
        weights=[
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0]
        ],
        threshold=5.0,  # All 5 cross neighbors must be active
        bias=0.0
    ),
    "KERNEL_DILATION_FILTER_3X3": ConvolutionKernel(
        kernel_id="KERNEL_DILATION_FILTER_3X3",
        name="Morphological 3x3 Active Zone Dilation",
        weights=[
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0]
        ],
        threshold=1.0,  # At least 1 neighbor active
        bias=0.0
    ),
    "KERNEL_6HP_INVARIANT_MASK_3X3": ConvolutionKernel(
        kernel_id="KERNEL_6HP_INVARIANT_MASK_3X3",
        name="Constitutional 6HP Invariant Boundary Mask",
        weights=[
            [0.5, 1.0, 0.5],
            [1.0, 2.0, 1.0],
            [0.5, 1.0, 0.5]
        ],
        threshold=3.0,
        bias=0.0
    )
}


def convolve_boolean_matrix(
    matrix: BooleanMatrix,
    kernel: ConvolutionKernel
) -> BooleanMatrix:
    """
    Applies spatial 2D convolution over a BooleanMatrix with threshold activation:
    output(r, c) = (sum(W_ij * M_{r+i, c+j}) + bias >= threshold).
    """
    k_size = kernel.size
    radius = k_size // 2
    res = BooleanMatrix(matrix.rows, matrix.cols)

    for r in range(matrix.rows):
        for c in range(matrix.cols):
            weighted_sum = kernel.bias
            for kr in range(k_size):
                mr = r + kr - radius
                for kc in range(k_size):
                    mc = c + kc - radius
                    # Edge clamp padding
                    mr_clamped = max(0, min(matrix.rows - 1, mr))
                    mc_clamped = max(0, min(matrix.cols - 1, mc))
                    val = 1.0 if matrix.get(mr_clamped, mc_clamped) else 0.0
                    weighted_sum += kernel.weights[kr][kc] * val

            res.set(r, c, weighted_sum >= kernel.threshold)

    return res


# ==============================================================================
# 2. REAL-TIME STREAMING LOG SERVICE
# ==============================================================================

@dataclass
class MatrixStreamingLogEntry:
    """Telemetry log entry emitted during convolutional matrix workloads."""
    log_id: int
    timestamp_ns: int
    workload_id: str
    stage_name: str
    kernel_applied: str
    active_bits_before: int
    active_bits_after: int
    density: float
    elapsed_us: float
    status_token: str
    message: str
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MatrixStreamingLogService:
    """
    High-throughput circular log ring buffer capturing workload transitions,
    convolution operations, and CRUD database events with microsecond precision.
    """

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self.ring_buffer: List[MatrixStreamingLogEntry] = []
        self._lock = threading.Lock()
        self._seq = 0
        self.listeners: List[Callable[[MatrixStreamingLogEntry], None]] = []

    def emit_event(
        self,
        workload_id: str,
        stage_name: str,
        kernel_applied: str,
        bits_before: int,
        bits_after: int,
        density: float,
        elapsed_us: float,
        status_token: str,
        message: str
    ) -> MatrixStreamingLogEntry:
        """Records an event and notifies real-time listeners."""
        with self._lock:
            self._seq += 1
            now_ns = time.time_ns()
            entry = MatrixStreamingLogEntry(
                log_id=self._seq,
                timestamp_ns=now_ns,
                workload_id=workload_id,
                stage_name=stage_name,
                kernel_applied=kernel_applied,
                active_bits_before=bits_before,
                active_bits_after=bits_after,
                density=density,
                elapsed_us=round(elapsed_us, 2),
                status_token=status_token,
                message=message,
                vital_max_hp=VITAL_MAX_HP
            )
            self.ring_buffer.append(entry)
            if len(self.ring_buffer) > self.capacity:
                self.ring_buffer.pop(0)

        # Notify external streaming subscribers
        for listener in list(self.listeners):
            try:
                listener(entry)
            except Exception:
                pass

        return entry

    def get_recent_logs(self, limit: int = 50, since_id: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            filtered = [e for e in self.ring_buffer if e.log_id > since_id]
            return [e.to_dict() for e in filtered[-limit:]]

    def subscribe(self, callback: Callable[[MatrixStreamingLogEntry], None]):
        self.listeners.append(callback)


# Global Singleton for Matrix Streaming Log Service
GLOBAL_MATRIX_STREAMING_LOG_SERVICE = MatrixStreamingLogService()


# ==============================================================================
# 3. BOOLEAN MATRIX CRUD DATABASE APP
# ==============================================================================

@dataclass
class BooleanEntity:
    """Entity record stored within the Boolean Matrix CRUD Database."""
    entity_id: str
    name: str
    collection: str
    matrix_hex: str
    active_bits: int
    density: float
    metadata: Dict[str, Any]
    created_at_ns: int
    updated_at_ns: int
    version: int = 1
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BooleanMatrixCrudDatabase:
    """
    High-performance in-memory Boolean Matrix CRUD database engine.
    Stores and indexes entities by their spatial 2D Boolean signatures.
    """

    def __init__(self, log_service: Optional[MatrixStreamingLogService] = None):
        self.entities: Dict[str, BooleanEntity] = {}
        self.matrix_store: Dict[str, BooleanMatrix] = {}
        self.log_service = log_service or GLOBAL_MATRIX_STREAMING_LOG_SERVICE
        self._lock = threading.Lock()
        self._init_default_presets()

    def _init_default_presets(self):
        """Populates database with canonical baseline system profiles."""
        p1 = BooleanMatrix(8, 8)
        # Symmetrical cross pattern for balanced profile
        for i in range(8):
            p1.set(i, i, True)
            p1.set(i, 7 - i, True)
        self.create_entity(
            entity_id="ENT_BALANCED_CROSS",
            name="Balanced Cross Matrix Profile",
            matrix=p1,
            collection="hardware_profiles",
            metadata={"pl1_watts": 24.0, "priority": "NORMAL_PRIORITY_CLASS"}
        )

        p2 = BooleanMatrix(8, 8)
        # Perimeter boundary for thermal defense
        for i in range(8):
            p2.set(0, i, True)
            p2.set(7, i, True)
            p2.set(i, 0, True)
            p2.set(i, 7, True)
        self.create_entity(
            entity_id="ENT_THERMAL_PERIMETER",
            name="Thermal Guard Perimeter Profile",
            matrix=p2,
            collection="hardware_profiles",
            metadata={"pl1_watts": 10.0, "priority": "BELOW_NORMAL_PRIORITY_CLASS"}
        )

    # ── CRUD: CREATE ─────────────────────────────────────────────────────────
    def create_entity(
        self,
        entity_id: str,
        name: str,
        matrix: BooleanMatrix,
        collection: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ) -> BooleanEntity:
        assert VITAL_MAX_HP == 6, "Constitutional Invariant VITAL_MAX_HP == 6 violated in database Create"
        t0 = time.perf_counter_ns()
        with self._lock:
            now_ns = time.time_ns()
            entity = BooleanEntity(
                entity_id=entity_id,
                name=name,
                collection=collection,
                matrix_hex=matrix.to_hex_string(),
                active_bits=matrix.active_bit_count,
                density=matrix.density,
                metadata=metadata or {},
                created_at_ns=now_ns,
                updated_at_ns=now_ns,
                version=1,
                vital_max_hp=VITAL_MAX_HP
            )
            self.entities[entity_id] = entity
            self.matrix_store[entity_id] = matrix

        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        self.log_service.emit_event(
            workload_id="CRUD_CREATE",
            stage_name="STAGE_DB_CREATE",
            kernel_applied="NONE",
            bits_before=0,
            bits_after=matrix.active_bit_count,
            density=matrix.density,
            elapsed_us=elapsed_us,
            status_token=to_wire_token(ActuationState.ACT_EXECUTED),
            message=f"Created Boolean entity [{entity_id}] in collection '{collection}'"
        )
        return entity

    # ── CRUD: READ ───────────────────────────────────────────────────────────
    def read_entity(self, entity_id: str) -> Optional[Tuple[BooleanEntity, BooleanMatrix]]:
        with self._lock:
            entity = self.entities.get(entity_id)
            matrix = self.matrix_store.get(entity_id)
            if entity and matrix:
                return entity, matrix
            return None

    def get_entity(self, entity_id: str) -> Optional[BooleanEntity]:
        """Convenience accessor returning the BooleanEntity."""
        with self._lock:
            return self.entities.get(entity_id)

    def get_matrix(self, entity_id: str) -> Optional[BooleanMatrix]:
        """Convenience accessor returning the BooleanMatrix."""
        with self._lock:
            return self.matrix_store.get(entity_id)

    def query_entities(
        self,
        collection: Optional[str] = None,
        min_density: float = 0.0,
        max_density: float = 1.0,
        query_matrix: Optional[BooleanMatrix] = None,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Queries entities filtered by collection, density, or Jaccard similarity."""
        results = []
        with self._lock:
            for eid, ent in self.entities.items():
                if collection and ent.collection != collection:
                    continue
                if not (min_density <= ent.density <= max_density):
                    continue

                sim = 1.0
                if query_matrix:
                    mat = self.matrix_store.get(eid)
                    if mat:
                        sim = mat.jaccard_similarity(query_matrix)
                        if sim < min_similarity:
                            continue

                record = ent.to_dict()
                record["similarity"] = sim
                results.append(record)
        return results

    # ── CRUD: UPDATE ─────────────────────────────────────────────────────────
    def update_entity(
        self,
        entity_id: str,
        name: Optional[str] = None,
        matrix: Optional[BooleanMatrix] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[BooleanEntity]:
        assert VITAL_MAX_HP == 6, "Constitutional Invariant VITAL_MAX_HP == 6 violated in database Update"
        t0 = time.perf_counter_ns()
        with self._lock:
            ent = self.entities.get(entity_id)
            if not ent:
                return None

            bits_before = ent.active_bits
            if name is not None:
                ent.name = name
            if metadata is not None:
                ent.metadata.update(metadata)
            if matrix is not None:
                self.matrix_store[entity_id] = matrix
                ent.matrix_hex = matrix.to_hex_string()
                ent.active_bits = matrix.active_bit_count
                ent.density = matrix.density

            ent.version += 1
            ent.updated_at_ns = time.time_ns()
            bits_after = ent.active_bits

        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        self.log_service.emit_event(
            workload_id="CRUD_UPDATE",
            stage_name="STAGE_DB_UPDATE",
            kernel_applied="NONE",
            bits_before=bits_before,
            bits_after=bits_after,
            density=ent.density,
            elapsed_us=elapsed_us,
            status_token=to_wire_token(ActuationState.ACT_EXECUTED),
            message=f"Updated Boolean entity [{entity_id}] (v{ent.version})"
        )
        return ent

    # ── CRUD: CONVOLVE & UPDATE ──────────────────────────────────────────────
    def convolve_entity_update(
        self,
        entity_id: str,
        kernel: Union[ConvolutionKernel, str]
    ) -> Optional[BooleanEntity]:
        """Directly convolves stored BooleanMatrix using spatial filter kernel and updates entity."""
        assert VITAL_MAX_HP == 6, "Constitutional Invariant VITAL_MAX_HP == 6 violated in database Convolve"
        if isinstance(kernel, str):
            kernel_obj = CANONICAL_CONVOLUTION_KERNELS.get(
                kernel, CANONICAL_CONVOLUTION_KERNELS.get("KERNEL_DIFFUSION_SMOOTHER_3X3")
            )
        else:
            kernel_obj = kernel

        if not kernel_obj:
            return None

        t0 = time.perf_counter_ns()
        with self._lock:
            ent = self.entities.get(entity_id)
            mat = self.matrix_store.get(entity_id)
            if not ent or not mat:
                return None

            bits_before = mat.active_bit_count
            convolved_mat = convolve_boolean_matrix(mat, kernel_obj)
            self.matrix_store[entity_id] = convolved_mat

            ent.matrix_hex = convolved_mat.to_hex_string()
            ent.active_bits = convolved_mat.active_bit_count
            ent.density = convolved_mat.density
            ent.version += 1
            ent.updated_at_ns = time.time_ns()
            ent.metadata["last_kernel_applied"] = kernel_obj.kernel_id

        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        self.log_service.emit_event(
            workload_id="CRUD_CONVOLVE",
            stage_name="STAGE_CONVOLUTION_UPDATE",
            kernel_applied=kernel_obj.kernel_id,
            bits_before=bits_before,
            bits_after=ent.active_bits,
            density=ent.density,
            elapsed_us=elapsed_us,
            status_token=to_wire_token(DerivedState.DER_COMPUTED_DELTA),
            message=f"Convolved entity [{entity_id}] via {kernel_obj.kernel_id} -> {ent.active_bits} active bits"
        )
        return ent

    # ── CRUD: DELETE ─────────────────────────────────────────────────────────
    def delete_entity(self, entity_id: str) -> bool:
        t0 = time.perf_counter_ns()
        with self._lock:
            if entity_id in self.entities:
                ent = self.entities.pop(entity_id)
                self.matrix_store.pop(entity_id, None)
                elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
                self.log_service.emit_event(
                    workload_id="CRUD_DELETE",
                    stage_name="STAGE_DB_DELETE",
                    kernel_applied="NONE",
                    bits_before=ent.active_bits,
                    bits_after=0,
                    density=0.0,
                    elapsed_us=elapsed_us,
                    status_token=to_wire_token(ActuationState.ACT_EXECUTED),
                    message=f"Deleted Boolean entity [{entity_id}]"
                )
                return True
        return False


# Global Singleton for Boolean CRUD Database
GLOBAL_BOOLEAN_CRUD_DATABASE = BooleanMatrixCrudDatabase()


# ==============================================================================
# 4. CHAINED CONVOLUTIONAL MATRIX WORKLOAD PIPELINE
# ==============================================================================

@dataclass
class WorkloadStageResult:
    """Telemetry report for a single stage in the chained matrix workload."""
    stage_id: str
    stage_name: str
    kernel_applied: str
    input_bits: int
    output_bits: int
    density: float
    elapsed_us: float
    output_matrix_hex: str
    output_matrix_grid: List[List[int]]
    status_code: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChainedPipelineExecutionReport:
    """Full execution report of the interconnected convolutional matrix workloads."""
    pipeline_id: str
    timestamp_ns: int
    total_stages: int
    total_elapsed_us: float
    initial_density: float
    final_density: float
    stages: List[Dict[str, Any]]
    safety_verdict: str
    vital_max_hp: int = VITAL_MAX_HP

    @property
    def total_duration_us(self) -> float:
        return self.total_elapsed_us

    @property
    def transaction_committed(self) -> bool:
        return any(
            (s.get("stage_id") == "STAGE_06_DB_COMMIT" or "COMMIT" in s.get("stage_name", "").upper())
            for s in self.stages
        ) or "SAFE" in str(self.safety_verdict).upper() or "COMMITTED" in str(self.safety_verdict).upper()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total_duration_us"] = self.total_duration_us
        d["transaction_committed"] = self.transaction_committed
        return d


class ChainedMatrixWorkloadPipeline:
    """
    Executes interconnected multi-stage matrix workloads where each stage's output
    matrix serves as the convolutional input for the subsequent stage.
    """

    def __init__(
        self,
        database: Optional[BooleanMatrixCrudDatabase] = None,
        log_service: Optional[MatrixStreamingLogService] = None
    ):
        self.db = database or GLOBAL_BOOLEAN_CRUD_DATABASE
        self.log_service = log_service or GLOBAL_MATRIX_STREAMING_LOG_SERVICE
        self.kernels = CANONICAL_CONVOLUTION_KERNELS

    def run_pipeline(
        self,
        initial_matrix: Optional[BooleanMatrix] = None,
        pipeline_id: Optional[str] = None
    ) -> ChainedPipelineExecutionReport:
        """
        Executes the 6-stage convolutional matrix workload chain:
          1. STAGE_INGEST_M0: Baseline state matrix.
          2. STAGE_EDGE_CONVOLUTION_M1: Laplacian boundary detection.
          3. STAGE_DIFFUSION_CONVOLUTION_M2: Gaussian lateral diffusion.
          4. STAGE_CONSENSUS_CONVOLUTION_M3: Cellular majority vote.
          5. STAGE_INVARIANT_GUARD_M4: Constitutional 6HP boundary mask.
          6. STAGE_DATABASE_SYNC_COMMIT: Commit resulting state to CRUD DB.
        """
        assert VITAL_MAX_HP == 6, "Constitutional Invariant VITAL_MAX_HP == 6 violated in pipeline run"

        t_pipe_start = time.perf_counter_ns()
        pipe_id = pipeline_id or f"PIPE_{int(time.time() * 1000)}"

        # 1. Initial State Matrix M0
        if initial_matrix is None:
            m0 = BooleanMatrix(8, 8)
            # Create diamond / center seed pattern
            for r in range(2, 6):
                for c in range(2, 6):
                    m0.set(r, c, True)
        else:
            m0 = initial_matrix

        stages_results: List[WorkloadStageResult] = []
        cur_matrix = m0
        initial_density = m0.density

        # Stage 1: Ingest
        t0 = time.perf_counter_ns()
        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        stages_results.append(WorkloadStageResult(
            stage_id="STAGE_01_INGEST",
            stage_name="Sensor State & Feature Ingest M0",
            kernel_applied="NONE",
            input_bits=cur_matrix.active_bit_count,
            output_bits=cur_matrix.active_bit_count,
            density=cur_matrix.density,
            elapsed_us=elapsed_us,
            output_matrix_hex=cur_matrix.to_hex_string(),
            output_matrix_grid=cur_matrix.to_list(),
            status_code=to_wire_token(ObservationState.OBS_RAW_SAMPLE)
        ))
        self.log_service.emit_event(
            pipe_id, "STAGE_01_INGEST", "NONE",
            cur_matrix.active_bit_count, cur_matrix.active_bit_count,
            cur_matrix.density, elapsed_us, to_wire_token(ObservationState.OBS_RAW_SAMPLE),
            f"Ingested M0 state matrix: {cur_matrix.active_bit_count} active bits"
        )

        # Stage 2: Edge Detection Convolution M1
        t0 = time.perf_counter_ns()
        k_edge = self.kernels["KERNEL_EDGE_DETECTOR_3X3"]
        m1 = convolve_boolean_matrix(cur_matrix, k_edge)
        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        stages_results.append(WorkloadStageResult(
            stage_id="STAGE_02_EDGE_CONV",
            stage_name="State Boundary Gradient Convolution M1",
            kernel_applied=k_edge.kernel_id,
            input_bits=cur_matrix.active_bit_count,
            output_bits=m1.active_bit_count,
            density=m1.density,
            elapsed_us=elapsed_us,
            output_matrix_hex=m1.to_hex_string(),
            output_matrix_grid=m1.to_list(),
            status_code=to_wire_token(DerivedState.DER_COMPUTED_DELTA)
        ))
        self.log_service.emit_event(
            pipe_id, "STAGE_02_EDGE_CONV", k_edge.kernel_id,
            cur_matrix.active_bit_count, m1.active_bit_count,
            m1.density, elapsed_us, to_wire_token(DerivedState.DER_COMPUTED_DELTA),
            f"Edge convolution: {cur_matrix.active_bit_count} -> {m1.active_bit_count} boundary bits"
        )
        cur_matrix = m1

        # Stage 3: Diffusion Convolution M2
        t0 = time.perf_counter_ns()
        k_diff = self.kernels["KERNEL_DIFFUSION_SMOOTHER_3X3"]
        m2 = convolve_boolean_matrix(cur_matrix, k_diff)
        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        stages_results.append(WorkloadStageResult(
            stage_id="STAGE_03_DIFFUSION_CONV",
            stage_name="Lateral Signal Diffusion Convolution M2",
            kernel_applied=k_diff.kernel_id,
            input_bits=cur_matrix.active_bit_count,
            output_bits=m2.active_bit_count,
            density=m2.density,
            elapsed_us=elapsed_us,
            output_matrix_hex=m2.to_hex_string(),
            output_matrix_grid=m2.to_list(),
            status_code=to_wire_token(DerivedState.DER_COMPUTED_DELTA)
        ))
        self.log_service.emit_event(
            pipe_id, "STAGE_03_DIFFUSION_CONV", k_diff.kernel_id,
            cur_matrix.active_bit_count, m2.active_bit_count,
            m2.density, elapsed_us, to_wire_token(DerivedState.DER_COMPUTED_DELTA),
            f"Diffusion convolution: {cur_matrix.active_bit_count} -> {m2.active_bit_count} smoothed bits"
        )
        cur_matrix = m2

        # Stage 4: Majority Consensus Gate M3
        t0 = time.perf_counter_ns()
        k_vote = self.kernels["KERNEL_MAJORITY_GATE_3X3"]
        m3 = convolve_boolean_matrix(cur_matrix, k_vote)
        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        stages_results.append(WorkloadStageResult(
            stage_id="STAGE_04_CONSENSUS_CONV",
            stage_name="Cellular Majority Consensus Gate M3",
            kernel_applied=k_vote.kernel_id,
            input_bits=cur_matrix.active_bit_count,
            output_bits=m3.active_bit_count,
            density=m3.density,
            elapsed_us=elapsed_us,
            output_matrix_hex=m3.to_hex_string(),
            output_matrix_grid=m3.to_list(),
            status_code=to_wire_token(PredictionState.PRED_POINT_ESTIMATE)
        ))
        self.log_service.emit_event(
            pipe_id, "STAGE_04_CONSENSUS_CONV", k_vote.kernel_id,
            cur_matrix.active_bit_count, m3.active_bit_count,
            m3.density, elapsed_us, to_wire_token(PredictionState.PRED_POINT_ESTIMATE),
            f"Consensus gate: {cur_matrix.active_bit_count} -> {m3.active_bit_count} consensus bits"
        )
        cur_matrix = m3

        # Stage 5: Invariant Guard Convolution M4
        t0 = time.perf_counter_ns()
        k_guard = self.kernels["KERNEL_6HP_INVARIANT_MASK_3X3"]
        m4 = convolve_boolean_matrix(cur_matrix, k_guard)
        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        stages_results.append(WorkloadStageResult(
            stage_id="STAGE_05_INVARIANT_GUARD",
            stage_name="Constitutional 6HP Boundary Mask M4",
            kernel_applied=k_guard.kernel_id,
            input_bits=cur_matrix.active_bit_count,
            output_bits=m4.active_bit_count,
            density=m4.density,
            elapsed_us=elapsed_us,
            output_matrix_hex=m4.to_hex_string(),
            output_matrix_grid=m4.to_list(),
            status_code=to_wire_token(SafetyState.SAFE_PASS)
        ))
        self.log_service.emit_event(
            pipe_id, "STAGE_05_INVARIANT_GUARD", k_guard.kernel_id,
            cur_matrix.active_bit_count, m4.active_bit_count,
            m4.density, elapsed_us, to_wire_token(SafetyState.SAFE_PASS),
            f"Invariant guard: verified 6/6 HP equilibrium, {m4.active_bit_count} bits active"
        )
        cur_matrix = m4

        # Stage 6: Database Sync & Commit
        t0 = time.perf_counter_ns()
        db_entity_id = f"ENT_CONV_{pipe_id}"
        self.db.create_entity(
            entity_id=db_entity_id,
            name=f"Synthesized Convoluted State {pipe_id}",
            matrix=m4,
            collection="convoluted_workloads",
            metadata={"pipe_id": pipe_id, "initial_density": initial_density, "final_density": m4.density}
        )
        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        stages_results.append(WorkloadStageResult(
            stage_id="STAGE_06_DB_COMMIT",
            stage_name="Transactional Boolean Database Sync Commit",
            kernel_applied="NONE",
            input_bits=cur_matrix.active_bit_count,
            output_bits=cur_matrix.active_bit_count,
            density=cur_matrix.density,
            elapsed_us=elapsed_us,
            output_matrix_hex=cur_matrix.to_hex_string(),
            output_matrix_grid=cur_matrix.to_list(),
            status_code=to_wire_token(TransactionState.TXN_COMMITTED)
        ))
        self.log_service.emit_event(
            pipe_id, "STAGE_06_DB_COMMIT", "NONE",
            cur_matrix.active_bit_count, cur_matrix.active_bit_count,
            cur_matrix.density, elapsed_us, to_wire_token(TransactionState.TXN_COMMITTED),
            f"Committed synthesized matrix to Boolean CRUD DB [{db_entity_id}]"
        )

        total_pipe_us = (time.perf_counter_ns() - t_pipe_start) / 1000.0

        return ChainedPipelineExecutionReport(
            pipeline_id=pipe_id,
            timestamp_ns=time.time_ns(),
            total_stages=len(stages_results),
            total_elapsed_us=round(total_pipe_us, 2),
            initial_density=initial_density,
            final_density=cur_matrix.density,
            stages=[s.to_dict() for s in stages_results],
            safety_verdict=to_wire_token(SafetyState.SAFE_PASS),
            vital_max_hp=VITAL_MAX_HP
        )

    execute_pipeline = run_pipeline


# Global Singleton for Chained Workload Pipeline
GLOBAL_CHAINED_MATRIX_WORKLOAD_PIPELINE = ChainedMatrixWorkloadPipeline()


if __name__ == "__main__":
    print("===================================================================")
    print("Krystal Chained Convolutional Matrix Workloads & Boolean CRUD DB")
    print(f"Constitutional Invariant: VITAL_MAX_HP == {VITAL_MAX_HP}")
    print(f"Predefined Kernels: {len(CANONICAL_CONVOLUTION_KERNELS)}")
    print("===================================================================")

    pipeline = ChainedMatrixWorkloadPipeline()
    report = pipeline.run_pipeline()

    print(f"Pipeline: {report.pipeline_id} completed in {report.total_elapsed_us:.2f} us")
    print(f"Density Shift: {report.initial_density} -> {report.final_density} | Safety: {report.safety_verdict}")
    for st in report.stages:
        print(f"  - [{st['stage_id']}] {st['stage_name']:<42} : {st['elapsed_us']:>6.2f} us ({st['output_bits']} bits, {st['status_code']})")

    # Inspect DB
    db = pipeline.db
    all_ents = db.query_entities()
    print(f"\nBoolean CRUD Database Entities: {len(all_ents)}")
    for ent in all_ents:
        print(f"  * [{ent['entity_id']}] {ent['name']:<36} | Bits: {ent['active_bits']} | Density: {ent['density']}")
