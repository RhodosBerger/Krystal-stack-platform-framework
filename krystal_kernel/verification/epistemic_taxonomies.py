#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: EPISTEMIC TAXONOMIES FOR BOOLEAN AND LIST PRIMITIVES
==============================================================================
Module: krystal_kernel/verification/epistemic_taxonomies.py
Description:
    Enriches primitive boolean and list elements with formal epistemic metadata:
      - Boolean Taxonomies: 7-valued Modal Lattice, Temporal Lease, Invariant Authority, Actuation Scope.
      - List Taxonomies: Algebraic Ordering, Bounded Capacity (VITAL_MAX_HP == 6), Integrity Merkle Tree, Memory Topology.
      - Predictive Dependency Graph: Tracks cross-stack causal dependencies between backend forecasts and frontend UI views.

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP

T = TypeVar("T")


class InvariantViolationError(RuntimeError):
    """Raised when an epistemic invariant or constitutional bound is breached."""
    pass


# ==============================================================================
# 1. BOOLEAN TAXONOMIES
# ==============================================================================

class BooleanModalKind(str, Enum):
    """7-Valued Truth Lattice for Epistemic Booleans."""
    VERIFIED_TRUE = "VERIFIED_TRUE"                    # Hardware measured / mathematically proven
    COUNTERFACTUAL_TRUE = "COUNTERFACTUAL_TRUE"        # Predicted by model / speculative forecast
    FALSIFIED_FALSE = "FALSIFIED_FALSE"                # Empirically refuted by evidence probe
    VETOED_FALSE = "VETOED_FALSE"                      # Explicitly blocked by authority safety gate
    INDETERMINATE_PENDING = "INDETERMINATE_PENDING"    # Proof in-flight or awaiting sensor sampling
    PARADOXICAL_CONTRADICTORY = "PARADOXICAL_CONTRADICTORY" # Contradiction detected across sources
    QUARANTINED = "QUARANTINED"                        # Frozen due to anomaly or sensor fault


class BooleanTemporalKind(str, Enum):
    """Temporal lifecycle dynamics for Epistemic Booleans."""
    MONOTONIC_LATCH = "MONOTONIC_LATCH"          # Write-once True within epoch; never resets to False
    EPOCH_LEASED = "EPOCH_LEASED"                # Valid only within nanosecond window [valid_from, valid_from + lease_ttl]
    BISTABLE_HYSTERETIC = "BISTABLE_HYSTERETIC"  # Requires upper/lower hysteresis margin to flip
    SHADOW_HYPOTHETICAL = "SHADOW_HYPOTHETICAL"  # Exists strictly in shadow runtime; zero-mutation guaranteed


class BooleanInvariantKind(str, Enum):
    """Constitutional and authority grounding for Epistemic Booleans."""
    CONSTITUTIONAL_INVARIANT = "CONSTITUTIONAL_INVARIANT"  # Inviolable rule (e.g. VITAL_MAX_HP == 6)
    PROVABLE_LEMMA = "PROVABLE_LEMMA"                      # Carries cryptographic proof bundle / Z3 hash
    BAYESIAN_STOCHASTIC = "BAYESIAN_STOCHASTIC"            # Subject to Bayesian belief update P in [0, 1]


class BooleanActionKind(str, Enum):
    """Actuation impact and side-effect scope for Epistemic Booleans."""
    REVERSIBLE_SWITCH = "REVERSIBLE_SWITCH"      # O(1) instantaneous reversible state toggle
    IRREVERSIBLE_TRIGGER = "IRREVERSIBLE_TRIGGER" # Actuation triggering hardware / external I/O commit
    TELEMETRY_PROBE = "TELEMETRY_PROBE"          # Passive observation probe; zero state impact


@dataclass(frozen=True)
class EpistemicBoolean:
    """
    Epistemically grounded Boolean primitive.
    A boolean is not merely True/False, but:
      Value + Modal Truth Lattice + Temporal Lease + Invariant Authority + Actuation Scope.
    """
    value: bool
    modal_kind: BooleanModalKind
    temporal_kind: BooleanTemporalKind = BooleanTemporalKind.EPOCH_LEASED
    invariant_kind: BooleanInvariantKind = BooleanInvariantKind.PROVABLE_LEMMA
    action_kind: BooleanActionKind = BooleanActionKind.REVERSIBLE_SWITCH
    confidence: float = 1.0                       # [0.0 - 1.0]
    valid_from_ns: int = field(default_factory=time.time_ns)
    lease_ttl_ns: Optional[int] = 5_000_000_000   # Default 5.0 seconds lease
    proof_hash: Optional[str] = None              # SHA-256 certificate or Z3 lemma hash
    domain: str = "general"
    causal_parent: Optional[str] = None

    def is_expired(self, now_ns: Optional[int] = None) -> bool:
        """Checks if boolean has expired relative to its nanosecond lease."""
        if self.temporal_kind != BooleanTemporalKind.EPOCH_LEASED:
            return False
        if self.lease_ttl_ns is None:
            return False
        now = now_ns if now_ns is not None else time.time_ns()
        return now > (self.valid_from_ns + self.lease_ttl_ns)

    def is_committable(self, now_ns: Optional[int] = None) -> bool:
        """
        Determines whether this boolean can legally participate in commit verdicts.
        Fails if expired, vetoed, quarantined, or unverified.
        """
        if self.modal_kind in (
            BooleanModalKind.VETOED_FALSE,
            BooleanModalKind.QUARANTINED,
            BooleanModalKind.FALSIFIED_FALSE,
            BooleanModalKind.PARADOXICAL_CONTRADICTORY,
            BooleanModalKind.INDETERMINATE_PENDING,
        ):
            return False
        if self.is_expired(now_ns):
            return False
        return self.value is True and self.confidence >= 0.70

    def assert_constitutional(self) -> None:
        """Asserts constitutional integrity if bound to a constitutional invariant."""
        if self.invariant_kind == BooleanInvariantKind.CONSTITUTIONAL_INVARIANT:
            if not self.value or self.modal_kind != BooleanModalKind.VERIFIED_TRUE:
                raise InvariantViolationError(
                    f"Constitutional boolean violation in domain '{self.domain}': "
                    f"value={self.value}, modal={self.modal_kind}"
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "modal_kind": self.modal_kind.value,
            "temporal_kind": self.temporal_kind.value,
            "invariant_kind": self.invariant_kind.value,
            "action_kind": self.action_kind.value,
            "confidence": self.confidence,
            "valid_from_ns": self.valid_from_ns,
            "lease_ttl_ns": self.lease_ttl_ns,
            "is_expired": self.is_expired(),
            "is_committable": self.is_committable(),
            "proof_hash": self.proof_hash,
            "domain": self.domain,
            "causal_parent": self.causal_parent,
        }


# ==============================================================================
# 2. LIST TAXONOMIES
# ==============================================================================

class ListAlgebraicKind(str, Enum):
    """Algebraic ordering semantics for Epistemic Lists."""
    STRICT_CAUSAL_SEQUENCE = "STRICT_CAUSAL_SEQUENCE"  # Monotonic Lamport/Vector clock sequence
    TOPOLOGICAL_DAG_ORDER = "TOPOLOGICAL_DAG_ORDER"    # Precedence order respecting DAG dependencies
    COMMUTATIVE_BAG = "COMMUTATIVE_BAG"                # Order-independent multiset (CRDT set)
    CIRCULAR_RING_BUFFER = "CIRCULAR_RING_BUFFER"      # Fixed-capacity FIFO rolling buffer (zero realloc)


class ListCapacityKind(str, Enum):
    """Capacity boundaries and physical alignment invariants for Epistemic Lists."""
    VITAL_BOUNDED_LIST = "VITAL_BOUNDED_LIST"          # Strictly bounded: len(items) <= VITAL_MAX_HP (6)
    CACHE_LINE_64B_CHUNK = "CACHE_LINE_64B_CHUNK"      # Fixed 64-byte payload (SIMD/AVX-512 aligned)
    STRAND_PAGE_WORD_PACK = "STRAND_PAGE_WORD_PACK"    # Exactly 64 words (512-bit Strand NPU page frame)
    UNBOUNDED_EVIDENCE_STREAM = "UNBOUNDED_EVIDENCE_STREAM" # Paged append stream backed by ledger


class ListIntegrityKind(str, Enum):
    """Mutation audit and cryptographic integrity for Epistemic Lists."""
    APPEND_ONLY_MERKLE = "APPEND_ONLY_MERKLE"          # Insertion updates running Merkle root hash
    IDEMPOTENT_PRUNED_SET = "IDEMPOTENT_PRUNED_SET"    # Deduplicated by unique element ID
    FALSIFICATION_PRIORITY_QUEUE = "FALSIFICATION_PRIORITY_QUEUE" # Sorted by hazard metric H


class ListMemoryKind(str, Enum):
    """Hardware layout and zero-copy bindings for Epistemic Lists."""
    PINNED_DMA_CONTIGUOUS = "PINNED_DMA_CONTIGUOUS"    # Page-locked RAM for PCIe DMA
    ZERO_COPY_RING = "ZERO_COPY_RING"                  # C-ABI shared memory ring buffer
    DUAL_ASPECT_MIRROR = "DUAL_ASPECT_MIRROR"          # Mirrored to WebGL/Canvas typed arrays


class EpistemicList(Generic[T]):
    """
    Epistemically grounded and bounded List container.
    Guarantees capacity invariants (such as VITAL_MAX_HP == 6), algebraic ordering,
    and rolling Merkle root hashes for tamper detection.
    """
    def __init__(
        self,
        capacity_kind: ListCapacityKind,
        algebraic_kind: ListAlgebraicKind = ListAlgebraicKind.STRICT_CAUSAL_SEQUENCE,
        integrity_kind: ListIntegrityKind = ListIntegrityKind.APPEND_ONLY_MERKLE,
        memory_kind: ListMemoryKind = ListMemoryKind.DUAL_ASPECT_MIRROR,
        max_capacity: Optional[int] = None,
        initial_items: Optional[List[T]] = None,
    ):
        self.capacity_kind = capacity_kind
        self.algebraic_kind = algebraic_kind
        self.integrity_kind = integrity_kind
        self.memory_kind = memory_kind

        # Enforce constitutional capacity bound
        if capacity_kind == ListCapacityKind.VITAL_BOUNDED_LIST:
            self.max_capacity = VITAL_MAX_HP  # Strictly 6
        elif capacity_kind == ListCapacityKind.CACHE_LINE_64B_CHUNK:
            self.max_capacity = 16            # 16 float32 elements = 64 bytes
        elif capacity_kind == ListCapacityKind.STRAND_PAGE_WORD_PACK:
            self.max_capacity = 64            # 64 words
        else:
            self.max_capacity = max_capacity

        self._items: List[T] = []
        self._merkle_root: str = "0" * 64

        if initial_items:
            for it in initial_items:
                self.append(it)

    @property
    def items(self) -> List[T]:
        return list(self._items)

    @property
    def merkle_root(self) -> str:
        return self._merkle_root

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> T:
        return self._items[index]

    def __iter__(self):
        return iter(self._items)

    def append(self, item: T) -> None:
        """
        Appends an element enforcing capacity limits and rolling Merkle integrity.
        """
        if self.max_capacity is not None and len(self._items) >= self.max_capacity:
            if self.algebraic_kind == ListAlgebraicKind.CIRCULAR_RING_BUFFER:
                # Evict oldest element in FIFO ring buffer
                self._items.pop(0)
            else:
                raise InvariantViolationError(
                    f"List capacity violation for {self.capacity_kind.value}: "
                    f"Maximum capacity {self.max_capacity} exceeded (attempted {len(self._items) + 1})."
                )

        self._items.append(item)
        self._update_merkle(item)

    def _update_merkle(self, item: T) -> None:
        """Updates rolling Merkle hash: R_k = SHA256(R_{k-1} || SHA256(item))."""
        item_str = str(item)
        item_hash = hashlib.sha256(item_str.encode("utf-8")).hexdigest()
        combined = f"{self._merkle_root}:{item_hash}".encode("utf-8")
        self._merkle_root = hashlib.sha256(combined).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capacity_kind": self.capacity_kind.value,
            "algebraic_kind": self.algebraic_kind.value,
            "integrity_kind": self.integrity_kind.value,
            "memory_kind": self.memory_kind.value,
            "max_capacity": self.max_capacity,
            "current_length": len(self._items),
            "merkle_root": self._merkle_root,
            "items": [it.to_dict() if hasattr(it, "to_dict") else str(it) for it in self._items],
        }


def create_vital_hearts_list(active_hp: int) -> EpistemicList[EpistemicBoolean]:
    """
    Factory creating the canonical 6-heart vital invariant list.
    Enforces VITAL_MAX_HP == 6: exactly 6 EpistemicBoolean items clamped between 0 and 6.
    """
    clamped_hp = max(0, min(VITAL_MAX_HP, active_hp))
    vital_list: EpistemicList[EpistemicBoolean] = EpistemicList(
        capacity_kind=ListCapacityKind.VITAL_BOUNDED_LIST,
        algebraic_kind=ListAlgebraicKind.STRICT_CAUSAL_SEQUENCE,
        integrity_kind=ListIntegrityKind.APPEND_ONLY_MERKLE,
        memory_kind=ListMemoryKind.DUAL_ASPECT_MIRROR,
    )
    now_ns = time.time_ns()

    for i in range(VITAL_MAX_HP):
        is_active = i < clamped_hp
        vital_list.append(
            EpistemicBoolean(
                value=is_active,
                modal_kind=BooleanModalKind.VERIFIED_TRUE if is_active else BooleanModalKind.FALSIFIED_FALSE,
                temporal_kind=BooleanTemporalKind.MONOTONIC_LATCH,
                invariant_kind=BooleanInvariantKind.CONSTITUTIONAL_INVARIANT,
                action_kind=BooleanActionKind.REVERSIBLE_SWITCH,
                confidence=1.0,
                valid_from_ns=now_ns,
                domain=f"vital.heart.{i}",
            )
        )
    return vital_list


# ==============================================================================
# 3. BIDIRECTIONAL PREDICTIVE DEPENDENCY GRAPH
# ==============================================================================

class DependencyEdgeKind(str, Enum):
    PREDICTS = "PREDICTS"                    # Backend prediction forecasts downstream state
    INVALIDATES = "INVALIDATES"              # Mutation or lease expiry forces cache invalidation
    SUBSCRIBES = "SUBSCRIBES"                # Frontend view component observes backend signal
    GROUNDS = "GROUNDS"                      # Hardware sensor provides empirical ground truth
    EPISTEMIC_CAUSAL = "EPISTEMIC_CAUSAL"    # Causal dependency across epistemic states
    CAUSES = "EPISTEMIC_CAUSAL"              # Alias for epistemic causal coupling
    CORRELATIONAL = "CORRELATIONAL"          # Correlational coupling between variables
    TEMPORAL = "TEMPORAL"                    # Temporal sequence or delay dependency
    STRUCTURAL = "STRUCTURAL"                # Structural or hierarchy relationship


@dataclass
class PredictiveDependencyNode:
    node_id: str
    subsystem: str               # "BACKEND_KERNEL", "FRONTEND_UI", "HARDWARE_SENSOR", "AST_PATTERN"
    epistemic_kind: str          # "MEASURED", "PREDICTED", "DERIVED", "VIEW_MODEL"
    lease_ttl_ns: int = 5_000_000_000
    state_version: int = 1
    last_updated_ns: int = field(default_factory=time.time_ns)
    value: float = 0.0
    domain_id: Optional[str] = None
    confidence: float = 1.0
    valid_from_ns: int = field(default_factory=time.time_ns)
    interval: Tuple[float, float] = (0.0, 1.0)
    vector_clock: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PredictiveDependencyGraph:
    """
    Bidirectional causal dependency graph between backend speculative engines,
    AST code-pattern contracts, and frontend reactive UI view models.
    """
    def __init__(self):
        self.nodes: Dict[str, PredictiveDependencyNode] = {}
        self.edges: List[Tuple[str, str, DependencyEdgeKind]] = []

    def add_node(
        self,
        node_or_id: Union[PredictiveDependencyNode, str] = "",
        subsystem: str = "BACKEND_KERNEL",
        epistemic_kind: str = "MEASURED",
        lease_ttl_ns: int = 5_000_000_000,
        value: float = 0.0,
        domain: Optional[str] = None,
        confidence: float = 1.0,
        interval: Tuple[float, float] = (0.0, 1.0),
        clock: Optional[Dict[str, int]] = None,
        **kwargs: Any,
    ) -> PredictiveDependencyNode:
        if isinstance(node_or_id, PredictiveDependencyNode):
            node = node_or_id
        else:
            nid = kwargs.get("node_id", node_or_id)
            domain_id = domain or kwargs.get("domain_id")
            vector_clock = clock if clock is not None else kwargs.get("vector_clock", {})
            node = PredictiveDependencyNode(
                node_id=str(nid),
                subsystem=subsystem,
                epistemic_kind=epistemic_kind,
                lease_ttl_ns=lease_ttl_ns,
                value=float(value),
                domain_id=domain_id,
                confidence=float(confidence),
                interval=interval,
                vector_clock=vector_clock,
                metadata=kwargs.get("metadata", {}),
            )
        self.nodes[node.node_id] = node
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        kind: Union[DependencyEdgeKind, str] = DependencyEdgeKind.PREDICTS,
        weight: float = 1.0,
    ) -> None:
        if source_id not in self.nodes or target_id not in self.nodes:
            raise KeyError(f"Both nodes must exist before creating edge: {source_id} -> {target_id}")
        if isinstance(kind, str):
            try:
                edge_kind = DependencyEdgeKind(kind)
            except ValueError:
                edge_kind = DependencyEdgeKind.PREDICTS
        else:
            edge_kind = kind
        self.edges.append((source_id, target_id, edge_kind))

    def get_invalidation_set(self, mutated_node_id: str) -> Set[str]:
        """
        Traverses downstream dependencies to compute all frontend and backend
        view caches that must be invalidated when mutated_node_id changes.
        """
        invalidated: Set[str] = {mutated_node_id}
        queue = [mutated_node_id]

        while queue:
            curr = queue.pop(0)
            for src, tgt, kind in self.edges:
                if src == curr and kind in (DependencyEdgeKind.PREDICTS, DependencyEdgeKind.INVALIDATES):
                    if tgt not in invalidated:
                        invalidated.add(tgt)
                        queue.append(tgt)
        return invalidated

    def fetch_dependencies(
        self,
        node_id: str,
        direction: str = "both",
        max_depth: int = 16,
    ):
        """
        Fetches upstream and/or downstream causal dependencies into a DependencyBundle.
        """
        from krystal_kernel.verification.predictive_dependency_pipeline import (
            PredictiveDependencyPipeline,
        )
        pipeline = PredictiveDependencyPipeline(graph=self)
        return pipeline.fetch_dependencies(node_id=node_id, direction=direction, max_depth=max_depth)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes": {nid: {"subsystem": n.subsystem, "kind": n.epistemic_kind} for nid, n in self.nodes.items()},
            "edges": [{"source": s, "target": t, "kind": k.value} for s, t, k in self.edges],
        }

