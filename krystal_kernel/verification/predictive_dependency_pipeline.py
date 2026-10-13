#!/usr/bin/env python3
r"""
==============================================================================
KRYSTAL-STACK: PREDICTIVE DEPENDENCY PIPELINE & ARITHMETIC RECONCILIATION
==============================================================================
Module: krystal_kernel/verification/predictive_dependency_pipeline.py
Description:
    High-performance pipeline for fetching, synchronizing, overlapping,
    complementing, and executing arithmetic operations (add, subtract,
    multiply, divide) across epistemic dependencies, with cross-domain
    transformations via the latent parameter x (Quadratic Variable Manifold).

Mathematical Foundation:
    1. Synchronize: Monotonic clock drift check (Δt_skew = |t_node - now|),
       lease TTL validity (now <= valid_from + lease_ttl), and vector clock reconciliation.
    2. Overlap: Jaccard similarity J(A, B) = |A ∩ B| / |A ∪ B|, temporal sliding window
       intersection O_time = max(0, min(end_A, end_B) - max(start_A, start_B)),
       and cosine value overlap cos(θ) = (A · B) / (||A|| ||B|| + ε).
    3. Complement: Algebraic complement x_comp = 1.0 - x, and domain set complement
       U \ B (padding missing dependencies with canonical defaults).
    4. Divide: Element-wise C_i = A_i / (B_i + sign(B_i)*ε) with strict ε = 10^-12
       numeric singularity protection.
    5. Multiply: Element-wise C_i = A_i * B_i and scalar scaling C_i = k * A_i with
       probabilistic confidence decay conf_C = conf_A * conf_B.
    6. Subtract: Element-wise C_i = A_i - B_i and residual L2 error ||A - B||_2.
    7. Add: Element-wise C_i = A_i + B_i and bundle value aggregation Σ A_i.
    8. Calculate via x: Forward/inverse solving through Quadratic Variable Manifold
       V(x) = a*x^2 + b*x + c, solving for latent x ∈ [0.0, 1.0], latent operations,
       and target domain projection, strictly preserving VITAL_MAX_HP == 6.

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.epistemic_taxonomies import (
    DependencyEdgeKind,
    PredictiveDependencyGraph,
    PredictiveDependencyNode,
)
from krystal_web_hub.economic_engine.quadratic_variable_transformer import (
    GLOBAL_QUADRATIC_TRANSFORMER,
    INV_GOLDEN_RATIO,
    QuadraticVariableTransformer,
)

EPSILON: float = 1e-12


# ==============================================================================
# 1. CORE DATA STRUCTURES
# ==============================================================================

@dataclass
class DependencyBundle:
    """
    Encapsulates a snapshot of interdependent causal nodes, edges, execution
    topology, numeric states, time intervals, and vector clocks.
    """
    root_node_id: str
    nodes: Dict[str, PredictiveDependencyNode] = field(default_factory=dict)
    edges: List[Tuple[str, str, DependencyEdgeKind]] = field(default_factory=list)
    topological_order: List[str] = field(default_factory=list)
    values: Dict[str, float] = field(default_factory=dict)
    intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    domains: Dict[str, str] = field(default_factory=dict)
    vector_clocks: Dict[str, Dict[str, int]] = field(default_factory=dict)
    confidences: Dict[str, float] = field(default_factory=dict)
    timestamp_ns: int = field(default_factory=time.time_ns)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Constitutional Invariant Verification
        assert VITAL_MAX_HP == 6, f"Constitutional breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"

        # Initialize defaults from node attributes if not explicitly populated
        for nid, node in self.nodes.items():
            if nid not in self.values:
                self.values[nid] = getattr(node, "value", 0.0)
            if nid not in self.intervals:
                self.intervals[nid] = getattr(node, "interval", (0.0, 1.0))
            if nid not in self.domains:
                dom = getattr(node, "domain_id", None)
                self.domains[nid] = dom if dom else "memory_latency_ns"
            if nid not in self.vector_clocks:
                self.vector_clocks[nid] = getattr(node, "vector_clock", {nid: node.state_version})
            if nid not in self.confidences:
                self.confidences[nid] = getattr(node, "confidence", 1.0)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def get_value(self, node_id: str, default: float = 0.0) -> float:
        return self.values.get(node_id, default)

    def set_value(self, node_id: str, value: float) -> None:
        self.values[node_id] = float(value)
        if node_id in self.nodes:
            self.nodes[node_id].value = float(value)

    def clone(self) -> "DependencyBundle":
        """Creates a deep copy of the dependency bundle."""
        cloned_nodes = {
            nid: PredictiveDependencyNode(
                node_id=n.node_id,
                subsystem=n.subsystem,
                epistemic_kind=n.epistemic_kind,
                lease_ttl_ns=n.lease_ttl_ns,
                state_version=n.state_version,
                last_updated_ns=n.last_updated_ns,
                value=getattr(n, "value", self.values.get(nid, 0.0)),
                domain_id=getattr(n, "domain_id", self.domains.get(nid, "memory_latency_ns")),
                confidence=getattr(n, "confidence", self.confidences.get(nid, 1.0)),
                valid_from_ns=getattr(n, "valid_from_ns", self.timestamp_ns),
                interval=getattr(n, "interval", self.intervals.get(nid, (0.0, 1.0))),
                vector_clock=dict(getattr(n, "vector_clock", self.vector_clocks.get(nid, {}))),
                metadata=dict(getattr(n, "metadata", {})),
            )
            for nid, n in self.nodes.items()
        }
        return DependencyBundle(
            root_node_id=self.root_node_id,
            nodes=cloned_nodes,
            edges=list(self.edges),
            topological_order=list(self.topological_order),
            values=dict(self.values),
            intervals=dict(self.intervals),
            domains=dict(self.domains),
            vector_clocks={k: dict(v) for k, v in self.vector_clocks.items()},
            confidences=dict(self.confidences),
            timestamp_ns=self.timestamp_ns,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_node_id": self.root_node_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "topological_order": self.topological_order,
            "values": {k: round(v, 4) for k, v in self.values.items()},
            "domains": self.domains,
            "intervals": self.intervals,
            "confidences": {k: round(v, 4) for k, v in self.confidences.items()},
            "vector_clocks": self.vector_clocks,
            "timestamp_ns": self.timestamp_ns,
            "vital_max_hp_rule": VITAL_MAX_HP,
        }


@dataclass
class SyncReport:
    """Telemetry report for pipeline synchronization."""
    is_synchronized: bool
    now_ns: int
    max_drift_ns: int
    observed_max_drift_ns: int
    expired_nodes: List[str]
    active_nodes: List[str]
    clock_skew_per_node: Dict[str, int]
    vector_clock_consensus: Dict[str, int]
    reconciled_bundle: DependencyBundle
    vital_max_hp: int = VITAL_MAX_HP


@dataclass
class OverlapReport:
    """Telemetry report for dependency intersection and overlap."""
    bundle_a_root: str
    bundle_b_root: str
    common_nodes: List[str]
    unique_a_nodes: List[str]
    unique_b_nodes: List[str]
    jaccard_similarity: float
    temporal_overlap_ratios: Dict[str, float]
    cosine_value_similarity: float
    overlapping_bundle: DependencyBundle
    vital_max_hp: int = VITAL_MAX_HP


@dataclass
class ComplementReport:
    """Telemetry report for algebraic and set complementation."""
    source_bundle_root: str
    universe_nodes: List[str]
    missing_nodes: List[str]
    algebraic_complement_x: Dict[str, float]
    algebraic_complement_values: Dict[str, float]
    completed_bundle: DependencyBundle
    vital_max_hp: int = VITAL_MAX_HP


# ==============================================================================
# 2. PREDICTIVE DEPENDENCY PIPELINE ENGINE
# ==============================================================================

class PredictiveDependencyPipeline:
    """
    Orchestration engine providing full lifecycle control over dependency graphs:
    fetching, synchronizing, overlapping, complementing, performing arithmetic
    operations, and evaluating cross-domain variables through latent x.
    """

    def __init__(
        self,
        graph: Optional[PredictiveDependencyGraph] = None,
        transformer: Optional[QuadraticVariableTransformer] = None,
    ):
        self.graph = graph if graph is not None else PredictiveDependencyGraph()
        self.transformer = transformer if transformer is not None else GLOBAL_QUADRATIC_TRANSFORMER
        assert VITAL_MAX_HP == 6, f"Constitutional breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"

    # ─── 1. FETCH DEPENDENCIES ────────────────────────────────────────────────

    def fetch_dependencies(
        self,
        node_id: str,
        direction: str = "both",
        max_depth: int = 16,
    ) -> DependencyBundle:
        """
        Traverses the causal dependency graph from node_id to fetch all upstream,
        downstream, or bidirectional dependencies up to max_depth.
        Returns a sorted DependencyBundle with full topology.
        """
        if node_id not in self.graph.nodes:
            raise KeyError(f"Root node '{node_id}' not found in dependency graph.")

        visited_nodes: Set[str] = {node_id}
        collected_edges: List[Tuple[str, str, DependencyEdgeKind]] = []
        frontier = [node_id]
        depth = 0

        norm_dir = direction.lower().strip()
        while frontier and depth < max_depth:
            next_frontier: Set[str] = set()
            depth += 1

            for curr in frontier:
                for src, tgt, kind in self.graph.edges:
                    # Upstream: target is curr -> source is parent/cause
                    if norm_dir in ("upstream", "both") and tgt == curr:
                        if (src, tgt, kind) not in collected_edges:
                            collected_edges.append((src, tgt, kind))
                        if src not in visited_nodes:
                            visited_nodes.add(src)
                            next_frontier.add(src)

                    # Downstream: source is curr -> target is child/effect
                    if norm_dir in ("downstream", "both") and src == curr:
                        if (src, tgt, kind) not in collected_edges:
                            collected_edges.append((src, tgt, kind))
                        if tgt not in visited_nodes:
                            visited_nodes.add(tgt)
                            next_frontier.add(tgt)

            frontier = list(next_frontier)

        # Build node dictionary
        nodes_dict = {nid: self.graph.nodes[nid] for nid in visited_nodes if nid in self.graph.nodes}

        # Compute topological execution order (Kahn's Algorithm on subgraph)
        in_degree: Dict[str, int] = {nid: 0 for nid in visited_nodes}
        adj_list: Dict[str, List[str]] = {nid: [] for nid in visited_nodes}
        for src, tgt, _ in collected_edges:
            if src in in_degree and tgt in in_degree:
                in_degree[tgt] += 1
                adj_list[src].append(tgt)

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        topo_order: List[str] = []
        while queue:
            curr = queue.pop(0)
            topo_order.append(curr)
            for neighbor in adj_list.get(curr, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Handle any residual cyclic nodes deterministically
        if len(topo_order) < len(visited_nodes):
            remaining = sorted(list(visited_nodes - set(topo_order)))
            topo_order.extend(remaining)

        # Extract values, intervals, domains, and vector clocks
        values = {}
        intervals = {}
        domains = {}
        clocks = {}
        confidences = {}

        for nid in visited_nodes:
            n = nodes_dict[nid]
            values[nid] = float(getattr(n, "value", 0.0))
            intervals[nid] = getattr(n, "interval", (0.0, 1.0))
            domains[nid] = getattr(n, "domain_id", None) or "memory_latency_ns"
            clocks[nid] = dict(getattr(n, "vector_clock", {nid: n.state_version}))
            confidences[nid] = float(getattr(n, "confidence", 1.0))

        return DependencyBundle(
            root_node_id=node_id,
            nodes=nodes_dict,
            edges=collected_edges,
            topological_order=topo_order,
            values=values,
            intervals=intervals,
            domains=domains,
            vector_clocks=clocks,
            confidences=confidences,
            timestamp_ns=time.time_ns(),
            metadata={"direction": direction, "depth_reached": depth},
        )

    # ─── 2. SYNCHRONIZE ───────────────────────────────────────────────────────

    def synchronize(
        self,
        bundle: DependencyBundle,
        now_ns: Optional[int] = None,
        max_drift_ns: int = 1_000_000_000,  # 1.0s max allowed clock skew
    ) -> SyncReport:
        """
        Synchronizes all dependencies in the bundle:
        - Validates monotonic clock skews and drift against max_drift_ns.
        - Identifies expired temporal leases (now_ns > valid_from_ns + lease_ttl_ns).
        - Reconciles vector clocks into a unified consensus state.
        - Emits a clean, synchronized clone with aligned clocks.
        """
        current_time = now_ns if now_ns is not None else time.time_ns()
        reconciled = bundle.clone()
        reconciled.timestamp_ns = current_time

        expired: List[str] = []
        active: List[str] = []
        skews: Dict[str, int] = {}
        consensus_clock: Dict[str, int] = {}
        observed_max_drift = 0

        for nid, node in reconciled.nodes.items():
            valid_from = getattr(node, "valid_from_ns", current_time)
            lease_ttl = getattr(node, "lease_ttl_ns", 5_000_000_000)
            last_updated = getattr(node, "last_updated_ns", current_time)

            # Check skew relative to now
            skew = abs(current_time - last_updated)
            skews[nid] = skew
            if skew > observed_max_drift:
                observed_max_drift = skew

            # Check lease expiry
            is_expired = current_time > (valid_from + lease_ttl)
            if is_expired:
                expired.append(nid)
                reconciled.confidences[nid] = round(reconciled.confidences[nid] * 0.5, 4)
            else:
                active.append(nid)

            # Reconcile vector clocks across nodes
            node_clock = reconciled.vector_clocks.get(nid, {})
            for key, val in node_clock.items():
                consensus_clock[key] = max(consensus_clock.get(key, 0), val)

        # Synchronize vector clocks to consensus
        for nid in reconciled.nodes:
            reconciled.vector_clocks[nid] = dict(consensus_clock)

        is_sync = (observed_max_drift <= max_drift_ns) and (len(expired) == 0)

        return SyncReport(
            is_synchronized=is_sync,
            now_ns=current_time,
            max_drift_ns=max_drift_ns,
            observed_max_drift_ns=observed_max_drift,
            expired_nodes=expired,
            active_nodes=active,
            clock_skew_per_node=skews,
            vector_clock_consensus=consensus_clock,
            reconciled_bundle=reconciled,
            vital_max_hp=VITAL_MAX_HP,
        )

    # ─── 3. OVERLAP ───────────────────────────────────────────────────────────

    def overlap(
        self,
        bundle_a: DependencyBundle,
        bundle_b: DependencyBundle,
    ) -> OverlapReport:
        """
        Calculates the topological, temporal, and spatial overlap between two bundles:
        - Jaccard set index: J(A, B) = |A ∩ B| / |A ∪ B|
        - Sliding window temporal overlap: max(0, min(end_A, end_B) - max(start_A, start_B))
        - Value vector cosine similarity: (A · B) / (||A|| ||B|| + ε)
        - Sub-bundle of overlapping intersection nodes.
        """
        nodes_a = set(bundle_a.nodes.keys())
        nodes_b = set(bundle_b.nodes.keys())

        common = sorted(list(nodes_a & nodes_b))
        unique_a = sorted(list(nodes_a - nodes_b))
        unique_b = sorted(list(nodes_b - nodes_a))
        union = nodes_a | nodes_b

        jaccard = len(common) / len(union) if union else 1.0

        # Temporal intervals overlap
        temporal_ratios: Dict[str, float] = {}
        dot_product = 0.0
        norm_a = 0.0
        norm_b = 0.0

        overlapping_nodes: Dict[str, PredictiveDependencyNode] = {}
        overlapping_values: Dict[str, float] = {}
        overlapping_intervals: Dict[str, Tuple[float, float]] = {}
        overlapping_domains: Dict[str, str] = {}
        overlapping_clocks: Dict[str, Dict[str, int]] = {}
        overlapping_confidences: Dict[str, float] = {}

        for nid in common:
            # Interval overlap
            int_a = bundle_a.intervals.get(nid, (0.0, 1.0))
            int_b = bundle_b.intervals.get(nid, (0.0, 1.0))
            start_max = max(int_a[0], int_b[0])
            end_min = min(int_a[1], int_b[1])
            duration_overlap = max(0.0, end_min - start_max)
            span_max = max(int_a[1] - int_a[0], int_b[1] - int_b[0], EPSILON)
            ratio = duration_overlap / span_max
            temporal_ratios[nid] = round(ratio, 4)

            # Value vector cosine overlap
            val_a = bundle_a.get_value(nid, 0.0)
            val_b = bundle_b.get_value(nid, 0.0)
            dot_product += val_a * val_b
            norm_a += val_a ** 2
            norm_b += val_b ** 2

            # Overlapping bundle properties: harmonic combination of values
            overlapping_nodes[nid] = bundle_a.nodes[nid]
            overlapping_values[nid] = round(0.5 * (val_a + val_b), 4)
            overlapping_intervals[nid] = (start_max, end_min)
            overlapping_domains[nid] = bundle_a.domains.get(nid, "memory_latency_ns")
            overlapping_clocks[nid] = bundle_a.vector_clocks.get(nid, {})
            overlapping_confidences[nid] = round(
                bundle_a.confidences.get(nid, 1.0) * bundle_b.confidences.get(nid, 1.0), 4
            )

        denom = (math.sqrt(norm_a) * math.sqrt(norm_b)) + EPSILON
        cosine_sim = dot_product / denom if common else 0.0

        # Filter edges for common nodes
        common_set = set(common)
        overlapping_edges = [
            (s, t, k) for s, t, k in bundle_a.edges if s in common_set and t in common_set
        ]

        overlap_bundle = DependencyBundle(
            root_node_id=f"overlap({bundle_a.root_node_id},{bundle_b.root_node_id})",
            nodes=overlapping_nodes,
            edges=overlapping_edges,
            topological_order=[nid for nid in bundle_a.topological_order if nid in common_set],
            values=overlapping_values,
            intervals=overlapping_intervals,
            domains=overlapping_domains,
            vector_clocks=overlapping_clocks,
            confidences=overlapping_confidences,
            timestamp_ns=time.time_ns(),
            metadata={"jaccard": round(jaccard, 4), "cosine_sim": round(cosine_sim, 4)},
        )

        return OverlapReport(
            bundle_a_root=bundle_a.root_node_id,
            bundle_b_root=bundle_b.root_node_id,
            common_nodes=common,
            unique_a_nodes=unique_a,
            unique_b_nodes=unique_b,
            jaccard_similarity=round(jaccard, 4),
            temporal_overlap_ratios=temporal_ratios,
            cosine_value_similarity=round(cosine_sim, 4),
            overlapping_bundle=overlap_bundle,
            vital_max_hp=VITAL_MAX_HP,
        )

    # ─── 4. COMPLEMENT ────────────────────────────────────────────────────────

    def complement(
        self,
        bundle: DependencyBundle,
        universe_ids: Optional[Set[str]] = None,
        default_value: float = 0.0,
        default_domain: str = "memory_latency_ns",
    ) -> ComplementReport:
        """
        Computes both algebraic complement (1.0 - x) and set-theoretic complement:
        - Algebraic complement: For each node, extracts latent x, computes x_comp = 1.0 - x,
          and calculates the corresponding physical domain value.
        - Set complement: Identifies missing dependencies from universe_ids, creating
          padded / completed nodes to ensure domain completeness.
        """
        completed = bundle.clone()

        # 1. Algebraic Complement via Latent Manifold x
        alg_comp_x: Dict[str, float] = {}
        alg_comp_vals: Dict[str, float] = {}

        for nid, val in bundle.values.items():
            dom_id = bundle.domains.get(nid, default_domain)
            if dom_id in self.transformer._domains:
                dom = self.transformer._domains[dom_id]
                x_val, _, _ = dom.solve_x(val)
                x = x_val if x_val is not None else 0.5
                x_comp = round(max(0.0, min(1.0, 1.0 - x)), 5)
                val_comp = dom.evaluate(x_comp)
            else:
                x_comp = round(max(0.0, min(1.0, 1.0 - (val / 100.0))), 5)
                val_comp = round(100.0 * x_comp, 4)

            alg_comp_x[nid] = x_comp
            alg_comp_vals[nid] = val_comp

        # 2. Set Complement (Padding missing nodes from universe)
        missing_nodes: List[str] = []
        u_list: List[str] = []
        if universe_ids:
            u_list = sorted(list(universe_ids))
            current_nodes = set(bundle.nodes.keys())
            missing = sorted(list(universe_ids - current_nodes))
            missing_nodes = missing

            for mid in missing:
                # Synthesize padded epistemic node
                padded_node = PredictiveDependencyNode(
                    node_id=mid,
                    subsystem="BACKEND_KERNEL",
                    epistemic_kind="DERIVED",
                    lease_ttl_ns=5_000_000_000,
                    state_version=1,
                    last_updated_ns=time.time_ns(),
                    value=default_value,
                    domain_id=default_domain,
                    confidence=0.70,  # Lower confidence for synthetic complement
                )
                completed.nodes[mid] = padded_node
                completed.values[mid] = default_value
                completed.domains[mid] = default_domain
                completed.intervals[mid] = (0.0, 1.0)
                completed.vector_clocks[mid] = {mid: 1}
                completed.confidences[mid] = 0.70
                completed.topological_order.append(mid)

        return ComplementReport(
            source_bundle_root=bundle.root_node_id,
            universe_nodes=u_list,
            missing_nodes=missing_nodes,
            algebraic_complement_x=alg_comp_x,
            algebraic_complement_values=alg_comp_vals,
            completed_bundle=completed,
            vital_max_hp=VITAL_MAX_HP,
        )

    # ─── 5. ARITHMETIC OPERATIONS: ADD, SUBTRACT, MULTIPLY, DIVIDE ───────────

    def add(
        self,
        bundle_a: DependencyBundle,
        operand: Union[DependencyBundle, float, Dict[str, float]],
    ) -> DependencyBundle:
        """
        Element-wise addition: C_i = A_i + B_i or scalar offset C_i = A_i + k.
        Confidence is calculated as min(conf_A, conf_B).
        """
        res = bundle_a.clone()
        if isinstance(operand, (int, float)):
            k = float(operand)
            for nid in res.values:
                res.values[nid] = round(res.values[nid] + k, 4)
        elif isinstance(operand, dict):
            for nid, k in operand.items():
                if nid in res.values:
                    res.values[nid] = round(res.values[nid] + float(k), 4)
        elif isinstance(operand, DependencyBundle):
            for nid, val_b in operand.values.items():
                if nid in res.values:
                    res.values[nid] = round(res.values[nid] + val_b, 4)
                    res.confidences[nid] = round(
                        min(res.confidences[nid], operand.confidences.get(nid, 1.0)), 4
                    )
        return res

    def subtract(
        self,
        bundle_a: DependencyBundle,
        operand: Union[DependencyBundle, float, Dict[str, float]],
    ) -> Tuple[DependencyBundle, float]:
        """
        Element-wise subtraction: C_i = A_i - B_i or scalar offset C_i = A_i - k.
        Returns (result_bundle, residual_L2_norm).
        """
        res = bundle_a.clone()
        diff_sq = 0.0

        if isinstance(operand, (int, float)):
            k = float(operand)
            for nid in res.values:
                diff = res.values[nid] - k
                res.values[nid] = round(diff, 4)
                diff_sq += diff ** 2
        elif isinstance(operand, dict):
            for nid, k in operand.items():
                if nid in res.values:
                    diff = res.values[nid] - float(k)
                    res.values[nid] = round(diff, 4)
                    diff_sq += diff ** 2
        elif isinstance(operand, DependencyBundle):
            for nid, val_b in operand.values.items():
                if nid in res.values:
                    diff = res.values[nid] - val_b
                    res.values[nid] = round(diff, 4)
                    diff_sq += diff ** 2
                    res.confidences[nid] = round(
                        min(res.confidences[nid], operand.confidences.get(nid, 1.0)), 4
                    )

        l2_residual = round(math.sqrt(diff_sq), 6)
        return res, l2_residual

    def multiply(
        self,
        bundle_a: DependencyBundle,
        operand: Union[DependencyBundle, float, Dict[str, float]],
    ) -> DependencyBundle:
        """
        Element-wise multiplication: C_i = A_i * B_i or scalar scaling C_i = k * A_i.
        Probabilistic confidence decays: conf_C = conf_A * conf_B.
        """
        res = bundle_a.clone()
        if isinstance(operand, (int, float)):
            k = float(operand)
            for nid in res.values:
                res.values[nid] = round(res.values[nid] * k, 4)
        elif isinstance(operand, dict):
            for nid, k in operand.items():
                if nid in res.values:
                    res.values[nid] = round(res.values[nid] * float(k), 4)
        elif isinstance(operand, DependencyBundle):
            for nid, val_b in operand.values.items():
                if nid in res.values:
                    res.values[nid] = round(res.values[nid] * val_b, 4)
                    res.confidences[nid] = round(
                        res.confidences[nid] * operand.confidences.get(nid, 1.0), 4
                    )
        return res

    def divide(
        self,
        bundle_a: DependencyBundle,
        operand: Union[DependencyBundle, float, Dict[str, float]],
        epsilon: float = EPSILON,
    ) -> DependencyBundle:
        """
        Element-wise division: C_i = A_i / (B_i + sign(B_i)*ε).
        Strict ε = 10^-12 protection guarantees finite numbers and prevents division by zero.
        """
        res = bundle_a.clone()

        def safe_div(numerator: float, denominator: float) -> float:
            sign = 1.0 if denominator >= 0.0 else -1.0
            eff_denom = denominator if abs(denominator) >= epsilon else sign * epsilon
            return numerator / eff_denom

        if isinstance(operand, (int, float)):
            k = float(operand)
            for nid in res.values:
                res.values[nid] = round(safe_div(res.values[nid], k), 4)
        elif isinstance(operand, dict):
            for nid, k in operand.items():
                if nid in res.values:
                    res.values[nid] = round(safe_div(res.values[nid], float(k)), 4)
        elif isinstance(operand, DependencyBundle):
            for nid, val_b in operand.values.items():
                if nid in res.values:
                    res.values[nid] = round(safe_div(res.values[nid], val_b), 4)
                    res.confidences[nid] = round(
                        res.confidences[nid] * operand.confidences.get(nid, 1.0), 4
                    )
        return res

    # ─── 6. CALCULATE VIA X (QUADRATIC VARIABLE MANIFOLD) ─────────────────────

    def calculate_via_x(
        self,
        bundle: DependencyBundle,
        target_domain: str = "vital_hp",
        delta_x: float = 0.0,
        latent_operation: str = "identity",
        alpha: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Transforms dependencies through the canonical latent variable x ∈ [0.0, 1.0]:
        1. Solves a_src * x^2 + b_src * x + (c_src - V_src) = 0 for each dependency node.
        2. Applies latent manifold transformation:
           - 'identity': x' = x
           - 'shift': x' = clamp(x + delta_x, 0.0, 1.0)
           - 'complement': x' = 1.0 - x
           - 'scale': x' = clamp(x * alpha, 0.0, 1.0)
           - 'harmonic': x' = (2 * x * phi^-1) / (x + phi^-1 + eps)
        3. Projects transformed x' into target_domain: V_tgt(x') = a_tgt * (x')^2 + b_tgt * x' + c_tgt.
        4. Invariant: Strictly enforces assert VITAL_MAX_HP == 6.
        """
        assert VITAL_MAX_HP == 6, f"Constitutional breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"

        if target_domain not in self.transformer._domains:
            raise KeyError(f"Target domain '{target_domain}' not found in QuadraticVariableTransformer.")

        tgt_dom = self.transformer._domains[target_domain]
        transformed_nodes: Dict[str, Dict[str, Any]] = {}
        transformed_values: List[float] = []

        for nid, val in bundle.values.items():
            src_domain_id = bundle.domains.get(nid, "memory_latency_ns")
            if src_domain_id not in self.transformer._domains:
                src_domain_id = "memory_latency_ns"
            src_dom = self.transformer._domains[src_domain_id]

            # 1. Solve for latent x
            chosen_x, delta, is_real = src_dom.solve_x(val)
            x_base = chosen_x if chosen_x is not None else 0.5

            # 2. Latent transformation
            op = latent_operation.lower().strip()
            if op == "shift":
                x_trans = max(0.0, min(1.0, x_base + delta_x))
            elif op == "complement":
                x_trans = max(0.0, min(1.0, 1.0 - x_base))
            elif op == "scale":
                x_trans = max(0.0, min(1.0, x_base * alpha))
            elif op == "harmonic":
                x_trans = (2.0 * x_base * INV_GOLDEN_RATIO) / (x_base + INV_GOLDEN_RATIO + EPSILON)
                x_trans = max(0.0, min(1.0, x_trans))
            else:  # identity
                x_trans = x_base

            x_trans = round(x_trans, 5)

            # 3. Project to target domain
            target_val = tgt_dom.evaluate(x_trans)
            transformed_values.append(target_val)

            transformed_nodes[nid] = {
                "source_domain": src_domain_id,
                "source_value": val,
                "source_unit": src_dom.unit,
                "latent_x_solved": x_base,
                "latent_x_transformed": x_trans,
                "discriminant_delta": delta,
                "is_real_root": is_real,
                "target_domain": target_domain,
                "target_value": target_val,
                "target_unit": tgt_dom.unit,
                "vital_hp_bound": target_val <= VITAL_MAX_HP if target_domain == "vital_hp" else True,
            }

        # Calculate bundle aggregate metrics
        val_sum = round(sum(transformed_values), 4)
        val_mean = round(val_sum / len(transformed_values), 4) if transformed_values else 0.0
        val_min = round(min(transformed_values), 4) if transformed_values else 0.0
        val_max = round(max(transformed_values), 4) if transformed_values else 0.0
        l2_norm = round(math.sqrt(sum(v ** 2 for v in transformed_values)), 4) if transformed_values else 0.0

        return {
            "root_node_id": bundle.root_node_id,
            "target_domain": target_domain,
            "latent_operation": latent_operation,
            "delta_x": delta_x,
            "transformed_nodes": transformed_nodes,
            "aggregate": {
                "sum": val_sum,
                "mean": val_mean,
                "min": val_min,
                "max": val_max,
                "l2_norm": l2_norm,
                "count": len(transformed_values),
            },
            "vital_max_hp_rule": VITAL_MAX_HP,
            "constitutional_compliance": True,
        }
