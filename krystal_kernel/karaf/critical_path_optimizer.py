#!/usr/bin/env python3
"""
==============================================================================
KARAF: ALGORITHM F — CASCADE CRITICAL PATH OPTIMIZER
==============================================================================
Module: krystal_kernel/karaf/critical_path_optimizer.py
Description:
    Analyzes ModelCascade DAG execution plans using CPM (Critical Path Method).
    
    Identifies:
      - critical_path_latency
      - critical_path (ordered list of node_ids)
      - critical_node (primary bottleneck on critical path)
      - slack per node (slack = latest_finish - earliest_finish)
      - parallelizable candidate segments

    Rules:
      - Parallel branches execute concurrently: total latency is max(branches),
        never the naive sum.
      - Optimizes ONLY nodes on the critical path that materially reduce end-to-end latency.
      - Flags false optimization attempts that target non-critical high-slack nodes.
==============================================================================
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.cascade.cascade_planner import CascadePlan, CascadeEdge
from krystal_kernel.cascade.model_node import ModelNodeSpec


@dataclass
class CriticalPathReport:
    plan_id: str
    total_nodes: int
    critical_path: List[str]
    critical_path_latency_ms: float
    critical_bottleneck_node: str
    node_slack_ms: Dict[str, float]
    parallelizable_segments: List[List[str]]
    recommended_target_node: str
    rationale: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "total_nodes": self.total_nodes,
            "critical_path": self.critical_path,
            "critical_path_latency_ms": self.critical_path_latency_ms,
            "critical_bottleneck_node": self.critical_bottleneck_node,
            "node_slack_ms": self.node_slack_ms,
            "parallelizable_segments": self.parallelizable_segments,
            "recommended_target_node": self.recommended_target_node,
            "rationale": self.rationale,
            "vital_max_hp": self.vital_max_hp
        }


class CascadeCriticalPathOptimizer:
    """
    Evaluates Critical Path Method (CPM) over model cascade DAGs.
    Computes earliest start (ES), earliest finish (EF), latest finish (LF),
    and total slack for every execution node.
    """

    def __init__(self):
        assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"
        self.vital_max_hp = VITAL_MAX_HP

    def analyze_plan(
        self,
        plan: CascadePlan,
        node_latencies: Optional[Dict[str, float]] = None
    ) -> CriticalPathReport:
        """
        Performs forward and backward CPM passes over the CascadePlan DAG.
        """
        nodes_dict: Dict[str, ModelNodeSpec] = {n.node_id: n for n in plan.nodes}
        if not nodes_dict:
            return CriticalPathReport(
                plan_id=plan.plan_id,
                total_nodes=0,
                critical_path=[],
                critical_path_latency_ms=0.0,
                critical_bottleneck_node="",
                node_slack_ms={},
                parallelizable_segments=[],
                recommended_target_node="",
                rationale="Empty cascade plan."
            )

        # Default estimated latency per node if not explicitly provided
        latencies: Dict[str, float] = {}
        for nid in nodes_dict:
            if node_latencies and nid in node_latencies:
                latencies[nid] = max(0.1, float(node_latencies[nid]))
            else:
                # Default heuristic based on role
                role = nodes_dict[nid].role.upper()
                if "REASON" in role:
                    latencies[nid] = 45.0
                elif "RAG" in role:
                    latencies[nid] = 20.0
                elif "VERIF" in role:
                    latencies[nid] = 10.0
                elif "CLASS" in role:
                    latencies[nid] = 5.0
                elif "SYNTH" in role:
                    latencies[nid] = 30.0
                else:
                    latencies[nid] = 15.0

        # Build adjacency graph
        adj_forward: Dict[str, List[str]] = {nid: [] for nid in nodes_dict}
        adj_reverse: Dict[str, List[str]] = {nid: [] for nid in nodes_dict}
        in_degree: Dict[str, int] = {nid: 0 for nid in nodes_dict}

        for edge in plan.edges:
            if edge.source in nodes_dict and edge.target in nodes_dict:
                adj_forward[edge.source].append(edge.target)
                adj_reverse[edge.target].append(edge.source)
                in_degree[edge.target] += 1

        # Topological sorting (Kahn's algorithm)
        queue = [nid for nid in nodes_dict if in_degree[nid] == 0]
        topo_order: List[str] = []

        while queue:
            curr = queue.pop(0)
            topo_order.append(curr)
            for neighbor in adj_forward[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If cyclic or disconnected, fall back to sequential
        if len(topo_order) < len(nodes_dict):
            topo_order = list(nodes_dict.keys())

        # Forward Pass: Earliest Start (ES) and Earliest Finish (EF)
        es: Dict[str, float] = {}
        ef: Dict[str, float] = {}

        for nid in topo_order:
            parents = adj_reverse[nid]
            if not parents:
                es[nid] = 0.0
            else:
                es[nid] = max(ef[p] for p in parents)
            ef[nid] = es[nid] + latencies[nid]

        total_latency = max(ef.values()) if ef else 0.0

        # Backward Pass: Latest Finish (LF) and Latest Start (LS)
        lf: Dict[str, float] = {}
        ls: Dict[str, float] = {}

        for nid in reversed(topo_order):
            children = adj_forward[nid]
            if not children:
                lf[nid] = total_latency
            else:
                lf[nid] = min(ls[c] for c in children)
            ls[nid] = lf[nid] - latencies[nid]

        # Calculate Slack = LF - EF (or LS - ES)
        slack: Dict[str, float] = {}
        critical_path_nodes: List[str] = []

        for nid in topo_order:
            s = max(0.0, round(lf[nid] - ef[nid], 3))
            slack[nid] = s
            if s <= 1e-3:  # On critical path
                critical_path_nodes.append(nid)

        # Find the bottleneck node on the critical path (node with largest latency)
        critical_bottleneck = ""
        max_crit_lat = -1.0
        for nid in critical_path_nodes:
            if latencies[nid] > max_crit_lat:
                max_crit_lat = latencies[nid]
                critical_bottleneck = nid

        # Identify parallelizable segments (nodes with overlapping ES/EF)
        parallel_groups: List[List[str]] = []
        visited_pairs: Set[Tuple[str, str]] = set()

        for i, n1 in enumerate(topo_order):
            for n2 in topo_order[i+1:]:
                # Two nodes are parallel if neither reaches the other and time intervals overlap
                if n2 not in adj_forward[n1] and n1 not in adj_forward[n2]:
                    # check time overlap
                    if not (ef[n1] <= es[n2] or ef[n2] <= es[n1]):
                        pair = tuple(sorted([n1, n2]))
                        if pair not in visited_pairs:
                            visited_pairs.add(pair)
                            parallel_groups.append([pair[0], pair[1]])

        rationale = (
            f"Critical path latency: {total_latency}ms across {len(critical_path_nodes)} nodes: "
            f"{' -> '.join(critical_path_nodes)}. "
            f"Primary bottleneck: '{critical_bottleneck}' ({max_crit_lat}ms). "
            f"Optimizing '{critical_bottleneck}' will directly decrease end-to-end latency."
        )

        return CriticalPathReport(
            plan_id=plan.plan_id,
            total_nodes=len(nodes_dict),
            critical_path=critical_path_nodes,
            critical_path_latency_ms=round(total_latency, 2),
            critical_bottleneck_node=critical_bottleneck,
            node_slack_ms=slack,
            parallelizable_segments=parallel_groups,
            recommended_target_node=critical_bottleneck,
            rationale=rationale,
            vital_max_hp=VITAL_MAX_HP
        )

    def validate_optimization_target(
        self,
        report: CriticalPathReport,
        proposed_node_id: str
    ) -> Tuple[bool, str]:
        """
        Validates whether optimizing proposed_node_id actually reduces critical path latency.
        Rejects optimization of non-critical high-slack nodes!
        """
        if proposed_node_id not in report.critical_path:
            slack_val = report.node_slack_ms.get(proposed_node_id, 0.0)
            return (
                False,
                f"FALSIFIED: Node '{proposed_node_id}' is NOT on the critical path (slack={slack_val}ms). "
                f"Optimizing it yields 0.0ms end-to-end gain. Target '{report.critical_bottleneck_node}' instead."
            )
        return (
            True,
            f"VALIDATED: Node '{proposed_node_id}' is on the critical path. Optimizing it reduces total latency."
        )
