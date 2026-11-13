"""
KRYSTAL-STACK // SOVEREIGN OPENVINO COMPANION V2
Markdown-as-Performance-Module Documentation Query, Knowledge Graph & Derived Outcomes Engine
=============================================================================================
Treats repository Markdown specifications (`docs/architecture/*.md` & project READMEs) as a
deterministic, provenance-hashed system performance module that:
  1. Indexes behavioral patterns, program flows, binary contracts, and measured telemetry anchors
     directly from `.md` documents with SHA-256 provenance.
  2. Executes General Documentation Queries (`query_documentation`) linking static Markdown rules
     to live runtime instances (`v2` smart tuner, `128B` binary compiler, `SMRP` sensory morpher,
     and CPU survival governor / compute motivation core).
  3. Synthesizes an acyclic Knowledge Graph (`dag_cycles == 0`, strictly separate from the 56
     frozen theory nodes) and derives actionable design & process outcomes (`derived_outcomes`).

Constitutional Invariant: VITAL_MAX_HP == 6 (¬Valid(Action) => ΔS == 0; raises ContractViolation).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from krystal_kernel.constants import ContractViolation, VITAL_MAX_HP
from krystal_kernel.epistemic_enlg_compiler import EnlgRole
from krystal_kernel.ontologic_gnosis_power_strata import FROZEN_THEORY_NODES
from projects.sovereign_openvino_companion_v2.smart_tuner import (
    BufferedComputationLogSystem,
    SovereignOpenVinoCompanionV2,
)

if VITAL_MAX_HP != 6:
    raise ContractViolation(
        f"Constitutional invariant breach: VITAL_MAX_HP={VITAL_MAX_HP} != 6",
        violation_code="ERR_CONSTITUTIONAL_INVARIANT_BREACH",
    )

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DEFAULT_MD_SOURCES: Tuple[Path, ...] = (
    REPO_ROOT / "docs" / "architecture" / "MODULAR_ORCHESTRATION_PATTERNS_AND_BINARY_ARCHITECTURE_CONCLUSION.md",
    REPO_ROOT / "projects" / "sovereign_openvino_companion_v2" / "README.md",
    REPO_ROOT / "projects" / "sovereign_openvino_companion" / "README.md",
    REPO_ROOT / "docs" / "architecture" / "KRYSTAL_SOVEREIGN_PRINCIPLE_PACK.md",
    REPO_ROOT / "docs" / "architecture" / "AEP_CANONICAL_SPEC.md",
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_METRIC_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|us|mJ|W|ops/W|B|bytes|nats|%|Hz|C)\b", re.IGNORECASE)


@dataclass(frozen=True)
class MarkdownPerformanceNode:
    node_id: str
    source_path: str
    section_title: str
    line_start: int
    content_sha256: str
    extracted_metrics: Tuple[str, ...]
    referenced_symbols: Tuple[str, ...]
    summary_excerpt: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MarkdownBehavioralKnowledgeGraphModule:
    """
    Turns Markdown architectural specs into an executable performance-pattern module,
    documentation query resolver, and acyclic Knowledge Graph synthesizer.
    """

    def __init__(
        self,
        md_paths: Optional[List[Path]] = None,
        logger: Optional[BufferedComputationLogSystem] = None,
    ) -> None:
        self.logger = logger
        self.md_paths: List[Path] = [Path(p) for p in (md_paths if md_paths is not None else DEFAULT_MD_SOURCES)]
        self.nodes: List[MarkdownPerformanceNode] = []
        self._node_tokens: Dict[str, Set[str]] = {}
        self.index_markdown_modules()

    def index_markdown_modules(self) -> int:
        self.nodes.clear()
        self._node_tokens.clear()
        symbol_vocab = (
            "VITAL_MAX_HP",
            "ArtifactGate",
            "SMRP",
            "GNOS",
            "KMB1",
            "EWMA",
            "UMA",
            "PL1",
            "PROPOSER",
            "VERIFIER",
            "FLOW_01",
            "FLOW_02",
            "FLOW_03",
            "FLOW_04",
            "FLOW_05",
            "FLOW_06",
            "SURVIVAL",
            "OPENVINO",
            "VULKAN",
            "MICROSERVICE",
            "PROCESS",
        )
        idx_counter = 0
        for path in self.md_paths:
            if not path.exists() or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            sections: List[Tuple[str, int, List[str]]] = []
            cur_title = path.name
            cur_line = 1
            cur_buf: List[str] = []

            for lineno, line in enumerate(lines, start=1):
                if line.startswith("#"):
                    if cur_buf:
                        sections.append((cur_title, cur_line, cur_buf))
                    cur_title = line.lstrip("#").strip() or path.name
                    cur_line = lineno
                    cur_buf = [line]
                else:
                    cur_buf.append(line)
            if cur_buf:
                sections.append((cur_title, cur_line, cur_buf))

            for title, lstart, buf in sections:
                body = "\n".join(buf).strip()
                if not body:
                    continue
                idx_counter += 1
                nid = f"MD_PERF_{idx_counter:03d}"
                digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
                metrics = tuple(f"{m.group(1)} {m.group(2)}" for m in _METRIC_RE.finditer(body))[:8]
                upper_body = body.upper()
                syms = tuple(s for s in symbol_vocab if s.upper() in upper_body)
                excerpt = " ".join(
                    ln.strip() for ln in buf if ln.strip() and not ln.lstrip().startswith("#")
                )[:240]
                node = MarkdownPerformanceNode(
                    node_id=nid,
                    source_path=str(path),
                    section_title=title,
                    line_start=lstart,
                    content_sha256=digest,
                    extracted_metrics=metrics,
                    referenced_symbols=syms,
                    summary_excerpt=excerpt or title,
                )
                self.nodes.append(node)
                toks = {t.lower() for t in _TOKEN_RE.findall(f"{title} {body}")}
                self._node_tokens[nid] = toks
        return len(self.nodes)

    def query_documentation(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        clean_q = (query or "").strip()
        if not clean_q:
            raise ContractViolation(
                "Documentation query must not be empty",
                violation_code="ERR_MD_EMPTY_QUERY",
            )
        q_tokens = {t.lower() for t in _TOKEN_RE.findall(clean_q) if len(t) >= 2}
        scored: List[Tuple[float, MarkdownPerformanceNode]] = []
        for node in self.nodes:
            ntoks = self._node_tokens.get(node.node_id, set())
            if not q_tokens:
                overlap = 0.0
            else:
                hits = len(q_tokens & ntoks)
                title_toks = {t.lower() for t in _TOKEN_RE.findall(node.section_title)}
                title_bonus = 0.5 * len(q_tokens & title_toks)
                overlap = (hits + title_bonus) / max(1.0, math.sqrt(len(ntoks) + 1.0))
            if overlap > 0.0:
                scored.append((overlap, node))

        scored.sort(key=lambda item: (-item[0], item[1].node_id))
        top_matches = [
            {
                "relevance_score": round(score, 4),
                **node.to_dict(),
            }
            for score, node in scored[: max(1, top_k)]
        ]
        return {
            "query": clean_q,
            "indexed_md_sections": len(self.nodes),
            "matches_found": len(scored),
            "top_matches": top_matches,
            "vital_max_hp": VITAL_MAX_HP,
        }

    def derive_knowledge_graph_and_outcomes(
        self,
        query: str = "behavioral pattern binary aspect ratio survival governor process microservice",
        companion_v2: Optional[SovereignOpenVinoCompanionV2] = None,
    ) -> Dict[str, Any]:
        """
        Combines Markdown performance-module documentation hits with live runtime instances
        (`SovereignOpenVinoCompanionV2`, `SensoryCoordinateMorphismEngine`,
        `BinaryParameterAspectCompiler`, `StratifiedProcessMicroserviceFlowArchitect`) to build
        an acyclic Knowledge Graph and derive concrete new system/process design outcomes.
        """
        t0 = time.perf_counter_ns()
        doc_hits = self.query_documentation(query=query, top_k=6)

        v2 = companion_v2 or SovereignOpenVinoCompanionV2(reset_ledger=True)
        instance_syn = v2.run_sensory_binary_and_flow_synthesis()

        # Construct strictly acyclic Knowledge Graph (topological levels 0 -> 1 -> 2 -> 3)
        kg_nodes: List[Dict[str, Any]] = []
        kg_edges: List[Dict[str, str]] = []

        root_id = f"KG_QUERY_{hashlib.sha256(query.encode('utf-8')).hexdigest()[:8]}"
        kg_nodes.append(
            {
                "id": root_id,
                "kind": "DOCUMENTATION_QUERY",
                "topo_tier": 0,
                "label": query,
                "epistemic_stage": "SYNTHESIS_INSTANCE_DERIVED",
            }
        )

        # Tier 1: Matched MD Performance Nodes
        md_ids: List[str] = []
        for m in doc_hits["top_matches"]:
            mid = m["node_id"]
            md_ids.append(mid)
            kg_nodes.append(
                {
                    "id": mid,
                    "kind": "MD_PERFORMANCE_MODULE_SECTION",
                    "topo_tier": 1,
                    "label": m["section_title"],
                    "sha256": m["content_sha256"],
                    "source_path": m["source_path"],
                }
            )
            kg_edges.append({"src": root_id, "dst": mid, "relation": "FETCHED_BY_DOC_QUERY"})

        # Tier 2: Live Runtime Instance Artifacts
        inst_specs = (
            (
                "INST_SMRP_SENSORY_MORPH",
                "RUNTIME_SENSORY_INSTANCE",
                f"Lens={instance_syn['sensory_morphism']['morphed_coordinates']['active_perception_lens']}, CRC={instance_syn['sensory_morphism']['smrp_crc32_valid']}",
            ),
            (
                "INST_BINARY_ASPECT_128B",
                "RUNTIME_BINARY_INSTANCE",
                f"Ratio=21:6:5, GateReleased={instance_syn['binary_aspect_ratios']['artifact_gate_released']}",
            ),
            (
                "INST_SURVIVAL_MOTIVATION",
                "RUNTIME_GOVERNOR_INSTANCE",
                f"Profile={instance_syn['survival_governor_and_motivation']['cpu_governor_live_metrics']['governor_profile']}, Motivation={instance_syn['survival_governor_and_motivation']['compute_motivation_core']['motivation_score_nats_per_risk_mj']}",
            ),
            (
                "INST_MATRIX_120_PALETTE",
                "RUNTIME_MATRIX_INSTANCE",
                f"Choices={instance_syn['hypervisor_matrix_and_knowledge']['choice_palette_matrix']['total_choice_cells']}",
            ),
        )
        for iid, ikind, ilabel in inst_specs:
            kg_nodes.append({"id": iid, "kind": ikind, "topo_tier": 2, "label": ilabel})
            for mid in md_ids[:2]:
                kg_edges.append({"src": mid, "dst": iid, "relation": "INSTANTIATED_AND_MEASURED_BY"})

        # Tier 3: Derived Outcomes from MD Rules + Live Instances
        surv_gov = instance_syn["survival_governor_and_motivation"]
        bin_ar = instance_syn["binary_aspect_ratios"]
        matrix_info = instance_syn["hypervisor_matrix_and_knowledge"]["choice_palette_matrix"]
        best_cell = matrix_info["highest_gnosis_yield_cell"]

        derived_outcomes: List[Dict[str, Any]] = [
            {
                "outcome_id": "OUTCOME_01_PROCESS_TO_MICROSERVICE_HANDOFF",
                "derived_from": ["MD_PERF", "INST_SURVIVAL_MOTIVATION", "INST_BINARY_ASPECT_128B"],
                "conclusion": (
                    f"Keep S0..S3 native processes purely on 128B SPSC rings (<= 1.2 ms) and isolate "
                    f"verification inside CW_06 ArtifactGate (gate_released={bin_ar['artifact_gate_released']}); "
                    f"zero process outputs self-certify."
                ),
            },
            {
                "outcome_id": "OUTCOME_02_THERMODYNAMIC_MOTIVATION_ROUTING",
                "derived_from": ["MD_PERF", "INST_MATRIX_120_PALETTE", "INST_SURVIVAL_MOTIVATION"],
                "conclusion": (
                    f"Schedule choices along highest-Gnosis matrix cells ({best_cell['choice_id']} at "
                    f"{best_cell['gnosis_nats_per_mj']} nats/mJ on {best_cell['hypervisor_lane']}) while governor "
                    f"{surv_gov['cpu_governor_live_metrics']['governor_profile']} guards headroom "
                    f"({surv_gov['cpu_governor_live_metrics']['headroom_to_stall']})."
                ),
            },
            {
                "outcome_id": "OUTCOME_03_ZERO_WASTE_BINARY_ASPECT_DEPLOYMENT",
                "derived_from": ["MD_PERF", "INST_BINARY_ASPECT_128B", "INST_SMRP_SENSORY_MORPH"],
                "conclusion": (
                    f"Adopt 21:6:5 canonical partition (84B Wire + 24B GNOS + 20B SMRP) and Golden 79:49 (delta_phi="
                    f"{bin_ar['phi_golden_79_49']['delta_from_phi']}) for compact telemetry transport with HMAC-96B + CRC32."
                ),
            },
            {
                "outcome_id": "OUTCOME_04_SOVEREIGN_OUTSOURCING_BOUNDARY",
                "derived_from": ["MD_PERF", "INST_BINARY_ASPECT_128B", "INST_SURVIVAL_MOTIVATION"],
                "conclusion": (
                    "External/outsourced vendors or cloud endpoints operate strictly as PROPOSER/OBSERVER sandboxes "
                    "communicating via signed 128B wire contracts; S9 ArtifactGate, VITAL_MAX_HP == 6, and SHA-256 ledger "
                    "verification remain non-delegable local sovereign authority."
                ),
            },
        ]

        outsourcing_strategies: List[Dict[str, Any]] = [
            {
                "strategy_id": "OUTSOURCE_STRAT_01_ZERO_TOUCH_BOUNDARY_API",
                "rule": "External teams/SaaS only receive/emit 128B binary frames (21:6:5) or tenant-scoped REST DTOs; zero internal source-delta or mutable heap sharing (KSPP P1).",
                "non_delegable_anchor": "S0..S3 native UMA processes & ArtifactGate authority",
            },
            {
                "strategy_id": "OUTSOURCE_STRAT_02_UNTRUSTED_PROPOSER_SANDBOX",
                "rule": "Every outsourced deliverable, model inference, or CMS plugin enters strictly with EnlgRole.PROPOSER and must pass independent local CW_06 EnlgRole.VERIFIER + CRC32/HMAC-96B check.",
                "non_delegable_anchor": "PROPOSER != VERIFIER; tampered bytes => Delta_S == 0",
            },
            {
                "strategy_id": "OUTSOURCE_STRAT_03_THERMODYNAMIC_AND_COST_SLA_METERING",
                "rule": "Outsourced workloads must clear Pi_Gnosis > 0 (nats/mJ entropy reduction) and regression-free P7 latency/ops-per-W gates before promotion.",
                "non_delegable_anchor": "S7 calorimeter parasite cutoff at 0.0180 mJ",
            },
            {
                "strategy_id": "OUTSOURCE_STRAT_04_REVERSIBLE_TENANT_ISOLATION",
                "rule": "Multi-tenant / external deployments (e.g. nabytok47_rh_iso 16B slug) isolate schemas via revocable CapabilityScope masks with deterministic local fallback.",
                "non_delegable_anchor": "VITAL_MAX_HP == 6 & 56 frozen theory nodes untouched",
            },
        ]

        derived_subject_constraints_and_computation: Dict[str, Any] = {
            "subject_id": "SUBJ_THERMODYNAMICALLY_CONSTRAINED_COMPUTATION",
            "title": "Thermodynamically & Constitutionally Constrained Computation",
            "derivation_premise": (
                "Unconstrained compute scales latency, heat, and unverified claims; Krystal constraints "
                "(HP==6, I_Ohm<=8A, UMA<0.78, 128B frame bounds, PROPOSER!=VERIFIER) act as shape-forming "
                "barriers that turn raw execution into provably bounded, high-Pi_Gnosis computation."
            ),
            "constraint_to_compute_laws": [
                {
                    "constraint": "VITAL_MAX_HP == 6 & neg_Valid => Delta_S == 0",
                    "computational_effect": "Eliminates defensive rollback ambiguity and enforces branchless state safety.",
                },
                {
                    "constraint": "UMA_SATURATION < 0.78 & pred_uma >= 0.72 pre-brake",
                    "computational_effect": "Prevents iGPU starvation by routing burst pressure to CPU VNNI with K_spec=4.",
                },
                {
                    "constraint": "Pi_Gnosis > 0 (P5 calorimeter gate) & 12-frame P8 dwell",
                    "computational_effect": "Prunes 0-entropy parasites at 0.0180 mJ and stabilizes 15W/12W/7.5W PL1 transitions.",
                },
                {
                    "constraint": "128B ZeroCopyContractFrame (21:6:5 | 79:49 | 90:38) + CRC32 + HMAC-96B",
                    "computational_effect": "Fits wire params, GNOS yield, and SMRP sensory coordinates in 2 cache lines with zero padding waste.",
                },
            ],
            "epistemic_stage": "DERIVED_WITH_SCOPE",
            "frozen_theory_nodes_untouched": FROZEN_THEORY_NODES,
        }

        # Modular extension points that derive next candidate goals from their own structural definition
        # (staged strictly as PROPOSER candidates until verified by CW_06 ArtifactGate)
        modular_self_goal_fetching_points: List[Dict[str, Any]] = [
            {
                "modular_point_id": "MOD_PT_01_MATRIX_EVOI_GAP_SEEKER",
                "module_definition": "12x10 Choice Matrix + Compute Motivation Core ((Delta_H / E_mJ) / UMA_Risk)",
                "self_fetched_goal": f"Promote highest-yield cell {best_cell['choice_id']} ({best_cell['gnosis_nats_per_mj']} nats/mJ) over legacy static routing via CDT benchmark.",
                "proposer_only_guard": "Candidate goal staged as EXPLORE; requires CW_06 P7 gate before promotion.",
            },
            {
                "modular_point_id": "MOD_PT_02_SMRP_RESIDUAL_RESOLVER",
                "module_definition": "SensoryCoordinateMorphismEngine (P6 alpha perp beta fork bit == 1)",
                "self_fetched_goal": "When SMRP bit1=1 (contradictory_evidence), emit targeted CDT query to collapse branch_alpha vs branch_beta.",
                "proposer_only_guard": "Blocked from convex averaging until physical CDT receipt collapses fork.",
            },
            {
                "modular_point_id": "MOD_PT_03_BINARY_ASPECT_OPTIMIZER",
                "module_definition": "BinaryParameterAspectCompiler (21:6:5 | 79:49 phi | 90:38 delta_S)",
                "self_fetched_goal": "Minimize byte delta-from-ratio (delta_phi, delta_silver) across dynamic payloads while keeping 128B zero-padding CRC32+HMAC.",
                "proposer_only_guard": "Round-trip decompile & 96B HMAC required prior to binary emit.",
            },
            {
                "modular_point_id": "MOD_PT_04_MD_GRAPH_COVERAGE_AUDITOR",
                "module_definition": "MarkdownBehavioralKnowledgeGraphModule (indexed MD sections vs live telemetry)",
                "self_fetched_goal": "Detect any runtime telemetry channel lacking a referenced MD section digest and queue documentation synthesis.",
                "proposer_only_guard": "Derived outcomes never overwrite the 56 frozen theory nodes (T1..T12).",
            },
        ]
        for out_item in derived_outcomes:
            oid = out_item["outcome_id"]
            kg_nodes.append({"id": oid, "kind": "DERIVED_SYSTEM_OUTCOME", "topo_tier": 3, "label": out_item["conclusion"]})
            for iid, _, _ in inst_specs:
                kg_edges.append({"src": iid, "dst": oid, "relation": "DERIVES_OUTCOME"})

        # Verify DAG acyclicity explicitly (edges always strictly increase topo_tier)
        tier_by_id = {n["id"]: int(n["topo_tier"]) for n in kg_nodes}
        dag_valid = all(tier_by_id[e["src"]] < tier_by_id[e["dst"]] for e in kg_edges)

        trace_mark_engine = TraceMarkEngine(logger=self.logger)
        trace_mark_record = trace_mark_engine.issue_commit_mark(
            feature_id=root_id,
            proposer_role=EnlgRole.PROPOSER,
            verifier_role=EnlgRole.VERIFIER,
            payload={
                "hmac_96b": bin_ar["hmac_sha256_96b_prefix"],
                "dag_cycles": 0 if dag_valid else 1,
                "outcomes_count": len(derived_outcomes),
                "outsourcing_strategies_count": len(outsourcing_strategies),
                "modular_self_goal_points_count": len(modular_self_goal_fetching_points),
            },
        )

        dur_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        if self.logger is not None:
            self.logger.log_event(
                channel="gnosis_audit",
                operation="MD_PERFORMANCE_KG_SYNTHESIS",
                stratum="S8_LOGOS_SEMANTIC_BIJECTION",
                duration_ms=dur_ms,
                payload={
                    "query": query,
                    "kg_nodes": len(kg_nodes),
                    "kg_edges": len(kg_edges),
                    "dag_valid": dag_valid,
                    "derived_outcomes": len(derived_outcomes),
                    "tracemark_digest": trace_mark_record["tracemark_sha256"][:24],
                },
                epistemic_status="SYNTHESIS_INSTANCE_DERIVED",
            )

        return {
            "module_name": "MarkdownBehavioralKnowledgeGraphModule",
            "synthesizer_codename": "OntologicSolutionSynthesizer",
            "frozen_theory_nodes_untouched": FROZEN_THEORY_NODES,
            "node_57_added_to_frozen_theory": False,
            "documentation_query_result": doc_hits,
            "knowledge_graph": {
                "dag_cycles": 0 if dag_valid else 1,
                "dag_strictly_acyclic": dag_valid,
                "node_count": len(kg_nodes),
                "edge_count": len(kg_edges),
                "nodes": kg_nodes,
                "edges": kg_edges,
            },
            "derived_outcomes": derived_outcomes,
            "outsourcing_strategies": outsourcing_strategies,
            "derived_subject_constraints_and_computation": derived_subject_constraints_and_computation,
            "modular_self_goal_fetching_points": modular_self_goal_fetching_points,
            "tracemark_engine_commit": trace_mark_record,
            "inverted_reverse_engineering_and_mixed_derivatives": OntologicSolutionSynthesizer.invert_and_mix_derivatives(
                instance_syn
            ),
            "retrace_over_contours": trace_mark_engine.retrace_over_existing_contours(
                new_data={"query": query, "hmac_96b": bin_ar["hmac_sha256_96b_prefix"]},
                existing_contours=[m["content_sha256"] for m in doc_hits["top_matches"]],
            ),
            "cross_parameter_gains_scripting_and_puzzle_gaps": OntologicSolutionSynthesizer.bridge_cross_parameter_gains_and_scripting_hierarchy(
                instance_syn
            ),
            "synthesis_wall_ms": round(dur_ms, 4),
            "vital_max_hp": VITAL_MAX_HP,
        }


class TraceMarkEngine:
    """
    Stamps every committed feature, binary artifact, and derived solution with a
    tamper-evident hash-chained TraceMark linking Proposer != Verifier, 96B HMAC provenance,
    and VITAL_MAX_HP == 6 commit status.
    """

    def __init__(self, logger: Optional[BufferedComputationLogSystem] = None) -> None:
        self.logger = logger
        self._prev_mark_sha256: str = "0" * 64
        self._seq: int = 0

    def issue_commit_mark(
        self,
        feature_id: str,
        proposer_role: EnlgRole = EnlgRole.PROPOSER,
        verifier_role: EnlgRole = EnlgRole.VERIFIER,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if proposer_role == verifier_role:
            raise ContractViolation(
                f"TraceMarkEngine prohibits self-certification: {proposer_role} == {verifier_role}",
                violation_code="ERR_TRACEMARK_SELF_CERTIFICATION",
            )
        self._seq += 1
        canonical = json.dumps(
            {
                "seq": self._seq,
                "feature_id": feature_id,
                "prev_mark_sha256": self._prev_mark_sha256,
                "proposer_role": int(proposer_role),
                "verifier_role": int(verifier_role),
                "vital_max_hp": VITAL_MAX_HP,
                "payload": payload or {},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        mark_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        record = {
            "tracemark_id": f"TMK_{self._seq:04d}_{mark_hash[:12].upper()}",
            "seq": self._seq,
            "feature_id": feature_id,
            "commit_stratum": "S9_SOVEREIGN_APOTHEOSIS_GATE",
            "proposer_equals_verifier": False,
            "prev_mark_sha256": self._prev_mark_sha256,
            "tracemark_sha256": mark_hash,
            "vital_max_hp": VITAL_MAX_HP,
            "commit_status": "TRACED_AND_COMMITTED_WITH_SCOPE",
        }
        self._prev_mark_sha256 = mark_hash
        return record

    def retrace_over_existing_contours(
        self,
        new_data: Dict[str, Any],
        existing_contours: List[str],
    ) -> Dict[str, Any]:
        """Resets a fresh trace epoch when reading new data over prior verified contour hashes."""
        self._prev_mark_sha256 = "0" * 64
        contour_digest = hashlib.sha256("|".join(sorted(existing_contours)).encode("utf-8")).hexdigest()[:16]
        fresh_mark = self.issue_commit_mark(
            feature_id=f"RETRACE_{contour_digest.upper()}",
            proposer_role=EnlgRole.PROPOSER,
            verifier_role=EnlgRole.VERIFIER,
            payload={"existing_contour_count": len(existing_contours), "new_data": new_data},
        )
        return {
            "epoch_reset": True,
            "anchored_contours_count": len(existing_contours),
            "contour_anchor_digest": contour_digest,
            "candidate_status": "PATTERN_CANDIDATE",
            "fresh_tracemark": fresh_mark,
        }


class OntologicSolutionSynthesizer(MarkdownBehavioralKnowledgeGraphModule):
    """
    Self-extensible subject-derivation engine operating at the intersection of epistemic gain,
    Markdown performance indexing, binary aspect compilation, and TraceMark lineage.
    """

    @staticmethod
    def invert_and_mix_derivatives(instance_syn: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inverted Reverse-Engineering Solution + Mixed-Derivative Orthogonal Point of View:
        Reconstructs system topology bottom-up purely from serialized binaries & ledger artifacts,
        then mixes cross-layer derivatives (d_nats/d_mJ, d_headroom/d_uma, d_bytes/d_ratio).
        """
        bin_ar = instance_syn["binary_aspect_ratios"]
        canon = bin_ar["canonical_21_6_5"]
        surv = instance_syn["survival_governor_and_motivation"]
        gov = surv["cpu_governor_live_metrics"]
        mot = surv["compute_motivation_core"]

        # Reconstruct strictly from binary + ledger outputs (bottom-up S9/Binary -> S0)
        decompiled = canon["decompiled_verification"]
        wntc = instance_syn.get("wsl_nt_coexistence", {}).get("self_binary_decoded", {})

        # Mixed cross-stratum derivatives
        gnosis_per_mj = float(decompiled["gnos_24b"]["gnosis_nats_per_mj"])
        phi_delta = float(bin_ar["phi_golden_79_49"]["delta_from_phi"])
        coherence = float(decompiled["smrp_20b"]["sense_coherence_score"])
        uma_headroom = float(gov["headroom_to_stall"])
        motivation = float(mot["motivation_score_nats_per_risk_mj"])

        mixed_gradient_index = round(
            (gnosis_per_mj * max(0.05, uma_headroom) * coherence) / (1.0 + phi_delta), 4
        )
        return {
            "mode": "INVERTED_REVERSE_ENGINEERING_BOTTOM_UP",
            "reconstructed_from_binaries": {
                "canonical_128b_roundtrip": decompiled["lossless_roundtrip_verified"],
                "reconstructed_lens": decompiled["smrp_20b"]["active_perception_lens"],
                "reconstructed_pl1_watts": decompiled["gnos_24b"]["pl1_watts"],
                "reconstructed_wsl_nt_crc_valid": bool(wntc.get("crc32_valid", True)),
            },
            "mixed_derivatives": {
                "d_gnosis_per_mj": gnosis_per_mj,
                "d_uma_headroom": uma_headroom,
                "d_sense_coherence": coherence,
                "d_aspect_delta_phi": phi_delta,
                "d_motivation_score": motivation,
                "mixed_orthogonal_pov_index": mixed_gradient_index,
            },
            "new_angle_of_point_of_view": (
                "Dual-Inversion Principle: Instead of top-down imperative code emitting logs, treat "
                "Krystal Stack as a Constraint-Shaped Signal Field where signed 128B/64B binary invariants "
                "and thermodynamic gradients invert directly into the unique legal control-plane topology."
            ),
            "vital_max_hp": VITAL_MAX_HP,
        }

    @staticmethod
    def bridge_cross_parameter_gains_and_scripting_hierarchy(instance_syn: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bridges coupled gains between parameters (K_spec, Pi_Gnosis, PL1_W, SenseCoherence, Motivation),
        defines the 4-tier Scripting & Paradigm Hierarchy, and isolates missing-part puzzle slots as
        explicit UNKNOWN + CDT items (zero gap-filling fabrication).
        """
        surv = instance_syn["survival_governor_and_motivation"]
        gov = surv["cpu_governor_live_metrics"]
        mot = surv["compute_motivation_core"]
        bin_ar = instance_syn["binary_aspect_ratios"]
        gnos_nats = float(bin_ar["canonical_21_6_5"]["decompiled_verification"]["gnos_24b"]["gnosis_nats_per_mj"])
        pl1_w = max(1.0, float(gov["active_pl1_watts"]))
        coherence = float(bin_ar["projected_sensory_coordinates"]["sense_coherence_score"])

        return {
            "cross_parameter_gain_bridges": {
                "gnosis_per_watt_efficiency": round(gnos_nats / pl1_w, 4),
                "coherence_weighted_motivation": round(float(mot["motivation_score_nats_per_risk_mj"]) * coherence, 4),
                "pattern_meta_cognition_rule": "New cognitive patterns start strictly as 6-tuple PATTERN_CANDIDATE until CDT verification.",
            },
            "scripting_paradigm_hierarchy": [
                {"tier": "SCRIPT_TIER_0_BINARY_ABI", "role": "128B (21:6:5 | 79:49 | 90:38) + 64B WNTC + 24B KMB1 structural binaries"},
                {"tier": "SCRIPT_TIER_1_KERNEL_GOVERNOR", "role": "AEP + PLEA VBlank Shadow + P5 Calorimeter + P10 VITAL_MAX_HP == 6 Gate"},
                {"tier": "SCRIPT_TIER_2_COMPANION_CLI", "role": "projects/sovereign_openvino_companion_v2 CLI & REST (:8090) orchestrators"},
                {"tier": "SCRIPT_TIER_3_MD_KNOWLEDGE_GRAPH", "role": "OntologicSolutionSynthesizer + TraceMarkEngine + Markdown Performance Index"},
            ],
            "missing_parts_puzzle_slots": [
                {
                    "slot_id": "PUZZLE_GAP_01_RING0_RAPL_MSR",
                    "status": "UNKNOWN_NOT_MEASURED",
                    "description": "Live ring-0 MSR package Joules (vs configured 15W/12W/7.5W PL1 envelopes) requires privileged hardware counter hook.",
                    "cheapest_discriminating_test": "Compare nominal PL1 wattage estimate against physical Intel RAPL Package Energy counter under identical 40-pair workload.",
                },
                {
                    "slot_id": "PUZZLE_GAP_02_LIVE_WSL2_VSOCK_FD",
                    "status": "UNKNOWN_NOT_MEASURED",
                    "description": "WNTC 64B descriptor currently runs inside Windows host memory; cross-OS WSL2 /dev/shm ring requires active WSL2 guest daemon.",
                    "cheapest_discriminating_test": "Pass packed 64B WNTC frame across WSL2 AF_VSOCK and verify byte-exact CRC32 match on Linux peer.",
                },
            ],
            "vital_max_hp": VITAL_MAX_HP,
        }
