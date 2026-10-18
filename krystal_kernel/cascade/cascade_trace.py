#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: MODEL CASCADE TRACE, RESULT & EVENT TELEMETRY MODELS
==============================================================================
Module: krystal_kernel/cascade/cascade_trace.py
Description:
    Granular event-level execution telemetry and forensic trace models capturing
    per-node lifecycle events, token counts, latencies, memory footprints,
    epistemic state classifications, and citation references.

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional

VITAL_MAX_HP: int = 6


@dataclass
class CascadeTraceEvent:
    """
    Granular execution lifecycle event emitted during cascade execution.
    Events:
      NODE_START, NODE_END, ROUTE_DECISION, RETRIEVAL, MODEL_INFERENCE,
      TOOL_CALL, VERIFY, FALLBACK, MERGE, FINAL_SYNTHESIS
    """
    event_id: str
    request_id: str
    node_id: str
    timestamp: float
    state_version: int
    parent_event: Optional[str]
    latency: float
    memory: float
    token_count: int
    status: str
    event_type: str
    details: Dict[str, Any] = field(default_factory=dict)
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, f"Invariant violation: vital_max_hp must be 6, got {self.vital_max_hp}"
        if not self.event_id:
            self.event_id = f"EVT_{uuid.uuid4().hex[:10].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricProvenance(str, Enum):
    """Rigorous epistemic classification of telemetry origins."""
    MEASURED  = "MEASURED"   # Physical timer, Win32 API, physical token stream
    ESTIMATED = "ESTIMATED"  # Algorithmic prediction, moving average, statistical curve
    SYNTHETIC = "SYNTHETIC"  # Heuristic mock, hardcoded constant, string split count


@dataclass
class CascadeNodeResult:
    """Atomic execution result produced by a single ModelNode."""
    node_id: str
    output: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    confidence: float = 1.0
    evidence_refs: List[str] = field(default_factory=list)
    status: str = "SUCCESS"  # SUCCESS, FALLBACK_TRIGGERED, VETOED, TIMEOUT, BYPASSED
    memory_mb: float = 0.0
    memory_used: float = 0.0
    error: Optional[str] = None
    next_route: Optional[str] = None
    epistemic_verdict: str = "PRED_MODEL_ANSWER"  # PRED_MODEL_ANSWER, EPI_VALIDATED, EPI_CONTRADICTORY, SAFE_BREACH, AUTH_VETOED
    provenance: str = "SYNTHETIC"  # MEASURED, ESTIMATED, SYNTHETIC
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, f"Invariant violation: vital_max_hp must be 6, got {self.vital_max_hp}"
        if self.memory_used == 0.0 and self.memory_mb > 0.0:
            self.memory_used = self.memory_mb
        elif self.memory_mb == 0.0 and self.memory_used > 0.0:
            self.memory_mb = self.memory_used

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["memory_used"] = self.memory_used or self.memory_mb
        d["memory_mb"] = self.memory_mb or self.memory_used
        d["provenance"] = self.provenance
        return d


@dataclass
class CascadeFinalResult:
    """
    Authoritative final outcome of a multi-model cascade execution graph.
    Reconstructs entire causal DAG, per-node outputs, evidence, and authority status.
    """
    request_id: str = ""
    plan_id: str = ""
    node_results: List[CascadeNodeResult] = field(default_factory=list)
    final_output: str = ""
    total_latency_ms: float = 0.0
    total_memory_peak: float = 0.0
    total_tokens: int = 0
    final_confidence: float = 1.0
    evidence_summary: List[str] = field(default_factory=list)
    authority_status: str = "AUTH_COMMIT_APPROVED"  # e.g. AUTH_COMMIT_APPROVED, AUTH_VETOED, SAFE_BREACH, WOULD_COMMIT
    trace_events: List[CascadeTraceEvent] = field(default_factory=list)
    shadow_mode: bool = False
    provenance_summary: Dict[str, int] = field(default_factory=lambda: {"MEASURED": 0, "ESTIMATED": 0, "SYNTHETIC": 0})
    vital_max_hp: int = VITAL_MAX_HP
    timestamp_ns: int = field(default_factory=time.time_ns)

    def __post_init__(self):
        assert self.vital_max_hp == 6, f"Invariant violation: vital_max_hp must be 6, got {self.vital_max_hp}"
        if not self.request_id:
            self.request_id = f"REQ_{uuid.uuid4().hex[:12].upper()}"
        if not self.provenance_summary or sum(self.provenance_summary.values()) == 0:
            self.provenance_summary = {"MEASURED": 0, "ESTIMATED": 0, "SYNTHETIC": 0}
            for nr in self.node_results:
                prov = getattr(nr, "provenance", "SYNTHETIC")
                self.provenance_summary[prov] = self.provenance_summary.get(prov, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "plan_id": self.plan_id,
            "final_output": self.final_output,
            "node_results": [nr.to_dict() for nr in self.node_results],
            "total_latency_ms": round(self.total_latency_ms, 3),
            "total_memory_peak": round(self.total_memory_peak, 2),
            "total_tokens": self.total_tokens,
            "final_confidence": round(self.final_confidence, 4),
            "evidence_summary": self.evidence_summary,
            "authority_status": self.authority_status,
            "shadow_mode": self.shadow_mode,
            "provenance_summary": self.provenance_summary,
            "trace_events": [te.to_dict() for te in self.trace_events],
            "vital_max_hp": self.vital_max_hp,
            "timestamp_ns": self.timestamp_ns
        }


# Fully compatible alias for existing components
@dataclass
class CascadeExecutionTrace(CascadeFinalResult):
    """Backwards-compatible subclass of CascadeFinalResult."""
    trace_id: str = ""
    prompt: str = ""
    strategy: str = "BALANCED"
    objective: str = "BALANCED"
    dual_gate_verdict: str = "AUTH_COMMIT_APPROVED"
    evidence_citations: List[str] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def __post_init__(self):
        if not self.trace_id:
            self.trace_id = self.request_id or f"TRC_{uuid.uuid4().hex[:12].upper()}"
        if not self.request_id:
            self.request_id = self.trace_id
        if self.peak_memory_mb == 0.0 and self.total_memory_peak > 0.0:
            self.peak_memory_mb = self.total_memory_peak
        elif self.total_memory_peak == 0.0 and self.peak_memory_mb > 0.0:
            self.total_memory_peak = self.peak_memory_mb
        if not self.evidence_citations and self.evidence_summary:
            self.evidence_citations = self.evidence_summary
        elif not self.evidence_summary and self.evidence_citations:
            self.evidence_summary = self.evidence_citations
        if not self.authority_status:
            self.authority_status = self.dual_gate_verdict
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["trace_id"] = self.trace_id
        d["prompt"] = self.prompt
        d["strategy"] = self.strategy
        d["objective"] = self.objective
        d["dual_gate_verdict"] = self.authority_status
        d["evidence_citations"] = self.evidence_citations or self.evidence_summary
        d["peak_memory_mb"] = self.total_memory_peak
        d["total_input_tokens"] = self.total_input_tokens
        d["total_output_tokens"] = self.total_output_tokens
        return d
