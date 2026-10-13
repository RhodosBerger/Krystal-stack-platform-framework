#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: MODEL CASCADE TRACE & RESULT MODELS
==============================================================================
Module: krystal_kernel/cascade/cascade_trace.py
Description:
    Forensic trace models capturing node-level execution telemetry, input/output
    tokens, latencies, confidence metrics, and citation evidence references.

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

VITAL_MAX_HP: int = 6


@dataclass
class CascadeNodeResult:
    """Atomic execution result produced by a single ModelNode."""
    node_id: str
    output: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    confidence: float
    evidence_refs: List[str] = field(default_factory=list)
    status: str = "SUCCESS"  # SUCCESS, FALLBACK_TRIGGERED, VETOED, TIMEOUT, BYPASSED
    memory_mb: float = 0.0
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, f"Invariant violation: vital_max_hp must be 6, got {self.vital_max_hp}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CascadeExecutionTrace:
    """Comprehensive forensic trace of a complete multi-model cascade execution."""
    trace_id: str
    plan_id: str
    prompt: str
    strategy: str
    objective: str
    final_output: str
    node_results: List[CascadeNodeResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    peak_memory_mb: float = 0.0
    final_confidence: float = 1.0
    dual_gate_verdict: str = "AUTH_COMMIT_APPROVED"
    evidence_citations: List[str] = field(default_factory=list)
    vital_max_hp: int = VITAL_MAX_HP
    timestamp_ns: int = field(default_factory=time.time_ns)

    def __post_init__(self):
        assert self.vital_max_hp == 6, f"Invariant violation: vital_max_hp must be 6, got {self.vital_max_hp}"
        if not self.trace_id:
            self.trace_id = f"TRC_{uuid.uuid4().hex[:12].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "plan_id": self.plan_id,
            "prompt": self.prompt,
            "strategy": self.strategy,
            "objective": self.objective,
            "final_output": self.final_output,
            "node_results": [nr.to_dict() for nr in self.node_results],
            "total_latency_ms": round(self.total_latency_ms, 3),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "final_confidence": round(self.final_confidence, 4),
            "dual_gate_verdict": self.dual_gate_verdict,
            "evidence_citations": self.evidence_citations,
            "vital_max_hp": self.vital_max_hp,
            "timestamp_ns": self.timestamp_ns
        }
