"""
Krystal-Bot Audiovisual Graph Diagnostics Adapter
=================================================
Milestone: V-AUDIOVISUAL-GRAPH-DIAGNOSTICS-01
Reference: docs/architecture/PROOF_CARRYING_GOVERNED_RUNTIME.md

Exposes the VisualStateSpaceGraph, GraphCollisionDiagnosticsDetector,
and LlmMemoryDiagnosticsEvaluator to krystal_bot agents and interactive services.
"""

from __future__ import annotations

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.audiovisual_graph_diagnostics import (
    DiagnosticSeverity,
    GraphTelemetrySnapshot,
    PhaseSpacePoint,
    DiagnosticAnomaly,
    DiagnosticReport,
    LlmDiagnosticSynthesis,
    VisualStateSpaceGraph,
    GraphCollisionDiagnosticsDetector,
    LlmMemoryDiagnosticsEvaluator,
)

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"

__all__ = [
    "DiagnosticSeverity",
    "GraphTelemetrySnapshot",
    "PhaseSpacePoint",
    "DiagnosticAnomaly",
    "DiagnosticReport",
    "LlmDiagnosticSynthesis",
    "VisualStateSpaceGraph",
    "GraphCollisionDiagnosticsDetector",
    "LlmMemoryDiagnosticsEvaluator",
]
