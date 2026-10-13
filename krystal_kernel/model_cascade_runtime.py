#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: MODEL CASCADE RUNTIME (ROOT FACADE)
==============================================================================
Module: krystal_kernel/model_cascade_runtime.py
Description:
    Root facade exposing the multi-model cascade orchestration layer:
      - ModelNode & ModelNodeSpec
      - ModelRouter & NodeRouteCandidate
      - CascadePlanner, CascadeEdge & CascadePlan
      - CascadeExecutor & CascadeMerger
      - CascadeNodeResult & CascadeExecutionTrace
      - ModelCascadeRuntime & GLOBAL_MODEL_CASCADE_RUNTIME

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

from krystal_kernel.cascade import (
    VITAL_MAX_HP,
    CascadeNodeResult,
    CascadeExecutionTrace,
    ModelNode,
    ModelNodeSpec,
    IntentClassifierNode,
    CoreRagNode,
    ReasoningNode,
    CodingSpecialistNode,
    VerificationNode,
    FinalSynthesisNode,
    ModelRouter,
    NodeRouteCandidate,
    RoutingObjectiveWeights,
    ROUTING_PROFILES,
    CascadeEdge,
    CascadePlan,
    CascadePlanner,
    CascadeMerger,
    CascadeExecutor,
    ModelCascadeRuntime,
    GLOBAL_MODEL_CASCADE_RUNTIME,
)

__all__ = [
    "VITAL_MAX_HP",
    "CascadeNodeResult",
    "CascadeExecutionTrace",
    "ModelNode",
    "ModelNodeSpec",
    "IntentClassifierNode",
    "CoreRagNode",
    "ReasoningNode",
    "CodingSpecialistNode",
    "VerificationNode",
    "FinalSynthesisNode",
    "ModelRouter",
    "NodeRouteCandidate",
    "RoutingObjectiveWeights",
    "ROUTING_PROFILES",
    "CascadeEdge",
    "CascadePlan",
    "CascadePlanner",
    "CascadeMerger",
    "CascadeExecutor",
    "ModelCascadeRuntime",
    "GLOBAL_MODEL_CASCADE_RUNTIME",
]
