#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: MODEL CASCADE & MULTI-MODEL EXECUTION FABRIC PACKAGE
==============================================================================
Module: krystal_kernel/cascade/__init__.py
Description:
    Exposes canonical abstractions, planner, router, executor, nodes, and
    the global orchestrator instance GLOBAL_MODEL_CASCADE_RUNTIME.

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.cascade.cascade_trace import CascadeNodeResult, CascadeExecutionTrace
from krystal_kernel.cascade.model_node import (
    ModelNode,
    ModelNodeSpec,
    IntentClassifierNode,
    CoreRagNode,
    ReasoningNode,
    CodingSpecialistNode,
    VerificationNode,
    FinalSynthesisNode
)
from krystal_kernel.cascade.model_router import (
    ModelRouter,
    NodeRouteCandidate,
    RoutingObjectiveWeights,
    ROUTING_PROFILES
)
from krystal_kernel.cascade.cascade_planner import CascadeEdge, CascadePlan, CascadePlanner
from krystal_kernel.cascade.cascade_executor import CascadeMerger, CascadeExecutor
from krystal_kernel.cascade.cascade_runtime import ModelCascadeRuntime, GLOBAL_MODEL_CASCADE_RUNTIME

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
