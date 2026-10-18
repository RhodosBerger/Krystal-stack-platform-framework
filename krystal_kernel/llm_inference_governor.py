"""
KRYSTAL-STACK // LLM INFERENCE GOVERNOR & BACKPRESSURE ENGINE
=============================================================================
Normative Specification: docs/architecture/LLM_GOVERNOR_EXTENSION.md
Classification: [IMPLEMENTED]

Core Implementation:
  1. LLMInferenceState - Authoritative state tracking token throughput & health.
  2. StaticBaselineGovernor - Greedy dispatch without bus awareness.
  3. AdaptiveInferenceGovernor - Chronometric shadow multiplexer & backpressure.
  4. Integration with AepTransactionalEngine (WorkloadType.INFERENCE).
  5. Invariant Preservation - VITAL_MAX_HP == 6.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.aep_engine import (
    AepTransactionalEngine,
    AepReceipt,
    AepVerdict,
    WorkloadType,
    ExecutionLane,
    VitalHpInvariantRule,
    compute_state_digest,
    VITAL_MAX_HP,
)
from krystal_kernel.slovak_llm_adapter import (
    SlovakLLMAdapter,
    InferenceRequest,
    InferenceResponse,
    ModelSpec,
)


class GovernorMode(str, enum.Enum):
    """Execution strategy governing inference dispatch."""
    STATIC_BASELINE = "STATIC_BASELINE"
    ADAPTIVE_GOVERNOR = "ADAPTIVE_GOVERNOR"


@dataclass
class LLMInferenceState:
    """Authoritative state for Layer 8 Inference Workload."""
    session_id: str
    active_model: str
    completed_requests: int = 0
    total_tokens_generated: int = 0
    total_prompt_tokens: int = 0
    accumulated_ttft_us: int = 0
    accumulated_execution_us: int = 0
    backpressure_drops: int = 0
    last_action: str = "INIT"
    vital_max_hp: int = VITAL_MAX_HP
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def transition_llm_inference(
    state: LLMInferenceState,
    action_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Pure candidate state transition function S' = T(St, A).
    Transforms previous state into candidate state based on inference telemetry.
    """
    new_state = copy.deepcopy(state.to_dict() if hasattr(state, "to_dict") else state)
    
    # If the payload indicates backpressure drop or invariant check
    action = action_payload.get("action", "INFERENCE_SUBMIT")
    new_state["last_action"] = action

    if action_payload.get("dropped", False):
        new_state["backpressure_drops"] = new_state.get("backpressure_drops", 0) + 1
        return new_state

    # Update state metrics on successful completion
    telemetry = action_payload.get("inference_telemetry", {})
    new_state["completed_requests"] = new_state.get("completed_requests", 0) + 1
    new_state["total_tokens_generated"] = (
        new_state.get("total_tokens_generated", 0) + telemetry.get("completion_tokens", 0)
    )
    new_state["total_prompt_tokens"] = (
        new_state.get("total_prompt_tokens", 0) + telemetry.get("prompt_tokens", 0)
    )
    new_state["accumulated_ttft_us"] = (
        new_state.get("accumulated_ttft_us", 0) + telemetry.get("ttft_us", 0)
    )
    new_state["accumulated_execution_us"] = (
        new_state.get("accumulated_execution_us", 0) + telemetry.get("total_duration_us", 0)
    )
    new_state["vital_max_hp"] = VITAL_MAX_HP

    # Expose actual device and telemetry for AepReceipt capture
    new_state["actual_device"] = telemetry.get("actual_device", "CPU")
    new_state["requested_device"] = telemetry.get("requested_device", "GPU")
    if telemetry.get("device_fallback_reason"):
        new_state["fallback_reason"] = telemetry.get("device_fallback_reason")
    new_state["inference_telemetry"] = telemetry

    return new_state


class AepInferenceGovernor:
    """
    Deterministic AEP Governor managing LLM inference pipelines.
    Supports STATIC_BASELINE (greedy) and ADAPTIVE_GOVERNOR (bus-aware) modes.
    """

    def __init__(
        self,
        mode: GovernorMode = GovernorMode.ADAPTIVE_GOVERNOR,
        adapter: Optional[SlovakLLMAdapter] = None,
        engine: Optional[AepTransactionalEngine] = None,
        max_pressure_threshold: float = 0.85,
    ) -> None:
        self.mode = mode
        self.adapter = adapter or SlovakLLMAdapter()
        self.engine = engine or AepTransactionalEngine(
            invariants=[VitalHpInvariantRule()],
            max_pressure_threshold=max_pressure_threshold,
        )
        self.current_state = LLMInferenceState(
            session_id=f"SESS-{time.time_ns()}",
            active_model=self.adapter.model_spec.model_id,
        )

    def dispatch_inference(
        self,
        prompt: str,
        max_tokens: int = 64,
        target_device: str = "GPU",
        current_bus_pressure: float = 0.10,
        deadline_budget_us: int = 500_000,
        enforce_headroom: bool = True,
    ) -> Tuple[bool, AepReceipt, Optional[InferenceResponse]]:
        """
        Dispatches inference through governor policy and AEP transaction gate.
        Enforces zero-dirty mutation on backpressure rejection.
        """
        runtime_id = f"AEP-INF-{self.mode.value}"

        # 1. Governor Decision Policy
        effective_max_tokens = max_tokens
        effective_device = target_device
        execution_lane = ExecutionLane.COMPUTE

        if self.mode == GovernorMode.STATIC_BASELINE:
            # Greedy dispatch: Ignore bus pressure completely
            effective_max_tokens = max_tokens
            effective_device = target_device
            execution_lane = ExecutionLane.COMPUTE

        elif self.mode == GovernorMode.ADAPTIVE_GOVERNOR:
            # Adaptive Chronometric Multiplexing
            if current_bus_pressure < 0.40:
                # Green Zone: Full throughput
                effective_max_tokens = max_tokens
                effective_device = target_device
                execution_lane = ExecutionLane.COMPUTE
            elif current_bus_pressure < 0.70:
                # Yellow Zone: Paced execution, clamp tokens by 50%
                effective_max_tokens = max(16, max_tokens // 2)
                effective_device = target_device
                execution_lane = ExecutionLane.COMPUTE
            elif current_bus_pressure < 0.85:
                # Orange Zone: Contention. Emergency compression + route to CPU
                # to liberate Iris Xe UMA bandwidth for render scanout
                effective_max_tokens = min(24, max_tokens)
                effective_device = "CPU"
                execution_lane = ExecutionLane.BACKGROUND
            else:
                # Red Zone: Critical saturation. Pre-emptively reject before running
                # to prevent bus starvation.
                pass

        # 2. Check engine backpressure rejection (L7)
        if current_bus_pressure > self.engine.max_pressure_threshold:
            # Backpressure rejection in transaction engine
            result = self.engine.execute_transition(
                runtime_id=runtime_id,
                workload=WorkloadType.INFERENCE,
                action="INFERENCE_DISCARD_BACKPRESSURE",
                current_state=self.current_state,
                transition_fn=transition_llm_inference,
                action_payload={"action": "INFERENCE_DISCARD_BACKPRESSURE", "dropped": True},
                policy_id=f"policy-{self.mode.value.lower()}",
                policy_revision=1,
                execution_lane=execution_lane,
                backend="slovak-nlp-qwen3-14b",
                device=effective_device,
                current_bus_pressure=current_bus_pressure,
                deadline_budget_us=deadline_budget_us,
            )
            # Delta S == 0 on discard
            return False, result.receipt, None

        # 3. Model Adapter Execution
        req = InferenceRequest(
            prompt=prompt,
            max_new_tokens=effective_max_tokens,
            target_device=effective_device,
        )

        try:
            inference_resp = self.adapter.execute_inference(
                req,
                enforce_headroom=enforce_headroom,
                bus_pressure=current_bus_pressure,
            )
        except Exception as e:
            # Handled via transaction engine contract violation
            result = self.engine.execute_transition(
                runtime_id=runtime_id,
                workload=WorkloadType.INFERENCE,
                action="INFERENCE_FAILED",
                current_state=self.current_state,
                transition_fn=lambda s, p: (_ for _ in ()).throw(e),
                action_payload={"error": str(e)},
                policy_id=f"policy-{self.mode.value.lower()}",
                policy_revision=1,
                execution_lane=execution_lane,
                backend="slovak-nlp-qwen3-14b",
                device=effective_device,
                current_bus_pressure=current_bus_pressure,
                deadline_budget_us=deadline_budget_us,
            )
            return False, result.receipt, None

        # 4. State Transition & Commit via AEP Engine
        action_payload = {
            "action": "INFERENCE_EXECUTE",
            "inference_telemetry": inference_resp.to_dict(),
        }

        result = self.engine.execute_transition(
            runtime_id=runtime_id,
            workload=WorkloadType.INFERENCE,
            action="INFERENCE_EXECUTE",
            current_state=self.current_state,
            transition_fn=transition_llm_inference,
            action_payload=action_payload,
            policy_id=f"policy-{self.mode.value.lower()}",
            policy_revision=1,
            execution_lane=execution_lane,
            backend="slovak-nlp-qwen3-14b",
            device=inference_resp.actual_device,
            current_bus_pressure=current_bus_pressure,
            deadline_budget_us=deadline_budget_us,
        )

        if result.committed:
            # Update authoritative state
            if isinstance(result.state, dict):
                self.current_state = LLMInferenceState(
                    session_id=self.current_state.session_id,
                    active_model=self.current_state.active_model,
                    completed_requests=result.state.get("completed_requests", 0),
                    total_tokens_generated=result.state.get("total_tokens_generated", 0),
                    total_prompt_tokens=result.state.get("total_prompt_tokens", 0),
                    accumulated_ttft_us=result.state.get("accumulated_ttft_us", 0),
                    accumulated_execution_us=result.state.get("accumulated_execution_us", 0),
                    backpressure_drops=result.state.get("backpressure_drops", 0),
                    last_action=result.state.get("last_action", "INFERENCE_EXECUTE"),
                    vital_max_hp=result.state.get("vital_max_hp", VITAL_MAX_HP),
                )
            return True, result.receipt, inference_resp
        else:
            return False, result.receipt, None
