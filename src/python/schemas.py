"""
Pydantic schemas matching Rust types for cross-language compatibility.
"""

from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


class ProposalType(str, Enum):
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    RULE_CREATION = "rule_creation"
    RULE_MODIFICATION = "rule_modification"
    RULE_DEACTIVATION = "rule_deactivation"
    SCORING_WEIGHT_UPDATE = "scoring_weight_update"


class PolicyProposal(BaseModel):
    """LLM-generated policy proposal for Rust to parse and validate."""
    proposal_id: str
    proposal_type: ProposalType
    target: str
    suggested_value: Any
    justification: str
    confidence: float = Field(ge=0.0, le=1.0)
    introspective_comment: Optional[str] = None
    related_metrics: Dict[str, float] = Field(default_factory=dict)


class ConditionOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GE = "ge"
    LT = "lt"
    LE = "le"
    CONTAINS = "contains"


class LogicalOp(str, Enum):
    AND = "and"
    OR = "or"


class Condition(BaseModel):
    metric: str
    operator: ConditionOperator
    value: Any
    logical_op: Optional[LogicalOp] = None


class ActionType(str, Enum):
    SET_CPU_AFFINITY = "set_cpu_affinity"
    SET_GOVERNOR = "set_governor"
    SET_GPU_POWER_LIMIT = "set_gpu_power_limit"
    SET_MEMORY_TIER = "set_memory_tier"
    TRIGGER_COOLDOWN = "trigger_cooldown"
    LOG_EVENT = "log_event"


class Action(BaseModel):
    action_type: ActionType
    params: Dict[str, Any] = Field(default_factory=dict)


class RuleSource(str, Enum):
    LLM_GENERATED = "llm_generated"
    USER_DEFINED = "user_defined"
    SYSTEM_DEFAULT = "system_default"


class MicroInferenceRule(BaseModel):
    """Declarative rule that LLM can generate for Rust to execute."""
    rule_id: str
    version: str
    source: RuleSource
    safety_tier: int = Field(ge=1, le=5)
    shadow_mode: bool = False
    conditions: List[Condition]
    actions: List[Action]
    justification: str


class TelemetrySnapshot(BaseModel):
    timestamp: datetime
    cpu_util: float
    gpu_util: float
    frametime_ms: float
    temp_cpu: int
    temp_gpu: int
    active_process_category: str = "unknown"


class Experience(BaseModel):
    """State-Action-Reward tuple for training."""
    id: str
    timestamp: datetime
    state: TelemetrySnapshot
    action: Action
    reward: float
    next_state: Optional[TelemetrySnapshot] = None


class MetacognitiveQuery(BaseModel):
    """Query structure for metacognitive analysis."""
    query_id: str
    query_type: str  # "performance_analysis", "policy_correlation", "anomaly_detection"
    time_window_start: datetime
    time_window_end: datetime
    focus_metrics: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)


class MetacognitiveResponse(BaseModel):
    """Structured response from metacognitive analysis."""
    query_id: str
    summary: str
    proposals: List[PolicyProposal] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_actions: List[str] = Field(default_factory=list)
