"""
PolicyProposalGenerator: Generate policy proposals and rules from LLM analysis.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

from .schemas import (
    PolicyProposal, ProposalType, MicroInferenceRule,
    Condition, ConditionOperator, LogicalOp,
    Action, ActionType, RuleSource
)


class PolicyProposalGenerator:
    """
    Generates PolicyProposals and MicroInferenceRules.

    Can be driven by:
    - LLM suggestions (parsed from metacognitive responses)
    - Template-based rule generation
    - Automated optimization based on statistics
    """

    def __init__(self):
        self.proposal_counter = 0
        self.rule_counter = 0

    def _next_proposal_id(self) -> str:
        self.proposal_counter += 1
        date_str = datetime.now().strftime("%Y%m%d")
        return f"pp-{date_str}-{self.proposal_counter:03d}"

    def _next_rule_id(self, prefix: str = "mir") -> str:
        self.rule_counter += 1
        return f"{prefix}-{uuid.uuid4().hex[:8]}"

    def create_parameter_proposal(
        self,
        target: str,
        value: Any,
        justification: str,
        confidence: float,
        related_metrics: Optional[Dict[str, float]] = None
    ) -> PolicyProposal:
        """Create a parameter adjustment proposal."""
        return PolicyProposal(
            proposal_id=self._next_proposal_id(),
            proposal_type=ProposalType.PARAMETER_ADJUSTMENT,
            target=target,
            suggested_value=value,
            justification=justification,
            confidence=confidence,
            related_metrics=related_metrics or {},
        )

    def create_rule_proposal(
        self,
        rule: MicroInferenceRule,
        justification: str,
        confidence: float
    ) -> PolicyProposal:
        """Create a rule creation proposal."""
        return PolicyProposal(
            proposal_id=self._next_proposal_id(),
            proposal_type=ProposalType.RULE_CREATION,
            target=rule.rule_id,
            suggested_value=rule.model_dump(),
            justification=justification,
            confidence=confidence,
        )

    def create_deactivation_proposal(
        self,
        rule_id: str,
        justification: str,
        confidence: float,
        related_metrics: Optional[Dict[str, float]] = None
    ) -> PolicyProposal:
        """Create a rule deactivation proposal."""
        return PolicyProposal(
            proposal_id=self._next_proposal_id(),
            proposal_type=ProposalType.RULE_DEACTIVATION,
            target=rule_id,
            suggested_value=None,
            justification=justification,
            confidence=confidence,
            introspective_comment="This rule has consistently underperformed or caused issues.",
            related_metrics=related_metrics or {},
        )

    def generate_gaming_rule(
        self,
        cpu_threshold: float = 0.70,
        thermal_headroom: int = 15
    ) -> MicroInferenceRule:
        """Generate a gaming optimization rule."""
        return MicroInferenceRule(
            rule_id=self._next_rule_id("mir-gaming"),
            version="1.0.0",
            source=RuleSource.LLM_GENERATED,
            safety_tier=2,
            shadow_mode=False,
            conditions=[
                Condition(
                    metric="active_process_category",
                    operator=ConditionOperator.EQ,
                    value="gaming",
                    logical_op=LogicalOp.AND,
                ),
                Condition(
                    metric="cpu_utilization",
                    operator=ConditionOperator.GT,
                    value=cpu_threshold,
                    logical_op=LogicalOp.AND,
                ),
                Condition(
                    metric="thermal_headroom_c",
                    operator=ConditionOperator.GT,
                    value=thermal_headroom,
                    logical_op=None,
                ),
            ],
            actions=[
                Action(
                    action_type=ActionType.SET_CPU_AFFINITY,
                    params={"target": "foreground_process", "cores": "p_cores_only"},
                ),
                Action(
                    action_type=ActionType.SET_GOVERNOR,
                    params={"governor": "performance"},
                ),
            ],
            justification="Optimize for low-latency gaming when thermal headroom permits.",
        )

    def generate_powersave_rule(
        self,
        idle_threshold: float = 0.30
    ) -> MicroInferenceRule:
        """Generate a power-saving rule for idle/low-load scenarios."""
        return MicroInferenceRule(
            rule_id=self._next_rule_id("mir-powersave"),
            version="1.0.0",
            source=RuleSource.LLM_GENERATED,
            safety_tier=3,
            shadow_mode=False,
            conditions=[
                Condition(
                    metric="cpu_utilization",
                    operator=ConditionOperator.LT,
                    value=idle_threshold,
                    logical_op=LogicalOp.AND,
                ),
                Condition(
                    metric="gpu_utilization",
                    operator=ConditionOperator.LT,
                    value=idle_threshold,
                    logical_op=None,
                ),
            ],
            actions=[
                Action(
                    action_type=ActionType.SET_GOVERNOR,
                    params={"governor": "powersave"},
                ),
            ],
            justification="Reduce power consumption during idle periods.",
        )

    def generate_thermal_protection_rule(
        self,
        temp_threshold: int = 85
    ) -> MicroInferenceRule:
        """Generate a thermal protection rule."""
        return MicroInferenceRule(
            rule_id=self._next_rule_id("mir-thermal"),
            version="1.0.0",
            source=RuleSource.SYSTEM_DEFAULT,
            safety_tier=5,  # Highest safety tier
            shadow_mode=False,
            conditions=[
                Condition(
                    metric="temp_cpu",
                    operator=ConditionOperator.GT,
                    value=temp_threshold,
                    logical_op=None,
                ),
            ],
            actions=[
                Action(
                    action_type=ActionType.TRIGGER_COOLDOWN,
                    params={"duration_ms": 5000},
                ),
                Action(
                    action_type=ActionType.SET_GOVERNOR,
                    params={"governor": "powersave"},
                ),
                Action(
                    action_type=ActionType.LOG_EVENT,
                    params={"event": "thermal_protection_triggered", "severity": "warning"},
                ),
            ],
            justification="Critical thermal protection to prevent hardware damage.",
        )

    def generate_production_rule(
        self,
        stability_priority: bool = True
    ) -> MicroInferenceRule:
        """Generate a production workload optimization rule."""
        return MicroInferenceRule(
            rule_id=self._next_rule_id("mir-production"),
            version="1.0.0",
            source=RuleSource.LLM_GENERATED,
            safety_tier=3,
            shadow_mode=False,
            conditions=[
                Condition(
                    metric="active_process_category",
                    operator=ConditionOperator.EQ,
                    value="production",
                    logical_op=LogicalOp.AND,
                ),
                Condition(
                    metric="cpu_utilization",
                    operator=ConditionOperator.GT,
                    value=0.50,
                    logical_op=None,
                ),
            ],
            actions=[
                Action(
                    action_type=ActionType.SET_GOVERNOR,
                    params={"governor": "schedutil" if stability_priority else "performance"},
                ),
                Action(
                    action_type=ActionType.SET_MEMORY_TIER,
                    params={"tier": "balanced", "prefetch": True},
                ),
            ],
            justification="Balance throughput and stability for production workloads.",
        )

    def generate_shadow_rule(
        self,
        base_rule: MicroInferenceRule
    ) -> MicroInferenceRule:
        """Create a shadow version of a rule for safe testing."""
        shadow = base_rule.model_copy()
        shadow.rule_id = f"{base_rule.rule_id}-shadow"
        shadow.shadow_mode = True
        shadow.justification = f"Shadow evaluation of: {base_rule.justification}"
        return shadow
