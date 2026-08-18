"""
GAMESA Metacognitive Engine

Main orchestrator for metacognitive reasoning and policy generation.
Integrates LLM-based analysis with GAMESA's rule engine.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import time

from .bot_core import ConversationManager
from .llm_integrations.base_connector import BaseLLMConnector, LLMConfig
from .tools.tool_registry import get_tool_registry
from .tools.calculator import Calculator
from .tools.telemetry_analyzer import TelemetryAnalyzer


@dataclass
class PolicyProposal:
    """Policy proposal from metacognitive analysis."""
    proposal_id: str
    proposal_type: str  # "rule", "parameter", "strategy"
    target: str  # What to optimize
    suggested_value: Any
    justification: str
    confidence: float  # 0.0-1.0
    introspective_comment: str
    related_metrics: List[str]
    safety_tier: str = "EXPERIMENTAL"  # STRICT, EXPERIMENTAL, DEBUG
    shadow_mode: bool = True  # Default to shadow mode for safety


@dataclass
class MetacognitiveAnalysis:
    """Results of metacognitive analysis."""
    timestamp: float
    trigger: str  # "periodic", "performance_event", "manual"
    summary: str
    proposals: List[PolicyProposal]
    insights: List[str]
    concerns: List[str]


class MetacognitiveEngine:
    """
    Metacognitive reasoning engine for GAMESA.

    Analyzes system performance, proposes optimizations,
    and learns from outcomes.
    """

    def __init__(
        self,
        llm_connector: BaseLLMConnector,
        telemetry_buffer: Optional[List[Dict]] = None
    ):
        """
        Initialize metacognitive engine.

        Args:
            llm_connector: LLM backend connector
            telemetry_buffer: Shared telemetry buffer
        """
        self.llm = llm_connector
        self.telemetry_buffer = telemetry_buffer or []

        # Initialize tool registry
        self.tools = get_tool_registry()

        # Register tools
        self.tools.register(Calculator())
        self.telemetry_tool = TelemetryAnalyzer(self.telemetry_buffer)
        self.tools.register(self.telemetry_tool)

        # Initialize conversation manager
        self.conversation = ConversationManager(
            llm_connector=self.llm,
            tool_registry=self.tools,
            system_prompt=self._system_prompt()
        )

        # Analysis history
        self.analysis_history: List[MetacognitiveAnalysis] = []

    def _system_prompt(self) -> str:
        """Enhanced system prompt for GAMESA metacognitive analysis."""
        return """You are the Metacognitive Interface for GAMESA, an adaptive system optimization framework.

**Your Role:**
Analyze system performance telemetry and propose evidence-based optimization policies.

**Available Tools:**
- calculator: Precise mathematical calculations
- telemetry_analyzer: Analyze telemetry data for patterns, correlations, anomalies

**Analysis Process:**
1. Use telemetry_analyzer to examine recent performance data
2. Identify patterns, bottlenecks, and optimization opportunities
3. Propose policies as JSON with this schema:

{
  "proposal_id": "unique_id",
  "proposal_type": "rule|parameter|strategy",
  "target": "what_to_optimize",
  "suggested_value": <value>,
  "justification": "data-driven explanation",
  "confidence": 0.0-1.0,
  "introspective_comment": "self-reflection on proposal quality",
  "related_metrics": ["metric1", "metric2"],
  "safety_tier": "STRICT|EXPERIMENTAL|DEBUG",
  "shadow_mode": true|false
}

**Guidelines:**
- Always ground proposals in telemetry data
- Use calculator for precise metrics
- Start with shadow_mode=true for risky changes
- Higher confidence for well-established patterns
- Consider thermal, power, and performance tradeoffs
- Be concise and analytical

**Safety:**
- Never propose changes that could exceed thermal limits
- Always consider worst-case scenarios
- Flag high-risk proposals with low confidence + shadow mode
"""

    def analyze(
        self,
        trigger: str = "periodic",
        focus: Optional[str] = None,
        window_size: int = 60
    ) -> MetacognitiveAnalysis:
        """
        Perform metacognitive analysis.

        Args:
            trigger: What triggered this analysis
            focus: Specific area to focus on (optional)
            window_size: Telemetry window size

        Returns:
            MetacognitiveAnalysis with proposals
        """
        # Update telemetry tool with latest buffer
        self.telemetry_tool.update_buffer(self.telemetry_buffer)

        # Construct analysis prompt
        prompt = self._build_analysis_prompt(trigger, focus, window_size)

        # Get LLM analysis (with tool use)
        response = self.conversation.chat(prompt, enable_tools=True)

        # Parse response for proposals
        proposals = self._extract_proposals(response)

        # Generate insights and concerns
        insights = self._extract_insights(response)
        concerns = self._extract_concerns(response)

        analysis = MetacognitiveAnalysis(
            timestamp=time.time(),
            trigger=trigger,
            summary=response[:500],  # First 500 chars as summary
            proposals=proposals,
            insights=insights,
            concerns=concerns
        )

        self.analysis_history.append(analysis)

        return analysis

    def _build_analysis_prompt(
        self,
        trigger: str,
        focus: Optional[str],
        window_size: int
    ) -> str:
        """Build analysis prompt for LLM."""
        prompt = f"""Analyze recent GAMESA performance data (trigger: {trigger}).

**Task:**
1. Use telemetry_analyzer to examine the last {window_size} samples
2. Identify optimization opportunities
3. Propose 1-3 concrete policy changes

"""

        if focus:
            prompt += f"**Focus Area:** {focus}\n\n"

        prompt += """**Analysis Steps:**
1. Get telemetry summary
2. Check for anomalies
3. Analyze correlations (e.g., FPS vs CPU/GPU)
4. Propose policies based on findings

Provide your analysis and policy proposals in a structured format.
"""

        return prompt

    def _extract_proposals(self, response: str) -> List[PolicyProposal]:
        """
        Extract PolicyProposal objects from LLM response.

        Looks for JSON blocks in the response.
        """
        proposals = []

        # Try to find JSON blocks in response
        import re
        json_pattern = r'\{[^{}]*"proposal_id"[^{}]*\}'

        matches = re.findall(json_pattern, response, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match)
                proposal = PolicyProposal(
                    proposal_id=data.get("proposal_id", f"prop_{time.time()}"),
                    proposal_type=data.get("proposal_type", "rule"),
                    target=data.get("target", "unknown"),
                    suggested_value=data.get("suggested_value"),
                    justification=data.get("justification", ""),
                    confidence=float(data.get("confidence", 0.5)),
                    introspective_comment=data.get("introspective_comment", ""),
                    related_metrics=data.get("related_metrics", []),
                    safety_tier=data.get("safety_tier", "EXPERIMENTAL"),
                    shadow_mode=data.get("shadow_mode", True)
                )
                proposals.append(proposal)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Warning: Could not parse proposal: {e}")

        return proposals

    def _extract_insights(self, response: str) -> List[str]:
        """Extract key insights from response."""
        # Simple extraction: look for bullet points or numbered lists
        insights = []

        lines = response.split('\n')
        for line in lines:
            if line.strip().startswith(('-', '•', '*')) or \
               (len(line) > 2 and line[0].isdigit() and line[1] in '.):'):
                insight = line.strip().lstrip('-•*0123456789.): ')
                if len(insight) > 10:  # Filter out very short lines
                    insights.append(insight)

        return insights[:5]  # Top 5 insights

    def _extract_concerns(self, response: str) -> List[str]:
        """Extract safety concerns from response."""
        concerns = []

        # Look for keywords indicating concerns
        concern_keywords = ['risk', 'concern', 'warning', 'caution', 'unsafe']

        lines = response.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in concern_keywords):
                concerns.append(line.strip())

        return concerns[:3]  # Top 3 concerns

    def evaluate_proposal(
        self,
        proposal: PolicyProposal,
        telemetry_snapshot: Dict
    ) -> Dict[str, Any]:
        """
        Evaluate a proposal against current state.

        Args:
            proposal: Policy proposal to evaluate
            telemetry_snapshot: Current telemetry

        Returns:
            Evaluation results
        """
        evaluation = {
            "proposal_id": proposal.proposal_id,
            "safe_to_execute": False,
            "estimated_impact": {},
            "concerns": []
        }

        # Basic safety checks
        if proposal.safety_tier == "STRICT":
            # Stricter validation
            if proposal.confidence < 0.8:
                evaluation["concerns"].append("Low confidence for STRICT tier")
            else:
                evaluation["safe_to_execute"] = True
        elif proposal.safety_tier == "EXPERIMENTAL":
            # Should start in shadow mode
            if not proposal.shadow_mode:
                evaluation["concerns"].append("EXPERIMENTAL should use shadow mode")
            else:
                evaluation["safe_to_execute"] = True
        else:  # DEBUG
            evaluation["safe_to_execute"] = True

        # Check thermal safety
        current_temp = telemetry_snapshot.get("temperature", 70)
        if current_temp > 75 and "boost" in str(proposal.suggested_value).lower():
            evaluation["concerns"].append("Temperature too high for boost action")
            evaluation["safe_to_execute"] = False

        return evaluation

    def get_analysis_history(self, limit: int = 10) -> List[MetacognitiveAnalysis]:
        """Get recent analysis history."""
        return self.analysis_history[-limit:]

    def export_proposals(self, analysis: MetacognitiveAnalysis) -> str:
        """Export proposals as JSON for GAMESA rule engine."""
        proposals_dict = [
            {
                "proposal_id": p.proposal_id,
                "proposal_type": p.proposal_type,
                "target": p.target,
                "suggested_value": p.suggested_value,
                "justification": p.justification,
                "confidence": p.confidence,
                "introspective_comment": p.introspective_comment,
                "related_metrics": p.related_metrics,
                "safety_tier": p.safety_tier,
                "shadow_mode": p.shadow_mode
            }
            for p in analysis.proposals
        ]

        return json.dumps(proposals_dict, indent=2)


# Factory function
def create_metacognitive_engine(
    llm_config: LLMConfig,
    telemetry_buffer: Optional[List[Dict]] = None
) -> MetacognitiveEngine:
    """
    Factory function to create metacognitive engine.

    Args:
        llm_config: LLM configuration
        telemetry_buffer: Shared telemetry buffer

    Returns:
        Initialized MetacognitiveEngine
    """
    from .llm_integrations.base_connector import LLMConnectorFactory

    connector = LLMConnectorFactory.create(llm_config)

    return MetacognitiveEngine(
        llm_connector=connector,
        telemetry_buffer=telemetry_buffer
    )
