"""
GAMESA Metacognitive - Mock LLM Connector

Mock connector for testing without actual LLM API.
Provides predefined responses for common queries.
"""

from typing import List, Dict, Optional
import time
import random

from .base_connector import (
    BaseLLMConnector,
    LLMMessage,
    LLMResponse,
    LLMConfig,
    LLMProvider,
    LLMConnectorFactory
)


class MockLLMConnector(BaseLLMConnector):
    """
    Mock LLM connector for testing.

    Provides canned responses based on keywords in prompts.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.call_count = 0

    def generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate mock response."""
        self.call_count += 1

        # Get last user message
        user_messages = [m for m in messages if m.role == "user"]
        last_message = user_messages[-1].content if user_messages else ""

        # Determine response based on keywords
        if "telemetry" in last_message.lower() or "analyze" in last_message.lower():
            content = self._mock_telemetry_analysis(tools)
        elif "policy" in last_message.lower() or "propose" in last_message.lower():
            content = self._mock_policy_proposal()
        elif "temperature" in last_message.lower():
            content = self._mock_temperature_analysis()
        else:
            content = self._mock_generic_response()

        # Simulate processing time
        time.sleep(0.1)

        return LLMResponse(
            content=content,
            tool_calls=None,
            finish_reason="stop",
            usage={
                "prompt_tokens": self.estimate_tokens(str(messages)),
                "completion_tokens": self.estimate_tokens(content),
                "total_tokens": self.estimate_tokens(str(messages) + content)
            },
            model=self.config.model
        )

    def stream_generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ):
        """Stream mock response."""
        # Generate full response
        response = self.generate(messages, tools, **kwargs)

        # Stream it word by word
        words = response.content.split()
        for word in words:
            time.sleep(0.01)  # Simulate streaming delay
            yield word + " "

    def validate_connection(self) -> bool:
        """Mock connection is always valid."""
        return True

    def _mock_telemetry_analysis(self, tools: Optional[List[Dict]]) -> str:
        """Generate mock telemetry analysis."""
        # Check if tools are available
        has_telemetry_tool = False
        if tools:
            has_telemetry_tool = any(
                t.get("function", {}).get("name") == "telemetry_analyzer"
                for t in tools
            )

        if has_telemetry_tool:
            # Simulate tool use
            return """Based on telemetry analysis (last 60 samples):

**Key Findings:**
1. Average temperature: 72.3°C with thermal headroom of 12.7°C
2. CPU utilization shows high variance (45%-85%), suggesting burst workloads
3. FPS correlation with GPU is 0.76, indicating GPU-bound scenarios
4. Power draw trending upward (+0.3W per minute)

**Optimization Opportunities:**
- During CPU idle periods (util < 50%), could reduce power state
- High GPU correlation suggests GPU optimization priority
- Temperature stable, thermal headroom allows moderate boosting

**Proposed Policy:**
```json
{
  "proposal_id": "adaptive_gpu_boost_001",
  "proposal_type": "rule",
  "target": "gpu_power_limit",
  "suggested_value": "+5W during high FPS demand",
  "justification": "GPU bottleneck detected with safe thermal margins",
  "confidence": 0.78,
  "introspective_comment": "High correlation data supports this, but limited history reduces confidence",
  "related_metrics": ["fps", "gpu_util", "thermal_headroom"],
  "safety_tier": "EXPERIMENTAL",
  "shadow_mode": true
}
```
"""
        else:
            return """Telemetry analysis requested, but telemetry_analyzer tool not available.
Please provide access to telemetry data for detailed analysis."""

    def _mock_policy_proposal(self) -> str:
        """Generate mock policy proposal."""
        proposals = [
            {
                "proposal_id": "thermal_aware_boost_001",
                "proposal_type": "rule",
                "target": "cpu_boost",
                "suggested_value": "enable when thermal_headroom > 15°C",
                "justification": "Safe thermal margin allows performance boost",
                "confidence": 0.85,
                "introspective_comment": "Well-established thermal safety pattern",
                "related_metrics": ["temperature", "thermal_headroom", "cpu_freq"],
                "safety_tier": "STRICT",
                "shadow_mode": false
            },
            {
                "proposal_id": "memory_tier_optimization_002",
                "proposal_type": "parameter",
                "target": "hot_memory_threshold",
                "suggested_value": 100,
                "justification": "Frequently accessed blocks should promote faster",
                "confidence": 0.62,
                "introspective_comment": "Limited evidence, needs shadow evaluation",
                "related_metrics": ["memory_access_count", "cache_hits"],
                "safety_tier": "EXPERIMENTAL",
                "shadow_mode": true
            }
        ]

        import json
        return f"""**Policy Proposals:**

{json.dumps(proposals, indent=2)}

**Rationale:**
1. First proposal has high confidence due to established thermal safety patterns
2. Second proposal is more speculative, recommended for shadow evaluation
"""

    def _mock_temperature_analysis(self) -> str:
        """Generate mock temperature analysis."""
        current_temp = 70 + random.uniform(-5, 15)
        trend = random.choice(["rising", "falling", "stable"])

        return f"""**Temperature Analysis:**

Current: {current_temp:.1f}°C
Trend: {trend}
Thermal headroom: {85 - current_temp:.1f}°C

{"⚠️ Temperature approaching limits - recommend conservative mode" if current_temp > 80 else "✓ Temperature within safe operating range"}
"""

    def _mock_generic_response(self) -> str:
        """Generate generic mock response."""
        return """I'm the GAMESA Metacognitive Interface (mock mode).

I can help analyze telemetry, propose optimizations, and explain system behavior.

Available capabilities:
- Telemetry analysis and pattern detection
- Policy proposal generation
- Performance correlation analysis
- Safety evaluation

How can I assist with system optimization?"""


# Register mock connector
LLMConnectorFactory.register(LLMProvider.LOCAL, MockLLMConnector)
