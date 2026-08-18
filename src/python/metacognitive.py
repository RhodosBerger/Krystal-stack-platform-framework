"""
MetacognitiveInterface: Self-reflecting analysis of system performance.

This is the "thinking about thinking" component - it analyzes logs,
experiences, and performance data to propose improvements.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from .schemas import (
    PolicyProposal, ProposalType, MetacognitiveQuery,
    MetacognitiveResponse, MicroInferenceRule, Experience
)
from .experience_store import ExperienceStore


@dataclass
class PerformanceSummary:
    """Aggregated performance data for LLM analysis."""
    time_window: str
    avg_frametime_ms: float
    frametime_variance: float
    avg_cpu_util: float
    avg_gpu_util: float
    avg_cpu_temp: float
    avg_gpu_temp: float
    action_count: int
    positive_reward_ratio: float
    emergency_cooldowns: int
    top_triggered_rules: List[str]


class MetacognitiveInterface:
    """
    Enables the Cognitive Stream (LLM) to analyze its own performance,
    understand policy impact, and propose improvements.

    Key responsibilities:
    - Aggregate data from events.log and ExperienceStore
    - Generate structured summaries for LLM analysis
    - Format introspective prompts
    - Parse and validate LLM responses
    """

    def __init__(
        self,
        experience_store: ExperienceStore,
        events_log_path: Optional[Path] = None
    ):
        self.experience_store = experience_store
        self.events_log_path = events_log_path or Path("events.log")
        self.analysis_history: List[MetacognitiveResponse] = []

    def trigger_analysis(
        self,
        query: MetacognitiveQuery
    ) -> PerformanceSummary:
        """
        Aggregate data and generate a performance summary for LLM analysis.
        """
        # Get experiences in time window
        experiences = self.experience_store.query_by_time_window(
            query.time_window_start,
            query.time_window_end
        )

        # Parse events log
        events = self._parse_events_log(
            query.time_window_start,
            query.time_window_end
        )

        # Compute aggregates
        return self._compute_summary(experiences, events, query)

    def _parse_events_log(
        self,
        start: datetime,
        end: datetime
    ) -> List[Dict[str, Any]]:
        """Parse JSONL events log for time window."""
        events = []

        if not self.events_log_path.exists():
            return events

        with open(self.events_log_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    ts = datetime.fromisoformat(event.get("ts", "").replace("Z", "+00:00"))
                    if start <= ts <= end:
                        events.append(event)
                except (json.JSONDecodeError, ValueError):
                    continue

        return events

    def _compute_summary(
        self,
        experiences: List[Experience],
        events: List[Dict[str, Any]],
        query: MetacognitiveQuery
    ) -> PerformanceSummary:
        """Compute aggregated performance summary."""
        # Extract telemetry events
        telemetry = [e for e in events if e.get("event") == "telemetry"]

        # Compute averages
        if telemetry:
            avg_frametime = sum(e.get("frametime_ms", 0) for e in telemetry) / len(telemetry)
            frametimes = [e.get("frametime_ms", 0) for e in telemetry]
            variance = sum((ft - avg_frametime) ** 2 for ft in frametimes) / len(frametimes)
            avg_cpu_util = sum(e.get("cpu_util", 0) for e in telemetry) / len(telemetry)
            avg_gpu_util = sum(e.get("gpu_util", 0) for e in telemetry) / len(telemetry)
            avg_cpu_temp = sum(e.get("temp_cpu", 0) for e in telemetry) / len(telemetry)
            avg_gpu_temp = sum(e.get("temp_gpu", 0) for e in telemetry) / len(telemetry)
        else:
            avg_frametime = variance = avg_cpu_util = avg_gpu_util = 0.0
            avg_cpu_temp = avg_gpu_temp = 0.0

        # Count emergency cooldowns
        cooldowns = len([e for e in events if e.get("event") == "emergency_cooldown"])

        # Compute reward ratio
        if experiences:
            positive_ratio = len([e for e in experiences if e.reward > 0]) / len(experiences)
        else:
            positive_ratio = 0.0

        # Find top triggered rules
        rule_counts: Dict[str, int] = {}
        for e in events:
            if e.get("event") == "action_taken":
                rule_id = e.get("rule_id", "unknown")
                rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1

        top_rules = sorted(rule_counts.keys(), key=lambda r: rule_counts[r], reverse=True)[:5]

        return PerformanceSummary(
            time_window=f"{query.time_window_start.isoformat()} to {query.time_window_end.isoformat()}",
            avg_frametime_ms=avg_frametime,
            frametime_variance=variance,
            avg_cpu_util=avg_cpu_util,
            avg_gpu_util=avg_gpu_util,
            avg_cpu_temp=avg_cpu_temp,
            avg_gpu_temp=avg_gpu_temp,
            action_count=len([e for e in events if e.get("event") == "action_taken"]),
            positive_reward_ratio=positive_ratio,
            emergency_cooldowns=cooldowns,
            top_triggered_rules=top_rules,
        )

    def generate_introspective_prompt(
        self,
        summary: PerformanceSummary,
        focus_questions: Optional[List[str]] = None
    ) -> str:
        """
        Generate a structured prompt for LLM metacognitive analysis.
        """
        questions = focus_questions or [
            "Which policies correlate with frametime improvements or degradations?",
            "Are there patterns in emergency cooldowns that suggest policy adjustments?",
            "What rule modifications would improve the positive reward ratio?",
            "Are any rules consistently underperforming and should be deactivated?",
        ]

        prompt = f"""## Metacognitive Analysis Request

### Performance Summary
- **Time Window:** {summary.time_window}
- **Avg Frametime:** {summary.avg_frametime_ms:.2f}ms (variance: {summary.frametime_variance:.2f})
- **CPU:** {summary.avg_cpu_util*100:.1f}% util, {summary.avg_cpu_temp:.1f}°C avg temp
- **GPU:** {summary.avg_gpu_util*100:.1f}% util, {summary.avg_gpu_temp:.1f}°C avg temp
- **Actions Taken:** {summary.action_count}
- **Positive Reward Ratio:** {summary.positive_reward_ratio*100:.1f}%
- **Emergency Cooldowns:** {summary.emergency_cooldowns}
- **Top Rules:** {', '.join(summary.top_triggered_rules) or 'None'}

### Introspective Questions
{chr(10).join(f'{i+1}. {q}' for i, q in enumerate(questions))}

### Required Response Format
Respond with a JSON object matching the PolicyProposal schema:
```json
{{
  "proposal_id": "pp-YYYYMMDD-NNN",
  "proposal_type": "parameter_adjustment|rule_creation|rule_modification|rule_deactivation|scoring_weight_update",
  "target": "<target_parameter_or_rule>",
  "suggested_value": <value>,
  "justification": "<reasoning>",
  "confidence": 0.0-1.0,
  "introspective_comment": "<self-reflection on this proposal>",
  "related_metrics": {{"metric_name": value}}
}}
```
"""
        return prompt

    def parse_llm_response(self, response: str) -> Optional[PolicyProposal]:
        """
        Parse and validate LLM response into a PolicyProposal.
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = response
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()

            data = json.loads(json_str)
            return PolicyProposal(**data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse LLM response: {e}")
            return None

    def record_analysis(self, response: MetacognitiveResponse) -> None:
        """Record analysis result for tracking confidence vs outcome."""
        self.analysis_history.append(response)

    def get_confidence_calibration(self) -> Dict[str, float]:
        """
        Analyze historical confidence vs actual outcomes.
        Used for self-improvement of confidence estimation.
        """
        if not self.analysis_history:
            return {"calibration_score": 0.0, "sample_size": 0}

        # This would require outcome tracking in production
        return {
            "calibration_score": 0.0,  # Placeholder
            "sample_size": len(self.analysis_history),
            "avg_confidence": sum(r.confidence for r in self.analysis_history) / len(self.analysis_history),
        }

    def identify_risky_patterns(self) -> List[str]:
        """
        Identify patterns that consistently lead to negative outcomes.
        Used for learning from mistakes.
        """
        negative_experiences = self.experience_store.query_negative_rewards()

        patterns = []

        # Check for repeated failures of same action type
        action_failures: Dict[str, int] = {}
        for exp in negative_experiences:
            action_type = exp.action.action_type.value
            action_failures[action_type] = action_failures.get(action_type, 0) + 1

        for action_type, count in action_failures.items():
            if count >= 3:
                patterns.append(f"Action '{action_type}' has failed {count} times")

        # Check for thermal-related failures
        thermal_failures = [
            exp for exp in negative_experiences
            if exp.state.temp_cpu > 85 or exp.state.temp_gpu > 80
        ]
        if len(thermal_failures) > 2:
            patterns.append(f"High thermal state correlated with {len(thermal_failures)} failures")

        return patterns
