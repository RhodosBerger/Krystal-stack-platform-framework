#!/usr/bin/env python3
"""
GAMESA Metacognitive Module - Demo Script

Demonstrates the metacognitive engine with mock LLM.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from metacognitive import (
    create_metacognitive_engine,
    LLMConfig,
    LLMProvider,
    get_tool_registry
)
from metacognitive.tools.calculator import Calculator
from metacognitive.tools.telemetry_analyzer import TelemetryAnalyzer
from metacognitive.llm_integrations.mock_connector import MockLLMConnector  # Register mock connector


def generate_mock_telemetry(count: int = 60):
    """Generate mock telemetry data for demonstration."""
    import random

    telemetry = []
    base_temp = 70

    for i in range(count):
        # Simulate realistic telemetry
        temp = base_temp + random.uniform(-5, 10) + (i / count) * 5  # Rising trend
        cpu_util = random.uniform(0.4, 0.9)
        gpu_util = random.uniform(0.5, 0.95)
        fps = 60 - (cpu_util * 10) if cpu_util > 0.8 else 60

        telemetry.append({
            "timestamp": time.time() - (count - i),
            "temperature": temp,
            "thermal_headroom": 85 - temp,
            "cpu_util": cpu_util,
            "gpu_util": gpu_util,
            "memory_util": random.uniform(0.5, 0.7),
            "power_draw": 15 + (cpu_util * 10) + (gpu_util * 8),
            "fps": fps,
            "latency": 10 + random.uniform(-2, 5)
        })

    return telemetry


def demo_tools():
    """Demonstrate tool usage."""
    print("=" * 70)
    print("DEMO 1: Tool Usage")
    print("=" * 70)
    print()

    registry = get_tool_registry()

    # Register tools
    registry.register(Calculator())

    # Calculator tool
    print("[Calculator Tool]")
    expressions = [
        "sqrt(144)",
        "2 * pi",
        "log(exp(5))",
        "sin(pi/2)"
    ]

    for expr in expressions:
        result = registry.execute("calculator", expression=expr)
        if result["success"]:
            print(f"  {expr} = {result['result']:.4f}")
        else:
            print(f"  {expr} = ERROR: {result['error']}")

    print()

    # Telemetry analyzer tool
    print("[Telemetry Analyzer Tool]")

    # Generate mock data
    telemetry = generate_mock_telemetry(60)

    # Update analyzer with data
    analyzer = TelemetryAnalyzer(telemetry)
    registry.tools["telemetry_analyzer"] = analyzer

    # Run queries
    queries = [
        ("temperature_stats", {}),
        ("power_trend", {}),
        ("fps_correlation", {}),
    ]

    for query_type, params in queries:
        result = registry.execute(
            "telemetry_analyzer",
            query_type=query_type,
            **params
        )
        print(f"\n  Query: {query_type}")
        if result["success"]:
            import json
            print(f"  Result: {json.dumps(result['result'], indent=4)}")
        else:
            print(f"  ERROR: {result['error']}")

    print()


def demo_metacognitive_analysis():
    """Demonstrate metacognitive analysis."""
    print("=" * 70)
    print("DEMO 2: Metacognitive Analysis")
    print("=" * 70)
    print()

    # Generate mock telemetry
    print("[Generating mock telemetry...]")
    telemetry = generate_mock_telemetry(60)
    print(f"  Generated {len(telemetry)} samples")
    print(f"  Temperature range: {min(t['temperature'] for t in telemetry):.1f}°C - "
          f"{max(t['temperature'] for t in telemetry):.1f}°C")
    print(f"  Avg FPS: {sum(t['fps'] for t in telemetry) / len(telemetry):.1f}")
    print()

    # Create engine with mock LLM
    print("[Creating metacognitive engine with mock LLM...]")
    config = LLMConfig(
        provider=LLMProvider.LOCAL,
        model="mock",
        temperature=0.7
    )

    engine = create_metacognitive_engine(config, telemetry)
    print("  Engine created successfully")
    print()

    # Run analysis
    print("[Running metacognitive analysis...]")
    analysis = engine.analyze(
        trigger="demo",
        focus="performance optimization",
        window_size=60
    )

    print(f"  Analysis complete (timestamp: {analysis.timestamp:.0f})")
    print()

    # Display summary
    print("[Analysis Summary]")
    print(f"  Trigger: {analysis.trigger}")
    print(f"  Summary: {analysis.summary[:200]}...")
    print()

    # Display insights
    if analysis.insights:
        print("[Key Insights]")
        for i, insight in enumerate(analysis.insights[:5], 1):
            print(f"  {i}. {insight}")
        print()

    # Display concerns
    if analysis.concerns:
        print("[Safety Concerns]")
        for i, concern in enumerate(analysis.concerns[:3], 1):
            print(f"  {i}. {concern}")
        print()

    # Display proposals
    print(f"[Policy Proposals: {len(analysis.proposals)}]")
    for i, proposal in enumerate(analysis.proposals, 1):
        print(f"\n  Proposal {i}: {proposal.proposal_id}")
        print(f"    Type: {proposal.proposal_type}")
        print(f"    Target: {proposal.target}")
        print(f"    Value: {proposal.suggested_value}")
        print(f"    Confidence: {proposal.confidence:.2f}")
        print(f"    Safety Tier: {proposal.safety_tier}")
        print(f"    Shadow Mode: {proposal.shadow_mode}")
        print(f"    Justification: {proposal.justification[:100]}...")

        # Evaluate proposal
        current_telemetry = telemetry[-1]
        evaluation = engine.evaluate_proposal(proposal, current_telemetry)

        print(f"    Evaluation:")
        print(f"      Safe to execute: {evaluation['safe_to_execute']}")
        if evaluation['concerns']:
            print(f"      Concerns: {', '.join(evaluation['concerns'])}")

    print()

    # Export proposals
    print("[Exporting proposals as JSON...]")
    json_export = engine.export_proposals(analysis)
    print(f"  Exported {len(analysis.proposals)} proposals")
    print()

    return analysis


def demo_conversation():
    """Demonstrate conversational interface."""
    print("=" * 70)
    print("DEMO 3: Conversational Interface")
    print("=" * 70)
    print()

    from metacognitive import ConversationManager
    from metacognitive.llm_integrations.mock_connector import MockLLMConnector

    # Setup
    config = LLMConfig(provider=LLMProvider.LOCAL, model="mock")
    llm = MockLLMConnector(config)
    conv = ConversationManager(llm)

    print("[Starting conversation with metacognitive interface...]")
    print()

    # Multi-turn conversation
    questions = [
        "What can you do?",
        "Analyze the current temperature",
        "Should we enable boost mode?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"User ({i}): {question}")
        response = conv.chat(question, enable_tools=False)
        print(f"Assistant ({i}): {response[:300]}...")
        print()

    print(f"[Conversation history: {len(conv.get_history())} turns]")
    print()


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "GAMESA METACOGNITIVE MODULE DEMO" + " " * 20 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    try:
        # Demo 1: Tools
        demo_tools()

        # Demo 2: Metacognitive Analysis
        analysis = demo_metacognitive_analysis()

        # Demo 3: Conversation
        demo_conversation()

        print("=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Review analysis proposals above")
        print("  2. Integrate with GAMESA rule engine")
        print("  3. Add real LLM connector (OpenAI, Anthropic)")
        print("  4. Deploy metacognitive analysis in production")
        print()

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
