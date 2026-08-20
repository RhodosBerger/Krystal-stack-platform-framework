# backend/core/benchmarks.py

from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class IndustryBenchmark:
    name: str
    metric: str
    value: float
    description: str
    source_id: str

class BenchmarkRegistry:
    """
    The 'Law Library' for the Auditor Agent.
    Stores the rigorous research data against which live performance is judged.
    """
    
    # Static registry derived from Deep Research
    BENCHMARKS = {
        "PATH_OPTIMIZATION": IndustryBenchmark(
            name="Expert CNC (Genetic)",
            metric="path_reduction_pct",
            value=18.63,
            description="Genetic Algorithm baseline for path shortening",
            source_id="[6]"
        ),
        "CYCLE_TIME_MTM": IndustryBenchmark(
            name="Non-linear GRG",
            metric="cycle_reduction_pct",
            value=60.0,
            description="Theoretical max for multi-spindle simultaneous machining",
            source_id="[5]"
        ),
        "SMOOTHING_GAIN": IndustryBenchmark(
            name="Splined Bezier",
            metric="machining_time_reduction_pct",
            value=9.0,
            description="Upper bound for pure curve smoothing gains",
            source_id="[7]"
        ),
        "INFERENCE_LATENCY": IndustryBenchmark(
            name="Neuro-C",
            metric="latency_reduction_pct",
            value=90.0,
            description="Target for Edge AI components",
            source_id="[8]"
        )
    }

    @staticmethod
    def evaluate_proposal(metric_type: str, proposed_value: float) -> dict:
        """
        Auditor uses this to Vote.
        If proposal < benchmark, Auditor is skeptical.
        If proposal > benchmark, Auditor is impressed but cautious.
        """
        benchmark = BenchmarkRegistry.BENCHMARKS.get(metric_type)
        
        if not benchmark:
            return {"status": "UNKNOWN_METRIC", "vote": 0.0}

        # Calculate deviation from industry standard
        deviation = proposed_value - benchmark.value
        
        if deviation > 0:
            return {
                "status": "EXCEEDS_INDUSTRY_STD", 
                "vote": 1.0, 
                "comment": f"Proposal beats {benchmark.name} by {deviation:.2f}%"
            }
        elif deviation > -5.0:
            return {
                "status": "COMPETITIVE", 
                "vote": 0.5, 
                "comment": f"Proposal near {benchmark.name} levels."
            }
        else:
            return {
                "status": "SUBOPTIMAL", 
                "vote": -1.0, 
                "comment": f"Proposal significantly below {benchmark.name} ({benchmark.value}%)."
            }
