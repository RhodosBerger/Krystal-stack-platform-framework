#!/usr/bin/env python3
"""
Hippocampus Aggregator.
The "Framework of Aggregations" for synthesizing experience.
"""

from typing import List, Dict, Any, Callable
from collections import defaultdict
import statistics
from hippocampus import Hippocampus, Episode

class AggregationFramework:
    def __init__(self, hippocampus_ref: Hippocampus):
        self.memory = hippocampus_ref

    def aggregate_by(self, key_selector: Callable[[Episode], str]) -> Dict[str, List[Episode]]:
        """
        Generic aggregation: Groups episodes by any lambda function.
        e.g., key_selector = lambda e: e.material
        """
        groups = defaultdict(list)
        for ep in self.memory.episodes:
            key = key_selector(ep)
            groups[key].append(ep)
        return groups

    def synthesize_stats(self, episodes: List[Episode]) -> Dict[str, float]:
        """
        Calculates aggregate statistics for a group of episodes.
        """
        if not episodes:
            return {}
        
        cortisols = [e.outcome_cortisol for e in episodes]
        dopamines = [e.outcome_dopamine for e in episodes]
        
        return {
            "count": len(episodes),
            "avg_cortisol": statistics.mean(cortisols),
            "max_cortisol": max(cortisols),
            "avg_dopamine": statistics.mean(dopamines),
            "consistency_score": 100.0 / (statistics.stdev(cortisols) + 1.0) if len(cortisols) > 1 else 100.0
        }

    def generate_report(self, dimension: str):
        """
        High-level report generator.
        dimension: 'material', 'strategy', 'combined'
        """
        print(f"\n--- Aggregation Report: {dimension.upper()} ---")
        
        selector = None
        if dimension == 'material':
            selector = lambda e: e.material
        elif dimension == 'strategy':
            selector = lambda e: e.action_taken
        elif dimension == 'combined':
            selector = lambda e: f"{e.material}::{e.action_taken}"
            
        if selector:
            groups = self.aggregate_by(selector)
            for group_key, eps in groups.items():
                stats = self.synthesize_stats(eps)
                print(f"[{group_key}] -> Count: {stats['count']}, AvgStress: {stats['avg_cortisol']:.1f}, Consistency: {stats['consistency_score']:.1f}")

# Usage
if __name__ == "__main__":
    # Mock Usage
    mem = Hippocampus("demo_memory.json") # Needs existing memory or seed
    agg = AggregationFramework(mem)
    agg.generate_report("combined")
