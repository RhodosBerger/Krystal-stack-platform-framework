#!/usr/bin/env python3
"""
Nightly Training: The Sleep Cycle.
Analyzes Hippocampus memories to update the Dopamine Policy.

"While the machine sleeps, it learns."
"""

import json
from collections import defaultdict
from hippocampus import Hippocampus, Episode

POLICY_FILE = "dopamine_policy.json"

class NightlyTrainer:
    def __init__(self, memory_file="machine_memory.json"):
        self.hippocampus = Hippocampus(memory_file)
        self.policy_updates = {}

    def dream_and_learn(self):
        """
        Main Learning Loop.
        1. Cluster memories by Material.
        2. Calculate Avg Cortisol vs Dopamine.
        3. Adjustment weights.
        """
        print("[NIGHTLY_TRAINING] Analysis Started...")
        
        # Group by Material -> Action
        material_stats = defaultdict(lambda: {"cortisol_sum": 0, "dopamine_sum": 0, "count": 0})
        
        for ep in self.hippocampus.episodes:
            key = f"{ep.material}::{ep.action_taken}"
            material_stats[key]["cortisol_sum"] += ep.outcome_cortisol
            material_stats[key]["dopamine_sum"] += ep.outcome_dopamine
            material_stats[key]["count"] += 1
            
        print(f"[NIGHTLY_TRAINING] Processed {len(self.hippocampus.episodes)} episodes.")
        
        # Calculate 'Net Score' for each strategy
        # Key: Material -> List of (Action, Score)
        strategy_rankings = defaultdict(list)
        
        for key, stats in material_stats.items():
            avg_cortisol = stats["cortisol_sum"] / stats["count"]
            avg_dopamine = stats["dopamine_sum"] / stats["count"]
            net_score = avg_dopamine - avg_cortisol # Simple Utility Function
            
            material, action = key.split("::")
            strategy_rankings[material].append((action, net_score, avg_cortisol))

        # Generate Policy
        new_policy = {}
        
        for material, rankings in strategy_rankings.items():
            # Sort by Net Score descending
            rankings.sort(key=lambda x: x[1], reverse=True)
            best_action, best_score, best_cortisol = rankings[0]
            
            print(f"   > ANALYSIS [{material}]: Winner is {best_action} (Score: {best_score:.1f})")
            
            new_policy[material] = {
                "preferred_strategy": best_action,
                "confidence_score": best_score,
                "caution_flag": best_cortisol > 40
            }
        
        self._save_policy(new_policy)

    def _save_policy(self, policy):
        with open(POLICY_FILE, 'w') as f:
            json.dump(policy, f, indent=2)
        print(f"[NIGHTLY_TRAINING] Policy updated and saved to {POLICY_FILE}")

# Usage
if __name__ == "__main__":
    # Seed some data for test
    mem = Hippocampus("test_memory.json")
    # Bad Titanium Runs
    for _ in range(3):
        mem.remember("Titanium", 5000, 2000, "ACTION_RUSH_MODE", 80.0, 5.0)
    # Good Aluminum Runs
    for _ in range(5):
        mem.remember("Aluminum", 8000, 3000, "ACTION_RUSH_MODE", 10.0, 50.0)
        
    trainer = NightlyTrainer("test_memory.json")
    trainer.dream_and_learn()
    
    # Check Result
    with open(POLICY_FILE, 'r') as f:
        print("\nPolicy File Content:")
        print(f.read())
