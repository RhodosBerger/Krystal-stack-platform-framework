#!/usr/bin/env python3
"""
The Hippocampus: Experience Store.
Records "Episodes" of machining for the Learning Loop.

"It remembers that Titanium made the tool scream."
"""

import json
import time
import os
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

HISTORY_FILE = "machine_memory.json"

@dataclass
class Episode:
    timestamp: float
    material: str
    rpm: float
    feed: float
    action_taken: str       # e.g., "ACTION_RUSH_MODE"
    outcome_cortisol: float # Stress Level (Penalty)
    outcome_dopamine: float # Reward Level
    notes: str = ""

class Hippocampus:
    def __init__(self, memory_file: str = HISTORY_FILE):
        self.memory_file = memory_file
        self.episodes: List[Episode] = self._load_memory()

    def _load_memory(self) -> List[Episode]:
        if not os.path.exists(self.memory_file):
            return []
        try:
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
                return [Episode(**e) for e in data]
        except Exception as e:
            print(f"[HIPPOCAMPUS] Memory Corrupt: {e}")
            return []

    def remember(self, material: str, rpm: float, feed: float, action: str, 
                 cortisol: float, dopamine: float, notes: str = ""):
        """
        Commit an experience to memory.
        """
        episode = Episode(
            timestamp=time.time(),
            material=material,
            rpm=rpm,
            feed=feed,
            action_taken=action,
            outcome_cortisol=cortisol,
            outcome_dopamine=dopamine,
            notes=notes
        )
        self.episodes.append(episode)
        self._save_memory()

    def _save_memory(self):
        try:
            with open(self.memory_file, 'w') as f:
                # Convert dataclasses to dicts
                json.dump([asdict(e) for e in self.episodes], f, indent=2)
        except Exception as e:
            print(f"[HIPPOCAMPUS] Write Error: {e}")

    def recall_trauma(self, material: str) -> List[Episode]:
        """
        Returns episodes with High Cortisol for this material.
        Used to warn the System before cutting.
        """
        return [e for e in self.episodes if e.material == material and e.outcome_cortisol > 50]

# Usage
if __name__ == "__main__":
    brain = Hippocampus("test_memory.json")
    brain.remember("Titanium", 5000, 2000, "ACTION_RUSH_MODE", 80.0, 0.0, "Tool chattered heavily")
    
    traumas = brain.recall_trauma("Titanium")
    if traumas:
        print(f"Recall: {len(traumas)} bad memories with Titanium.")
