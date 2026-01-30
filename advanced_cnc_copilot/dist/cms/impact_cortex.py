#!/usr/bin/env python3
"""
THE IMPACT CORTEX
The "Brain" of the Unified Perceptron.

Purpose: To digest normalized 'SenseDatum' streams and 
calculate high-level Impact Scores (Safety, Quality).
"""

from typing import List, Dict, Any
from sensory_cortex import SenseDatum

class ImpactScore:
    def __init__(self, safety: float, quality: float, efficiency: float):
        self.safety = safety        # 0-100 (100 = Safe)
        self.quality = quality      # 0-100 (100 = Perfect)
        self.efficiency = efficiency # 0-100 (100 = Max Output)
        
    def __repr__(self):
        return f"[IMPACT] Safety:{self.safety:.1f} | Quality:{self.quality:.1f} | Eff:{self.efficiency:.1f}"

class ImpactCortex:
    """
    Analyzes the "Impact" of the current sensory state.
    """
    def __init__(self):
        # Weights for the "Personality"
        self.w_vib = 50.0   # Impact of vibration on Quality
        self.w_load = 30.0  # Impact of load on Safety
        
    def process(self, sensory_stream: List[SenseDatum]) -> ImpactScore:
        """
        The Thinking Process.
        """
        if not sensory_stream:
            return ImpactScore(100, 100, 0)
            
        # Aggregate Metrics from all senses
        total_vib = 0.0
        max_load = 0.0
        avg_speed = 0.0
        
        for datum in sensory_stream:
            total_vib += datum.vibration_level
            max_load = max(max_load, datum.load_factor)
            avg_speed += datum.speed_metric
            
        count = len(sensory_stream)
        avg_vib = total_vib / count
        
        # --- LOGIC CORE ---
        
        # 1. SAFETY CALCULATION
        # Safety drops as Load increases and Vibration increases
        # Score = 100 - (Load*40 + Vib*60)
        safety_score = 100.0 - (max_load * 40.0 + avg_vib * 60.0)
        safety_score = max(0, min(100, safety_score))
        
        # 2. QUALITY CALCULATION
        # Quality is inversely proportional to Vibration
        # Score = 100 - (Vib * 100)
        quality_score = 100.0 - (avg_vib * 100.0)
        quality_score = max(0, min(100, quality_score))
        
        # 3. EFFICIENCY CALCULATION
        # Efficiency is Speed * Load (Work Done)
        # Score = Speed * 100
        efficiency_score = avg_speed * 100.0
        
        return ImpactScore(safety_score, quality_score, efficiency_score)

# Usage Example
if __name__ == "__main__":
    from sensory_cortex import FanucParser, SolidworksParser, SensoryCortex
    
    # 1. Sense
    cortex = SensoryCortex()
    cortex.register_parser(FanucParser({}))
    cortex.register_parser(SolidworksParser({}))
    
    inputs = {
        "fanuc": {"load": 80, "rpm": 12000, "vibration": 0.4}, # High Load/Vib
        "solidworks": {"curvature": 0.05} # Low complexity
    }
    
    data_stream = cortex.collect_all(inputs)
    
    # 2. Think
    brain = ImpactCortex()
    impact = brain.process(data_stream)
    
    print("\n--- UNIFIED PERCEPTRON RESULT ---")
    print(f"Inputs: {len(data_stream)} Sources")
    print(impact)
    
    if impact.safety < 50:
        print(">>> DECISION: INITIATE SAFETY PROTOCOLS")
