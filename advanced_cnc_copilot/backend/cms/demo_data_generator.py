#!/usr/bin/env python3
"""
DEMO DATA GENERATOR
Generates realistic CNC telemetry for dashboard demonstration.
"""

import random
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List
import math

class DemoDataGenerator:
    def __init__(self):
        self.timestamp = datetime.now()
        self.cycle_count = 0
        self.tool_life = 100.0
        self.base_rpm = 8000
        
    def generate_telemetry(self) -> Dict:
        """Generate one realistic telemetry snapshot"""
        
        # Simulate varying load based on sine wave (simulates toolpath)
        t = time.time()
        load_variation = 30 * math.sin(t * 0.5) + 60  # 30-90% range
        
        # Vibration correlates with load
        vibration = 0.02 + (load_variation / 1000)
        
        # Add some random spikes (chatter events)
        if random.random() < 0.05:  # 5% chance
            vibration += random.uniform(0.05, 0.15)
            
        # Dopamine decreases when load/vibration high
        dopamine = max(10, 100 - (load_variation * 0.5) - (vibration * 200))
        cortisol = 100 - dopamine  # Inverse relationship
        
        # Economics calculation
        cycle_time = 7.2  # minutes
        cost_per_part = 25.73
        throughput = 60 / cycle_time  # parts per hour
        
        return {
            "timestamp": datetime.now().isoformat(),
            "machine_id": "CNC_VMC_01",
            "program": "BRACKET_ALU_V3.NC",
            "block": random.randint(1200, 1500),
            
            # Core metrics
            "rpm": self.base_rpm + random.randint(-200, 200),
            "load": round(load_variation, 1),
            "vibration": {
                "x": round(random.uniform(0.01, 0.04), 3),
                "y": round(random.uniform(0.01, 0.04), 3),
                "z": round(vibration, 3)
            },
            "feed_rate": random.randint(2200, 2800),
            
            # Temperature
            "spindle_temp": round(40 + (load_variation * 0.3), 1),
            "coolant_temp": round(18 + random.uniform(-2, 2), 1),
            
            # Position
            "position": {
                "x": round(random.uniform(-100, 200), 3),
                "y": round(random.uniform(-80, 80), 3),
                "z": round(random.uniform(-50, 5), 3)
            },
            
            # Tool info
            "tool_id": "T03",
            "tool_life": round(self.tool_life, 1),
            
            # Neuro-engine
            "dopamine": round(dopamine, 1),
            "cortisol": round(cortisol, 1),
            "serotonin": round(random.uniform(50, 80), 1),
            
            # Signaling
            "signal": "GREEN" if load_variation < 85 and vibration < 0.1 else ("AMBER" if load_variation < 95 else "RED"),
            
            # Economics
            "cycle_time_actual": round(cycle_time + random.uniform(-0.5, 0.5), 2),
            "cost_per_part": round(cost_per_part + random.uniform(-2, 2), 2),
            "throughput": round(throughput, 1),
            "parts_completed": self.cycle_count,
            
            # Logic flow
            "logic_flow": {
                "sensory": round(load_variation, 0),
                "reward": round(dopamine, 0),
                "signal": "GREEN" if load_variation < 85 else "AMBER",
                "economics": round(cost_per_part, 1)
            }
        }
    
    def generate_historical_batch(self, count: int = 100) -> List[Dict]:
        """Generate historical data points"""
        data = []
        for i in range(count):
            # Degrade tool life over time
            self.tool_life = max(10, 100 - (i * 0.5))
            if i % 10 == 0:
                self.cycle_count += 1
            
            telemetry = self.generate_telemetry()
            telemetry["timestamp"] = (datetime.now() - timedelta(minutes=count-i)).isoformat()
            data.append(telemetry)
            
        return data
    
    def generate_projects_dataset(self, count: int = 20) -> List[Dict]:
        """Generate historical projects for LLM training"""
        materials = ["Aluminum_6061", "Steel_1045", "Titanium", "Brass"]
        strategies = ["AGGRESSIVE", "BALANCED", "CONSERVATIVE"]
        
        projects = []
        for i in range(count):
            material = random.choice(materials)
            strategy = random.choice(strategies)
            
            # Base params by material
            base_rpm = {
                "Aluminum_6061": 8000,
                "Steel_1045": 3500,
                "Titanium": 2000,
                "Brass": 5000
            }[material]
            
            # Adjust by strategy
            rpm_mult = {"AGGRESSIVE": 1.2, "BALANCED": 1.0, "CONSERVATIVE": 0.8}[strategy]
            
            projects.append({
                "project_id": f"PROJ_2024_{1000+i:04d}",
                "material": material,
                "strategy": strategy,
                "params": {
                    "rpm": int(base_rpm * rpm_mult),
                    "feed": random.randint(1500, 3000),
                    "depth_of_cut": round(random.uniform(0.5, 3.0), 2)
                },
                "outcome": {
                    "cycle_time": round(random.uniform(5, 15), 2),
                    "quality_score": round(random.uniform(0.7, 0.98), 3),
                    "tool_life_consumed": round(random.uniform(0.05, 0.30), 3),
                    "success": random.random() > 0.1
                }
            })
            
        return projects

# API endpoints to serve this data
def generate_demo_api_responses():
    """Generate all demo responses for API"""
    gen = DemoDataGenerator()
    
    return {
        "/api/telemetry/current": gen.generate_telemetry(),
        "/api/telemetry/history": gen.generate_historical_batch(100),
        "/api/projects": gen.generate_projects_dataset(20),
        "/api/health": {
            "status": "healthy",
            "database": "connected",
            "redis": "connected",
            "hal_mode": "mock"
        }
    }

if __name__ == "__main__":
    gen = DemoDataGenerator()
    
    # Test: Generate and print one snapshot
    print("=== SINGLE TELEMETRY SNAPSHOT ===")
    print(json.dumps(gen.generate_telemetry(), indent=2))
    
    # Test: Generate batch
    print("\n=== HISTORICAL BATCH (5 samples) ===")
    batch = gen.generate_historical_batch(5)
    for item in batch:
        print(f"{item['timestamp']}: RPM={item['rpm']}, Load={item['load']}%, Signal={item['signal']}")
    
    # Test: Projects
    print("\n=== PROJECTS DATASET (3 samples) ===")
    projects = gen.generate_projects_dataset(3)
    for proj in projects:
        print(f"{proj['project_id']}: {proj['material']} @ {proj['params']['rpm']} RPM → Quality: {proj['outcome']['quality_score']}")
    
    print("\n✅ Demo data generator ready!")
