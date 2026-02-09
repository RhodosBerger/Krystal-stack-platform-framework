#!/usr/bin/env python3
"""
FANUC RISE API
REST Interface for the Cognitive Manufacturing System.

Run with: uvicorn cms.fanuc_api:app --reload
"""

import sys
import os
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Imports from Core Logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.cms.sensory_cortex import SensoryCortex, HALParser, SolidworksParser, SenseDatum
from backend.cms.hal_fanuc import FanucAdapter
from backend.cms.impact_cortex import ImpactCortex, ImpactScore
from backend.cms.dopamine_engine import DopamineEngine
from backend.cms.cnc_vino_optimizer import CNCOptimizer
from backend.cms.protocol_conductor import ProtocolConductor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Fanuc Rise API", version="1.0.0")
conductor_ai = ProtocolConductor()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL STATE (The Brain in a Box) ---
cortex = SensoryCortex()
cortex.register_parser(HALParser(FanucAdapter())) # Use HAL
cortex.register_parser(SolidworksParser({}))

impact_brain = ImpactCortex()
dopamine_brain = DopamineEngine()
optimizer = CNCOptimizer()

# --- MODELS ---
class SimulationInput(BaseModel):
    fanuc_data: Dict[str, float] = {"load": 0, "rpm": 0, "vibration": 0.0}
    sw_data: Dict[str, float] = {"curvature": 0.0}

class OptimizedResult(BaseModel):
    ir_lines: List[str]

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "ONLINE", "system": "FANUC_RISE_NEURO_CORE"}

@app.post("/perceive")
def perceive_environment(sim_input: SimulationInput):
    """
    Feeds raw data into the Sensory Cortex and gets the Impact Score.
    """
    inputs = {
        "fanuc": sim_input.fanuc_data,
        "solidworks": sim_input.sw_data
    }
    
    # 1. Sense
    stream = cortex.collect_all(inputs)
    
    # 2. Think (Impact)
    impact = impact_brain.process(stream)
    
    # 3. Feel (Dopamine - Simplified update)
    # We update the dopamine engine based on the 'Safety' of the impact
    action = dopamine_brain.evaluate_stimuli(
        speed_factor=stream[0].speed_metric * 2.0, # Approximate
        vibration_level=stream[0].vibration_level,
        deviation_score=0.1, # Mock deviation
        result_quality=impact.quality / 100.0
    )
    
    return {
        "sensation_count": len(stream),
        "impact": {
            "safety": impact.safety,
            "quality": impact.quality,
            "efficiency": impact.efficiency
        },
        "neuro_state": vars(dopamine_brain.state),
        "recommended_action": action
    }

@app.post("/optimize")
def optimize_gcode(gcode: List[str], material: str = "Steel4140"):
    """
    Runs the CNC-VINO Optimizer on raw G-Code.
    """
    try:
        ir = optimizer.optimize_model(gcode, material)
        return {"optimized_ir": ir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conduct")
def conduct_protocol(request: Dict[str, str]):
    """
    Creative UI Endpoint: Conducts a protocol based on user prompt.
    """
    try:
        name = request.get("name", "Unnamed_Protocol")
        prompt = request.get("prompt", "")
        return conductor_ai.conduct_scenario(name, prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
