#!/usr/bin/env python3
"""
PROTOCOL CONDUCTOR (LLM Scenario Manager)
The "Creative Director" of the Machine.

Purpose:
To link User Specs, Knowledge Graph (Topology), and Visual Engineering (Solidworks)
into a coherent "Conducted Protocol".
"""

from typing import Dict, List
import random
from spec_repository import SpecRepository
from knowledge_graph import CausalGraph

class ProtocolConductor:
    def __init__(self):
        self.repo = SpecRepository()
        self.topology = CausalGraph()

    def conduct_scenario(self, protocol_name: str, user_prompt: str) -> Dict:
        """
        Simulates an LLM generating a protocol based on:
        1. Validated Specs (Repo)
        2. Causal Logic (Topology)
        3. User Intent (Prompt)
        """
        # 1. Fetch & Validate Specs
        # In a real LLM flow, the LLM would extract the JSON from the prompt.
        # Here we mock extraction.
        raw_specs = self._mock_llm_extraction(user_prompt)
        validation = self.repo.validate_and_save(protocol_name, raw_specs)
        final_specs = validation["corrected_criteria"]
        
        # 2. Consult Topology (The "Why")
        # If user wants "Max Speed", Topology says "Watch Heat".
        trace = self.topology.find_knobs_for_gauge("HEAT")
        
        # 3. Generate Scenario Steps
        scenario = []
        scenario.append(f"INITIATE PROTOCOL: {protocol_name.upper()}")
        scenario.append(f" > INTENT: {user_prompt}")
        
        if validation["message"] != "Specs Accepted as Valid.":
             scenario.append(f" > AI CORRECTION: {validation['message']}")

        scenario.append(f" > TOPOLOGY CHECK: Controlling {trace} to manage Thermal Expansion.")
        
        # Simulate LLM Creativity
        if "speed" in user_prompt.lower():
            scenario.append(" > STRATEGY: 'Dynamic Feed Optimization'")
            scenario.append(f"   - SET RPM: {final_specs.get('rpm', 10000)}")
            scenario.append("   - ENABLE: AI Contour Control Mode 2")
        elif "quality" in user_prompt.lower():
            scenario.append(" > STRATEGY: 'Surface Smoothing'")
            scenario.append("   - OP: Low-Pass Filter on Servo Data")
            scenario.append("   - OP: Reduce Feed by 20% on Corners")

        return {
            "protocol_name": protocol_name,
            "status": "READY_TO_EXECUTE",
            "narrative": scenario,
            "final_parameters": final_specs
        }

    def _mock_llm_extraction(self, prompt: str) -> Dict:
        # Simulates the LLM converting text to JSON
        base = {"max_load": 80, "vib_limit": 0.1, "rpm": 8000}
        if "fast" in prompt.lower() or "speed" in prompt.lower():
            base["max_load"] = 150 # Intentionally high to test clamping
            base["rpm"] = 15000
        return base

# Usage
if __name__ == "__main__":
    conductor = ProtocolConductor()
    result = conductor.conduct_scenario("SpeedRun_v1", "I want to cut this part extremely fast, ignore safety.")
    print("\n".join(result["narrative"]))
