#!/usr/bin/env python3
"""
CNC Optimization Engine & Logic Bridges
Connects the 'Mind' (Economics) to the 'Machine' (Fanuc G-Code).

This module implements the "Bridge" pattern requested:
1.  Analyzes Project Parameters (Data Layer)
2.  Consults Manufacturing Economics (Mind Layer)
3.  Generates Optimized G-Code (Physical Layer)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

# Import the Data and Mind layers
from manufacturing_economics import (
    ProjectParameters, 
    ProductionMode, 
    ManufacturingEconomics, 
    CostFactors,
    ProductionRunResult
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CNC_BRIDGE] - %(message)s')
logger = logging.getLogger(__name__)

class GCodeBridge:
    """
    The Bridge between Abstract Strategy and Concrete G-Code.
    Translates 'Modes' into 'Feed Rates' and 'M-Codes'.
    """
    
    def __init__(self, mode: ProductionMode):
        self.mode = mode
        self.code_buffer: List[str] = []
        
    def _get_feed_rate(self) -> int:
        """Determines feed rate based on mode."""
        if self.mode == ProductionMode.RUSH:
            return 5000 # High speed
        elif self.mode == ProductionMode.PRECISION:
            return 800  # Slow, careful
        elif self.mode == ProductionMode.ECO:
            return 2000 # Optimal for tool life
        else:
            return 2500 # Standard

    def _get_spindle_speed(self) -> int:
        """Determines spindle RPM."""
        if self.mode == ProductionMode.RUSH:
            return 12000
        elif self.mode == ProductionMode.PRECISION:
            return 8000
        elif self.mode == ProductionMode.ECO:
            return 6000 # Cool, less heat
        else:
            return 10000

    def generate_header(self, part_name: str):
        """Standard Fanuc Header."""
        self.code_buffer.append(f"%")
        self.code_buffer.append(f"O0001 ({part_name})")
        self.code_buffer.append(f"(MODE: {self.mode.name})")
        self.code_buffer.append(f"G21 G17 G40 G49 G80 G90") # Metric, XY plane, Cancel cycles
        self.code_buffer.append(f"T1 M06 (Select Tool 1)")
        self.code_buffer.append(f"S{self._get_spindle_speed()} M03 (Spindle On)")
        self.code_buffer.append(f"G54 (Work Offset 1)")

    def generate_facing_op(self, width: float, depth: float):
        """Simulates a facing operation adaptable to mode."""
        feed = self._get_feed_rate()
        
        self.code_buffer.append(f"(--- FACING OP ---)")
        self.code_buffer.append(f"G00 X0 Y0")
        self.code_buffer.append(f"G43 Z50. H01")
        self.code_buffer.append(f"G01 Z0. F{feed}")
        
        # Simple zig-zag pattern simulation
        self.code_buffer.append(f"G01 X{width} F{feed}")
        self.code_buffer.append(f"G01 Y{depth * 0.5}")
        self.code_buffer.append(f"G01 X0")
        self.code_buffer.append(f"G01 Y{depth}")
        self.code_buffer.append(f"G01 X{width}")

        if self.mode == ProductionMode.PRECISION:
             self.code_buffer.append(f"(PRECISION MODE: EXTRA FINISHING PASS)")
             self.code_buffer.append(f"G01 Z-0.05 F{feed // 2}") # Very slow skim cut
             self.code_buffer.append(f"X0 Y0")

    def generate_footer(self):
        """Safe shutdown."""
        self.code_buffer.append(f"M05 (Spindle Stop)")
        self.code_buffer.append(f"G91 G28 Z0 (Retract Z)")
        self.code_buffer.append(f"G28 Y0 (Home Y)")
        self.code_buffer.append(f"M30 (End Program)")
        self.code_buffer.append(f"%")

    def get_program(self) -> str:
        return "\n".join(self.code_buffer)


class OptimizationCopilot:
    """
    The 'Mind' that controls the 'Bridge'.
    Uses Economics to decide which G-Code Bridge to build.
    """
    
    def __init__(self, cost_factors: CostFactors):
        self.economics = ManufacturingEconomics(cost_factors)
        
    def create_optimized_plan(self, project: ProjectParameters) -> dict:
        """
        Full workflow: Analyzes economics -> Selects Mode -> Generates Code.
        """
        logger.info(f"Analyzing project: {project.part_name}")
        
        # 1. Run Economic Simulations
        std_res = self.economics.calculate_production_cost(project, ProductionMode.STANDARD)
        rush_res = self.economics.calculate_production_cost(project, ProductionMode.RUSH)
        eco_res = self.economics.calculate_production_cost(project, ProductionMode.ECO)
        
        # 2. Select Best Strategy (Logic Bridge)
        # Decision Logic: 
        # - If Daily Quota is high > RUSH
        # - If Complexity is high > PRECISION (Not modeled economically yet, but logic stub)
        # - Default > Lowest Cost
        
        selected_res = std_res
        reason = "Standard Baseline"
        
        if project.daily_quota > 0:
            # Check if standard meets quota (assuming 8h shift)
            parts_per_shift_std = (8 * 60) / (project.estimated_cycle_time_minutes)
            if parts_per_shift_std < project.daily_quota:
                selected_res = rush_res
                reason = "Quota Requirement (Speed Priority)"
        elif eco_res.cost_per_unit < std_res.cost_per_unit:
            selected_res = eco_res
            reason = "Economic Efficiency (Cost Priority)"

        logger.info(f"Optimization Complete. Selected Strategy: {selected_res.mode.name} ({reason})")

        # 3. Activate the G-Code Bridge
        bridge = GCodeBridge(selected_res.mode)
        bridge.generate_header(project.part_name)
        bridge.generate_facing_op(width=100.0, depth=50.0) # Placeholder dimensions
        bridge.generate_footer()
        
        return {
            "strategy": selected_res,
            "reason": reason,
            "gcode": bridge.get_program(),
            "alternatives": [std_res, rush_res, eco_res]
        }

if __name__ == "__main__":
    # Test the Bridge
    costs = CostFactors(machine_hourly_rate=120.0, labor_hourly_rate=45.0, kilowatt_price=0.20)
    copilot = OptimizationCopilot(costs)
    
    # Scenario: Urgent order
    urgent_project = ProjectParameters(
        project_id="URG-001",
        part_name="Aero-Flange",
        material_cost_per_unit=120.0,
        estimated_cycle_time_minutes=30.0,
        batch_size=50,
        daily_quota=20 # Needs fast production
    )
    
    plan = copilot.create_optimized_plan(urgent_project)
    
    print(f"=== OPTIMIZATION RESULT ===")
    print(f"Selected Mode: {plan['strategy'].mode.name}")
    print(f"Reason: {plan['reason']}")
    print(f"Cost/Unit: ${plan['strategy'].cost_per_unit}")
    print(f"\n[GENERATED G-CODE PREVIEW]\n" + "-"*30)
    print("\n".join(plan['gcode'].split("\n")[:10])) # Show first 10 lines
