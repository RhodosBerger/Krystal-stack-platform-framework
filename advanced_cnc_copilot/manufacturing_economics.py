#!/usr/bin/env python3
"""
Manufacturing Economics Engine for CNC Copilot
Adapted from: business_model_framework.py

This module calculates production costs, optimizes for economic efficiency,
and compares "Human" vs "AI" production plans.
"""

import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import asyncio
from enum import Enum

# Import MessageBus for The Shadow Council
try:
    from cms.message_bus import global_bus, Message
except ImportError:
    global_bus = None # Fallback for standalone runs

# Configure logging to match parent project style
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CNC_ECON] - %(message)s')
logger = logging.getLogger(__name__)

class ProductionMode(Enum):
    """Machining strategies adapted from OptimizationProfiles."""
    STANDARD = "standard"          # Balanced speed/quality
    PRECISION = "precision"        # High quality, slower (Higher cost)
    RUSH = "rush"                  # Max speed, higher tool wear (Hidden cost)
    ECO = "eco"                    # Max tool life, slowest speed

@dataclass
class CostFactors:
    """Base cost parameters for a production facility."""
    machine_hourly_rate: float      # Cost to run CNC machine per hour
    labor_hourly_rate: float        # Operator cost per hour
    kilowatt_price: float           # Electricity cost
    tool_wear_factor: float = 1.0   # Multiplier for tool depreciation

@dataclass
class ProjectParameters:
    """
    Input parameters for a specific part/project.
    Analogous to a 'SaaS Subscription' tier but for a physical object.
    """
    project_id: str
    part_name: str
    material_cost_per_unit: float
    estimated_cycle_time_minutes: float # Human estimate
    batch_size: int
    daily_quota: int = 0
    
    # "Solidworks" simulated data
    complexity_score: float = 1.0   # 1.0 = simple, 5.0 = 5-axis complex

@dataclass
class ProductionRunResult:
    """The financial result of a production run."""
    run_id: str
    mode: ProductionMode
    total_cost: float
    cost_per_unit: float
    total_time_hours: float
    profit_margin: float
    is_viable: bool
    ai_optimization_notes: str
    
    # NEW: The "Score" (Reward)
    economic_score: float = 0.0
    longevity_impact: float = 0.0 # Cost of wear

class ManufacturingEconomics:
    """
    The 'Business Model' for the factory floor.
    Calculates the 'Revenue' (Value of parts) vs 'Expenses' (Production cost).
    """
    
    def __init__(self, cost_factors: CostFactors):
        self.cost_factors = cost_factors
        self.history: List[ProductionRunResult] = []
        self._framework_id = f"MFG_ECON_{uuid.uuid4().hex[:8].upper()}"
        logger.info(f"Initialized Manufacturing Economics Engine: {self._framework_id}")

    def calculate_production_cost(self, 
                                  params: ProjectParameters, 
                                  mode: ProductionMode = ProductionMode.STANDARD) -> ProductionRunResult:
        """
        Calculates the full cost of a production run based on the selected mode.
        This is the core 'Adaptability' logic - changing the mode changes the economics.
        """
        
        # 1. Apply Mode Modifiers (The "AI" Logic)
        time_modifier = 1.0
        wear_modifier = 1.0
        
        if mode == ProductionMode.RUSH:
            time_modifier = 0.7  # 30% faster
            wear_modifier = 2.5  # High tool wear
        elif mode == ProductionMode.PRECISION:
            time_modifier = 1.5  # 50% slower
            wear_modifier = 0.8  # Gentle on tools
        elif mode == ProductionMode.ECO:
            time_modifier = 1.2
            wear_modifier = 0.5  # Very gentle
            
        # 2. Calculate Resource Usage
        adjusted_cycle_time = params.estimated_cycle_time_minutes * time_modifier
        total_machining_minutes = adjusted_cycle_time * params.batch_size
        total_hours = total_machining_minutes / 60.0
        
        # 3. Calculate Financials
        # Machine Cost (including power/wear)
        machine_cost = (self.cost_factors.machine_hourly_rate * total_hours) * wear_modifier
        
        # Labor Cost (Operator only needs to load/unload, usually part-time attention, but simplifying)
        labor_cost = self.cost_factors.labor_hourly_rate * total_hours * 0.5 # 50% attention
        
        # Material Cost
        material_total = params.material_cost_per_unit * params.batch_size
        
        total_cost = machine_cost + labor_cost + material_total
        cost_per_unit = total_cost / params.batch_size
        
        # 4. Calculate NEW Economic Score (Reward System)
        # Revenue assumption (User didn't specify, so we assume $50 value/part + material)
        assumed_revenue = 50.0 + params.material_cost_per_unit
        profit = (assumed_revenue - cost_per_unit) * params.batch_size
        
        # Volume Reward (Logarithmic scale: more parts = higher confidence score)
        import math
        volume_score = math.log(params.batch_size + 1) * 10
        
        # Longevity Penalty (Simulated based on wear modifier)
        # If wear_modifier > 1.0 (Rush), we assume spindle/machine damage cost
        spindle_replacement_cost = 15000.0
        longevity_loss = 0.0
        if wear_modifier > 1.1:
            # 1% life loss per 1000 hours of Rush mode?
            longevity_loss = (total_hours / 1000.0) * spindle_replacement_cost
            
        final_score = (profit / 100) + volume_score - (longevity_loss / 10)

        # 5. Generate Result
        result = ProductionRunResult(
            run_id=f"RUN_{uuid.uuid4().hex[:6].upper()}",
            mode=mode,
            total_cost=round(total_cost, 2),
            cost_per_unit=round(cost_per_unit, 2),
            total_time_hours=round(total_hours, 2),
            profit_margin=round(profit, 2), # Placeholder
            is_viable=profit > 0, # Placeholder
            ai_optimization_notes=f"Mode {mode.name}: Time Mod {time_modifier}, Wear Mod {wear_modifier}",
            economic_score=round(final_score, 2),
            longevity_impact=round(longevity_loss, 2)
        )
        
        self.history.append(result)
        logger.info(f"Calculated run for {params.part_name} [{mode.name}]: Score={result.economic_score} (Vol={params.batch_size})")
        return result

    def compare_human_vs_ai(self, params: ProjectParameters) -> str:
        """
        Generates a text report comparing 'Standard' (Human) vs optimized modes.
        """
        human_result = self.calculate_production_cost(params, ProductionMode.STANDARD)
        rush_result = self.calculate_production_cost(params, ProductionMode.RUSH)
        eco_result = self.calculate_production_cost(params, ProductionMode.ECO)
        
        report = f"""
        === MANUFACTURING ECONOMIC ANALYSIS: {params.part_name} ===
        Batch Size: {params.batch_size} | Material: ${params.material_cost_per_unit}/unit
        
        [HUMAN BASELINE] (Standard Mode)
        - Cost/Unit: ${human_result.cost_per_unit}
        - Total Time: {human_result.total_time_hours} hrs
        
        [AI OPTIMIZED SCENARIOS]
        1. RUSH MODE (High Speed, High Wear)
           - Cost/Unit: ${rush_result.cost_per_unit} ({self._calc_diff(human_result.cost_per_unit, rush_result.cost_per_unit)})
           - Total Time: {rush_result.total_time_hours} hrs
           - Score: {rush_result.economic_score} (Long. Impact: -${rush_result.longevity_impact})
           
        2. ECO MODE (Tool Preservation)
           - Cost/Unit: ${eco_result.cost_per_unit} ({self._calc_diff(human_result.cost_per_unit, eco_result.cost_per_unit)})
           - Total Time: {eco_result.total_time_hours} hrs
           - Score: {eco_result.economic_score} (Long. Impact: -${eco_result.longevity_impact})
           
        RECOMMENDATION:
        {self._recommend_strategy(human_result, rush_result, eco_result)}
        """
        return report

    def _calc_diff(self, base, new) -> str:
        diff = ((new - base) / base) * 100
        return f"{diff:+.1f}%"

    def _recommend_strategy(self, human, rush, eco) -> str:
        # Simple logic: If Rush is cheaper (due to high labor/machine hourly cost overriding tool wear), suggest it.
        if rush.cost_per_unit < human.cost_per_unit:
            return ">>> RUSH MODE RECOMMENDED: Labor/Machine savings outweigh tool wear costs."
        elif eco.cost_per_unit < human.cost_per_unit:
            return ">>> ECO MODE RECOMMENDED: Tool savings outweigh extra machine time."
        else:
            return ">>> STANDARD MODE RECOMMENDED: Optimal balance found."

class AccountantWrapper:
    """
    The Shadow Council Member: 'The Accountant'.
    Listens to the MessageBus and critiques plans financially.
    """
    def __init__(self):
        # Default costs for the council
        costs = CostFactors(machine_hourly_rate=150.0, labor_hourly_rate=50.0, kilowatt_price=0.15)
        self.engine = ManufacturingEconomics(costs)
        
        if global_bus:
            global_bus.subscribe("DRAFT_PLAN", self.audit_plan)
            logger.info("The Accountant has joined the Council.")

    async def audit_plan(self, msg: Message):
        """
        Triggered when 'The Creator' proposes a plan.
        """
        plan = msg.payload
        logger.info(f"Accountant analyzing plan for: {plan.get('material', 'Unknown')}")
        
        # Simulate quick calculation
        # In real app, we'd map 'plan' dict to 'ProjectParameters'
        # Here we just emit a generic impact analysis
        
        impact = {
            "estimated_cost": 150.00, # Placeholder
            "notes": "Plan uses Titanium. High tool wear expected.",
            "status": "CAUTION"
        }
        
        await global_bus.publish("ECONOMIC_IMPACT", impact, sender_id="ACCOUNTANT")

if __name__ == "__main__":
    # Rudimentary Test
    costs = CostFactors(machine_hourly_rate=150.0, labor_hourly_rate=50.0, kilowatt_price=0.15)
    project = ProjectParameters(
        project_id="P001", 
        part_name="Titanium Bracket", 
        material_cost_per_unit=25.0, 
        estimated_cycle_time_minutes=45.0, 
        batch_size=100
    )
    
    engine = ManufacturingEconomics(costs)
    print(engine.compare_human_vs_ai(project))
