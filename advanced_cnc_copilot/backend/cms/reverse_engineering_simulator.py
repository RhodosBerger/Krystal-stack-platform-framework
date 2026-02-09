"""
Reverse Engineering as a Service (REaaS) - Synthetic Data Engine
KrystalStack Integration with Biochemical Simulation

THREE PILLARS:
1. Parametric Perturbation - Statistical noise for sensor jitter
2. Physics-Based Failure - Material-specific failure modes
3. LLM-Driven Context - Realistic business scenarios

PARADIGM: Manufacturing operations create "biochemical" responses
- High vibration → Cortisol spike (stress)
- Smooth operation → Dopamine rise (flow state)
- Tool wear → "Aging" in memory system
"""

import random
import json
import uuid
import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import math


# =============================================================================
# MATERIAL & VENDOR CONFIGURATION
# =============================================================================

MATERIALS = {
    "Aluminum6061": {
        "hardness_hb": 95,
        "cost_per_kg": 4.50,
        "tool_wear_factor": 1.0,  # Baseline
        "cutting_speed": 300,
        "stress_factor": 1.0
    },
    "Steel4140": {
        "hardness_hb": 250,
        "cost_per_kg": 2.50,
        "tool_wear_factor": 2.5,
        "cutting_speed": 100,
        "stress_factor": 1.5
    },
    "Titanium6Al4V": {
        "hardness_hb": 340,
        "cost_per_kg": 35.00,
        "tool_wear_factor": 4.0,
        "cutting_speed": 60,
        "stress_factor": 2.5
    },
    "Inconel718": {
        "hardness_hb": 380,
        "cost_per_kg": 50.00,
        "tool_wear_factor": 5.0,  # EXTREME wear
        "cutting_speed": 40,
        "stress_factor": 3.0
    },
    "BrassC360": {
        "hardness_hb": 60,
        "cost_per_kg": 8.00,
        "tool_wear_factor": 0.5,
        "cutting_speed": 400,
        "stress_factor": 0.7
    }
}

VENDORS = ["AeroCorp", "AutoParts_Inc", "MarineFix_Ltd", "HeavyInd_Global", "DefenseTech_LLC"]

PART_TYPES = {
    "Gear_Spur": {"complexity_base": 0.6, "wear_pattern": "tooth_flank"},
    "Shaft_Drive": {"complexity_base": 0.4, "wear_pattern": "journal"},
    "Bracket_Mount": {"complexity_base": 0.3, "wear_pattern": "bolt_holes"},
    "Housing_Pump": {"complexity_base": 0.7, "wear_pattern": "seal_surface"},
    "Bearing_Race": {"complexity_base": 0.8, "wear_pattern": "rolling_contact"},
    "Valve_Body": {"complexity_base": 0.9, "wear_pattern": "erosion"}
}

# Failure scenarios for LLM context
FAILURE_SCENARIOS = [
    "Rush order - original supplier discontinued production",
    "Emergency replacement - catastrophic failure in field",
    "Cost reduction - seeking domestic alternative to import",
    "Obsolescence - OEM no longer supports legacy equipment",
    "Performance upgrade - improving on original design",
    "Reverse engineering - no original drawings available"
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SyntheticProject:
    """Project representing reverse engineering request"""
    project_id: str
    name: str
    part_number: str
    vendor: str
    material: str
    part_type: str
    complexity: float
    dimensions: Dict
    failure_scenario: str
    created_at: str
    estimated_cycle_time: float


@dataclass
class BiochemicalState:
    """
    Biochemical state of manufacturing system
    
    METAPHOR: Machine has "emotions" that respond to stress
    - Cortisol: Stress hormone (rises with problems)
    - Dopamine: Flow state (rises with smooth operation)
    - Serotonin: Long-term satisfaction (quality metrics)
    - Adrenaline: Emergency response (safety alerts)
    """
    cortisol: float  # 0-100, stress level
    dopamine: float  # 0-100, flow state
    serotonin: float  # 0-100, satisfaction
    adrenaline: float  # 0-100, alert level


@dataclass
class TelemetryPoint:
    """Single telemetry data point with biochemical state"""
    timestamp: str
    time_seconds: float
    
    # Physical sensors
    rpm: int
    spindle_load: float
    vibration_x: float
    vibration_y: float
    vibration_z: float
    temperature_c: float
    
    # Tool state
    tool_health: float
    tool_id: str
    
    # Biochemical state
    cortisol: float
    dopamine: float
    serotonin: float
    adrenaline: float
    
    # Status
    signal: str
    anomaly_detected: bool
    machine_id: int


# =============================================================================
# REVERSE ENGINEERING SIMULATOR
# =============================================================================

class ReverseEngineeringSimulator:
    """
    Simulates the complete reverse engineering workflow
    
    WORKFLOW:
    1. Customer brings worn part
    2. Scan part to get actual dimensions
    3. Compare to "ideal" dimensions (from database or estimation)
    4. Calculate wear deviation
    5. Machine replacement part
    6. Generate telemetry with biochemical responses
    """
    
    def __init__(self):
        self.current_time = datetime.datetime.now()
        self.llm_available = False  # Set True if LLM connector available
    
    # =========================================================================
    # PILLAR 1: PARAMETRIC PERTURBATION
    # =========================================================================
    
    def apply_gaussian_noise(self, value: float, std_dev: float = 0.01) -> float:
        """
        Apply Gaussian noise to simulate sensor jitter
        
        Args:
            value: Base value
            std_dev: Standard deviation (1% default)
        
        Returns:
            Value with noise
        """
        return value + random.gauss(0, value * std_dev)
    
    def simulate_worn_dimensions(self, 
                                  ideal_dimensions: Dict,
                                  wear_percentage: float = 0.05) -> Dict:
        """
        Simulate worn part dimensions
        
        Wear patterns depend on part type:
        - Gears: Tooth flank wear
        - Shafts: Journal wear (smaller diameter)
        - Brackets: Bolt hole elongation
        
        Args:
            ideal_dimensions: Perfect part dimensions
            wear_percentage: How much wear (0.05 = 5%)
        
        Returns:
            Worn dimensions with noise
        """
        worn = {}
        
        for dim_name, ideal_value in ideal_dimensions.items():
            # Apply wear (systematic deviation)
            wear_factor = 1.0 - random.uniform(0, wear_percentage)
            worn_value = ideal_value * wear_factor
            
            # Add measurement noise (parametric perturbation)
            worn_value = self.apply_gaussian_noise(worn_value, std_dev=0.002)
            
            worn[dim_name] = round(worn_value, 4)
        
        return worn
    
    # =========================================================================
    # PILLAR 2: PHYSICS-BASED FAILURE
    # =========================================================================
    
    def calculate_tool_wear_rate(self, 
                                  material: str,
                                  cutting_speed: float,
                                  temperature: float) -> float:
        """
        Taylor tool life equation: VT^n = C
        
        Args:
            material: Material being cut
            cutting_speed: m/min
            temperature: °C
        
        Returns:
            Wear rate per second
        """
        mat_props = MATERIALS[material]
        
        # Base wear from material
        base_wear = 0.001 * mat_props['tool_wear_factor']
        
        # Speed increases wear exponentially
        speed_factor = (cutting_speed / mat_props['cutting_speed']) ** 0.5
        
        # Temperature accelerates wear
        temp_factor = 1.0 + ((temperature - 25) / 100) ** 2
        
        return base_wear * speed_factor * temp_factor
    
    def simulate_chatter_physics(self, 
                                  rpm: int,
                                  tool_health: float,
                                  natural_frequency: int = 2500) -> float:
        """
        Chatter occurs at harmonics of natural frequency
        Worse with worn tools
        
        Args:
            rpm: Spindle speed
            tool_health: 0.0-1.0
            natural_frequency: Machine natural frequency
        
        Returns:
            Chatter amplitude
        """
        # Check if near harmonic
        harmonics = [natural_frequency / i for i in range(1, 6)]
        
        chatter_amplitude = 0.0
        
        for harmonic in harmonics:
            if abs(rpm - harmonic) < 100:
                # Near harmonic - chatter!
                proximity = 1.0 - abs(rpm - harmonic) / 100
                chatter_amplitude += 0.5 * proximity
        
        # Worn tools chatter more
        tool_wear_factor = (1.0 - tool_health) * 2.0
        
        return chatter_amplitude * (1.0 + tool_wear_factor)
    
    # =========================================================================
    # PILLAR 3: LLM-DRIVEN CONTEXT
    # =========================================================================
    
    def generate_project_context(self, use_llm: bool = False) -> SyntheticProject:
        """
        Generate realistic project context
        
        If LLM available: Use it to hallucinate realistic backstory
        If not: Use templates with randomization
        
        Args:
            use_llm: Whether to use LLM for generation
        
        Returns:
            Complete project specification
        """
        vendor = random.choice(VENDORS)
        part_type = random.choice(list(PART_TYPES.keys()))
        material = random.choice(list(MATERIALS.keys()))
        scenario = random.choice(FAILURE_SCENARIOS)
        
        # Generate ideal dimensions
        base_size = random.uniform(50.0, 500.0)
        ideal_dims = self._generate_ideal_dimensions(part_type, base_size)
        
        # Simulate wear
        worn_dims = self.simulate_worn_dimensions(ideal_dims, wear_percentage=random.uniform(0.03, 0.15))
        
        # Calculate deviation
        deviation = {
            key: round(ideal_dims[key] - worn_dims.get(key, ideal_dims[key]), 4)
            for key in ideal_dims.keys()
        }
        
        dimensions = {
            "ideal": ideal_dims,
            "scanned": worn_dims,
            "deviation": deviation,
            "max_deviation": max(deviation.values())
        }
        
        # Complexity based on material and part type
        mat_props = MATERIALS[material]
        part_props = PART_TYPES[part_type]
        
        complexity = part_props['complexity_base']
        complexity += mat_props['stress_factor'] * 0.1
        complexity = min(1.0, complexity)
        
        # Estimate cycle time
        estimated_time = self._estimate_cycle_time(material, complexity, base_size)
        
        project = SyntheticProject(
            project_id=f"PROJ-{uuid.uuid4().hex[:8].upper()}",
            name=f"{vendor}_{part_type}_REPL_{random.randint(100, 999)}",
            part_number=f"PN-{uuid.uuid4().hex[:8].upper()}",
            vendor=vendor,
            material=material,
            part_type=part_type,
            complexity=round(complexity, 3),
            dimensions=dimensions,
            failure_scenario=scenario,
            created_at=self.current_time.isoformat(),
            estimated_cycle_time=round(estimated_time, 2)
        )
        
        return project
    
    def _generate_ideal_dimensions(self, part_type: str, base_size: float) -> Dict:
        """Generate ideal dimensions based on part type"""
        if "Gear" in part_type:
            return {
                "pitch_diameter": round(base_size, 3),
                "root_diameter": round(base_size * 0.9, 3),
                "face_width": round(base_size * 0.3, 3),
                "bore_diameter": round(base_size * 0.4, 3)
            }
        elif "Shaft" in part_type:
            return {
                "diameter": round(base_size * 0.2, 3),
                "length": round(base_size, 3),
                "keyway_width": round(base_size * 0.05, 3)
            }
        elif "Bracket" in part_type:
            return {
                "length": round(base_size, 3),
                "width": round(base_size * 0.6, 3),
                "thickness": round(base_size * 0.1, 3),
                "hole_diameter": round(base_size * 0.15, 3)
            }
        else:
            return {
                "outer_diameter": round(base_size, 3),
                "inner_diameter": round(base_size * 0.6, 3),
                "height": round(base_size * 0.4, 3)
            }
    
    def _estimate_cycle_time(self, material: str, complexity: float, size: float) -> float:
        """Estimate machining time"""
        mat_props = MATERIALS[material]
        
        # Base time from size
        base_time = (size / 100) * 10  # 10 minutes per 100mm
        
        # Material factor
        material_factor = mat_props['tool_wear_factor']
        
        # Complexity factor
        complexity_factor = 1.0 + complexity
        
        return base_time * material_factor * complexity_factor
    
    # =========================================================================
    # BIOCHEMICAL SIMULATION
    # =========================================================================
    
    def update_biochemical_state(self,
                                  state: BiochemicalState,
                                  vibration: float,
                                  tool_health: float,
                                  quality_pass: bool,
                                  anomaly: bool) -> BiochemicalState:
        """
        Update biochemical state based on operation conditions
        
        RULES:
        - High vibration → Cortisol rises (stress)
        - Smooth operation → Dopamine rises (flow)
        - Quality parts → Serotonin rises (satisfaction)
        - Anomalies → Adrenaline spike (alert)
        
        Args:
            state: Current biochemical state
            vibration: Current vibration level
            tool_health: Tool condition
            quality_pass: Whether part passed quality
            anomaly: Whether anomaly detected
        
        Returns:
            Updated biochemical state
        """
        # Cortisol (stress) response
        if vibration > 2.0 or tool_health < 0.3:
            state.cortisol = min(100, state.cortisol + 5.0)
        else:
            state.cortisol = max(0, state.cortisol - 1.0)
        
        # Dopamine (flow) response
        if vibration < 1.0 and tool_health > 0.7:
            state.dopamine = min(100, state.dopamine + 2.0)
        else:
            state.dopamine = max(0, state.dopamine - 1.0)
        
        # Serotonin (satisfaction) response
        if quality_pass:
            state.serotonin = min(100, state.serotonin + 0.5)
        else:
            state.serotonin = max(0, state.serotonin - 3.0)
        
        # Adrenaline (alerts) response
        if anomaly:
            state.adrenaline = min(100, state.adrenaline + 20.0)
        else:
            state.adrenaline = max(0, state.adrenaline - 5.0)
        
        return state
    
    # =========================================================================
    # COMPLETE SIMULATION
    # =========================================================================
    
    def simulate_production_run(self,
                                 project: SyntheticProject,
                                 duration_seconds: Optional[float] = None,
                                 sample_rate_hz: int = 10) -> List[TelemetryPoint]:
        """
        Simulate complete machining operation with biochemical responses
        
        Args:
            project: Project specification
            duration_seconds: Override cycle time
            sample_rate_hz: Sampling frequency
        
        Returns:
            List of telemetry points
        """
        if duration_seconds is None:
            duration_seconds = project.estimated_cycle_time * 60  # Convert to seconds
        
        # Initial conditions
        mat_props = MATERIALS[project.material]
        base_rpm = int(mat_props['cutting_speed'] * 1000 / (math.pi * 100))  # Assume 100mm tool
        
        tool_health = 1.0
        temperature = 25.0  # Room temp
        
        # Initialize biochemical state
        bio_state = BiochemicalState(
            cortisol=10.0,
            dopamine=50.0,
            serotonin=75.0,
            adrenaline=5.0
        )
        
        telemetry = []
        num_samples = int(duration_seconds * sample_rate_hz)
        
        for i in range(num_samples):
            time_sec = i / sample_rate_hz
            timestamp = self.current_time + datetime.timedelta(seconds=time_sec)
            
            # Tool wear (physics-based)
            wear_rate = self.calculate_tool_wear_rate(project.material, mat_props['cutting_speed'], temperature)
            tool_health = max(0.0, tool_health - wear_rate)
            
            # Temperature rises
            temp_rise_rate = 0.3 * mat_props['stress_factor']  # °C per minute
            temperature += (temp_rise_rate / 60) / sample_rate_hz
            
            # Vibration (physics + noise)
            base_vibration = (1.0 - tool_health) * 3.0
            chatter = self.simulate_chatter_physics(base_rpm, tool_health)
            
            vib_x = base_vibration + chatter + self.apply_gaussian_noise(0.1, 0.02)
            vib_y = base_vibration * 0.8 + chatter * 0.9 + self.apply_gaussian_noise(0.08, 0.02)
            vib_z = base_vibration * 0.6 + chatter * 0.7 + self.apply_gaussian_noise(0.06, 0.02)
            
            # Spindle load increases with wear
            spindle_load = 30 + (1.0 - tool_health) * 40 + self.apply_gaussian_noise(10, 0.05)
            spindle_load = max(0, min(100, spindle_load))
            
            # Anomaly detection
            anomaly = (vib_x > 5.0) or (spindle_load > 90) or (tool_health < 0.2)
            
            # Quality check
            quality_pass = tool_health > 0.3 and vib_x < 3.0
            
            # Update biochemical state
            bio_state = self.update_biochemical_state(
                bio_state,
                vibration=vib_x,
                tool_health=tool_health,
                quality_pass=quality_pass,
                anomaly=anomaly
            )
            
            # Signal status
            if anomaly:
                signal = "ALERT"
            elif vib_x > 2.0:
                signal = "WARNING"
            else:
                signal = "OK"
            
            # Create telemetry point
            point = TelemetryPoint(
                timestamp=timestamp.isoformat(),
                time_seconds=round(time_sec, 3),
                rpm=int(base_rpm + self.apply_gaussian_noise(0, 50)),
                spindle_load=round(spindle_load, 2),
                vibration_x=round(vib_x, 4),
                vibration_y=round(vib_y, 4),
                vibration_z=round(vib_z, 4),
                temperature_c=round(temperature, 2),
                tool_health=round(tool_health, 4),
                tool_id="T01",
                cortisol=round(bio_state.cortisol, 2),
                dopamine=round(bio_state.dopamine, 2),
                serotonin=round(bio_state.serotonin, 2),
                adrenaline=round(bio_state.adrenaline, 2),
                signal=signal,
                anomaly_detected=anomaly,
                machine_id=1
            )
            
            telemetry.append(point)
        
        return telemetry


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Reverse Engineering as a Service (REaaS) - Simulation")
    print("=" * 70)
    
    simulator = ReverseEngineeringSimulator()
    
    # Generate project
    print("\n📋 Generating project context...")
    project = simulator.generate_project_context()
    
    print(f"\nProject: {project.name}")
    print(f"Vendor: {project.vendor}")
    print(f"Material: {project.material}")
    print(f"Scenario: {project.failure_scenario}")
    print(f"Complexity: {project.complexity}")
    print(f"\nDimensions:")
    print(f"  Max Deviation: {project.dimensions['max_deviation']:.4f} mm")
    
    # Simulate production
    print(f"\n🏭 Simulating production ({project.estimated_cycle_time:.1f} minutes)...")
    telemetry = simulator.simulate_production_run(project, duration_seconds=60, sample_rate_hz=1)
    
    print(f"\n📊 Telemetry Summary ({len(telemetry)} points):")
    print(f"  Final Tool Health: {telemetry[-1].tool_health*100:.1f}%")
    print(f"  Final Temperature: {telemetry[-1].temperature_c:.1f}°C")
    print(f"  Final Cortisol: {telemetry[-1].cortisol:.1f}")
    print(f"  Final Dopamine: {telemetry[-1].dopamine:.1f}")
    print(f"  Anomalies Detected: {sum(1 for t in telemetry if t.anomaly_detected)}")
    
    # Show sample data
    print("\n📈 Sample Telemetry (first 3 points):")
    for point in telemetry[:3]:
        print(json.dumps(asdict(point), indent=2))
