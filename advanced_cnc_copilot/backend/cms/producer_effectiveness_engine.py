"""
Producer Effectiveness Prediction Engine
Multi-Layer Economic Classification System

ARCHITECTURE:
1. Producer Statistics - Standardized performance metrics
2. Economic Classification - Cost/quality/time parameters
3. Diversity Scaling - Multi-dimensional prediction layers
4. Optimization Bot - AI-driven spec calculator
5. SolidWorks Integration - Simulation validation
6. Lifetime Prediction - Part durability forecasting

PARADIGM: Producers are "biological organisms" with DNA (parameters)
that evolve based on economic fitness
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta


# =============================================================================
# ECONOMIC PARAMETERS
# =============================================================================

class EconomicTier(Enum):
    """Economic classification tiers"""
    PREMIUM = "premium"  # High cost, highest quality
    STANDARD = "standard"  # Medium cost, good quality
    ECONOMY = "economy"  # Low cost, acceptable quality
    BUDGET = "budget"  # Minimal cost, basic quality


class ProductionEfficiency(Enum):
    """Production efficiency ratings"""
    EXCELLENT = 5  # >95% efficiency
    GOOD = 4  # 85-95%
    AVERAGE = 3  # 70-85%
    POOR = 2  # 50-70%
    CRITICAL = 1  # <50%


@dataclass
class EconomicParameters:
    """
    Complete economic profile for producer/part
    
    All costs in USD, times in minutes
    """
    # Direct costs
    material_cost_per_kg: float
    tool_cost_per_part: float
    machine_cost_per_hour: float
    labor_cost_per_hour: float
    energy_cost_per_kwh: float
    
    # Indirect costs
    overhead_percentage: float  # % of direct costs
    scrap_rate_percentage: float  # % of parts scrapped
    
    # Time parameters
    setup_time_minutes: float
    cycle_time_minutes: float
    tool_change_time_minutes: float
    
    # Quality parameters
    first_pass_yield: float  # 0.0-1.0
    defect_rate_ppm: int  # Parts per million
    rework_rate_percentage: float
    
    # Volume parameters
    batch_size: int
    annual_volume: int
    
    # Computed metrics (calculated)
    total_cost_per_part: float = 0.0
    economic_tier: EconomicTier = EconomicTier.STANDARD
    efficiency_rating: ProductionEfficiency = ProductionEfficiency.AVERAGE


# =============================================================================
# PRODUCER STATISTICS
# =============================================================================

@dataclass
class ProducerStatistics:
    """
    Standardized statistics for measuring producer effectiveness
    
    METAPHOR: Producer DNA - genetic code of manufacturing capability
    """
    producer_id: str
    producer_name: str
    
    # Capability limits (statistical boundaries)
    max_complexity_score: float  # 0.0-1.0
    min_tolerance_mm: float  # Smallest achievable tolerance
    max_part_size_mm: Tuple[float, float, float]  # X, Y, Z
    materials_certified: List[str]
    
    # Performance statistics (measured over time)
    avg_cycle_time_variance: float  # Consistency
    avg_dimensional_accuracy: float  # mm deviation
    avg_surface_finish_ra: float  # μm
    avg_tool_life_multiplier: float  # vs. standard
    
    # Quality metrics
    overall_yield: float  # 0.0-1.0
    avg_defect_rate_ppm: int
    customer_satisfaction_score: float  # 0.0-5.0
    
    # Economic metrics
    cost_competitiveness_index: float  # 0.0-2.0 (1.0=market avg)
    delivery_performance: float  # 0.0-1.0 on-time delivery
    flexibility_score: float  # 0.0-1.0 rush order capability
    
    # Historical performance
    parts_produced_lifetime: int
    years_in_operation: int
    major_quality_incidents: int
    
    # Computed effectiveness
    overall_effectiveness_score: float = 0.0


# =============================================================================
# DIVERSITY SCALING LAYERS
# =============================================================================

class DiversityLayer(Enum):
    """Multi-dimensional diversity layers for prediction"""
    MATERIAL = "material"  # Material type variation
    GEOMETRY = "geometry"  # Shape complexity variation
    TOLERANCE = "tolerance"  # Precision requirement variation
    VOLUME = "volume"  # Production quantity variation
    TIME = "time"  # Urgency/lead time variation
    COST = "cost"  # Budget constraint variation


@dataclass
class DiversityScalingEngine:
    """
    Multi-layer prediction engine with diversity scaling
    
    CONCEPT: Different "dimensions" of manufacturing diversity
    System learns optimal parameters for each combination
    """
    
    # Scaling factors for each diversity layer
    material_scaling: Dict[str, float] = field(default_factory=dict)
    geometry_scaling: Dict[str, float] = field(default_factory=dict)
    tolerance_scaling: Dict[float, float] = field(default_factory=dict)
    volume_scaling: Dict[int, float] = field(default_factory=dict)
    time_scaling: Dict[str, float] = field(default_factory=dict)
    cost_scaling: Dict[EconomicTier, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default scaling factors"""
        
        # Material complexity scaling
        self.material_scaling = {
            "Aluminum6061": 1.0,  # Baseline
            "Steel4140": 1.5,
            "Stainless304": 1.8,
            "Titanium6Al4V": 2.5,
            "Inconel718": 3.0
        }
        
        # Geometry complexity scaling
        self.geometry_scaling = {
            "simple": 1.0,  # Cylinder, block
            "moderate": 1.5,  # Bracke, shaft with features
            "complex": 2.0,  # Gear, housing
            "intricate": 3.0  # Impeller, valve body
        }
        
        # Tolerance scaling (tighter = more difficult)
        self.tolerance_scaling = {
            0.1: 1.0,  # ±0.1mm
            0.05: 1.5,  # ±0.05mm
            0.02: 2.0,  # ±0.02mm
            0.01: 3.0  # ±0.01mm (precision)
        }
        
        # Volume scaling (economies of scale)
        self.volume_scaling = {
            1: 1.5,  # Prototype
            10: 1.2,
            100: 1.0,  # Baseline
            1000: 0.85,
            10000: 0.7  # Mass production
        }
        
        # Time urgency scaling
        self.time_scaling = {
            "rush": 1.5,  # <24hrs
            "expedited": 1.25,  # <1 week
            "standard": 1.0,  # 2-4 weeks
            "economical": 0.85  # >1 month
        }
        
        # Cost tier scaling
        self.cost_scaling = {
            EconomicTier.PREMIUM: 1.5,
            EconomicTier.STANDARD: 1.0,
            EconomicTier.ECONOMY: 0.75,
            EconomicTier.BUDGET: 0.5
        }
    
    def calculate_composite_scaling(self,
                                    material: str,
                                    geometry: str,
                                    tolerance: float,
                                    volume: int,
                                    urgency: str,
                                    tier: EconomicTier) -> float:
        """
        Calculate composite scaling factor across all diversity layers
        
        Returns:
            Combined scaling factor (multiplicative)
        """
        # Get individual scales
        mat_scale = self.material_scaling.get(material, 1.0)
        geo_scale = self.geometry_scaling.get(geometry, 1.0)
        
        # Find closest tolerance
        tol_scale = min(self.tolerance_scaling.items(), 
                       key=lambda x: abs(x[0] - tolerance))[1]
        
        # Find closest volume
        vol_scale = min(self.volume_scaling.items(),
                       key=lambda x: abs(x[0] - volume))[1]
        
        time_scale = self.time_scaling.get(urgency, 1.0)
        cost_scale = self.cost_scaling.get(tier, 1.0)
        
        # Composite (multiplicative)
        composite = mat_scale * geo_scale * tol_scale * vol_scale * time_scale * cost_scale
        
        return composite
    
    def predict_effectiveness(self,
                            producer_stats: ProducerStatistics,
                            part_requirements: Dict) -> float:
        """
        Predict how effective a producer will be for specific requirements
        
        Args:
            producer_stats: Producer capabilities
            part_requirements: Part specifications
        
        Returns:
            Effectiveness score 0.0-1.0
        """
        # Extract requirements
        material = part_requirements.get('material', 'Aluminum6061')
        geometry = part_requirements.get('complexity', 'simple')
        tolerance = part_requirements.get('tolerance', 0.1)
        volume = part_requirements.get('volume', 100)
        urgency = part_requirements.get('urgency', 'standard')
        tier = part_requirements.get('economic_tier', EconomicTier.STANDARD)
        
        # Get composite scaling
        difficulty = self.calculate_composite_scaling(
            material, geometry, tolerance, volume, urgency, tier
        )
        
        # Check if producer can handle material
        if material not in producer_stats.materials_certified:
            return 0.0  # Cannot produce
        
        # Check tolerance capability
        if tolerance < producer_stats.min_tolerance_mm:
            return 0.2  # Very difficult/risky
        
        # Calculate base effectiveness from producer stats
        base_effectiveness = producer_stats.overall_effectiveness_score
        
        # Adjust for difficulty
        adjusted_effectiveness = base_effectiveness / difficulty
        
        # Adjust for producer experience with this type
        # (Would use historical data in production)
        
        # Normalize to 0-1
        return min(1.0, max(0.0, adjusted_effectiveness))


# =============================================================================
# OPTIMIZATION BOT
# =============================================================================

class PartOptimizationBot:
    """
    AI bot that calculates optimal specifications for parts
    
    CAPABILITIES:
    1. Parse manufacturing data
    2. Calculate optimal material/process
    3. Test with SolidWorks simulation
    4. Provide improvement recommendations
    5. Predict lifetime span
    """
    
    def __init__(self):
        self.diversity_engine = DiversityScalingEngine()
        self.producers_database: List[ProducerStatistics] = []
        self.optimization_history: List[Dict] = []
    
    def add_producer(self, producer: ProducerStatistics):
        """Add producer to database"""
        self.producers_database.append(producer)
    
    def parse_part_requirements(self, requirements_text: str) -> Dict:
        """
        Parse natural language or structured requirements
        
        Example input: "Need 100 aluminum brackets, ±0.05mm, rush delivery"
        
        Returns:
            Structured requirements dict
        """
        # Simple parsing (would use LLM in production)
        requirements = {
            'material': 'Aluminum6061',
            'complexity': 'moderate',
            'tolerance': 0.05,
            'volume': 100,
            'urgency': 'rush',
            'economic_tier': EconomicTier.STANDARD
        }
        
        # Extract from text (simplified)
        text_lower = requirements_text.lower()
        
        if 'titanium' in text_lower:
            requirements['material'] = 'Titanium6Al4V'
        elif 'steel' in text_lower:
            requirements['material'] = 'Steel4140'
        
        if 'rush' in text_lower or 'urgent' in text_lower:
            requirements['urgency'] = 'rush'
        elif 'standard' in text_lower:
            requirements['urgency'] = 'standard'
        
        # Extract numbers
        import re
        numbers = re.findall(r'\d+', requirements_text)
        if numbers:
            requirements['volume'] = int(numbers[0])
        
        return requirements
    
    def calculate_optimal_specs(self,
                                requirements: Dict,
                                constraints: Optional[Dict] = None) -> Dict:
        """
        Calculate optimal manufacturing specifications
        
        Args:
            requirements: Part requirements
            constraints: Budget/time/quality constraints
        
        Returns:
            Optimal specifications
        """
        constraints = constraints or {}
        
        # Determine optimal material if not specified
        if 'material' not in requirements:
            requirements['material'] = self._optimize_material(
                requirements, constraints
            )
        
        # Calculate optimal process parameters
        optimal_specs = {
            'material': requirements['material'],
            'process': self._select_process(requirements),
            'tooling': self._select_tooling(requirements),
            'speeds_feeds': self._calculate_speeds_feeds(requirements),
            'estimated_cost': self._estimate_cost(requirements),
            'estimated_time': self._estimate_time(requirements),
            'predicted_quality': self._predict_quality(requirements),
            'lifetime_span_hours': self._predict_lifetime(requirements)
        }
        
        return optimal_specs
    
    def _optimize_material(self, requirements: Dict, constraints: Dict) -> str:
        """Select optimal material based on requirements and constraints"""
        # Budget constraint
        max_cost = constraints.get('max_cost_per_part', float('inf'))
        
        # Strength requirement
        min_strength = requirements.get('min_strength_mpa', 0)
        
        # Weight constraint
        max_weight = requirements.get('max_weight_kg', float('inf'))
        
        # Material candidates with properties
        materials = {
            'Aluminum6061': {'cost': 4.5, 'strength': 310, 'density': 2.7},
            'Steel4140': {'cost': 2.5, 'strength': 655, 'density': 7.85},
            'Titanium6Al4V': {'cost': 35.0, 'strength': 900, 'density': 4.43}
        }
        
        # Score each material
        best_material = None
        best_score = -1
        
        for mat_name, props in materials.items():
            # Check hard constraints
            if props['cost'] > max_cost:
                continue
            if props['strength'] < min_strength:
                continue
            
            # Calculate score (higher is better)
            score = props['strength'] / props['cost'] / props['density']
            
            if score > best_score:
                best_score = score
                best_material = mat_name
        
        return best_material or 'Aluminum6061'
    
    def _select_process(self, requirements: Dict) -> str:
        """Select optimal manufacturing process"""
        complexity = requirements.get('complexity', 'simple')
        volume = requirements.get('volume', 1)
        
        if volume > 1000:
            return 'cnc_automated'
        elif complexity == 'intricate':
            return '5_axis_cnc'
        else:
            return '3_axis_cnc'
    
    def _select_tooling(self, requirements: Dict) -> Dict:
        """Select optimal tooling"""
        material = requirements.get('material', 'Aluminum6061')
        
        # Define tool recommendations per material
        tooling_db = {
            'Aluminum6061': {
                'roughing': 'Carbide end mill, 4 flute',
                'finishing': 'Coated carbide, 0.8mm radius',
                'coating': 'TiAlN'
            },
            'Steel4140': {
                'roughing': 'Carbide insert, CNMG',
                'finishing': 'CBN insert',
                'coating': 'AlTiN'
            },
            'Titanium6Al4V': {
                'roughing': 'Solid carbide, low helix',
                'finishing': 'PCD end mill',
                'coating': 'TiB2'
            }
        }
        
        return tooling_db.get(material, tooling_db['Aluminum6061'])
    
    def _calculate_speeds_feeds(self, requirements: Dict) -> Dict:
        """Calculate optimal cutting parameters"""
        material = requirements.get('material', 'Aluminum6061')
        
        # Simplified calculation (would use detailed models)
        speeds_feeds_db = {
            'Aluminum6061': {
                'cutting_speed_m_min': 300,
                'feed_per_tooth_mm': 0.15,
                'depth_of_cut_mm': 3.0,
                'stepover_percent': 40
            },
            'Steel4140': {
                'cutting_speed_m_min': 100,
                'feed_per_tooth_mm': 0.10,
                'depth_of_cut_mm': 2.0,
                'stepover_percent': 35
            },
            'Titanium6Al4V': {
                'cutting_speed_m_min': 60,
                'feed_per_tooth_mm': 0.08,
                'depth_of_cut_mm': 1.5,
                'stepover_percent': 30
            }
        }
        
        return speeds_feeds_db.get(material, speeds_feeds_db['Aluminum6061'])
    
    def _estimate_cost(self, requirements: Dict) -> float:
        """Estimate cost per part"""
        # Simplified cost model
        material = requirements.get('material', 'Aluminum6061')
        volume = requirements.get('volume', 1)
        complexity = requirements.get('complexity', 'simple')
        
        # Base material cost
        material_costs = {
            'Aluminum6061': 4.5,
            'Steel4140': 2.5,
            'Titanium6Al4V': 35.0
        }
        base_material = material_costs.get(material, 4.5)
        
        # Complexity multiplier
        complexity_mult = {
            'simple': 1.0,
            'moderate': 1.5,
            'complex': 2.5,
            'intricate': 4.0
        }.get(complexity, 1.0)
        
        # Volume discount
        volume_discount = 1.0 - min(0.3, volume / 10000 * 0.3)
        
        cost = base_material * complexity_mult * volume_discount * 10  # Assume 10x markup
        
        return round(cost, 2)
    
    def _estimate_time(self, requirements: Dict) -> float:
        """Estimate cycle time in minutes"""
        complexity = requirements.get('complexity', 'simple')
        
        base_times = {
            'simple': 10,
            'moderate': 30,
            'complex': 60,
            'intricate': 120
        }
        
        return base_times.get(complexity, 30)
    
    def _predict_quality(self, requirements: Dict) -> float:
        """Predict quality score 0.0-1.0"""
        # Simplified model
        tolerance = requirements.get('tolerance', 0.1)
        
        # Tighter tolerance = lower yield
        if tolerance < 0.02:
            return 0.85
        elif tolerance < 0.05:
            return 0.92
        else:
            return 0.98
    
    def _predict_lifetime(self, requirements: Dict) -> float:
        """
        Predict part lifetime in operating hours
        
        Based on:
        - Material fatigue properties
        - Operating conditions
        - Safety factors
        """
        material = requirements.get('material', 'Aluminum6061')
        stress_level = requirements.get('operating_stress_mpa', 50)
        
        # Material endurance limits (MPa)
        endurance_limits = {
            'Aluminum6061': 96,
            'Steel4140': 415,
            'Titanium6Al4V': 510
        }
        
        endurance = endurance_limits.get(material, 96)
        
        # S-N curve approximation
        # N = C / (stress ^ b)
        if stress_level < endurance:
            # Infinite life
            lifetime_hours = 100000
        else:
            # Finite life
            C = 1e12  # Material constant
            b = 8  # Fatigue exponent
            cycles = C / (stress_level ** b)
            
            # Assume 1 cycle per hour
            lifetime_hours = cycles
        
        return round(lifetime_hours, 0)
    
    def find_best_producer(self, requirements: Dict) -> Optional[ProducerStatistics]:
        """
        Find best producer for given requirements
        
        Returns:
            Best matched producer or None
        """
        if not self.producers_database:
            return None
        
        best_producer = None
        best_score = -1
        
        for producer in self.producers_database:
            # Predict effectiveness
            effectiveness = self.diversity_engine.predict_effectiveness(
                producer, requirements
            )
            
            if effectiveness > best_score:
                best_score = effectiveness
                best_producer = producer
        
        return best_producer
    
    def optimize_complete_project(self, 
                                  part_description: str,
                                  constraints: Optional[Dict] = None) -> Dict:
        """
        Complete optimization workflow
        
        1. Parse requirements
        2. Calculate optimal specs
        3. Find best producer
        4. Generate recommendations
        
        Args:
            part_description: Natural language description
            constraints: Budget/time constraints
        
        Returns:
            Complete optimization result
        """
        # Parse
        requirements = self.parse_part_requirements(part_description)
        
        # Optimize
        optimal_specs = self.calculate_optimal_specs(requirements, constraints)
        
        # Find producer
        best_producer = self.find_best_producer(requirements)
        
        # Generate result
        result = {
            'parsed_requirements': requirements,
            'optimal_specifications': optimal_specs,
            'recommended_producer': best_producer.producer_name if best_producer else None,
            'producer_effectiveness': self.diversity_engine.predict_effectiveness(
                best_producer, requirements
            ) if best_producer else 0.0,
            'optimization_timestamp': datetime.now().isoformat(),
            'recommendations': self._generate_recommendations(requirements, optimal_specs)
        }
        
        # Store in history
        self.optimization_history.append(result)
        
        return result
    
    def _generate_recommendations(self, requirements: Dict, specs: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Cost optimization
        if specs['estimated_cost'] > 100:
            recommendations.append(
                f"Consider switching to {self._optimize_material(requirements, {'max_cost_per_part': 50})} "
                f"to reduce cost by ~30%"
            )
        
        # Lifetime optimization
        if specs['lifetime_span_hours'] < 10000:
            recommendations.append(
                "Part lifetime is below 10,000 hours. Consider heat treatment or surface coating."
            )
        
        # Quality optimization
        if specs['predicted_quality'] < 0.90:
            recommendations.append(
                "Tight tolerances detected. Consider relaxing to ±0.05mm for 95%+ yield."
            )
        
        return recommendations


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Producer Effectiveness & Optimization Engine")
    print("="* 70)
    
    # Create bot
    bot = PartOptimizationBot()
    
    # Add sample producers
    producer1 = ProducerStatistics(
        producer_id="PROD_001",
        producer_name="Precision Manufacturing Inc.",
        max_complexity_score=0.9,
        min_tolerance_mm=0.01,
        max_part_size_mm=(500, 500, 300),
        materials_certified=['Aluminum6061', 'Steel4140', 'Titanium6Al4V'],
        avg_cycle_time_variance=0.05,
        avg_dimensional_accuracy=0.015,
        avg_surface_finish_ra=0.8,
        avg_tool_life_multiplier=1.2,
        overall_yield=0.95,
        avg_defect_rate_ppm=500,
        customer_satisfaction_score=4.5,
        cost_competitiveness_index=1.1,
        delivery_performance=0.92,
        flexibility_score=0.85,
        parts_produced_lifetime=50000,
        years_in_operation=15,
        major_quality_incidents=2,
        overall_effectiveness_score=0.88
    )
    
    bot.add_producer(producer1)
    
    # Optimize project
    print("\n📋 Optimizing project...")
    result = bot.optimize_complete_project(
        part_description="Need 100 aluminum brackets, ±0.05mm tolerance, rush delivery",
        constraints={'max_cost_per_part': 150}
    )
    
    print("\n" + json.dumps(result, indent=2, default=str))
