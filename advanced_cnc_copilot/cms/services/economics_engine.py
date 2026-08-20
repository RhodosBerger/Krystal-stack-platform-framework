from typing import Dict, List, Any
from datetime import datetime, timedelta
import logging
import math

from ..repositories.telemetry_repository import TelemetryRepository

logger = logging.getLogger(__name__)


class EconomicsEngine:
    """
    Implements 'The Great Translation': Mapping SaaS metrics to Manufacturing Physics
    Churn → Tool Wear, CAC → Setup Time
    Optimizes for Profit Rate (Pr) rather than just cycle time
    """
    
    def __init__(self, repository: TelemetryRepository):
        self.repository = repository
        self.base_power_cost_per_kwh = 0.12  # $0.12 per kWh
        self.labor_cost_per_hour = 35.0     # $35 per labor hour
        self.tool_cost_per_hour = 2.5       # $2.50 per hour of operation
        self.maintenance_cost_per_hour = 1.0 # $1.00 per hour
        self.base_machine_depreciation_per_hour = 5.0  # $5.00 per hour
    
    def calculate_profit_rate(self, sales_price: float, costs: Dict[str, float], time_hours: float) -> float:
        """
        Calculate Profit Rate: Pr = (Sales_Price - Cost) / Time
        Implements the directive requirement for economic optimization
        """
        total_cost = sum(costs.values())
        profit = sales_price - total_cost
        profit_rate = profit / time_hours if time_hours > 0 else 0.0
        
        return profit_rate
    
    def calculate_churn_risk(self, tool_wear_rate: float, tool_life_hours: float = 100.0) -> float:
        """
        Map tool wear rate to a 'Churn Score' as required by directive
        Higher scores indicate higher risk of tool failure/quality issues (like customer churn)
        """
        # Calculate tool wear as percentage of total tool life
        max_acceptable_wear_rate = 1.0 / tool_life_hours  # Max wear rate before tool needs replacement
        churn_score = min(1.0, tool_wear_rate / max_acceptable_wear_rate)
        
        return churn_score
    
    def get_operational_mode(self, churn_score: float, profit_rate: float) -> str:
        """
        Implement logic switch: If Churn Score > Threshold, switch to ECONOMY_MODE
        Otherwise, allow RUSH_MODE to preserve assets while maximizing productivity
        """
        churn_threshold = 0.7  # High risk threshold
        profit_threshold = 10.0  # High profit rate threshold per hour
        
        if churn_score > churn_threshold:
            return "ECONOMY_MODE"  # Conservative to protect equipment
        elif profit_rate > profit_threshold and churn_score < 0.5:
            return "RUSH_MODE"     # Aggressive when safe and profitable
        else:
            return "BALANCED_MODE"  # Moderate approach
    
    def analyze_job_economics(self, job_data: Dict) -> Dict:
        """
        Analyze the economic aspects of a specific job
        Returns metrics for decision making
        """
        # Extract job parameters
        estimated_time_hours = job_data.get('estimated_duration_hours', 1.0)
        actual_time_hours = job_data.get('actual_duration_hours', estimated_time_hours)
        sales_price = job_data.get('sales_price', 100.0)
        material_cost = job_data.get('material_cost', 20.0)
        labor_hours = job_data.get('labor_hours', actual_time_hours)
        machine_id = job_data.get('machine_id', 1)
        
        # Calculate tool wear based on telemetry data
        tool_wear_rate = self._calculate_tool_wear_rate(machine_id, actual_time_hours)
        
        # Calculate various costs
        labor_cost = labor_hours * self.labor_cost_per_hour
        tool_cost = actual_time_hours * self.tool_cost_per_hour
        maintenance_cost = actual_time_hours * self.maintenance_cost_per_hour
        energy_cost = self._calculate_energy_cost(machine_id, actual_time_hours)
        depreciation_cost = actual_time_hours * self.base_machine_depreciation_per_hour
        
        costs = {
            'material': material_cost,
            'labor': labor_cost,
            'tool_wear': tool_cost,
            'maintenance': maintenance_cost,
            'energy': energy_cost,
            'depreciation': depreciation_cost
        }
        
        # Calculate economic metrics
        profit_rate = self.calculate_profit_rate(sales_price, costs, actual_time_hours)
        churn_score = self.calculate_churn_risk(tool_wear_rate)
        operational_mode = self.get_operational_mode(churn_score, profit_rate)
        
        # Calculate efficiency metrics
        profit_margin = (sales_price - sum(costs.values())) / sales_price if sales_price > 0 else 0.0
        cost_per_part = sum(costs.values()) / job_data.get('part_count', 1)
        
        return {
            'profit_rate': profit_rate,
            'churn_score': churn_score,
            'operational_mode': operational_mode,
            'profit_margin': profit_margin,
            'cost_per_part': cost_per_part,
            'total_costs': costs,
            'estimated_vs_actual_time_ratio': actual_time_hours / estimated_time_hours if estimated_time_hours > 0 else 1.0,
            'recommendations': self._generate_recommendations(profit_rate, churn_score, profit_margin)
        }
    
    def _calculate_tool_wear_rate(self, machine_id: int, duration_hours: float) -> float:
        """
        Calculate tool wear rate based on telemetry data
        Uses historical data to estimate wear
        """
        if duration_hours <= 0:
            return 0.0
        
        # Get recent telemetry data to estimate wear
        recent_data = self.repository.get_recent_by_machine(machine_id, minutes=int(duration_hours * 60))
        
        if not recent_data:
            # Default wear rate if no data available
            return 0.01  # 0.01 units of wear per hour
        
        # Calculate average spindle load and vibration to estimate wear
        total_spindle_load = 0.0
        total_vibration = 0.0
        valid_records = 0
        
        for r in recent_data:
            try:
                spindle_load_val = getattr(r, 'spindle_load', 0.0) or 0.0
                if isinstance(spindle_load_val, (int, float)):
                    total_spindle_load += float(spindle_load_val)
                else:
                    total_spindle_load += 0.0
                
                vibration_val = getattr(r, 'vibration_x', 0.0) or 0.0
                if isinstance(vibration_val, (int, float)):
                    total_vibration += float(vibration_val)
                else:
                    total_vibration += 0.0
                
                valid_records += 1
            except Exception as e:
                logger.warning(f"Error processing telemetry record for wear calculation: {e}")
                continue
        
        if valid_records == 0:
            return 0.01  # Default wear rate if no valid records
        
        avg_spindle_load = total_spindle_load / valid_records
        avg_vibration = total_vibration / valid_records
        
        # Estimate wear rate based on operating conditions
        # Higher load and vibration increase wear
        base_wear_rate = 0.005  # Base wear rate per hour
        load_factor = max(0.5, min(3.0, avg_spindle_load / 50.0))  # Factor based on average load
        vibration_factor = max(0.8, min(2.0, 1.0 + (avg_vibration / 2.0)))  # Factor based on vibration
        
        estimated_wear_rate = base_wear_rate * load_factor * vibration_factor
        
        return min(0.1, estimated_wear_rate)  # Cap at 0.1 units/hour
    
    def _calculate_energy_cost(self, machine_id: int, duration_hours: float) -> float:
        """
        Calculate energy cost based on machine usage
        """
        # Get recent telemetry to estimate power consumption
        recent_data = self.repository.get_recent_by_machine(machine_id, minutes=int(duration_hours * 60))
        
        if not recent_data:
            # Default energy consumption if no data
            return duration_hours * 10.0 * self.base_power_cost_per_kwh  # 10 kW average
        
        # Calculate average power based on spindle load
        total_load = 0.0
        valid_records = 0
        
        for r in recent_data:
            try:
                spindle_load_val = getattr(r, 'spindle_load', 0.0) or 0.0
                if isinstance(spindle_load_val, (int, float)):
                    total_load += float(spindle_load_val)
                    valid_records += 1
                else:
                    total_load += 0.0
            except Exception as e:
                logger.warning(f"Error processing telemetry record for energy calculation: {e}")
                continue
        
        if valid_records == 0:
            avg_load = 50.0  # Default average load
        else:
            avg_load = total_load / valid_records
        
        # Estimate power consumption: base power + load-dependent power
        base_power_kw = 5.0  # kW when idle
        load_power_factor = 15.0  # Additional kW at 100% load
        avg_power_kw = base_power_kw + (avg_load / 100.0) * load_power_factor
        
        energy_consumption_kwh = avg_power_kw * duration_hours
        energy_cost = energy_consumption_kwh * self.base_power_cost_per_kwh
        
        return energy_cost
    
    def _generate_recommendations(self, profit_rate: float, churn_score: float, profit_margin: float) -> List[str]:
        """
        Generate economic recommendations based on calculated metrics
        """
        recommendations = []
        
        if churn_score > 0.7:
            recommendations.append("HIGH CHURN RISK: Consider switching to ECONOMY mode to reduce tool wear")
            recommendations.append("Schedule preventive maintenance soon to avoid unplanned downtime")
        
        if profit_rate < 5.0:
            recommendations.append("LOW PROFIT RATE: Consider optimizing process parameters or adjusting pricing")
            recommendations.append("Review material costs and supplier contracts")
        
        profit_margin_threshold = 0.15  # 15% minimum profit margin
        if profit_margin < profit_margin_threshold:
            recommendations.append(f"PROFIT MARGIN BELOW THRESHOLD ({profit_margin_threshold*100}%): Evaluate cost reduction opportunities")
        
        if churn_score < 0.3 and profit_rate > 15.0:
            recommendations.append("OPTIMAL CONDITIONS: Consider increasing production volume if demand allows")
            recommendations.append("RUSH mode may be appropriate for future similar jobs")
        
        if not recommendations:
            recommendations.append("Current operations appear economically optimal")
        
        return recommendations
    
    def calculate_quadratic_mantinel(self, path_segments: List[Dict]) -> Dict:
        """
        Implement the Quadratic Mantinel: PermissibleSpeed = f(Curvature^2)
        Use a 'Tolerance Band' approach to maintain momentum through corners
        """
        results = {
            'segments': [],
            'recommended_speed_profile': [],
            'total_path_time': 0.0
        }
        
        total_time = 0.0
        
        for i, segment in enumerate(path_segments):
            # Extract segment properties
            curvature = segment.get('curvature', 0.0)  # 1/mm, radius = 1/curvature
            segment_length = segment.get('length_mm', 10.0)  # Length of segment in mm
            material_properties = segment.get('material', {})
            
            # Calculate permissible speed based on curvature (Quadratic Mantinel)
            # Higher curvature (tighter turn) requires lower speed
            # Speed is inversely proportional to curvature squared
            base_max_speed = material_properties.get('max_feed_rate_mm_min', 2000.0)
            
            if curvature == 0:  # Straight path
                permissible_speed = base_max_speed
            else:
                # Quadratic relationship: speed decreases with square of curvature
                radius_mm = 1.0 / curvature if curvature != 0 else float('inf')
                
                # Use tolerance band approach - don't stop completely at corners
                min_radius_for_full_speed = 5.0  # mm - below this, we start reducing speed
                max_curvature_for_operation = 0.5  # 1/mm - extreme tightness we can handle
                
                if radius_mm < min_radius_for_full_speed:
                    # Reduce speed significantly for very tight corners
                    speed_reduction_factor = min_radius_for_full_speed / max(radius_mm, 0.1)
                    permissible_speed = base_max_speed * (speed_reduction_factor ** 2)
                elif curvature > max_curvature_for_operation:
                    # For extremely tight curves, use minimum safe speed
                    permissible_speed = base_max_speed * 0.1  # 10% of max speed
                else:
                    # Normal quadratic relationship
                    # Use a tolerance band to maintain momentum
                    ideal_speed_reduction = min(1.0, (curvature / max_curvature_for_operation) ** 2)
                    permissible_speed = base_max_speed * (1 - 0.8 * ideal_speed_reduction)  # Don't slow down too much
            
            # Ensure speed is within safe bounds
            min_safe_speed = material_properties.get('min_safe_feed_rate_mm_min', 50.0)
            permissible_speed = max(min_safe_speed, min(base_max_speed, permissible_speed))
            
            # Calculate time for this segment
            segment_time = (segment_length / permissible_speed) * 60 if permissible_speed > 0 else float('inf')
            total_time += segment_time
            
            results['segments'].append({
                'segment_id': i,
                'curvature': curvature,
                'radius_mm': 1.0 / curvature if curvature != 0 else float('inf'),
                'length_mm': segment_length,
                'permissible_speed_mm_min': permissible_speed,
                'estimated_time_sec': segment_time * 60 if segment_time != float('inf') else 0
            })
            
            results['recommended_speed_profile'].append(permissible_speed)
        
        results['total_path_time'] = total_time
        return results
    
    def get_pricing_recommendations(self, job_complexity: float, market_demand: str, competition_level: str) -> Dict:
        """
        Provide pricing recommendations based on job characteristics and market conditions
        """
        base_multiplier = 1.0
        
        # Adjust for complexity
        if job_complexity > 0.8:
            complexity_adjustment = 1.4  # 40% premium for high complexity
        elif job_complexity > 0.5:
            complexity_adjustment = 1.2  # 20% premium for medium complexity
        else:
            complexity_adjustment = 1.0  # No premium for low complexity
        
        # Adjust for market demand
        demand_multipliers = {
            'high': 1.3,   # 30% premium when demand is high
            'medium': 1.1, # 10% premium when demand is medium
            'low': 0.9     # 10% discount when demand is low
        }
        demand_adjustment = demand_multipliers.get(market_demand.lower(), 1.0)
        
        # Adjust for competition
        competition_multipliers = {
            'high': 0.9,   # 10% discount when competition is high
            'medium': 1.0, # No adjustment for medium competition
            'low': 1.2     # 20% premium when competition is low
        }
        competition_adjustment = competition_multipliers.get(competition_level.lower(), 1.0)
        
        final_multiplier = complexity_adjustment * demand_adjustment * competition_adjustment
        
        return {
            'complexity_adjustment': complexity_adjustment,
            'demand_adjustment': demand_adjustment,
            'competition_adjustment': competition_adjustment,
            'final_pricing_multiplier': final_multiplier,
            'pricing_strategy': self._determine_pricing_strategy(final_multiplier)
        }
    
    def _determine_pricing_strategy(self, multiplier: float) -> str:
        """
        Determine pricing strategy based on the calculated multiplier
        """
        if multiplier > 1.3:
            return "PREMIUM PRICING: Charge premium for complex jobs in high-demand markets"
        elif multiplier > 1.1:
            return "SOLID PREMIUM: Justified premium for complexity or demand"
        elif multiplier > 0.95:
            return "COMPETITIVE PRICING: Fair market price"
        else:
            return "PROMOTIONAL PRICING: Aggressive pricing to win business"