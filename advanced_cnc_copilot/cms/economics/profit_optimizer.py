from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import logging


@dataclass
class ManufacturingMetrics:
    """Manufacturing-specific metrics mapped from SaaS concepts"""
    tool_wear_rate: float  # Like "Churn" in SaaS (rate of customer loss = tool loss)
    setup_time_hours: float  # Like "CAC" in SaaS (Customer Acquisition Cost = Setup Cost)
    cycle_time_hours: float
    sale_price: float
    tool_wear_cost: float
    material_cost: float
    labor_cost: float
    energy_cost: float


class ProfitOptimizer:
    """
    Implements "The Great Translation" - mapping SaaS metrics to Manufacturing Physics.
    Calculates Profit Rate (Pr) = (SalePrice - Costs) / Time
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_parallel_efficiency(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate efficiency for parallel machining plans.
        Maps CAC (Customer Acquisition Cost) to Setup Time.
        Calculates Profit Rate (Pr) = (SalePrice - ToolWearCost) / CycleTime.
        """
        # Extract plan parameters
        operations = plan.get('operations', [])
        sale_price = plan.get('sale_price', 1000.0)
        material_cost = plan.get('material_cost', 200.0)
        labor_hours = plan.get('labor_hours', 1.0)
        estimated_time = plan.get('estimated_time_hours', 1.0)
        actual_time = plan.get('actual_time_hours', estimated_time)
        
        # Calculate tool wear based on operations
        tool_wear_cost = self._calculate_tool_wear_cost(operations)
        
        # Calculate labor cost
        labor_cost = labor_hours * 50.0  # $50/hour labor rate (example)
        
        # Calculate energy cost based on machine usage
        energy_cost = self._calculate_energy_cost(operations)
        
        # Total costs
        total_costs = {
            'material': material_cost,
            'labor': labor_cost,
            'tool_wear': tool_wear_cost,
            'energy': energy_cost
        }
        
        # Calculate profit
        total_cost = sum(total_costs.values())
        profit = sale_price - total_cost
        
        # Calculate profit rate (Pr)
        profit_rate = profit / actual_time if actual_time > 0 else 0.0
        
        # Calculate tool wear rate (equivalent to SaaS "Churn")
        tool_wear_rate = tool_wear_cost / actual_time if actual_time > 0 else 0.0
        
        # Determine operational mode based on risk/reward
        operational_mode = self._determine_operational_mode(profit_rate, tool_wear_rate)
        
        # Calculate efficiency metrics
        profit_margin = profit / sale_price if sale_price > 0 else 0.0
        cost_per_part = total_cost / plan.get('part_count', 1)
        
        result = {
            'plan_id': plan.get('id', 'unknown'),
            'profit_rate': profit_rate,
            'profit_margin': profit_margin,
            'tool_wear_rate': tool_wear_rate,  # Equivalent to SaaS "Churn"
            'setup_time': plan.get('setup_time_hours', 0.0),  # Equivalent to SaaS "CAC"
            'cycle_time': actual_time,
            'estimated_vs_actual_time_ratio': actual_time / estimated_time if estimated_time > 0 else 1.0,
            'sale_price': sale_price,
            'total_costs': total_costs,
            'profit': profit,
            'operational_mode': operational_mode,
            'cost_per_part': cost_per_part,
            'recommendations': self._generate_recommendations(profit_rate, tool_wear_rate, operational_mode)
        }
        
        self.logger.info(f"Profit optimization calculated for plan {result['plan_id']}: Pr={profit_rate:.2f}, Churn={tool_wear_rate:.3f}")
        return result
    
    def _calculate_tool_wear_cost(self, operations: List[Dict[str, Any]]) -> float:
        """
        Calculate tool wear cost based on operations performed.
        Maps to SaaS "Churn" concept.
        """
        total_wear_cost = 0.0
        
        for operation in operations:
            # Factors affecting tool wear: speed, feed, material, etc.
            speed_factor = operation.get('rpm', 1000) / 10000  # Normalize to 0-1 range
            feed_factor = operation.get('feed_rate', 500) / 2000  # Normalize to 0-1 range
            duration_hours = operation.get('duration_hours', 0.1)
            
            # Calculate wear based on operating conditions
            wear_rate = (speed_factor * 0.4 + feed_factor * 0.3 + 0.3)  # Base wear rate
            wear_cost = wear_rate * duration_hours * 100.0  # $100 per hour of wear at max rate
            
            total_wear_cost += wear_cost
        
        return total_wear_cost
    
    def _calculate_energy_cost(self, operations: List[Dict[str, Any]]) -> float:
        """
        Calculate energy cost based on machine usage during operations.
        """
        total_energy_cost = 0.0
        energy_rate_per_kwh = 0.12  # $0.12 per kWh
        
        for operation in operations:
            duration_hours = operation.get('duration_hours', 0.1)
            power_draw_kw = operation.get('power_draw_kw', 5.0)  # Average power draw
            
            energy_consumed_kwh = power_draw_kw * duration_hours
            energy_cost = energy_consumed_kwh * energy_rate_per_kwh
            
            total_energy_cost += energy_cost
        
        return total_energy_cost
    
    def _determine_operational_mode(self, profit_rate: float, tool_wear_rate: float) -> str:
        """
        Determine operational mode based on profit rate and tool wear rate.
        Maps to SaaS concepts: High "Churn" = High tool wear rate.
        """
        # Define thresholds
        high_profit_threshold = 10.0  # High profit rate per hour
        high_wear_threshold = 0.5     # High tool wear rate per hour (like high churn)
        
        if tool_wear_rate > high_wear_threshold:
            # High tool wear = High "Churn" -> Switch to ECONOMY mode
            return "ECONOMY_MODE"
        elif profit_rate > high_profit_threshold and tool_wear_rate < 0.3:
            # High profit, low wear -> RUSH mode is safe
            return "RUSH_MODE"
        else:
            # Balanced approach
            return "BALANCED_MODE"
    
    def _generate_recommendations(self, profit_rate: float, tool_wear_rate: float, operational_mode: str) -> List[str]:
        """
        Generate recommendations based on calculated metrics.
        """
        recommendations = []
        
        if tool_wear_rate > 0.5:
            recommendations.append("HIGH TOOL WEAR DETECTED: Consider switching to ECONOMY mode to preserve tool life")
            recommendations.append("Schedule preventive maintenance soon to avoid unplanned downtime")
        
        if profit_rate < 5.0:
            recommendations.append("LOW PROFIT RATE: Consider optimizing process parameters or adjusting pricing")
            recommendations.append("Review material costs and supplier contracts")
        
        if operational_mode == "ECONOMY_MODE":
            recommendations.append("Current mode is ECONOMY: Prioritizing tool life over speed")
            recommendations.append("Consider using coated tools for better wear resistance")
        elif operational_mode == "RUSH_MODE":
            recommendations.append("Current mode is RUSH: Aggressive optimization for speed")
            recommendations.append("Monitor tool wear closely in this mode")
        else:
            recommendations.append("Current mode is BALANCED: Moderate approach balancing speed and tool life")
        
        if not recommendations:
            recommendations.append("Current operations appear economically optimal")
        
        return recommendations
    
    def optimize_for_rush_mode(self, base_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize a plan for "Rush Mode" while respecting safety constraints.
        This corresponds to aggressive manufacturing when safe to do so.
        """
        optimized_plan = base_plan.copy()
        
        # Increase speeds and feeds for rush mode
        for operation in optimized_plan.get('operations', []):
            # Increase RPM by 10% for rush mode
            operation['rpm'] = min(operation.get('rpm', 1000) * 1.1, operation.get('max_rpm', 12000))
            # Increase feed rate by 10% for rush mode
            operation['feed_rate'] = min(operation.get('feed_rate', 500) * 1.1, operation.get('max_feed', 2000))
        
        # Recalculate metrics for the optimized plan
        efficiency_result = self.calculate_parallel_efficiency(optimized_plan)
        
        # Verify that safety constraints are still satisfied
        if efficiency_result['tool_wear_rate'] > 0.7:  # Too high wear for rush mode
            self.logger.warning("Rush mode optimization would exceed tool wear limits, reverting to balanced mode")
            efficiency_result['operational_mode'] = "BALANCED_MODE"
            efficiency_result['recommendations'].append("RUSH MODE NOT SAFE: Tool wear too high, reverted to BALANCED mode")
        
        return efficiency_result
    
    def optimize_for_economy_mode(self, base_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize a plan for "Economy Mode" to minimize tool wear and maximize longevity.
        """
        optimized_plan = base_plan.copy()
        
        # Decrease speeds and feeds for economy mode
        for operation in optimized_plan.get('operations', []):
            # Decrease RPM by 15% for economy mode
            operation['rpm'] = max(operation.get('rpm', 1000) * 0.85, operation.get('min_rpm', 500))
            # Decrease feed rate by 15% for economy mode
            operation['feed_rate'] = max(operation.get('feed_rate', 500) * 0.85, operation.get('min_feed', 100))
        
        # Recalculate metrics for the optimized plan
        efficiency_result = self.calculate_parallel_efficiency(optimized_plan)
        
        # Ensure we're in economy mode
        efficiency_result['operational_mode'] = "ECONOMY_MODE"
        efficiency_result['recommendations'].append("Optimized for ECONOMY MODE: Prioritizing tool life and safety")
        
        return efficiency_result


# Example usage
if __name__ == "__main__":
    optimizer = ProfitOptimizer()
    
    sample_plan = {
        'id': 'sample_plan_001',
        'operations': [
            {
                'id': 'op_001',
                'type': 'face_mill',
                'rpm': 3000,
                'feed_rate': 800,
                'duration_hours': 0.5,
                'power_draw_kw': 8.5,
                'max_rpm': 12000,
                'max_feed': 2000,
                'min_rpm': 500,
                'min_feed': 100
            },
            {
                'id': 'op_002',
                'type': 'drill',
                'rpm': 2000,
                'feed_rate': 400,
                'duration_hours': 0.25,
                'power_draw_kw': 5.2,
                'max_rpm': 12000,
                'max_feed': 2000,
                'min_rpm': 500,
                'min_feed': 100
            }
        ],
        'sale_price': 1500.0,
        'material_cost': 300.0,
        'labor_hours': 1.0,
        'estimated_time_hours': 1.0,
        'actual_time_hours': 0.75,
        'part_count': 1,
        'setup_time_hours': 0.5
    }
    
    result = optimizer.calculate_parallel_efficiency(sample_plan)
    print("Profit Optimization Result:")
    print(f"  Profit Rate: ${result['profit_rate']:.2f}/hour")
    print(f"  Tool Wear Rate: {result['tool_wear_rate']:.3f}/hour")
    print(f"  Operational Mode: {result['operational_mode']}")
    print(f"  Recommendations: {result['recommendations']}")