"""
Economics Calculator
"""

from decimal import Decimal
from django.utils import timezone

class EconomicsCalculator:
    """Calculate job economics"""
    
    def __init__(self, job):
        self.job = job
        self.hourly_rate = Decimal('85.00')  # €/hour
        self.material_cost_per_piece = self.get_material_cost()
    
    def get_material_cost(self):
        """Get material cost from project"""
        material_costs = {
            'Aluminum 6061': Decimal('15.00'),
            'Steel 1045': Decimal('12.00'),
            'Stainless 304': Decimal('25.00'),
            'Titanium': Decimal('120.00'),
            'Brass': Decimal('18.00'),
        }
        return material_costs.get(self.job.project.material, Decimal('10.00'))
    
    def calculate_machine_time_cost(self):
        """Machine time cost"""
        if self.job.completed_at and self.job.started_at:
            duration_hours = (self.job.completed_at - self.job.started_at).total_seconds() / 3600
            total_cost = Decimal(str(duration_hours)) * self.hourly_rate
            cost_per_part = total_cost / self.job.completed_quantity if self.job.completed_quantity > 0 else Decimal('0')
            return total_cost, cost_per_part
        return Decimal('0'), Decimal('0')
    
    def calculate_material_cost(self):
        """Total material cost"""
        return self.material_cost_per_piece * self.job.quantity
    
    def calculate_tool_cost(self):
        """Tool cost based on usage"""
        from .models import Tool
        
        # Assume we know which tools were used (simplified)
        # In real implementation, track tool usage per job
        estimated_tool_cost = Decimal('0.50')  # €/part average
        return estimated_tool_cost * self.job.completed_quantity
    
    def calculate_energy_cost(self):
        """Energy cost (if tracked)"""
        # Simplified: 0.15 kWh/part @ €0.12/kWh
        kwh_per_part = Decimal('0.15')
        cost_per_kwh = Decimal('0.12')
        return kwh_per_part * cost_per_kwh * self.job.completed_quantity
    
    def create_record(self):
        """Create economic record"""
        from .models import EconomicRecord
        
        machine_total, machine_per_part = self.calculate_machine_time_cost()
        material_cost = self.calculate_material_cost()
        tool_cost = self.calculate_tool_cost()
        energy_cost = self.calculate_energy_cost()
        
        total_cost = machine_total + material_cost + tool_cost + energy_cost
        cost_per_part = total_cost / self.job.completed_quantity if self.job.completed_quantity > 0 else Decimal('0')
        
        # Calculate cycle time
        if self.job.completed_at and self.job.started_at:
            total_minutes = (self.job.completed_at - self.job.started_at).total_seconds() / 60
            cycle_time = total_minutes / self.job.completed_quantity if self.job.completed_quantity > 0 else 0
        else:
            cycle_time = 0
        
        # Traditional cost estimate (for comparison)
        traditional_cycle_time = self.job.project.estimated_cycle_time
        traditional_cost_per_part = (
            (traditional_cycle_time / 60 * self.hourly_rate) +
            self.material_cost_per_piece +
            Decimal('0.72')  # Traditional tool cost
        )
        
        savings = traditional_cost_per_part - cost_per_part
        
        record, created = EconomicRecord.objects.update_or_create(
            job=self.job,
            defaults={
                'machine_time_cost': machine_total,
                'material_cost': material_cost,
                'tool_cost': tool_cost,
                'energy_cost': energy_cost,
                'total_cost': total_cost,
                'cost_per_part': cost_per_part,
                'cycle_time': cycle_time,
                'traditional_cost': traditional_cost_per_part,
                'savings': savings
            }
        )
        
        return record
