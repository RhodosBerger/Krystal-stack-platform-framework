"""
Anti-Fragile Marketplace - Economic Auditor
Calculates the "Cost of Ignorance" vs. the "Value of Shared Knowledge" in the fleet
and produces Fleet Savings Reports that quantify ROI of the Hive Mind in real dollars.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import uuid
import json
from decimal import Decimal


@dataclass
class CostOfIgnoranceEvent:
    """Records an event where a machine experienced a failure that could have been prevented by shared knowledge"""
    event_id: str
    affected_machine_id: str
    timestamp: datetime
    failure_type: str  # tool_breakage, thermal_damage, vibration_damage, etc.
    estimated_cost: Decimal  # Estimated monetary cost of the failure
    prevented_by: str  # Which shared knowledge could have prevented this
    knowledge_available_since: datetime  # When this knowledge was shared in the Hive
    hours_delayed: float  # How many hours after knowledge was available did this occur
    severity_level: str  # minor, moderate, major, catastrophic
    recovery_time_hours: float  # Time needed to recover from failure


@dataclass
class FleetSavingsReport:
    """Report showing the economic value of shared knowledge in the fleet"""
    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    total_potential_savings: Decimal
    actual_savings_realized: Decimal
    cost_of_ignorance: Decimal
    events_prevented_count: int
    events_occurred_count: int
    fleet_efficiency_improvement: float
    roi_percentage: float
    detailed_breakdown: Dict[str, Any]


class EconomicAuditor:
    """
    Economic Auditor - Calculates the value of shared knowledge in the fleet
    
    Calculates:
    - Cost of Ignorance: Money lost when machines repeat preventable failures
    - Value of Shared Knowledge: Money saved by avoiding repeat failures
    - Fleet Savings: Quantified ROI of the Hive Mind system
    """
    
    def __init__(self):
        self.cost_of_ignorance_events = []
        self.prevented_events = []
        self.knowledge_base = {}  # Maps failure patterns to prevention strategies
        self.machine_efficiency_tracking = {}
    
    def record_cost_of_ignorance_event(self, machine_id: str, failure_type: str, 
                                      estimated_cost: float, prevented_by: str,
                                      knowledge_available_since: datetime,
                                      severity_level: str = "moderate") -> CostOfIgnoranceEvent:
        """
        Record an event where a machine experienced a failure that could have been prevented
        by shared knowledge from the Hive.
        
        Args:
            machine_id: ID of the machine that experienced the failure
            failure_type: Type of failure that occurred
            estimated_cost: Estimated monetary cost of the failure
            prevented_by: Which shared knowledge could have prevented this
            knowledge_available_since: When this knowledge was available in the Hive
            severity_level: How severe the failure was
            
        Returns:
            The recorded CostOfIgnoranceEvent
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        # Calculate how long the preventive knowledge was available before the failure
        hours_delayed = (timestamp - knowledge_available_since).total_seconds() / 3600.0
        
        # Estimate recovery time based on severity
        severity_recovery_map = {
            "minor": 0.5,
            "moderate": 2.0,
            "major": 8.0,
            "catastrophic": 24.0
        }
        recovery_time = severity_recovery_map.get(severity_level, 2.0)
        
        event = CostOfIgnoranceEvent(
            event_id=event_id,
            affected_machine_id=machine_id,
            timestamp=timestamp,
            failure_type=failure_type,
            estimated_cost=Decimal(str(estimated_cost)),
            prevented_by=prevented_by,
            knowledge_available_since=knowledge_available_since,
            hours_delayed=hours_delayed,
            severity_level=severity_level,
            recovery_time_hours=recovery_time
        )
        
        self.cost_of_ignorance_events.append(event)
        
        print(f"[ECONOMIC_AUDITOR] Recorded Cost of Ignorance event: "
              f"Machine {machine_id} suffered {failure_type} costing ${estimated_cost:,.2f} "
              f"that could have been prevented by '{prevented_by}' knowledge "
              f"available {hours_delayed:.1f} hours earlier")
        
        return event
    
    def record_prevented_event(self, machine_id: str, failure_type: str,
                              prevented_by: str, savings_amount: float,
                              efficiency_improvement: float) -> Dict[str, Any]:
        """
        Record an event where shared knowledge prevented a potential failure.
        
        Args:
            machine_id: ID of the machine that was protected
            failure_type: Type of failure that was prevented
            prevented_by: Which shared knowledge prevented the failure
            savings_amount: Estimated amount saved by preventing the failure
            efficiency_improvement: Efficiency improvement from the prevention
            
        Returns:
            Dictionary with details of the prevented event
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        prevented_event = {
            'event_id': event_id,
            'machine_id': machine_id,
            'failure_type': failure_type,
            'prevented_by': prevented_by,
            'savings_amount': Decimal(str(savings_amount)),
            'efficiency_improvement': efficiency_improvement,
            'timestamp': timestamp
        }
        
        self.prevented_events.append(prevented_event)
        
        print(f"[ECONOMIC_AUDITOR] Prevented event: Machine {machine_id} avoided "
              f"{failure_type} saving ${savings_amount:,.2f} thanks to '{prevented_by}' knowledge")
        
        return prevented_event
    
    def calculate_fleet_savings_report(self, period_days: int = 30) -> FleetSavingsReport:
        """
        Generate a comprehensive Fleet Savings Report showing the economic value
        of shared knowledge in the fleet.
        
        Args:
            period_days: Number of days to include in the report (default 30)
            
        Returns:
            FleetSavingsReport with comprehensive economic analysis
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        
        # Filter events within the reporting period
        period_co_events = [
            event for event in self.cost_of_ignorance_events
            if start_date <= event.timestamp <= end_date
        ]
        
        period_prevented_events = [
            event for event in self.prevented_events
            if start_date <= event['timestamp'] <= end_date
        ]
        
        # Calculate total cost of ignorance
        total_cost_of_ignorance = sum(event.estimated_cost for event in period_co_events)
        
        # Calculate total savings from prevented events
        total_savings = sum(event['savings_amount'] for event in period_prevented_events)
        
        # Calculate potential savings if all ignorance events had been prevented
        potential_savings = total_cost_of_ignorance + total_savings
        
        # Calculate efficiency improvements
        avg_efficiency_improvement = (
            sum(event['efficiency_improvement'] for event in period_prevented_events) /
            len(period_prevented_events) if period_prevented_events else 0.0
        )
        
        # Calculate ROI percentage
        # Assuming some investment cost for the Hive system
        system_investment_cost = Decimal('50000.00')  # Placeholder for actual system cost
        net_benefit = total_savings - total_cost_of_ignorance
        roi_percentage = (
            (net_benefit / system_investment_cost * 100) if system_investment_cost > 0 
            else 0.0
        )
        
        report_id = str(uuid.uuid4())
        
        report = FleetSavingsReport(
            report_id=report_id,
            generated_at=end_date,
            period_start=start_date,
            period_end=end_date,
            total_potential_savings=Decimal(str(potential_savings)),
            actual_savings_realized=Decimal(str(total_savings)),
            cost_of_ignorance=Decimal(str(total_cost_of_ignorance)),
            events_prevented_count=len(period_prevented_events),
            events_occurred_count=len(period_co_events),
            fleet_efficiency_improvement=avg_efficiency_improvement,
            roi_percentage=float(roi_percentage),
            detailed_breakdown={
                'cost_of_ignorance_events': [
                    {
                        'machine_id': event.affected_machine_id,
                        'failure_type': event.failure_type,
                        'cost': float(event.estimated_cost),
                        'prevented_by': event.prevented_by,
                        'hours_delayed': event.hours_delayed,
                        'severity': event.severity_level
                    } for event in period_co_events
                ],
                'prevented_events': [
                    {
                        'machine_id': event['machine_id'],
                        'failure_type': event['failure_type'],
                        'savings': float(event['savings_amount']),
                        'prevented_by': event['prevented_by'],
                        'efficiency_improvement': event['efficiency_improvement']
                    } for event in period_prevented_events
                ],
                'cost_breakdown_by_type': self._calculate_cost_breakdown(period_co_events),
                'savings_breakdown_by_type': self._calculate_savings_breakdown(period_prevented_events)
            }
        )
        
        return report
    
    def _calculate_cost_breakdown(self, events: List[CostOfIgnoranceEvent]) -> Dict[str, Decimal]:
        """Calculate cost breakdown by failure type."""
        breakdown = {}
        for event in events:
            if event.failure_type not in breakdown:
                breakdown[event.failure_type] = Decimal('0.00')
            breakdown[event.failure_type] += event.estimated_cost
        return breakdown
    
    def _calculate_savings_breakdown(self, events: List[Dict]) -> Dict[str, Decimal]:
        """Calculate savings breakdown by prevention type."""
        breakdown = {}
        for event in events:
            prevention_type = event['prevented_by']
            if prevention_type not in breakdown:
                breakdown[prevention_type] = Decimal('0.00')
            breakdown[prevention_type] += event['savings_amount']
        return breakdown
    
    def calculate_savings_from_shared_trauma(self, machine_id: str, 
                                           trauma_avoided: str) -> Decimal:
        """
        Calculate the specific savings when a machine avoids a trauma that was
        shared by another machine in the fleet.
        
        Args:
            machine_id: ID of the machine that avoided the trauma
            trauma_avoided: Description of the trauma that was avoided
            
        Returns:
            Decimal representing the calculated savings amount
        """
        # Standard costs for different types of failures
        standard_costs = {
            'tool_breakage': Decimal('150.00'),
            'thermal_damage': Decimal('500.00'),
            'vibration_damage': Decimal('300.00'),
            'collosion_damage': Decimal('1000.00'),
            'quality_defect': Decimal('75.00'),
            'downtime_loss': Decimal('200.00')  # Per hour of unplanned downtime
        }
        
        # Find the closest matching standard cost
        matched_cost = Decimal('0.00')
        for key, cost in standard_costs.items():
            if key.lower() in trauma_avoided.lower():
                matched_cost = cost
                break
        
        # Add efficiency bonus for avoiding the trauma
        efficiency_bonus = matched_cost * Decimal('0.2')  # 20% bonus for prevention
        
        total_savings = matched_cost + efficiency_bonus
        
        # Record the prevention event
        self.record_prevented_event(
            machine_id=machine_id,
            failure_type=trauma_avoided,
            prevented_by="Fleet-Wide Trauma Sharing",
            savings_amount=float(total_savings),
            efficiency_improvement=0.05  # 5% efficiency improvement
        )
        
        return total_savings
    
    def get_fleet_roi_metrics(self) -> Dict[str, Any]:
        """
        Get high-level ROI metrics for the entire fleet.
        
        Returns:
            Dictionary with key ROI metrics
        """
        total_co_events = len(self.cost_of_ignorance_events)
        total_prevented = len(self.prevented_events)
        
        total_co_cost = sum(event.estimated_cost for event in self.cost_of_ignorance_events)
        total_savings = sum(event['savings_amount'] for event in self.prevented_events)
        
        # Calculate prevention rate
        prevention_rate = (
            total_prevented / (total_prevented + total_co_events) * 100
            if (total_prevented + total_co_events) > 0 else 0.0
        )
        
        # Calculate average cost per ignorance event
        avg_co_cost = (
            total_co_cost / total_co_events if total_co_events > 0 else Decimal('0.00')
        )
        
        # Calculate average savings per prevented event
        avg_savings = (
            total_savings / total_prevented if total_prevented > 0 else Decimal('0.00')
        )
        
        return {
            'total_events_prevented': total_prevented,
            'total_cost_of_ignorance_events': total_co_events,
            'cumulative_cost_of_ignorance': float(total_co_cost),
            'cumulative_savings': float(total_savings),
            'prevention_rate_percentage': round(prevention_rate, 2),
            'average_cost_per_ignored_event': float(avg_co_cost),
            'average_savings_per_prevented_event': float(avg_savings),
            'net_positive_impact': float(total_savings - total_co_cost),
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def generate_executive_summary(self) -> str:
        """
        Generate an executive summary of the economic impact of the Hive Mind system.
        
        Returns:
            String with executive summary of economic impact
        """
        metrics = self.get_fleet_roi_metrics()
        
        summary = f"""
FLEET ECONOMIC IMPACT SUMMARY
============================

Period: Since system inception
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

KEY METRICS:
- Events Prevented: {metrics['total_events_prevented']:,}
- Cost of Ignorance Events: {metrics['total_cost_of_ignorance_events']:,}
- Cumulative Savings: ${metrics['cumulative_savings']:,.2f}
- Total Cost of Ignorance: ${metrics['cumulative_cost_of_ignorance']:,.2f}
- Net Positive Impact: ${metrics['net_positive_impact']:,.2f}
- Prevention Rate: {metrics['prevention_rate_percentage']:.1f}%

FINANCIAL IMPACT:
- Average cost per ignored event: ${metrics['average_cost_per_ignored_event']:,.2f}
- Average savings per prevented event: ${metrics['average_savings_per_prevented_event']:,.2f}

The Hive Mind system has prevented {metrics['total_events_prevented']:,} potentially costly failures
and generated a net positive economic impact of ${metrics['net_positive_impact']:,.2f}.
The {metrics['prevention_rate_percentage']:.1f}% prevention rate demonstrates the value
of shared knowledge in reducing operational costs and improving fleet efficiency.
"""
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    print("Economic Auditor initialized successfully.")
    print("Ready to calculate Cost of Ignorance vs. Value of Shared Knowledge.")
    
    # Example usage would be:
    # auditor = EconomicAuditor()
    # 
    # # Record a cost of ignorance event
    # auditor.record_cost_of_ignorance_event(
    #     machine_id="M001",
    #     failure_type="tool_breakage",
    #     estimated_cost=150.00,
    #     prevented_by="Inconel-718 aggressive cutting parameters",
    #     knowledge_available_since=datetime.utcnow() - timedelta(hours=5),
    #     severity_level="moderate"
    # )
    # 
    # # Record a prevented event
    # auditor.record_prevented_event(
    #     machine_id="M002",
    #     failure_type="thermal_damage",
    #     prevented_by="Coolant flow monitoring",
    #     savings_amount=500.00,
    #     efficiency_improvement=0.08
    # )
    # 
    # # Generate fleet savings report
    # report = auditor.calculate_fleet_savings_report(period_days=30)
    # print(f"Generated report with ID: {report.report_id}")
    # print(f"Total potential savings: ${report.total_potential_savings}")
    # 
    # # Get executive summary
    # summary = auditor.generate_executive_summary()
    # print(summary)