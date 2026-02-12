from typing import Dict, Any, List, Tuple
from datetime import datetime
import logging

from ..models import Telemetry
from ..services.dopamine_engine import DopamineEngine
from ..services.economics_engine import EconomicsEngine
from ..repositories.telemetry_repository import TelemetryRepository
from ..vulkan_ingestor import grid_state


logger = logging.getLogger(__name__)


class DecisionPolicy:
    """
    Defines policies for the Shadow Council's decision-making process
    Implements the 'Death Penalty Function' and constraint validation
    """
    
    def __init__(self):
        # Define operational limits and constraints
        self.constraints = {
            'max_spindle_load_percent': 95.0,
            'max_vibration_level': 2.0,
            'max_temperature_celsius': 70.0,
            'min_coolant_flow_rate': 0.5,
            'max_feed_rate_mm_min': 5000.0,
            'max_rpm': 12000.0,
            'physics_constraint_matrix': {
                # Define relationships between parameters that must be validated
                'feed_rate_vs_rpm': lambda feed, rpm: feed / rpm < 0.5,  # Chip load constraint
                'spindle_load_vs_temperature': lambda load, temp: temp < 50 + (load * 0.2),  # Heat generation constraint
            }
        }
    
    def validate_constraint(self, param_name: str, value: float) -> bool:
        """Validate a single parameter against its constraint"""
        if param_name in self.constraints:
            constraint_limit = self.constraints[param_name]
            if isinstance(constraint_limit, (int, float)):
                return value <= constraint_limit
        return True  # No constraint defined, assume valid
    
    def validate_physics_match(self, parameters: Dict[str, float]) -> bool:
        """Validate that parameters satisfy physics-based constraints"""
        # Check individual constraints
        for param, value in parameters.items():
            if not self.validate_constraint(param, value):
                return False
        
        # Check physics relationships
        physics_checks = self.constraints.get('physics_constraint_matrix', {})
        for check_name, check_func in physics_checks.items():
            try:
                if check_name == 'feed_rate_vs_rpm':
                    feed = parameters.get('feed_rate', 1000.0)
                    rpm = parameters.get('rpm', 2000.0)
                    if not check_func(feed, rpm):
                        return False
                elif check_name == 'spindle_load_vs_temperature':
                    load = parameters.get('spindle_load', 50.0)
                    temp = parameters.get('temperature', 35.0)
                    if not check_func(load, temp):
                        return False
            except KeyError:
                # Missing parameters for physics check, skip validation
                continue
            except Exception as e:
                logger.warning(f"Physics validation error: {e}")
                continue
        
        return True


class CreatorAgent:
    """
    The Creator Agent - Proposes new strategies and optimizations
    Implements probabilistic suggestions based on LLM or optimization algorithms
    """
    
    def __init__(self, repository: TelemetryRepository):
        self.repository = repository
        self.aggressive_optimization_threshold = 0.8  # Confidence threshold for aggressive changes
    
    def propose_optimization(self, current_state: Dict[str, Any], machine_id: int) -> Dict[str, Any]:
        """
        Propose an optimization based on current state and historical data
        """
        # Analyze current state to determine optimization opportunities
        current_efficiency = current_state.get('efficiency_score', 0.5)
        current_stress = current_state.get('stress_level', 0.3)
        
        # Fetch historical data for this machine
        recent_data = self.repository.get_recent_by_machine(machine_id, minutes=30)
        
        if not recent_data:
            # No historical data, propose conservative changes
            return self._propose_conservative_change(current_state)
        
        # Analyze historical trends to propose improvements
        proposed_change = self._analyze_historical_trends(recent_data, current_state)
        
        return proposed_change
    
    def _propose_conservative_change(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Propose a conservative change when lacking historical data"""
        return {
            'timestamp': datetime.utcnow(),
            'proposed_parameters': {
                'spindle_load': min(85.0, current_state.get('spindle_load', 50.0) * 1.05),
                'feed_rate': min(4000.0, current_state.get('feed_rate', 1000.0) * 1.02),
                'rpm': min(10000.0, current_state.get('rpm', 2000.0) * 1.01),
            },
            'confidence': 0.6,
            'optimization_target': 'efficiency',
            'reasoning': 'Conservative optimization based on current state'
        }
    
    def _analyze_historical_trends(self, recent_data: List, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze historical trends to propose specific optimizations"""
        # Calculate average performance metrics
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
                logger.warning(f"Error processing telemetry record for trend analysis: {e}")
                continue
        
        if valid_records == 0:
            avg_spindle_load = 50.0  # Default if no data
            avg_vibration = 0.5
        else:
            avg_spindle_load = total_spindle_load / valid_records
            avg_vibration = total_vibration / valid_records
        
        # Determine optimization direction based on trends
        optimization_direction = 'efficiency' if avg_spindle_load < 75 else 'throughput'
        
        # Propose specific changes based on analysis
        proposed_params = {}
        
        if avg_vibration < 0.8 and avg_spindle_load < 60:
            # System appears stable with room for improvement, can increase aggressiveness
            proposed_params['spindle_load'] = min(90.0, avg_spindle_load * 1.15)
            proposed_params['feed_rate'] = min(4500.0, current_state.get('feed_rate', 1000.0) * 1.05)
            proposed_params['rpm'] = min(11000.0, current_state.get('rpm', 2000.0) * 1.05)
            confidence = 0.85
        elif avg_vibration > 1.2 or avg_spindle_load > 85:
            # System appears stressed, recommend conservative approach
            proposed_params['spindle_load'] = max(50.0, avg_spindle_load * 0.9)
            proposed_params['feed_rate'] = max(800.0, current_state.get('feed_rate', 1000.0) * 0.95)
            proposed_params['rpm'] = max(1500.0, current_state.get('rpm', 2000.0) * 0.95)
            confidence = 0.75
        else:
            # Moderate optimization
            proposed_params['spindle_load'] = min(85.0, avg_spindle_load * 1.03)
            proposed_params['feed_rate'] = min(4000.0, current_state.get('feed_rate', 1000.0) * 1.02)
            proposed_params['rpm'] = min(10000.0, current_state.get('rpm', 2000.0) * 1.02)
            confidence = 0.7
        
        return {
            'timestamp': datetime.utcnow(),
            'proposed_parameters': proposed_params,
            'confidence': confidence,
            'optimization_target': optimization_direction,
            'reasoning': f'Optimization based on historical trends: avg_vibration={avg_vibration:.2f}, avg_spindle_load={avg_spindle_load:.2f}'
        }


class AuditorAgent:
    """
    The Auditor Agent - Implements deterministic validation of proposals
    Applies the 'Death Penalty Function' for constraint violations
    """
    
    def __init__(self, decision_policy: DecisionPolicy):
        self.policy = decision_policy
        self.logger = logging.getLogger(__name__)
    
    def validate_proposal(self, proposal: Dict[str, Any], current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a proposal against physics and safety constraints
        Implements the 'Death Penalty Function' for constraint violations
        """
        proposed_params = proposal.get('proposed_parameters', {})
        confidence = proposal.get('confidence', 0.5)
        
        # Validate against all constraints
        validation_result = self._validate_parameters(proposed_params, current_state)
        
        # Create validation report
        result = {
            'proposal_id': proposal.get('timestamp', datetime.utcnow()).isoformat(),
            'is_approved': validation_result['is_valid'],
            'fitness_score': validation_result['fitness'],
            'constraint_violations': validation_result['violations'],
            'reasoning_trace': validation_result['reasoning'],
            'confidence_after_validation': confidence * validation_result['fitness'] if validation_result['is_valid'] else 0.0,
            'validation_timestamp': datetime.utcnow()
        }
        
        if not validation_result['is_valid']:
            self.logger.warning(f"Auditor rejected proposal with violations: {validation_result['violations']}")
        
        return result
    
    def _validate_parameters(self, proposed_params: Dict[str, Any], current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate proposed parameters against all constraints"""
        violations = []
        reasoning = []
        
        # Validate each proposed parameter against constraints
        for param_name, proposed_value in proposed_params.items():
            if not self.policy.validate_constraint(param_name, proposed_value):
                violations.append({
                    'parameter': param_name,
                    'proposed_value': proposed_value,
                    'constraint_limit': self.policy.constraints.get(param_name, 'undefined'),
                    'reason': f'{param_name} exceeds safety limit'
                })
                reasoning.append(f"REJECTED: {param_name}={proposed_value} exceeds limit")
        
        # Validate physics relationships
        if not self.policy.validate_physics_match(proposed_params):
            violations.append({
                'parameter': 'physics_constraints',
                'proposed_value': 'multiple',
                'constraint_limit': 'physics laws',
                'reason': 'Proposed parameters violate physics relationships'
            })
            reasoning.append("REJECTED: Proposed parameters violate physics relationships")
            
        # Validate Grid Saturation (Vulkan)
        if grid_state.get("saturation", 0.0) > 0.9:
             violations.append({
                 'parameter': 'grid_saturation',
                 'proposed_value': 'N/A',
                 'constraint_limit': '90%',
                 'reason': 'Vulkan Grid Memory Saturated'
             })
             reasoning.append(f"REJECTED: Grid Saturation at {grid_state['saturation']:.2%}")

        
        # Calculate fitness score
        if violations:
            # Apply 'Death Penalty' - if any constraint is violated, fitness is 0
            fitness = 0.0
            is_valid = False
        else:
            # If all constraints pass, calculate fitness based on efficiency gains
            fitness = self._calculate_fitness_score(proposed_params, current_state)
            is_valid = True
        
        return {
            'is_valid': is_valid,
            'fitness': fitness,
            'violations': violations,
            'reasoning': reasoning
        }
    
    def _calculate_fitness_score(self, proposed_params: Dict[str, Any], current_state: Dict[str, Any]) -> float:
        """Calculate fitness score for valid proposals"""
        # Base fitness on improvement potential
        fitness = 0.8  # Start with high fitness for valid proposals
        
        # Adjust based on parameter improvements
        for param, proposed_val in proposed_params.items():
            current_val = current_state.get(param, 0.0)
            if param in ['spindle_load', 'feed_rate', 'rpm']:
                # Higher values generally improve efficiency
                if proposed_val > current_val:
                    fitness = min(1.0, fitness + 0.1)
        
        return min(1.0, fitness)


class AccountantAgent:
    """
    The Accountant Agent - Economic evaluation and cost-benefit analysis
    Evaluates proposals based on economic impact
    """
    
    def __init__(self, economics_engine: EconomicsEngine):
        self.economics_engine = economics_engine
    
    def evaluate_economic_impact(self, proposal: Dict[str, Any], current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the economic impact of a proposal
        """
        proposed_params = proposal.get('proposed_parameters', {})
        
        # Create job data for economic analysis
        job_data = {
            'estimated_duration_hours': current_state.get('estimated_duration_hours', 1.0),
            'actual_duration_hours': current_state.get('actual_duration_hours', 1.0),
            'sales_price': current_state.get('sales_price', 1000.0),
            'material_cost': current_state.get('material_cost', 200.0),
            'labor_hours': current_state.get('labor_hours', 1.0),
            'part_count': current_state.get('part_count', 1),
            'machine_id': current_state.get('machine_id', 1)
        }
        
        # Calculate potential economic benefits with proposed parameters
        benefit_analysis = self.economics_engine.analyze_job_economics(job_data)
        
        return {
            'proposal_id': proposal.get('timestamp', datetime.utcnow()).isoformat(),
            'economic_analysis': benefit_analysis,
            'projected_profit_rate': self.economics_engine.calculate_profit_rate(
                job_data.get('sales_price', 0.0),
                benefit_analysis.get('total_costs', {}),
                job_data.get('actual_duration_hours', 1.0)
            ),
            'projected_churn_risk': self.economics_engine.calculate_churn_risk(
                benefit_analysis.get('total_costs', {}).get('tool_wear', 0.0) / job_data.get('actual_duration_hours', 1.0)
            ),
            'financial_recommendation': self.economics_engine._generate_recommendations(
                benefit_analysis.get('profit_rate', 0.0),
                benefit_analysis.get('churn_score', 0.0),
                benefit_analysis.get('profit_margin', 0.0)
            ),
            'evaluation_timestamp': datetime.utcnow()
        }


class ShadowCouncil:
    """
    The Shadow Council Governance Pattern
    Implements the three-agent system: Creator, Auditor, Accountant
    Creator (Probabilistic): Proposes optimizations
    Auditor (Deterministic): Validates against constraints
    Accountant (Economic): Evaluates economic impact
    """
    
    def __init__(self, creator: CreatorAgent, auditor: AuditorAgent, decision_policy: DecisionPolicy):
        self.creator = creator
        self.auditor = auditor
        self.accountant = None  # Will be set separately
        self.policy = decision_policy
        self.logger = logging.getLogger(__name__)
    
    def evaluate_strategy(self, current_state: Dict[str, Any], machine_id: int) -> Dict[str, Any]:
        """
        Evaluate a strategy through the Shadow Council process
        1. Creator proposes an optimization
        2. Auditor validates against constraints
        3. Accountant evaluates economic impact
        """
        # Step 1: Creator proposes optimization
        proposal = self.creator.propose_optimization(current_state, machine_id)
        
        # Step 2: Auditor validates proposal
        validation_result = self.auditor.validate_proposal(proposal, current_state)
        
        # Step 3: Accountant evaluates economic impact if proposal passes audit
        if validation_result['is_approved'] and self.accountant:
            economic_result = self.accountant.evaluate_economic_impact(proposal, current_state)
        else:
            economic_result = {
                'economic_analysis': {},
                'projected_profit_rate': 0.0,
                'projected_churn_risk': 1.0,
                'financial_recommendation': 'REJECT DUE TO SAFETY VIOLATIONS'
            }
        
        # Step 4: Combine results
        council_decision = {
            'proposal': proposal,
            'validation': validation_result,
            'economic_evaluation': economic_result,
            'council_approval': validation_result['is_approved'],
            'final_fitness': validation_result['fitness_score'],
            'decision_timestamp': datetime.utcnow(),
            'reasoning_trace': validation_result['reasoning_trace'],
            'economic_impact': economic_result.get('projected_profit_rate', 0.0),
            'risk_assessment': economic_result.get('projected_churn_risk', 1.0)
        }
        
        if council_decision['council_approval']:
            self.logger.info(f"Shadow Council APPROVED proposal with fitness {council_decision['final_fitness']}")
        else:
            self.logger.warning(f"Shadow Council REJECTED proposal due to constraints: {validation_result['constraint_violations']}")
        
        return council_decision
    
    def set_accountant(self, accountant: AccountantAgent):
        """Set the accountant agent for economic evaluation"""
        self.accountant = accountant
    
    def execute_nightmare_training(self, machine_id: int, duration_hours: float = 1.0):
        """
        Execute nightmare training during idle time
        Replays telemetry logs with injected failure scenarios
        """
        # Get historical telemetry for simulation
        historical_data = self.creator.repository.get_recent_by_machine(
            machine_id, 
            minutes=int(duration_hours * 60)
        )
        
        if not historical_data:
            self.logger.warning(f"No historical data for nightmare training on machine {machine_id}")
            return
        
        # Inject failure scenarios into historical data
        failure_scenarios = [
            {'type': 'spindle_stall', 'time_index': len(historical_data)//2},
            {'type': 'vibration_spike', 'time_index': len(historical_data)//3},
            {'type': 'thermal_overload', 'time_index': 2*len(historical_data)//3}
        ]
        
        for scenario in failure_scenarios:
            # Simulate the system's response to the failure
            self.logger.info(f"Injecting {scenario['type']} at index {scenario['time_index']} for nightmare training")
            
            # Process through Shadow Council to see how it responds to failure scenarios
            if scenario['time_index'] < len(historical_data):
                telemetry_at_failure = historical_data[scenario['time_index']]
                current_metrics = {
                    'spindle_load': getattr(telemetry_at_failure, 'spindle_load', 50.0) or 0.0,
                    'vibration_x': getattr(telemetry_at_failure, 'vibration_x', 0.5) or 0.0,
                    'temperature': getattr(telemetry_at_failure, 'temperature', 35.0) or 35.0,
                    'machine_id': machine_id
                }
                
                # Evaluate how the system responds to this failure scenario
                council_decision = self.evaluate_strategy(current_metrics, machine_id)
                
                self.logger.info(f"Nightmare training response: {council_decision['council_approval']}")
        
        self.logger.info(f"Nightmare training completed for machine {machine_id}")