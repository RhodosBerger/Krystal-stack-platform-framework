"""
Shadow Council Governance - Complete Implementation
Orchestrates the decision-making process between Creator (probabilistic), 
Auditor (deterministic), and Accountant (economic) agents.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from dataclasses import dataclass
import uuid


@dataclass
class StrategyProposal:
    """A proposed strategy from the Creator Agent"""
    proposal_id: str
    strategy_name: str
    proposed_parameters: Dict[str, Any]
    intent: str
    timestamp: datetime
    creator_confidence: float
    creator_reasoning: str


@dataclass
class ValidationOutcome:
    """Outcome of Auditor's validation"""
    proposal_id: str
    is_approved: bool
    fitness_score: float
    reasoning_trace: List[str]
    constraint_violations: List[Dict[str, Any]]
    validation_timestamp: datetime
    death_penalty_applied: bool
    death_penalty_reason: str


@dataclass
class EconomicAssessment:
    """Economic evaluation from the Accountant Agent"""
    proposal_id: str
    churn_risk: float
    projected_profit_rate: float
    tool_wear_cost: float
    recommended_mode: str
    economic_timestamp: datetime
    roi_projection: float


@dataclass
class CouncilDecision:
    """Final decision from the Shadow Council"""
    decision_id: str
    proposal: StrategyProposal
    validation: ValidationOutcome
    economic_assessment: EconomicAssessment
    council_approval: bool
    final_fitness: float
    reasoning_trace: List[str]
    decision_timestamp: datetime
    decision_confidence: float


class CreatorAgent:
    """
    The Creator Agent - Probabilistic AI that proposes optimizations
    Based on biological "Id" - creative but potentially unsafe without governance
    """
    
    def __init__(self, llm_interface=None):
        self.llm_interface = llm_interface
        self.logger = logging.getLogger(__name__)
        
    def propose_strategy(self, intent: str, current_state: Dict[str, Any]) -> StrategyProposal:
        """
        Generate a strategy proposal based on operator intent and current state.
        
        Args:
            intent: Operator's intent (e.g., "face mill aluminum block aggressively")
            current_state: Current machine state and constraints
            
        Returns:
            StrategyProposal with parameters and confidence metrics
        """
        # In a real system, this would connect to an LLM
        # For now, we'll simulate with heuristic-based optimization
        
        # Extract current parameters
        current_rpm = current_state.get('rpm', 4000)
        current_feed = current_state.get('feed_rate', 2000)
        material = current_state.get('material', 'steel')
        
        # Propose optimized parameters based on intent
        if 'aggressive' in intent.lower():
            # Aggressive parameters for higher efficiency
            proposed_rpm = min(current_rpm * 1.15, 12000)  # 15% increase, capped at max
            proposed_feed = min(current_feed * 1.2, 5000)  # 20% increase, capped at max
        elif 'conservative' in intent.lower() or 'safe' in intent.lower():
            # Conservative parameters for safety
            proposed_rpm = current_rpm * 0.9  # 10% decrease
            proposed_feed = current_feed * 0.85  # 15% decrease
        else:
            # Balanced parameters
            proposed_rpm = current_rpm * 1.05  # 5% increase
            proposed_feed = current_feed * 1.1   # 10% increase
        
        # Create proposal
        proposal = StrategyProposal(
            proposal_id=f"PROP_{uuid.uuid4().hex[:8]}",
            strategy_name=f"Auto-Optimized-{material}-{intent[:20].replace(' ', '_')}",
            proposed_parameters={
                'rpm': proposed_rpm,
                'feed_rate': proposed_feed,
                'spindle_load': current_state.get('spindle_load', 60.0),
                'temperature': current_state.get('temperature', 35.0),
                'vibration_x': current_state.get('vibration_x', 0.2),
                'vibration_y': current_state.get('vibration_y', 0.2),
                'coolant_flow': current_state.get('coolant_flow', 1.5),
                'material': material,
                'operation_type': current_state.get('operation_type', 'generic_mill')
            },
            intent=intent,
            timestamp=datetime.utcnow(),
            creator_confidence=0.85,  # High confidence in proposed optimization
            creator_reasoning=f"Based on intent '{intent}', proposing {('aggressive' if 'aggressive' in intent.lower() else 'balanced' if 'balanced' in intent.lower() else 'conservative')} parameters for {material} operation"
        )
        
        self.logger.info(f"Creator proposed: {proposal.strategy_name} with confidence {proposal.creator_confidence}")
        
        return proposal


class AuditorAgent:
    """
    The Auditor Agent - Deterministic validator with physics constraints
    Implements the "Death Penalty Function" where any constraint violation results in fitness=0
    Based on biological "Superego" - strict moral/physical constraints
    """
    
    def __init__(self):
        self.physics_constraints = {
            'max_spindle_load_percent': 95.0,
            'max_temperature_celsius': 70.0,
            'max_vibration_g_force': 2.0,
            'max_feed_rate_mm_min': 5000.0,
            'max_rpm': 12000.0,
            'min_coolant_flow_rate': 0.5,
            'min_curvature_radius_mm': 0.5,  # For Quadratic Mantinel
        }
        self.logger = logging.getLogger(__name__)
    
    def validate_proposal(self, proposal: StrategyProposal, 
                         current_state: Dict[str, Any]) -> ValidationOutcome:
        """
        Validate a proposal against hard physics constraints using deterministic checks.
        
        Args:
            proposal: The strategy proposal to validate
            current_state: Current machine state for context
            
        Returns:
            ValidationOutcome with approval decision and reasoning trace
        """
        violations = []
        reasoning_trace = []
        
        reasoning_trace.append("AUDIT_START: Initiating Physics Check...")
        
        # Check each proposed parameter against physics constraints
        for param_name, proposed_value in proposal.proposed_parameters.items():
            if param_name in self.physics_constraints:
                constraint_limit = self.physics_constraints[param_name]
                
                if isinstance(proposed_value, (int, float)) and isinstance(constraint_limit, (int, float)):
                    if proposed_value > constraint_limit:
                        violations.append({
                            'parameter': param_name,
                            'proposed_value': proposed_value,
                            'constraint_limit': constraint_limit,
                            'reason': f'{param_name} exceeds physical limit'
                        })
                        reasoning_trace.append(f"VIOLATION: {param_name}={proposed_value} > limit={constraint_limit}")
        
        # Check Quadratic Mantinel constraint (feed rate vs curvature)
        if 'path_curvature_radius' in proposal.proposed_parameters:
            curvature_radius = proposal.proposed_parameters['path_curvature_radius']
            feed_rate = proposal.proposed_parameters.get('feed_rate', current_state.get('feed_rate', 1000))
            
            # Calculate max safe feed based on curvature (Quadratic Mantinel)
            max_safe_feed = 1500.0 * (curvature_radius ** 0.5)  # Example formula
            
            if feed_rate > max_safe_feed:
                violations.append({
                    'parameter': 'feed_rate_vs_curvature',
                    'proposed_value': f"feed={feed_rate}, radius={curvature_radius}",
                    'constraint_limit': f"max_safe_feed={max_safe_feed}",
                    'reason': 'Exceeds Quadratic Mantinel constraint'
                })
                reasoning_trace.append(f"QUADRATIC_MANTELINEL VIOLATION: Feed {feed_rate} exceeds safe limit {max_safe_feed:.1f} for curvature radius {curvature_radius}mm")
        
        # Apply Death Penalty if any violations exist
        if violations:
            # Any constraint violation results in fitness=0 (Death Penalty function)
            outcome = ValidationOutcome(
                proposal_id=proposal.proposal_id,
                is_approved=False,
                fitness_score=0.0,
                reasoning_trace=reasoning_trace,
                constraint_violations=violations,
                validation_timestamp=datetime.utcnow(),
                death_penalty_applied=True,
                death_penalty_reason=f"Constraint violations detected: {[v['parameter'] for v in violations]}"
            )
            self.logger.warning(f"Death Penalty applied to proposal {proposal.proposal_id}: {outcome.death_penalty_reason}")
        else:
            # Calculate fitness based on efficiency if no violations
            fitness_score = self._calculate_efficiency_fitness(proposal.proposed_parameters, current_state)
            outcome = ValidationOutcome(
                proposal_id=proposal.proposal_id,
                is_approved=True,
                fitness_score=fitness_score,
                reasoning_trace=reasoning_trace,
                constraint_violations=[],
                validation_timestamp=datetime.utcnow(),
                death_penalty_applied=False,
                death_penalty_reason=""
            )
            self.logger.info(f"Proposal {proposal.proposal_id} approved with fitness {fitness_score:.3f}")
        
        reasoning_trace.append(f"FINAL_OUTCOME: {'APPROVED' if outcome.is_approved else 'REJECTED'}")
        
        return outcome
    
    def _calculate_efficiency_fitness(self, proposed_params: Dict[str, Any], 
                                    current_state: Dict[str, Any]) -> float:
        """
        Calculate fitness score based on operational efficiency when physics constraints are satisfied.
        """
        # Base fitness is high if all constraints pass
        base_fitness = 0.8
        
        # Adjust for efficiency parameters
        feed_rate = proposed_params.get('feed_rate', current_state.get('feed_rate', 1000))
        rpm = proposed_params.get('rpm', current_state.get('rpm', 2000))
        
        # Normalize to 0-1 scale based on machine capabilities
        normalized_feed = min(1.0, feed_rate / self.physics_constraints['max_feed_rate_mm_min'])
        normalized_rpm = min(1.0, rpm / self.physics_constraints['max_rpm'])
        
        # Efficiency bonus based on utilization of machine capabilities
        efficiency_bonus = (normalized_feed * 0.1) + (normalized_rpm * 0.1)
        
        # Combined fitness
        fitness = min(1.0, base_fitness + efficiency_bonus)
        
        return max(0.0, fitness)


class AccountantAgent:
    """
    The Accountant Agent - Economic evaluator of proposals
    Based on biological "Ego" - balances Id's creativity with Superego's constraints
    """
    
    def __init__(self):
        # Economic constants
        self.hourly_machine_cost = 85.00  # USD/hr (Depreciation + Labor)
        self.tool_replacement_cost = 150.00  # USD per tool
        self.tool_max_life_minutes = 120.0  # Expected life at nominal load
        self.logger = logging.getLogger(__name__)
    
    def evaluate_economic_impact(self, proposal: StrategyProposal, 
                               current_state: Dict[str, Any]) -> EconomicAssessment:
        """
        Evaluate the economic impact of a proposal based on the "Great Translation" mapping.
        
        Args:
            proposal: The strategy proposal to evaluate
            current_state: Current machine state for context
            
        Returns:
            EconomicAssessment with financial metrics
        """
        # Extract parameters
        rpm = proposal.proposed_parameters.get('rpm', current_state.get('rpm', 4000))
        feed_rate = proposal.proposed_parameters.get('feed_rate', current_state.get('feed_rate', 2000))
        vibration = proposal.proposed_parameters.get('vibration_x', current_state.get('vibration_x', 0.2))
        material_factor = self._get_material_factor(proposal.proposed_parameters.get('material', 'steel'))
        
        # THE GREAT TRANSLATION: Physics -> Churn (Tool Wear)
        # Higher RPM/Feed + Hard material = Faster "Churn"
        load_factor = (rpm * feed_rate) / 1000000.0
        stress_multiplier = material_factor * (1 + vibration)
        
        # Real-time Tool Wear Rate (equivalent to Customer Churn Rate)
        churn_rate = load_factor * stress_multiplier
        
        # Calculate costs (OpEx)
        estimated_cycle_time_min = 10000 / feed_rate if feed_rate > 0 else 100  # Simplified geometry
        
        time_cost = (estimated_cycle_time_min / 60) * self.hourly_machine_cost
        tool_wear_cost = (estimated_cycle_time_min / self.tool_max_life_minutes) * self.tool_replacement_cost * churn_rate
        
        total_cost = time_cost + tool_wear_cost
        
        # Calculate profit rate
        part_price = current_state.get('part_price', 500.00)
        gross_margin = part_price - total_cost
        profit_rate_per_hour = (gross_margin / estimated_cycle_time_min) * 60 if estimated_cycle_time_min > 0 else 0
        
        # Determine recommended operational mode
        recommended_mode = self._determine_operational_mode(churn_rate, profit_rate_per_hour)
        
        # Calculate ROI projection
        roi_projection = (profit_rate_per_hour / self.hourly_machine_cost) * 100 if self.hourly_machine_cost > 0 else 0
        
        assessment = EconomicAssessment(
            proposal_id=proposal.proposal_id,
            churn_risk=round(churn_rate, 2),
            projected_profit_rate=round(profit_rate_per_hour, 2),
            tool_wear_cost=round(tool_wear_cost, 2),
            recommended_mode=recommended_mode,
            economic_timestamp=datetime.utcnow(),
            roi_projection=round(roi_projection, 2)
        )
        
        self.logger.info(f"Proposal {proposal.proposal_id} - Churn Risk: {assessment.churn_risk}, "
                        f"Projected Profit: ${assessment.projected_profit_rate}/hr, Mode: {assessment.recommended_mode}")
        
        return assessment
    
    def _get_material_factor(self, material: str) -> float:
        """Get material-specific stress factor for the Great Translation."""
        material_factors = {
            'aluminum': 0.7,      # Softer material, less wear
            'steel': 1.0,         # Standard material
            'titanium': 1.3,      # Harder material, more wear
            'inconel': 1.5,       # Very hard material, much more wear
            'cast_iron': 0.9,     # Brittle but dense
            'brass': 0.6          # Soft and lubricious
        }
        
        return material_factors.get(material.lower(), 1.0)
    
    def _determine_operational_mode(self, churn_rate: float, profit_rate: float) -> str:
        """Determine the recommended operational mode based on economic physics."""
        if churn_rate > 2.0:
            return "ECONOMY_MODE"  # Tool is wearing too fast, be conservative
        elif profit_rate > 200.0 and churn_rate < 1.2:
            return "RUSH_MODE"     # High margin, low wear -> maximize throughput
        else:
            return "BALANCED_MODE"  # Balance safety and performance


class ShadowCouncil:
    """
    The Shadow Council - Governance system that orchestrates Creator, Auditor, and Accountant
    Implements the "Invisible Church" reasoning trace that explains all decisions.
    """
    
    def __init__(self, creator_agent: CreatorAgent, auditor_agent: AuditorAgent, 
                 accountant_agent: AccountantAgent):
        self.creator = creator_agent
        self.auditor = auditor_agent
        self.accountant = accountant_agent
        self.logger = logging.getLogger(__name__)
        
    def evaluate_strategy(self, intent: str, current_machine_state: Dict[str, Any], 
                         machine_id: int) -> CouncilDecision:
        """
        Complete Shadow Council evaluation process:
        1. Creator proposes strategy based on intent
        2. Auditor validates with physics constraints (Death Penalty)
        3. Accountant evaluates economic impact
        4. Council renders final decision
        
        Args:
            intent: Operator's intent for the operation
            current_machine_state: Current state of the machine
            machine_id: ID of the machine being controlled
            
        Returns:
            CouncilDecision with final approval and reasoning trace
        """
        self.logger.info(f"Shadow Council evaluating intent: '{intent}' for machine {machine_id}")
        
        # 1. Creator proposes optimization strategy
        proposal = self.creator.propose_strategy(intent, current_machine_state)
        
        # 2. Auditor validates against physics constraints
        validation = self.auditor.validate_proposal(proposal, current_machine_state)
        
        # 3. Accountant evaluates economic impact
        economic_assessment = None
        if validation.is_approved:
            economic_assessment = self.accountant.evaluate_economic_impact(proposal, current_machine_state)
        else:
            # If Auditor rejects, no economic evaluation needed
            economic_assessment = EconomicAssessment(
                proposal_id=proposal.proposal_id,
                churn_risk=0.0,
                projected_profit_rate=0.0,
                tool_wear_cost=0.0,
                recommended_mode="MANUAL_OVERRIDE_REQUIRED",
                economic_timestamp=datetime.utcnow(),
                roi_projection=0.0
            )
        
        # 4. Combine all evaluations for final decision
        council_approval = validation.is_approved
        final_fitness = validation.fitness_score
        
        # Create comprehensive reasoning trace (The "Invisible Church")
        reasoning_trace = []
        reasoning_trace.extend([
            f"SHADOW_COUNCIL_EVALUATION for machine {machine_id}",
            f"Intent: {intent}",
            f"Proposed Strategy: {proposal.strategy_name}",
            f"Creator Confidence: {proposal.creator_confidence:.2f}",
            f"Creator Reasoning: {proposal.creator_reasoning}"
        ])
        reasoning_trace.extend(validation.reasoning_trace)
        reasoning_trace.append(f"Economic Assessment: Churn Risk={economic_assessment.churn_risk}, "
                              f"Profit Rate=${economic_assessment.projected_profit_rate}/hr")
        reasoning_trace.append(f"Recommended Mode: {economic_assessment.recommended_mode}")
        reasoning_trace.append(f"Council Decision: {'APPROVED' if council_approval else 'REJECTED'}")
        
        # Calculate decision confidence based on agreement between agents
        if council_approval:
            # If approved, confidence is based on economic assessment and creator confidence
            decision_confidence = (proposal.creator_confidence + 
                                 (1.0 if economic_assessment.recommended_mode == "RUSH_MODE" else 0.7)) / 2
        else:
            # If rejected, high confidence in safety decision
            decision_confidence = 0.95
        
        decision = CouncilDecision(
            decision_id=f"DEC_{uuid.uuid4().hex[:8]}",
            proposal=proposal,
            validation=validation,
            economic_assessment=economic_assessment,
            council_approval=council_approval,
            final_fitness=final_fitness,
            reasoning_trace=reasoning_trace,
            decision_timestamp=datetime.utcnow(),
            decision_confidence=min(1.0, decision_confidence)
        )
        
        self.logger.info(f"Shadow Council decision: {decision.council_approval}, "
                        f"fitness: {decision.final_fitness:.3f}, "
                        f"confidence: {decision.decision_confidence:.3f}")
        
        return decision
    
    def get_reasoning_trace(self, decision: CouncilDecision) -> str:
        """
        Get the complete reasoning trace for a decision (The "Invisible Church").
        This provides transparency into how the Council arrived at its decision.
        """
        trace_lines = [
            f"SHADOW COUNCIL REASONING TRACE",
            f"Decision ID: {decision.decision_id}",
            f"Timestamp: {decision.decision_timestamp.isoformat()}",
            f"Overall Outcome: {'APPROVED' if decision.council_approval else 'REJECTED'}",
            f"Final Fitness: {decision.final_fitness:.3f}",
            f"Decision Confidence: {decision.decision_confidence:.3f}",
            "",
            "CREATOR AGENT REASONING:",
            f"  - Strategy: {decision.proposal.strategy_name}",
            f"  - Intent: {decision.proposal.intent}",
            f"  - Confidence: {decision.proposal.creator_confidence:.3f}",
            f"  - Reasoning: {decision.proposal.creator_reasoning}",
            "",
            "AUDITOR AGENT REASONING:",
        ]
        
        for reasoning_line in decision.validation.reasoning_trace:
            trace_lines.append(f"  - {reasoning_line}")
        
        trace_lines.extend([
            "",
            "ACCOUNTANT AGENT REASONING:",
            f"  - Churn Risk: {decision.economic_assessment.churn_risk:.2f}x normal",
            f"  - Projected Profit Rate: ${decision.economic_assessment.projected_profit_rate}/hr",
            f"  - Tool Wear Cost: ${decision.economic_assessment.tool_wear_cost}",
            f"  - Recommended Mode: {decision.economic_assessment.recommended_mode}",
            f"  - ROI Projection: {decision.economic_assessment.roi_projection:.1f}%",
            "",
            "COUNCIL SYNTHESIS:",
            f"  - Final Decision: {'APPROVED' if decision.council_approval else 'REJECTED'}",
            f"  - Deterministic Validation: {'PASSED' if decision.validation.is_approved else 'FAILED'}",
            f"  - Economic Viability: {'POSITIVE' if decision.economic_assessment.projected_profit_rate > 0 else 'NEGATIVE'}",
        ])
        
        return "\n".join(trace_lines)


# Example usage
if __name__ == "__main__":
    print("Shadow Council Governance initialized successfully.")
    print("Ready to orchestrate Creator, Auditor, and Accountant agents.")
    
    # Initialize the three agents
    creator = CreatorAgent()
    auditor = AuditorAgent()
    accountant = AccountantAgent()
    
    # Create the Shadow Council
    council = ShadowCouncil(creator, auditor, accountant)
    
    # Example 1: Conservative intent (should pass all checks)
    print("\n--- Example 1: Conservative Operation ---")
    conservative_intent = "face mill aluminum block conservatively to ensure tool longevity"
    conservative_state = {
        'rpm': 4000,
        'feed_rate': 2000,
        'spindle_load': 65.0,
        'temperature': 38.0,
        'vibration_x': 0.3,
        'vibration_y': 0.2,
        'coolant_flow': 1.8,
        'material': 'aluminum',
        'operation_type': 'face_mill',
        'part_price': 450.00
    }
    
    decision1 = council.evaluate_strategy(conservative_intent, conservative_state, machine_id=1)
    print(f"Decision: {decision1.council_approval}")
    print(f"Fitness: {decision1.final_fitness:.3f}")
    print(f"Recommended Mode: {decision1.economic_assessment.recommended_mode}")
    print(f"Projected Profit: ${decision1.economic_assessment.projected_profit_rate}/hr")
    
    # Example 2: Aggressive intent that might trigger Death Penalty
    print("\n--- Example 2: Aggressive Operation ---")
    aggressive_intent = "face mill inconel block aggressively to maximize speed"
    aggressive_state = {
        'rpm': 11000,
        'feed_rate': 4500,
        'spindle_load': 85.0,
        'temperature': 55.0,
        'vibration_x': 1.2,
        'vibration_y': 1.0,
        'coolant_flow': 1.2,
        'material': 'inconel',
        'operation_type': 'face_mill',
        'part_price': 800.00,
        'path_curvature_radius': 0.3  # Small radius triggering Quadratic Mantinel
    }
    
    decision2 = council.evaluate_strategy(aggressive_intent, aggressive_state, machine_id=2)
    print(f"Decision: {decision2.council_approval}")
    print(f"Fitness: {decision2.final_fitness:.3f}")
    print(f"Recommended Mode: {decision2.economic_assessment.recommended_mode}")
    print(f"Projected Profit: ${decision2.economic_assessment.projected_profit_rate}/hr")
    print(f"Constraint Violations: {len(decision2.validation.constraint_violations)}")
    
    # Show reasoning trace for the second decision
    print("\n--- Reasoning Trace (Invisible Church) ---")
    print(council.get_reasoning_trace(decision2))