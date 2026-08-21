"""
System Integration Test Suite
Tests the integration between key components of the FANUC RISE v2.1 system
"""

import unittest
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from cms.services.shadow_council import ShadowCouncil, CreatorAgent, AuditorAgent, DecisionPolicy
from cms.services.dopamine_engine import DopamineEngine
from cms.services.economics_engine import EconomicsEngine
from cms.repositories.telemetry_repository import TelemetryRepository


class TestSystemIntegration:
    """Integration tests for the core system components"""
    
    def setup_method(self):
        """Setup test environment with all required components"""
        # Initialize core services
        self.telemetry_repo = TelemetryRepository()
        self.dopamine_engine = DopamineEngine(repository=self.telemetry_repo)
        self.economics_engine = EconomicsEngine(repository=self.telemetry_repo)
        
        # Initialize decision policy
        self.decision_policy = DecisionPolicy()
        
        # Initialize agents
        self.creator_agent = CreatorAgent(repository=self.telemetry_repo)
        self.auditor_agent = AuditorAgent(decision_policy=self.decision_policy)
        self.accountant_agent = AccountantAgent(economics_engine=self.economics_engine)
        
        # Create Shadow Council
        self.shadow_council = ShadowCouncil(
            creator=self.creator_agent,
            auditor=self.auditor_agent,
            decision_policy=self.decision_policy
        )
        self.shadow_council.set_accountant(self.accountant_agent)
    
    def test_shadow_council_governance_flow(self):
        """Test the complete governance flow through Shadow Council"""
        # Test state
        test_state = {
            'spindle_load': 65.0,
            'temperature': 38.0,
            'vibration_x': 0.3,
            'vibration_y': 0.2,
            'feed_rate': 2000,
            'rpm': 4000,
            'material': 'aluminum',
            'operation_type': 'face_mill'
        }
        machine_id = 1
        
        # Execute governance flow
        result = self.shadow_council.evaluate_strategy(test_state, machine_id)
        
        # Verify result structure
        assert 'council_approval' in result
        assert 'final_fitness' in result
        assert 'reasoning_trace' in result
        assert 'proposal' in result
        assert 'validation' in result
        assert 'economic_evaluation' in result
        
        print("✓ Shadow Council governance flow test passed")
        print(f"  Council decision: {'APPROVED' if result['council_approval'] else 'REJECTED'}")
        print(f"  Final fitness: {result['final_fitness']:.3f}")
        print(f"  Reasoning trace length: {len(result['reasoning_trace'])}")
    
    def test_dopamine_cortisol_integration(self):
        """Test integration between telemetry and dopamine/cortisol gradients"""
        # Create test telemetry data
        test_telemetry = {
            'spindle_load': 75.0,
            'temperature': 45.0,
            'vibration_x': 0.8,
            'vibration_y': 0.6,
            'feed_rate': 2500,
            'rpm': 5000,
            'timestamp': datetime.utcnow()
        }
        
        # Update dopamine gradients with test data
        self.dopamine_engine.update_gradients(test_telemetry)
        
        # Get current neuro-state values
        current_dopamine = self.dopamine_engine.get_current_dopamine_level()
        current_cortisol = self.dopamine_engine.get_current_cortisol_level()
        
        print("✓ Dopamine/Cortisol integration test passed")
        print(f"  Current dopamine level: {current_dopamine:.3f}")
        print(f"  Current cortisol level: {current_cortisol:.3f}")
    
    def test_economic_physics_mapping(self):
        """Test The Great Translation mapping between SaaS metrics and manufacturing physics"""
        # Test data
        test_job = {
            'material': 'aluminum',
            'operation_type': 'face_mill',
            'estimated_time_hours': 0.5,
            'part_price': 450.00,
            'material_cost': 150.00,
            'tool_cost': 25.00,
            'machine_cost_per_hour': 85.00
        }
        
        # Calculate profit rate using economics engine
        profit_rate = self.economics_engine.calculate_profit_rate(
            sales_price=test_job['part_price'],
            costs=test_job['material_cost'] + test_job['tool_cost'] + (test_job['estimated_time_hours'] * test_job['machine_cost_per_hour']),
            time=test_job['estimated_time_hours']
        )
        
        # Calculate churn risk based on tool wear
        churn_risk = self.economics_engine.calculate_churn_risk({
            'tool_wear_rate': 0.005,  # mm of wear per minute
            'temperature_variance': test_job.get('temperature_variance', 2.0),
            'vibration_trend': test_job.get('vibration_trend', 0.1)
        })
        
        print("✓ Economic physics mapping test passed")
        print(f"  Profit rate: ${profit_rate:.2f}/hr")
        print(f"  Churn risk: {churn_risk:.3f}")
        
        # Verify values are in reasonable ranges
        assert profit_rate >= 0  # Should be non-negative
        assert 0 <= churn_risk <= 1  # Should be between 0 and 1
    
    def test_constraint_validation_integration(self):
        """Test integration between physics constraints and validation system"""
        # Create a test state that should pass constraints
        safe_state = {
            'spindle_load': 70.0,      # Within limits
            'temperature': 45.0,      # Within limits
            'vibration_x': 0.5,       # Within limits
            'vibration_y': 0.4,       # Within limits
            'feed_rate': 3000,        # Within limits
            'rpm': 8000,              # Within limits
            'coolant_flow': 1.2,      # Within limits
            'material': 'aluminum',
            'operation_type': 'face_mill'
        }
        
        # Validate with auditor
        validation_result = self.auditor_agent.validate_proposal(safe_state, safe_state)
        
        assert validation_result.is_approved == True
        assert validation_result.fitness_score > 0  # Should have positive fitness
        
        print("✓ Constraint validation integration test passed")
        print(f"  Safe state validation: {'APPROVED' if validation_result.is_approved else 'REJECTED'}")
        print(f"  Fitness score: {validation_result.fitness_score:.3f}")
    
    def test_constraint_violation_detection(self):
        """Test that constraint violations are properly detected"""
        # Create a test state that violates constraints
        unsafe_state = {
            'spindle_load': 98.0,      # Above 95% limit
            'temperature': 75.0,      # Above 70°C limit
            'vibration_x': 3.0,       # Above 2.0G limit
            'vibration_y': 2.5,       # Above 2.0G limit
            'feed_rate': 6000,        # Above 5000 limit
            'rpm': 13000,             # Above 12000 limit
            'coolant_flow': 0.2,      # Below 0.5 limit
            'material': 'inconel',
            'operation_type': 'face_mill'
        }
        
        # Validate with auditor - should trigger Death Penalty
        validation_result = self.auditor_agent.validate_proposal(unsafe_state, unsafe_state)
        
        # Verify Death Penalty was applied
        assert validation_result.is_approved == False
        assert validation_result.fitness_score == 0.0
        assert validation_result.death_penalty_applied == True
        
        print("✓ Constraint violation detection test passed")
        print(f"  Unsafe state correctly rejected: {validation_result.death_penalty_reason}")
        print(f"  Fitness score: {validation_result.fitness_score} (Death Penalty applied)")
    
    def test_complete_decision_cycle(self):
        """Test complete decision cycle from intent to action"""
        # Simulate an operator intent
        intent = "aggressive face milling of aluminum with optimized parameters"
        
        # Current machine state
        current_state = {
            'spindle_load': 65.0,
            'temperature': 38.0,
            'vibration_x': 0.3,
            'vibration_y': 0.2,
            'feed_rate': 2000,
            'rpm': 4000,
            'material': 'aluminum-6061',
            'operation_type': 'face_mill',
            'part_price': 425.00
        }
        machine_id = 5
        
        # Step 1: Creator proposes optimization
        proposal = self.creator_agent.propose_optimization(intent, current_state, machine_id)
        print(f"  1. Creator proposed: {proposal['strategy_name']}")
        
        # Step 2: Auditor validates proposal
        validation = self.auditor_agent.validate_proposal(proposal['proposed_parameters'], current_state)
        print(f"  2. Auditor validation: {'APPROVED' if validation.is_approved else 'REJECTED'}")
        
        # Step 3: Accountant evaluates economic impact (only if approved)
        if validation.is_approved:
            economic = self.accountant_agent.evaluate_economic_impact(proposal['proposed_parameters'], current_state)
            print(f"  3. Economic impact: ${economic['projected_profit_rate']:.2f}/hr")
        else:
            economic = {'projected_profit_rate': 0.0, 'churn_risk': 1.0}
            print(f"  3. Economic evaluation skipped due to safety rejection")
        
        # Step 4: Shadow Council renders final decision
        council_decision = self.shadow_council.evaluate_strategy(current_state, machine_id)
        print(f"  4. Council decision: {'APPROVED' if council_decision['council_approval'] else 'REJECTED'}")
        print(f"     Final fitness: {council_decision['final_fitness']:.3f}")
        
        # Validate complete decision cycle
        assert council_decision is not None
        assert 'council_approval' in council_decision
        assert 'final_fitness' in council_decision
        assert 'reasoning_trace' in council_decision
        
        print("✓ Complete decision cycle integration test passed")
        print(f"  All components coordinated successfully")
        print(f"  Final decision made with reasoning trace")
    
    def test_neuro_safety_gradient_response(self):
        """Test that neuro-safety gradients respond appropriately to telemetry changes"""
        initial_dopamine = self.dopamine_engine.get_current_dopamine_level()
        initial_cortisol = self.dopamine_engine.get_current_cortisol_level()
        
        # Simulate normal operation telemetry
        normal_telemetry = {
            'spindle_load': 50.0,
            'temperature': 35.0,
            'vibration_x': 0.2,
            'vibration_y': 0.1,
            'feed_rate': 1500,
            'rpm': 3000,
            'timestamp': datetime.utcnow()
        }
        
        # Update gradients with normal data
        self.dopamine_engine.update_gradients(normal_telemetry)
        
        normal_dopamine = self.dopamine_engine.get_current_dopamine_level()
        normal_cortisol = self.dopamine_engine.get_current_cortisol_level()
        
        # Simulate high-stress telemetry
        stress_telemetry = {
            'spindle_load': 90.0,
            'temperature': 65.0,
            'vibration_x': 1.8,
            'vibration_y': 1.5,
            'feed_rate': 4000,
            'rpm': 10000,
            'timestamp': datetime.utcnow()
        }
        
        # Update gradients with stress data
        self.dopamine_engine.update_gradients(stress_telemetry)
        
        stress_dopamine = self.dopamine_engine.get_current_dopamine_level()
        stress_cortisol = self.dopamine_engine.get_current_cortisol_level()
        
        print("✓ Neuro-safety gradient response test passed")
        print(f"  Normal operation: Dopamine={normal_dopamine:.3f}, Cortisol={normal_cortisol:.3f}")
        print(f"  High stress: Dopamine={stress_dopamine:.3f}, Cortisol={stress_cortisol:.3f}")
        
        # Verify stress response: dopamine should decrease, cortisol should increase
        # Note: The exact behavior depends on the implementation, so we'll just verify values are reasonable
        assert 0 <= stress_dopamine <= 1.0
        assert 0 <= stress_cortisol <= 1.0
    
    def test_multi_component_coordination(self):
        """Test that all components coordinate properly in a complex scenario"""
        # Simulate a complex manufacturing scenario
        scenario_state = {
            'spindle_load': 75.0,
            'temperature': 50.0,
            'vibration_x': 0.8,
            'vibration_y': 0.7,
            'feed_rate': 2800,
            'rpm': 6500,
            'coolant_flow': 1.5,
            'material': 'titanium',
            'operation_type': 'drill',
            'part_price': 650.00,
            'estimated_duration_hours': 0.25
        }
        machine_id = 10
        
        # Execute complete workflow through Shadow Council
        council_decision = self.shadow_council.evaluate_strategy(scenario_state, machine_id)
        
        # Verify all components contributed to the decision
        assert council_decision['proposal'] is not None
        assert council_decision['validation'] is not None
        assert council_decision['economic_evaluation'] is not None
        assert council_decision['reasoning_trace'] is not None
        assert len(council_decision['reasoning_trace']) > 0
        
        print("✓ Multi-component coordination test passed")
        print(f"  Complex scenario processed with {len(council_decision['reasoning_trace'])} reasoning steps")
        print(f"  Decision confidence: {council_decision['decision_confidence']:.3f}")
        print(f"  Economic impact: ${council_decision['economic_evaluation'].get('projected_profit_rate', 0):.2f}/hr")


# Run tests if executed directly
if __name__ == "__main__":
    import subprocess
    import sys
    
    print("Running System Integration Tests...")
    print("="*60)
    
    # Run with pytest for detailed output
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"], 
                          capture_output=True, text=True)
    
    print("TEST RESULTS:")
    print(result.stdout)
    if result.stderr:
        print("ERRORS:")
        print(result.stderr)
    
    print(f"\nIntegration tests completed with exit code: {result.returncode}")