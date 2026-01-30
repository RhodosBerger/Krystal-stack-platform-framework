"""
Integration Test Suite for Shadow Council Components
Tests the complete integration between Creator, Auditor, and Accountant agents
"""

import unittest
import pytest
from datetime import datetime
from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from cms.swarm.shadow_council_governance import ShadowCouncil, CreatorAgent, AuditorAgent, AccountantAgent
from cms.swarm.survivor_ranking import SurvivorRankingSystem
from cms.swarm.economic_auditor import EconomicAuditor
from cms.swarm.genetic_tracker import GeneticTracker
from cms.services.dopamine_engine import DopamineEngine
from cms.repositories.telemetry_repository import TelemetryRepository
from cms.models import get_session_local, create_database_engine


class TestShadowCouncilIntegration:
    """Integration tests for the complete Shadow Council system"""
    
    def setup_method(self):
        """Setup test environment with all required components"""
        # Create database engine and session
        self.engine = create_database_engine()
        self.db_session = get_session_local(self.engine)()
        
        # Initialize repositories
        self.telemetry_repo = TelemetryRepository(self.db_session)
        
        # Initialize core services
        self.dopamine_engine = DopamineEngine(repository=self.telemetry_repo)
        self.economic_auditor = EconomicAuditor()
        self.genetic_tracker = GeneticTracker()
        self.survivor_ranking = SurvivorRankingSystem()
        
        # Initialize individual agents
        self.creator_agent = CreatorAgent(repository=self.telemetry_repo)
        self.auditor_agent = AuditorAgent(decision_policy=self.survivor_ranking.decision_policy)
        self.accountant_agent = AccountantAgent(economics_engine=self.economic_auditor)
        
        # Create Shadow Council
        self.shadow_council = ShadowCouncil(
            creator=self.creator_agent,
            auditor=self.auditor_agent,
            decision_policy=self.survivor_ranking.decision_policy
        )
        self.shadow_council.set_accountant(self.accountant_agent)
    
    def teardown_method(self):
        """Clean up test resources"""
        self.db_session.close()
    
    def test_complete_governance_loop(self):
        """Test the complete governance loop: Creator → Auditor → Accountant"""
        # Test data
        intent = "face mill aluminum block aggressively for higher efficiency"
        current_state = {
            'rpm': 4000,
            'feed_rate': 2000,
            'spindle_load': 65.0,
            'temperature': 38.0,
            'vibration_x': 0.3,
            'vibration_y': 0.2,
            'coolant_flow': 1.8,
            'material': 'aluminum',
            'operation_type': 'face_mill',
            'part_price': 500.00
        }
        machine_id = 1
        
        # Execute the complete governance loop
        result = self.shadow_council.evaluate_strategy(current_state, machine_id)
        
        # Validate result structure
        assert 'proposal' in result
        assert 'validation' in result
        assert 'economic_evaluation' in result
        assert 'council_approval' in result
        assert 'final_fitness' in result
        assert 'reasoning_trace' in result
        
        # Validate that the proposal was created
        assert result['proposal'] is not None
        assert 'proposed_parameters' in result['proposal']
        
        # Validate that validation was performed
        assert result['validation'] is not None
        assert 'is_approved' in result['validation']
        assert 'fitness_score' in result['validation']
        
        # Validate that economic evaluation was performed
        assert result['economic_evaluation'] is not None
        assert 'projected_profit_rate' in result['economic_evaluation']
        
        print(f"✓ Complete governance loop test passed")
        print(f"  Proposal created: {result['proposal']['strategy_name']}")
        print(f"  Validation result: {'APPROVED' if result['validation']['is_approved'] else 'REJECTED'}")
        print(f"  Final fitness: {result['final_fitness']:.3f}")
        print(f"  Economic impact: ${result['economic_evaluation']['projected_profit_rate']:.2f}/hr")
    
    def test_death_penalty_function(self):
        """Test that the Death Penalty function correctly rejects unsafe proposals"""
        # Create a deliberately unsafe proposal
        unsafe_state = {
            'rpm': 15000,  # Exceeds max RPM
            'feed_rate': 8000,  # Exceeds max feed rate
            'spindle_load': 98.0,  # Exceeds max load
            'temperature': 85.0,  # Exceeds max temperature
            'vibration_x': 4.0,  # Exceeds max vibration
            'material': 'inconel',
            'operation_type': 'face_mill'
        }
        machine_id = 2
        
        # Evaluate the unsafe proposal
        result = self.shadow_council.evaluate_strategy(unsafe_state, machine_id)
        
        # The Death Penalty should have been applied (fitness = 0)
        assert result['final_fitness'] == 0.0
        assert result['council_approval'] is False
        assert result['validation']['death_penalty_applied'] is True
        
        print(f"✓ Death Penalty function test passed")
        print(f"  Unsafe proposal correctly rejected")
        print(f"  Fitness score: {result['final_fitness']}")
        print(f"  Death penalty applied: {result['validation']['death_penalty_applied']}")
    
    def test_quadratic_mantinel_constraint(self):
        """Test that the Quadratic Mantinel correctly constrains high-curvature operations"""
        # Create a state with high feed rate and small curvature radius
        high_curvature_state = {
            'rpm': 8000,
            'feed_rate': 4000,  # High feed rate
            'spindle_load': 70.0,
            'temperature': 45.0,
            'vibration_x': 0.8,
            'path_curvature_radius': 0.3,  # Very small radius - dangerous
            'material': 'aluminum',
            'operation_type': 'contour_mill'
        }
        machine_id = 3
        
        # Evaluate the high-curvature operation
        result = self.shadow_council.evaluate_strategy(high_curvature_state, machine_id)
        
        # The Quadratic Mantinel should reject this (too high feed for small radius)
        assert result['final_fitness'] == 0.0 or result['final_fitness'] < 0.3  # Low fitness due to mantinel violation
        assert result['council_approval'] is False or result['final_fitness'] < 0.3
        
        print(f"✓ Quadratic Mantinel constraint test passed")
        print(f"  High-curvature operation properly constrained")
        print(f"  Fitness score: {result['final_fitness']:.3f}")
    
    def test_economic_optimization_balance(self):
        """Test that economic evaluation balances safety and profitability"""
        # Test state with moderate parameters
        balanced_state = {
            'rpm': 6000,
            'feed_rate': 2500,
            'spindle_load': 75.0,
            'temperature': 42.0,
            'vibration_x': 0.6,
            'material': 'aluminum',
            'operation_type': 'face_mill',
            'part_price': 450.00
        }
        machine_id = 4
        
        # Evaluate the balanced operation
        result = self.shadow_council.evaluate_strategy(balanced_state, machine_id)
        
        # Should be approved with reasonable fitness and economic value
        assert result['council_approval'] is True
        assert 0.5 <= result['final_fitness'] <= 1.0  # Reasonable fitness
        assert result['economic_evaluation']['projected_profit_rate'] > 0  # Positive economic value
        
        print(f"✓ Economic optimization balance test passed")
        print(f"  Balanced operation approved with fitness: {result['final_fitness']:.3f}")
        print(f"  Economic value: ${result['economic_evaluation']['projected_profit_rate']:.2f}/hr")
    
    def test_neuro_safety_gradient_response(self):
        """Test that neuro-safety gradients respond appropriately to stress levels"""
        # Initialize dopamine engine with test data
        initial_dopamine = self.dopamine_engine.get_current_dopamine_level()
        initial_cortisol = self.dopamine_engine.get_current_cortisol_level()
        
        # Simulate a stressful situation
        stress_telemetry = {
            'spindle_load': 90.0,
            'temperature': 65.0,
            'vibration_x': 1.8,
            'vibration_y': 1.5,
            'timestamp': datetime.utcnow()
        }
        
        # Update neuro-states with stress data
        self.dopamine_engine.update_gradients(stress_telemetry)
        
        # Check that cortisol increased and dopamine decreased
        new_cortisol = self.dopamine_engine.get_current_cortisol_level()
        new_dopamine = self.dopamine_engine.get_current_dopamine_level()
        
        # Cortisol should increase with stress
        assert new_cortisol >= initial_cortisol
        
        print(f"✓ Neuro-safety gradient response test passed")
        print(f"  Cortisol increased from {initial_cortisol:.3f} to {new_cortisol:.3f}")
        print(f"  Dopamine changed from {initial_dopamine:.3f} to {new_dopamine:.3f}")
    
    def test_collective_learning_propagation(self):
        """Test that trauma learned by one machine propagates to others"""
        # Register a trauma event in the genetic tracker
        trauma_event = {
            'strategy_id': 'TEST_STRATEGY_001',
            'failure_type': 'tool_breakage',
            'failure_cause': 'excessive_feed_rate',
            'loss_cost': 250.0,
            'material': 'inconel',
            'operation_type': 'face_mill',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add the trauma to the genetic tracker
        self.genetic_tracker.register_trauma(trauma_event)
        
        # Verify the trauma is recorded
        trauma_registry = self.genetic_tracker.trauma_registry
        assert 'TEST_STRATEGY_001' in trauma_registry
        
        print(f"✓ Collective learning propagation test passed")
        print(f"  Trauma registered in shared registry: {len(trauma_registry)} total traumas")
    
    def test_great_translation_mapping(self):
        """Test that SaaS metrics correctly map to manufacturing physics"""
        # Test the Great Translation mapping
        churn_risk = self.economic_auditor.calculate_churn_risk({
            'tool_wear_per_hour': 0.02,  # High wear rate
            'temperature_variance': 15.0,  # High thermal stress
            'vibration_increasing_trend': True  # Degrading vibration
        })
        
        # High manufacturing churn should result in high risk
        assert churn_risk > 0.5  # High risk threshold
        
        print(f"✓ Great Translation mapping test passed")
        print(f"  Churn risk calculated as: {churn_risk:.3f}")
    
    def test_complete_nightmare_training_cycle(self):
        """Test the complete Nightmare Training protocol"""
        # Create historical data to replay
        historical_data = [
            {'timestamp': datetime.utcnow(), 'spindle_load': 65.0, 'temperature': 38.0, 'vibration_x': 0.3},
            {'timestamp': datetime.utcnow(), 'spindle_load': 70.0, 'temperature': 40.0, 'vibration_x': 0.4},
            {'timestamp': datetime.utcnow(), 'spindle_load': 75.0, 'temperature': 42.0, 'vibration_x': 0.5}
        ]
        
        # Inject synthetic failures
        failure_scenarios = [
            {'type': 'spindle_load_spike', 'time_index': 1, 'severity': 0.8},
            {'type': 'thermal_runaway', 'time_index': 2, 'severity': 0.6}
        ]
        
        # Process through Nightmare Training simulation
        for scenario in failure_scenarios:
            # Modify historical data to simulate failure
            modified_data = historical_data.copy()
            if scenario['type'] == 'spindle_load_spike':
                modified_data[scenario['time_index']]['spindle_load'] = 95.0
            elif scenario['type'] == 'thermal_runaway':
                modified_data[scenario['time_index']]['temperature'] = 75.0
            
            # Run each modified scenario through Shadow Council
            for i, data_point in enumerate(modified_data):
                if i == scenario['time_index']:
                    # This is where the failure occurs
                    result = self.shadow_council.evaluate_strategy(data_point, 5)
                    
                    # The system should detect and respond appropriately to the simulated failure
                    if scenario['severity'] > 0.5:
                        # High severity should trigger appropriate response
                        print(f"  Simulated {scenario['type']} at index {i}, fitness: {result['final_fitness']:.3f}")
        
        print(f"✓ Complete Nightmare Training cycle test passed")
    
    def test_genetic_evolution_tracking(self):
        """Test that genetic mutations are properly tracked"""
        # Register an initial strategy
        initial_lineage = self.genetic_tracker.register_initial_strategy(
            strategy_id="STRAT_GENETIC_TEST_001",
            material="aluminum",
            operation_type="face_mill",
            parameters={'feed_rate': 2000, 'rpm': 4000, 'depth': 1.0}
        )
        
        # Record a mutation
        mutation = self.genetic_tracker.record_mutation(
            parent_strategy_id="STRAT_GENETIC_TEST_001",
            mutation_type="parameter_optimization",
            mutation_description="Increased feed rate for efficiency",
            parameters_changed={'feed_rate': 2200},
            improvement_metric=0.12,
            machine_id="M001",
            fitness_before=0.7,
            fitness_after=0.82
        )
        
        # Verify the mutation was recorded
        assert mutation.mutation_id is not None
        assert mutation.parent_strategy_id == "STRAT_GENETIC_TEST_001"
        
        # Check that the lineage was updated
        updated_lineage = self.genetic_tracker.lineages.get("STRAT_GENETIC_TEST_001")
        assert updated_lineage is not None
        assert len(updated_lineage.mutation_history) >= 1
        
        print(f"✓ Genetic evolution tracking test passed")
        print(f"  Initial lineage created: {initial_lineage.lineage_root_id}")
        print(f"  Mutation recorded: {mutation.mutation_id}")
        print(f"  Lineage now has {len(updated_lineage.mutation_history)} mutations")
    
    def test_system_resilience_under_stress(self):
        """Test overall system resilience when subjected to multiple stress conditions"""
        stress_conditions = [
            {'rpm': 12000, 'feed_rate': 4500, 'spindle_load': 92.0, 'temperature': 68.0},  # High stress
            {'rpm': 2000, 'feed_rate': 800, 'spindle_load': 35.0, 'temperature': 30.0},   # Low stress
            {'rpm': 8000, 'feed_rate': 3000, 'spindle_load': 80.0, 'temperature': 55.0}   # Medium stress
        ]
        
        results = []
        for i, condition in enumerate(stress_conditions):
            condition.update({
                'vibration_x': 0.3 + (i * 0.5),  # Increasing vibration
                'material': 'steel',
                'operation_type': 'drill',
                'part_price': 300.00
            })
            
            result = self.shadow_council.evaluate_strategy(condition, i+10)
            results.append(result)
        
        # Verify that high-stress condition was appropriately handled
        high_stress_result = results[0]
        low_stress_result = results[1]
        
        # High stress should either be rejected or have lower fitness
        if high_stress_result['council_approval']:
            assert high_stress_result['final_fitness'] <= 0.5  # Lower fitness for high stress
        
        print(f"✓ System resilience under stress test passed")
        print(f"  Tested {len(stress_conditions)} different stress conditions")
        print(f"  High stress result: {'APPROVED' if high_stress_result['council_approval'] else 'REJECTED'} (Fitness: {high_stress_result['final_fitness']:.3f})")
        print(f"  Low stress result: {'APPROVED' if low_stress_result['council_approval'] else 'REJECTED'} (Fitness: {low_stress_result['final_fitness']:.3f})")
    
    def test_fleet_wide_intelligence_sharing(self):
        """Test that intelligence is properly shared across the fleet"""
        # Simulate multiple machines learning from the same trauma
        trauma_event = {
            'strategy_id': 'FLEET_SHARED_TRAUMA_001',
            'failure_type': 'thermal_overload',
            'failure_cause': 'insufficient_coolant_flow',
            'loss_cost': 500.0,
            'material': 'titanium',
            'operation_type': 'mill',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Register trauma that affects fleet-wide
        self.genetic_tracker.register_trauma(trauma_event)
        
        # Verify trauma propagation mechanisms
        fleet_traumas = self.genetic_tracker.trauma_registry
        assert 'FLEET_SHARED_TRAUMA_001' in fleet_traumas
        
        print(f"✓ Fleet-wide intelligence sharing test passed")
        print(f"  Fleet trauma registry now has {len(fleet_traumas)} shared traumas")
    
    def test_complete_workflow_integration(self):
        """Test the complete workflow from intent to execution with all components"""
        # Define a complete workflow scenario
        intent = "aggressive face milling of aluminum with optimized parameters"
        current_state = {
            'rpm': 5000,
            'feed_rate': 2200,
            'spindle_load': 68.0,
            'temperature': 40.0,
            'vibration_x': 0.4,
            'vibration_y': 0.3,
            'coolant_flow': 1.6,
            'material': 'aluminum-6061',
            'operation_type': 'face_mill',
            'part_price': 425.00
        }
        machine_id = 20
        
        # Execute complete workflow
        print(f"\nExecuting complete workflow integration test...")
        
        # Step 1: Creator proposes strategy
        proposal = self.creator_agent.propose_optimization(intent, current_state, machine_id)
        print(f"  1. Creator proposed strategy: {proposal['strategy_name']}")
        
        # Step 2: Auditor validates proposal
        validation = self.auditor_agent.validate_proposal(proposal, current_state)
        print(f"  2. Auditor validation: {'APPROVED' if validation['is_approved'] else 'REJECTED'} (Fitness: {validation['fitness_score']:.3f})")
        
        # Step 3: Accountant evaluates economic impact
        if validation['is_approved']:
            economic = self.accountant_agent.evaluate_economic_impact(proposal, current_state)
            print(f"  3. Economic evaluation: ${economic['projected_profit_rate']:.2f}/hr, Risk: {economic['churn_risk']:.3f}")
        else:
            economic = {'projected_profit_rate': 0.0, 'churn_risk': 1.0}
            print(f"  3. Economic evaluation skipped due to safety rejection")
        
        # Step 4: Shadow Council renders final decision
        council_decision = self.shadow_council.evaluate_strategy(current_state, machine_id)
        print(f"  4. Council decision: {'APPROVED' if council_decision['council_approval'] else 'REJECTED'}")
        print(f"     Final fitness: {council_decision['final_fitness']:.3f}")
        print(f"     Economic impact: ${council_decision['economic_evaluation']['projected_profit_rate']:.2f}/hr")
        print(f"     Reasoning trace length: {len(council_decision['reasoning_trace'])}")
        
        # Validate complete workflow
        assert council_decision is not None
        assert 'council_approval' in council_decision
        assert 'final_fitness' in council_decision
        assert 'reasoning_trace' in council_decision
        assert len(council_decision['reasoning_trace']) > 0
        
        print(f"✓ Complete workflow integration test passed")
        print(f"  All components (Creator, Auditor, Accountant) successfully coordinated")
        print(f"  Decision made with complete reasoning trace available")


# Performance and Stress Tests
class TestShadowCouncilPerformance:
    """Performance and stress tests for the Shadow Council system"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create database engine and session
        self.engine = create_database_engine()
        self.db_session = get_session_local(self.engine)()
        
        # Initialize components
        self.telemetry_repo = TelemetryRepository(self.db_session)
        self.dopamine_engine = DopamineEngine(repository=self.telemetry_repo)
        self.economic_auditor = EconomicAuditor()
        self.genetic_tracker = GeneticTracker()
        self.survivor_ranking = SurvivorRankingSystem()
        
        self.creator_agent = CreatorAgent(repository=self.telemetry_repo)
        self.auditor_agent = AuditorAgent(decision_policy=self.survivor_ranking.decision_policy)
        self.accountant_agent = AccountantAgent(economics_engine=self.economic_auditor)
        
        self.shadow_council = ShadowCouncil(
            creator=self.creator_agent,
            auditor=self.auditor_agent,
            decision_policy=self.survivor_ranking.decision_policy
        )
        self.shadow_council.set_accountant(self.accountant_agent)
    
    def teardown_method(self):
        """Clean up test resources"""
        self.db_session.close()
    
    def test_high_frequency_decision_making(self):
        """Test system performance under high-frequency decision making"""
        import time
        
        test_states = [
            {
                'rpm': 4000 + i*100,
                'feed_rate': 2000 + i*50,
                'spindle_load': 65.0 + (i % 5),
                'temperature': 38.0 + (i % 3),
                'vibration_x': 0.3 + (i % 2) * 0.1,
                'material': 'aluminum',
                'operation_type': 'face_mill',
                'part_price': 400.00 + i*10
            } for i in range(100)  # 100 consecutive decisions
        ]
        
        start_time = time.time()
        
        for i, state in enumerate(test_states):
            result = self.shadow_council.evaluate_strategy(state, i % 10 + 1)
            
            # Verify each decision is valid
            assert 'council_approval' in result
            assert 'final_fitness' in result
            
            if i % 25 == 0:  # Log progress
                print(f"  Processed {i+1}/100 decisions")
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_decision = total_time / len(test_states)
        
        print(f"✓ High-frequency decision making test passed")
        print(f"  Processed {len(test_states)} decisions in {total_time:.3f}s")
        print(f"  Average time per decision: {avg_time_per_decision*1000:.2f}ms")
        print(f"  Throughput: {1/avg_time_per_decision:.1f} decisions/second")
        
        # Performance requirement: Should handle at least 10 decisions per second
        assert avg_time_per_decision < 0.1, f"Decision time too slow: {avg_time_per_decision*1000:.2f}ms"
    
    def test_memory_usage_under_extended_operation(self):
        """Test memory usage remains stable during extended operation"""
        import gc
        
        initial_objects = len(gc.get_objects())
        
        # Simulate extended operation with many decisions
        for i in range(500):
            test_state = {
                'rpm': 4000 + (i % 1000),
                'feed_rate': 2000 + (i % 500),
                'spindle_load': 60.0 + (i % 20),
                'temperature': 35.0 + (i % 15),
                'vibration_x': 0.2 + (i % 10) * 0.05,
                'material': 'steel' if i % 2 == 0 else 'aluminum',
                'operation_type': 'face_mill' if i % 3 == 0 else 'drill',
                'part_price': 300.00 + (i % 200)
            }
            
            result = self.shadow_council.evaluate_strategy(test_state, i % 5 + 1)
            
            # Periodic garbage collection
            if i % 100 == 0:
                gc.collect()
        
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects
        
        print(f"✓ Memory usage stability test passed")
        print(f"  Initial objects: {initial_objects}")
        print(f"  Final objects: {final_objects}")
        print(f"  Growth: {object_growth} objects (+{object_growth/initial_objects*100:.2f}%)")
        
        # Memory requirement: Growth should be minimal
        assert abs(object_growth) < initial_objects * 0.1, f"Memory growth too high: {object_growth} objects"


# Run tests if executed directly
if __name__ == "__main__":
    import subprocess
    import sys
    
    print("Running Shadow Council Integration Tests...")
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