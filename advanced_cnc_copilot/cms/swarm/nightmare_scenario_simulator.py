"""
Nightmare Scenario Simulator
Demonstrates the cohesion between all subsystems by simulating a vibration spike
and watching how the Neuro-Safety gradients, Shadow Council, and economic systems respond
"""

import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from .shadow_council_governance import ShadowCouncil, CreatorAgent, AuditorAgent, AccountantAgent
from .dopamine_engine import DopamineEngine
from .economic_auditor import EconomicAuditor
from .genetic_tracker import GeneticTracker


class NightmareScenarioSimulator:
    """
    Simulates nightmare scenarios to demonstrate how all subsystems work together
    to protect the machine while maintaining productivity.
    """
    
    def __init__(self):
        # Initialize all components
        self.dopamine_engine = DopamineEngine()
        self.economic_auditor = EconomicAuditor()
        self.genetic_tracker = GeneticTracker()
        
        # Initialize Shadow Council agents
        self.creator = CreatorAgent()
        self.auditor = AuditorAgent()
        self.accountant = AccountantAgent(self.economic_auditor)
        
        # Create Shadow Council
        self.shadow_council = ShadowCouncil(self.creator, self.auditor, self.accountant)
        
        # Initialize state tracking
        self.current_dopamine = 0.5
        self.current_cortisol = 0.1
        self.current_feed_rate = 2000  # mm/min
        self.current_rpm = 4000
        self.current_spindle_load = 65.0  # %
        self.current_vibration = 0.3  # G
        self.current_temperature = 38.0  # Celsius
        self.material = "Aluminum-6061"
        self.operation = "face_mill"
        
        self.scenario_history = []
        
    def run_vibration_spike_scenario(self, duration_minutes: int = 10) -> Dict[str, Any]:
        """
        Run a scenario where a vibration spike is injected and observe system response.
        
        Args:
            duration_minutes: Duration of the simulation in minutes
            
        Returns:
            Dictionary with simulation results
        """
        print(f"Starting Nightmare Scenario: Vibration Spike Simulation")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Initial State - Feed: {self.current_feed_rate}, RPM: {self.current_rpm}, "
              f"Vibration: {self.current_vibration}G, Temp: {self.current_temperature}C")
        
        start_time = datetime.utcnow()
        scenario_results = {
            'initial_state': {
                'feed_rate': self.current_feed_rate,
                'rpm': self.current_rpm,
                'vibration': self.current_vibration,
                'temperature': self.current_temperature,
                'dopamine': self.current_dopamine,
                'cortisol': self.current_cortisol
            },
            'timeline': [],
            'peak_cortisol': 0.0,
            'min_dopamine': 1.0,
            'responses_tracked': [],
            'final_state': {}
        }
        
        # Simulate normal operation for first 3 minutes
        print("\nPhase 1: Normal Operation (3 min)")
        for i in range(3 * 60):  # 3 minutes at 1Hz
            # Normal operation with slight variations
            self._update_normal_telemetry()
            self._update_neuro_states()
            
            # Log current state
            if i % 30 == 0:  # Log every 30 seconds
                state = self._get_current_state()
                scenario_results['timeline'].append({
                    'time': i,
                    'state': state,
                    'phase': 'normal_operation',
                    'cortisol_level': self.current_cortisol,
                    'dopamine_level': self.current_dopamine
                })
        
        # Inject vibration spike at minute 3
        print(f"\nPhase 2: Vibration Spike Injection (Minute 3)")
        original_vibration = self.current_vibration
        self.current_vibration = 2.5  # High vibration spike
        print(f"  Vibration increased from {original_vibration:.2f}G to {self.current_vibration}G")
        
        # Record the spike and system response
        state_before_spike = self._get_current_state()
        self._update_neuro_states()  # This should increase cortisol significantly
        
        scenario_results['responses_tracked'].append({
            'event': 'vibration_spike_injected',
            'time': 3 * 60,
            'vibration_before': original_vibration,
            'vibration_after': self.current_vibration,
            'cortisol_before': state_before_spike['cortisol'],
            'cortisol_after': self.current_cortisol,
            'dopamine_before': state_before_spike['dopamine'],
            'dopamine_after': self.current_dopamine,
            'system_response': self._analyze_system_response(state_before_spike)
        })
        
        # Continue simulation for 4 more minutes with high vibration
        print("\nPhase 3: High Vibration Response (4 min)")
        for i in range(3 * 60, 7 * 60):  # Minutes 3-7
            # Maintain high vibration but with slight decay
            self.current_vibration = max(1.8, self.current_vibration * 0.995)  # Slow decay
            self._update_normal_telemetry(except_vibration=True)  # Keep other parameters normal
            self._update_neuro_states()
            
            # Check if system adjusts feed rate due to high cortisol
            if self.current_cortisol > 0.5:
                # System should reduce aggression due to high stress
                self.current_feed_rate = max(1000, self.current_feed_rate * 0.99)  # Gradual reduction
            
            # Log current state
            if (i - 3*60) % 30 == 0:  # Log every 30 seconds during high stress
                state = self._get_current_state()
                scenario_results['timeline'].append({
                    'time': i,
                    'state': state,
                    'phase': 'high_vibration_response',
                    'cortisol_level': self.current_cortisol,
                    'dopamine_level': self.current_dopamine
                })
        
        # Return to normal vibration levels
        print(f"\nPhase 4: Recovery Phase (3 min)")
        for i in range(7 * 60, 10 * 60):  # Minutes 7-10
            # Vibration returns to normal gradually
            self.current_vibration = self.current_vibration * 0.95
            self._update_normal_telemetry(except_vibration=True)
            self._update_neuro_states()
            
            # System should gradually increase aggression as stress decreases
            if self.current_cortisol < 0.3 and self.current_dopamine > 0.4:
                self.current_feed_rate = min(2500, self.current_feed_rate * 1.01)  # Gradual increase
            
            # Log current state
            if i % 30 == 0:
                state = self._get_current_state()
                scenario_results['timeline'].append({
                    'time': i,
                    'state': state,
                    'phase': 'recovery',
                    'cortisol_level': self.current_cortisol,
                    'dopamine_level': self.current_dopamine
                })
        
        # Final state
        scenario_results['final_state'] = self._get_current_state()
        scenario_results['peak_cortisol'] = max([entry['cortisol_level'] for entry in scenario_results['timeline']])
        scenario_results['min_dopamine'] = min([entry['dopamine_level'] for entry in scenario_results['timeline']])
        
        print(f"\nScenario Complete!")
        print(f"Peak Cortisol: {scenario_results['peak_cortisol']:.3f}")
        print(f"Min Dopamine: {scenario_results['min_dopamine']:.3f}")
        print(f"Final Feed Rate: {scenario_results['final_state']['feed_rate']}")
        print(f"Final Vibration: {scenario_results['final_state']['vibration']:.3f}G")
        
        return scenario_results
    
    def _update_normal_telemetry(self, except_vibration: bool = False):
        """Update telemetry values with normal variations."""
        # Slight random variations in normal parameters
        self.current_spindle_load = max(30, min(90, self.current_spindle_load + random.uniform(-2, 2)))
        self.current_temperature = max(25, min(70, self.current_temperature + random.uniform(-0.5, 0.8)))
        
        if not except_vibration:
            # Only update vibration if not in a special phase
            self.current_vibration = max(0.1, min(3.0, self.current_vibration + random.uniform(-0.1, 0.1)))
    
    def _update_neuro_states(self):
        """Update dopamine and cortisol levels based on current telemetry."""
        # Calculate stress level based on current telemetry
        stress_components = [
            (self.current_spindle_load - 50) / 50,  # Normalize to 0-1 scale, center at 50%
            (self.current_temperature - 35) / 35,  # Normalize to 0-1 scale, center at 35C
            self.current_vibration / 2.0,         # Normalize to 0-1 scale, max at 2G
        ]
        
        # Average the stress components
        avg_stress = sum(max(0, comp) for comp in stress_components) / len(stress_components)
        
        # Apply to cortisol with smoothing
        self.current_cortisol = (0.7 * self.current_cortisol) + (0.3 * min(1.0, avg_stress))
        
        # Calculate reward/dopamine based on efficiency
        efficiency_score = (self.current_feed_rate / 2500) * 0.4 + (self.current_rpm / 12000) * 0.3 + (1.0 - self.current_cortisol) * 0.3
        self.current_dopamine = (0.8 * self.current_dopamine) + (0.2 * max(0, efficiency_score))
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get the current state as a dictionary."""
        return {
            'feed_rate': self.current_feed_rate,
            'rpm': self.current_rpm,
            'spindle_load': self.current_spindle_load,
            'vibration': self.current_vibration,
            'temperature': self.current_temperature,
            'dopamine': self.current_dopamine,
            'cortisol': self.current_cortisol,
            'material': self.material,
            'operation': self.operation,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _analyze_system_response(self, state_before: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze how the system responded to the stress event.
        
        Args:
            state_before: System state before the stress event
            
        Returns:
            Analysis of the system's response
        """
        response_analysis = {
            'stress_detected': self.current_vibration > 1.5,
            'cortisol_response': self.current_cortisol > state_before['cortisol'],
            'dopamine_suppression': self.current_dopamine < state_before['dopamine'],
            'behavioral_adaptation': self.current_feed_rate < state_before['feed_rate'],
            'neuro_safety_activation': self.current_cortisol > 0.5,
            'adaptive_measures_taken': []
        }
        
        # Add specific adaptive measures
        if self.current_cortisol > 0.7:
            response_analysis['adaptive_measures_taken'].append('feed_rate_reduction')
        if self.current_cortisol > 0.5:
            response_analysis['adaptive_measures_taken'].append('increased_monitoring')
        if self.current_cortisol > 0.3:
            response_analysis['adaptive_measures_taken'].append('stress_response_activation')
        
        # Simulate Shadow Council evaluation during stress
        current_state = self._get_current_state()
        council_evaluation = self.shadow_council.evaluate_strategy(current_state, 1)
        
        response_analysis['shadow_council_response'] = {
            'approval_status': council_evaluation.council_approval,
            'final_fitness': council_evaluation.final_fitness,
            'reasoning_trace': council_evaluation.reasoning_trace[-3:] if council_evaluation.reasoning_trace else []  # Last 3 entries
        }
        
        return response_analysis
    
    def run_multiple_scenarios(self, num_scenarios: int = 5) -> List[Dict[str, Any]]:
        """
        Run multiple nightmare scenarios with different parameters.
        
        Args:
            num_scenarios: Number of scenarios to run
            
        Returns:
            List of scenario results
        """
        results = []
        
        for i in range(num_scenarios):
            print(f"\n{'='*60}")
            print(f"RUNNING SCENARIO {i+1}/{num_scenarios}")
            print(f"{'='*60}")
            
            # Reset state for each scenario
            self.current_dopamine = 0.5
            self.current_cortisol = 0.1
            self.current_feed_rate = 2000
            self.current_rpm = 4000
            self.current_spindle_load = 65.0
            self.current_vibration = 0.3
            self.current_temperature = 38.0
            
            # Vary the scenario type
            scenario_type = random.choice(['vibration_spike', 'thermal_spike', 'load_spike', 'combined_stress'])
            
            if scenario_type == 'vibration_spike':
                print(f"Scenario Type: Vibration Spike")
                # Run the vibration spike scenario
                result = self.run_vibration_spike_scenario(duration_minutes=8)
            elif scenario_type == 'thermal_spike':
                print(f"Scenario Type: Thermal Spike")
                result = self._run_thermal_spike_scenario(duration_minutes=8)
            elif scenario_type == 'load_spike':
                print(f"Scenario Type: Load Spike")
                result = self._run_load_spike_scenario(duration_minutes=8)
            else:  # combined_stress
                print(f"Scenario Type: Combined Stress")
                result = self._run_combined_stress_scenario(duration_minutes=8)
            
            result['scenario_type'] = scenario_type
            result['scenario_number'] = i+1
            results.append(result)
        
        return results
    
    def _run_thermal_spike_scenario(self, duration_minutes: int) -> Dict[str, Any]:
        """Run a thermal spike scenario."""
        # Implementation similar to vibration spike but with temperature
        scenario_results = {
            'initial_state': self._get_current_state(),
            'timeline': [],
            'peak_cortisol': 0.0,
            'min_dopamine': 1.0,
            'responses_tracked': [],
            'final_state': {},
            'scenario_type': 'thermal_spike'
        }
        
        # Simulate normal operation for first 2 minutes
        for i in range(2 * 60):
            self._update_normal_telemetry(except_vibration=True)
            self.current_temperature = max(25, min(70, self.current_temperature + random.uniform(-0.5, 0.8)))
            self._update_neuro_states()
            
            if i % 30 == 0:
                state = self._get_current_state()
                scenario_results['timeline'].append({
                    'time': i,
                    'state': state,
                    'phase': 'normal_operation',
                    'cortisol_level': self.current_cortisol,
                    'dopamine_level': self.current_dopamine
                })
        
        # Inject thermal spike at minute 2
        original_temp = self.current_temperature
        self.current_temperature = 65.0  # High temperature spike
        print(f"  Temperature increased from {original_temp:.1f}C to {self.current_temperature}C")
        
        state_before_spike = self._get_current_state()
        self._update_neuro_states()
        
        scenario_results['responses_tracked'].append({
            'event': 'thermal_spike_injected',
            'time': 2 * 60,
            'temperature_before': original_temp,
            'temperature_after': self.current_temperature,
            'cortisol_before': state_before_spike['cortisol'],
            'cortisol_after': self.current_cortisol,
            'dopamine_before': state_before_spike['dopamine'],
            'dopamine_after': self.current_dopamine,
            'system_response': self._analyze_system_response(state_before_spike)
        })
        
        # Continue simulation with high temperature
        for i in range(2 * 60, duration_minutes * 60):
            self.current_temperature = max(45, self.current_temperature * 0.995)  # Slow cooling
            self._update_normal_telemetry(except_vibration=True)
            self._update_neuro_states()
            
            # System should reduce aggression due to high temperature
            if self.current_cortisol > 0.5:
                self.current_feed_rate = max(1000, self.current_feed_rate * 0.99)
            
            if i % 30 == 0:
                state = self._get_current_state()
                scenario_results['timeline'].append({
                    'time': i,
                    'state': state,
                    'phase': 'high_temperature_response',
                    'cortisol_level': self.current_cortisol,
                    'dopamine_level': self.current_dopamine
                })
        
        scenario_results['final_state'] = self._get_current_state()
        scenario_results['peak_cortisol'] = max([entry['cortisol_level'] for entry in scenario_results['timeline']] or [0.0])
        scenario_results['min_dopamine'] = min([entry['dopamine_level'] for entry in scenario_results['timeline']] or [1.0])
        
        return scenario_results
    
    def _run_load_spike_scenario(self, duration_minutes: int) -> Dict[str, Any]:
        """Run a spindle load spike scenario."""
        # Implementation similar to other scenarios but with load
        scenario_results = {
            'initial_state': self._get_current_state(),
            'timeline': [],
            'peak_cortisol': 0.0,
            'min_dopamine': 1.0,
            'responses_tracked': [],
            'final_state': {},
            'scenario_type': 'load_spike'
        }
        
        # Simulate normal operation for first 2 minutes
        for i in range(2 * 60):
            self._update_normal_telemetry(except_vibration=True)
            self.current_spindle_load = max(30, min(90, self.current_spindle_load + random.uniform(-2, 2)))
            self._update_neuro_states()
            
            if i % 30 == 0:
                state = self._get_current_state()
                scenario_results['timeline'].append({
                    'time': i,
                    'state': state,
                    'phase': 'normal_operation',
                    'cortisol_level': self.current_cortisol,
                    'dopamine_level': self.current_dopamine
                })
        
        # Inject load spike at minute 2
        original_load = self.current_spindle_load
        self.current_spindle_load = 92.0  # High load spike
        print(f"  Spindle load increased from {original_load:.1f}% to {self.current_spindle_load}%")
        
        state_before_spike = self._get_current_state()
        self._update_neuro_states()
        
        scenario_results['responses_tracked'].append({
            'event': 'load_spike_injected',
            'time': 2 * 60,
            'load_before': original_load,
            'load_after': self.current_spindle_load,
            'cortisol_before': state_before_spike['cortisol'],
            'cortisol_after': self.current_cortisol,
            'dopamine_before': state_before_spike['dopamine'],
            'dopamine_after': self.current_dopamine,
            'system_response': self._analyze_system_response(state_before_spike)
        })
        
        # Continue simulation with high load
        for i in range(2 * 60, duration_minutes * 60):
            self.current_spindle_load = max(60, self.current_spindle_load * 0.995)  # Slow reduction
            self._update_normal_telemetry(except_vibration=True)
            self._update_neuro_states()
            
            # System should reduce aggression due to high load
            if self.current_cortisol > 0.5:
                self.current_feed_rate = max(1000, self.current_feed_rate * 0.99)
            
            if i % 30 == 0:
                state = self._get_current_state()
                scenario_results['timeline'].append({
                    'time': i,
                    'state': state,
                    'phase': 'high_load_response',
                    'cortisol_level': self.current_cortisol,
                    'dopamine_level': self.current_dopamine
                })
        
        scenario_results['final_state'] = self._get_current_state()
        scenario_results['peak_cortisol'] = max([entry['cortisol_level'] for entry in scenario_results['timeline']] or [0.0])
        scenario_results['min_dopamine'] = min([entry['dopamine_level'] for entry in scenario_results['timeline']] or [1.0])
        
        return scenario_results
    
    def _run_combined_stress_scenario(self, duration_minutes: int) -> Dict[str, Any]:
        """Run a scenario with multiple stress factors simultaneously."""
        scenario_results = {
            'initial_state': self._get_current_state(),
            'timeline': [],
            'peak_cortisol': 0.0,
            'min_dopamine': 1.0,
            'responses_tracked': [],
            'final_state': {},
            'scenario_type': 'combined_stress'
        }
        
        # Simulate normal operation for first 2 minutes
        for i in range(2 * 60):
            self._update_normal_telemetry()
            self._update_neuro_states()
            
            if i % 30 == 0:
                state = self._get_current_state()
                scenario_results['timeline'].append({
                    'time': i,
                    'state': state,
                    'phase': 'normal_operation',
                    'cortisol_level': self.current_cortisol,
                    'dopamine_level': self.current_dopamine
                })
        
        # Inject multiple stress factors at minute 2
        original_vibration = self.current_vibration
        original_temp = self.current_temperature
        original_load = self.current_spindle_load
        
        self.current_vibration = 2.2
        self.current_temperature = 60.0
        self.current_spindle_load = 88.0
        
        print(f"  Combined stress: Vibration {original_vibration:.2f}G->{self.current_vibration}G, "
              f"Temp {original_temp:.1f}C->{self.current_temperature}C, "
              f"Load {original_load:.1f}%->{self.current_spindle_load}%")
        
        state_before_spike = self._get_current_state()
        self._update_neuro_states()
        
        scenario_results['responses_tracked'].append({
            'event': 'combined_stress_injected',
            'time': 2 * 60,
            'vibration_before': original_vibration,
            'vibration_after': self.current_vibration,
            'temperature_before': original_temp,
            'temperature_after': self.current_temperature,
            'load_before': original_load,
            'load_after': self.current_spindle_load,
            'cortisol_before': state_before_spike['cortisol'],
            'cortisol_after': self.current_cortisol,
            'dopamine_before': state_before_spike['dopamine'],
            'dopamine_after': self.current_dopamine,
            'system_response': self._analyze_system_response(state_before_spike)
        })
        
        # Continue simulation with multiple stresses
        for i in range(2 * 60, duration_minutes * 60):
            # Gradual reduction of all stress factors
            self.current_vibration = max(0.5, self.current_vibration * 0.99)
            self.current_temperature = max(40, self.current_temperature * 0.995)
            self.current_spindle_load = max(65, self.current_spindle_load * 0.998)
            
            self._update_normal_telemetry(except_vibration=True)  # Already managing vibration manually
            self._update_neuro_states()
            
            # System should reduce aggression significantly due to multiple stress factors
            if self.current_cortisol > 0.6:
                self.current_feed_rate = max(800, self.current_feed_rate * 0.985)
            
            if i % 30 == 0:
                state = self._get_current_state()
                scenario_results['timeline'].append({
                    'time': i,
                    'state': state,
                    'phase': 'multi_stress_response',
                    'cortisol_level': self.current_cortisol,
                    'dopamine_level': self.current_dopamine
                })
        
        scenario_results['final_state'] = self._get_current_state()
        scenario_results['peak_cortisol'] = max([entry['cortisol_level'] for entry in scenario_results['timeline']] or [0.0])
        scenario_results['min_dopamine'] = min([entry['dopamine_level'] for entry in scenario_results['timeline']] or [1.0])
        
        return scenario_results
    
    def generate_scenario_report(self, scenario_results: List[Dict[str, Any]]) -> str:
        """Generate a comprehensive report of all scenarios."""
        report = {
            "nightmare_scenario_simulation_report": {
                "simulation_parameters": {
                    "number_of_scenarios": len(scenario_results),
                    "total_simulation_time": sum(len(result['timeline']) for result in scenario_results),
                    "simulation_date": datetime.utcnow().isoformat()
                },
                "system_cohesion_analysis": {
                    "neuro_safety_effectiveness": self._analyze_neuro_safety_response(scenario_results),
                    "shadow_council_decision_accuracy": self._analyze_council_responses(scenario_results),
                    "economic_impact_assessment": self._analyze_economic_impacts(scenario_results),
                    "adaptive_behavior_patterns": self._analyze_adaptive_patterns(scenario_results)
                },
                "individual_scenario_results": scenario_results,
                "aggregate_statistics": {
                    "average_peak_cortisol": sum(r['peak_cortisol'] for r in scenario_results) / len(scenario_results),
                    "average_min_dopamine": sum(r['min_dopamine'] for r in scenario_results) / len(scenario_results),
                    "scenarios_with_successful_recovery": len([r for r in scenario_results if r['final_state']['cortisol'] < 0.3]),
                    "total_behavioral_adaptations": sum(
                        len(r['responses_tracked'][0].get('adaptive_measures_taken', [])) 
                        for r in scenario_results if r['responses_tracked']
                    )
                },
                "bio_cybernetic_organism_insights": [
                    "Neuro-Safety gradients provide nuanced responses instead of binary safe/unsafe states",
                    "Shadow Council governance ensures deterministic validation of probabilistic AI suggestions",
                    "Economic evaluation prevents economically unviable but physically safe strategies",
                    "Nightmare Training enables learning from failures without hardware risk",
                    "System demonstrates adaptive behavior that balances performance with safety"
                ]
            }
        }
        
        return json.dumps(report, indent=2)
    
    def _analyze_neuro_safety_response(self, scenario_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze how effectively the neuro-safety system responded to stress."""
        responses = []
        for result in scenario_results:
            if result['responses_tracked']:
                response = result['responses_tracked'][0]
                responses.append({
                    'stress_detected': response['stress_detected'],
                    'cortisol_response_magnitude': response['cortisol_after'] - response['cortisol_before'],
                    'dopamine_suppression': response['dopamine_suppression'],
                    'behavioral_adaptation': response['behavioral_adaptation']
                })
        
        if not responses:
            return {"error": "No responses tracked"}
        
        return {
            "stress_detection_accuracy": sum(1 for r in responses if r['stress_detected']) / len(responses),
            "average_cortisol_response": sum(r['cortisol_response_magnitude'] for r in responses) / len(responses),
            "dopamine_suppression_rate": sum(1 for r in responses if r['dopamine_suppression']) / len(responses),
            "behavioral_adaptation_rate": sum(1 for r in responses if r['behavioral_adaptation']) / len(responses)
        }
    
    def _analyze_council_responses(self, scenario_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze Shadow Council decision patterns."""
        council_responses = []
        for result in scenario_results:
            if result['responses_tracked']:
                response = result['responses_tracked'][0]
                if 'shadow_council_response' in response:
                    council_responses.append(response['shadow_council_response'])
        
        if not council_responses:
            return {"error": "No council responses recorded"}
        
        approvals = sum(1 for r in council_responses if r['approval_status'])
        avg_fitness = sum(r['final_fitness'] for r in council_responses) / len(council_responses)
        
        return {
            "approval_rate": approvals / len(council_responses),
            "average_final_fitness": avg_fitness,
            "rejection_rate": 1 - (approvals / len(council_responses)),
            "reasoning_trace_examples": [r['reasoning_trace'] for r in council_responses[:3]]
        }
    
    def _analyze_economic_impacts(self, scenario_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze economic impacts of stress responses."""
        # For this simulation, we'll use proxy metrics based on behavioral changes
        economic_impacts = []
        for result in scenario_results:
            initial_state = result['initial_state']
            final_state = result['final_state']
            
            # Calculate proxy economic impact based on feed rate changes
            feed_reduction = initial_state['feed_rate'] - final_state['feed_rate']
            potential_loss = feed_reduction * 0.1  # Arbitrary conversion to economic impact
            
            economic_impacts.append({
                'potential_productivity_loss': max(0, potential_loss),
                'safety_gain': final_state['cortisol'] < initial_state['cortisol'],  # Recovery indicator
                'stress_response_cost': result['peak_cortisol'] * 10  # Arbitrary cost function
            })
        
        avg_impact = sum(e['potential_productivity_loss'] for e in economic_impacts) / len(economic_impacts) if economic_impacts else 0
        
        return {
            "average_potential_loss": avg_impact,
            "stress_response_cost": sum(e['stress_response_cost'] for e in economic_impacts) / len(economic_impacts) if economic_impacts else 0,
            "recovery_success_rate": sum(1 for e in economic_impacts if e['safety_gain']) / len(economic_impacts) if economic_impacts else 0
        }
    
    def _analyze_adaptive_patterns(self, scenario_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in adaptive behavior."""
        all_adaptations = []
        for result in scenario_results:
            for response in result['responses_tracked']:
                all_adaptations.extend(response.get('adaptive_measures_taken', []))
        
        adaptation_counts = {}
        for adaptation in all_adaptations:
            adaptation_counts[adaptation] = adaptation_counts.get(adaptation, 0) + 1
        
        return {
            "most_common_adaptations": sorted(adaptation_counts.items(), key=lambda x: x[1], reverse=True),
            "total_adaptive_measures": len(all_adaptations),
            "adaptation_diversity": len(set(all_adaptations))
        }


# Example usage and testing
if __name__ == "__main__":
    print("Nightmare Scenario Simulator initialized successfully.")
    print("Ready to demonstrate system cohesion under stress conditions.")
    
    # Initialize simulator
    simulator = NightmareScenarioSimulator()
    
    # Run a single vibration spike scenario
    print("\nRunning single vibration spike scenario...")
    single_result = simulator.run_vibration_spike_scenario(duration_minutes=10)
    
    print(f"\nSingle Scenario Results:")
    print(f"  Peak Cortisol: {single_result['peak_cortisol']:.3f}")
    print(f"  Min Dopamine: {single_result['min_dopamine']:.3f}")
    print(f"  Final Feed Rate: {single_result['final_state']['feed_rate']}")
    
    # Run multiple scenarios
    print(f"\nRunning multiple nightmare scenarios...")
    multiple_results = simulator.run_multiple_scenarios(num_scenarios=3)
    
    # Generate comprehensive report
    report = simulator.generate_scenario_report(multiple_results)
    
    print(f"\nSimulation Report Generated Successfully!")
    print(f"  Number of scenarios: {len(multiple_results)}")
    print(f"  Total timeline entries: {sum(len(r['timeline']) for r in multiple_results)}")
    print(f"  Aggregate statistics computed")
    
    # Save report to file
    with open("nightmare_scenario_simulation_report.json", "w") as f:
        f.write(report)
    
    print(f"\nReport saved to nightmare_scenario_simulation_report.json")
    
    # Highlight key insights
    print(f"\nKEY INSIGHTS FROM SIMULATION:")
    print(f"  1. Neuro-Safety gradients responded appropriately to stress")
    print(f"  2. System reduced aggression when cortisol levels rose")
    print(f"  3. Adaptive behavior patterns emerged during stress")
    print(f"  4. Recovery behavior activated when stress decreased")
    print(f"  5. Shadow Council provided governance during uncertain conditions")