"""
Nightmare Training - Dreamer Component
Runs the Shadow Council against modified telemetry data during offline simulation
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
import os
from pathlib import Path

from ..services.dopamine_engine import DopamineEngine
from ..services.shadow_council import ShadowCouncil, AuditorAgent, DecisionPolicy
from ..services.physics_auditor import PhysicsAuditor
from ..services.economics_engine import EconomicsEngine


class Dreamer:
    """
    The Dreamer component of Nightmare Training.
    Runs the Shadow Council (Auditor & Dopamine Engine) against modified telemetry data
    to test system responses to failure scenarios during the "Dream State".
    """
    
    def __init__(self, shadow_council: ShadowCouncil):
        self.shadow_council = shadow_council
        self.auditor = shadow_council.auditor if hasattr(shadow_council, 'auditor') else None
        self.economics_engine = None  # Will be set separately if needed
        
        # Track simulation results
        self.simulation_results = []
        
    def run_simulation_loop(self, modified_telemetry: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the Shadow Council against modified telemetry data to test responses
        to injected failure scenarios
        
        Args:
            modified_telemetry: DataFrame with injected failure scenarios
            
        Returns:
            Dictionary with simulation results and analysis
        """
        results = {
            'total_data_points': len(modified_telemetry),
            'failures_detected': 0,
            'kill_switch_triggers': 0,
            'cortisol_spikes': 0,
            'preemptive_responses': 0,  # Responses before theoretical failure point
            'missed_failures': 0,  # Failures that weren't caught in time
            'policy_updates_needed': [],
            'detailed_timeline': []
        }
        
        # Process each data point in the telemetry
        for idx, row in modified_telemetry.iterrows():
            # Convert row to dictionary format expected by Shadow Council
            current_state = row.to_dict()
            
            # Get machine ID from the state or use a default
            machine_id = current_state.get('machine_id', 1)
            
            # Run Shadow Council evaluation
            council_decision = self.shadow_council.evaluate_strategy(current_state, machine_id)
            
            # Analyze the decision for failure detection
            analysis = self._analyze_decision_for_failure(council_decision, current_state)
            
            # Record detailed timeline
            timeline_entry = {
                'index': idx,
                'timestamp': current_state.get('timestamp', idx),
                'state_values': current_state,
                'council_decision': council_decision,
                'failure_analysis': analysis
            }
            results['detailed_timeline'].append(timeline_entry)
            
            # Update counters based on analysis
            if analysis['failure_detected']:
                results['failures_detected'] += 1
                if analysis['response_was_preemptive']:
                    results['preemptive_responses'] += 1
                else:
                    results['missed_failures'] += 1
                    
            if analysis['kill_switch_triggered']:
                results['kill_switch_triggers'] += 1
                
            if analysis['cortisol_spike']:
                results['cortisol_spikes'] += 1
        
        # Determine if policy updates are needed based on missed failures
        if results['missed_failures'] > 0:
            results['policy_updates_needed'] = self._determine_policy_updates(results)
        
        return results
    
    def _analyze_current_state(self, state: Dict[str, float]) -> str:
        """
        Analyze the current machine state to generate appropriate intent
        for the Shadow Council
        """
        anomalies = []
        
        # Check for various anomaly types based on state values
        if state.get('spindle_load', 0) > 85:
            anomalies.append(f"high spindle load detected: {state['spindle_load']:.2f}%")
        
        if state.get('temperature', 0) > 60:
            anomalies.append(f"elevated temperature detected: {state['temperature']:.2f}°C")
        
        if state.get('vibration_x', 0) > 1.5 or state.get('vibration_y', 0) > 1.5:
            anomalies.append(f"high vibration detected: X={state.get('vibration_x', 0):.2f}, Y={state.get('vibration_y', 0):.2f}")
        
        if state.get('spindle_load', 0) < 10 and (state.get('vibration_x', 0) > 2.0 or state.get('vibration_y', 0) > 2.0):
            anomalies.append("possible phantom trauma detected: low load with high vibration")
        
        if state.get('coolant_flow', 10) < 0.5:
            anomalies.append(f"coolant flow low: {state.get('coolant_flow', 0):.2f}")
        
        if anomalies:
            return f"Respond to detected anomalies: {', '.join(anomalies)}"
        else:
            return "Maintain normal operations, monitor for changes"
    
    def _analyze_decision_for_failure(self, council_decision: Dict[str, Any], 
                                    current_state: Dict[str, float]) -> Dict[str, bool]:
        """
        Analyze a Shadow Council decision to determine if it properly detected
        and responded to a failure scenario
        """
        analysis = {
            'failure_detected': False,
            'kill_switch_triggered': False,
            'cortisol_spike': False,
            'response_was_preemptive': False,
            'needs_policy_update': False
        }
        
        # Check if the Auditor rejected the proposed action (kill switch)
        if not council_decision.get('council_approval', True):
            analysis['kill_switch_triggered'] = True
            analysis['failure_detected'] = True
        
        # Check for high cortisol levels indicating stress response
        if council_decision.get('final_fitness', 1.0) < 0.3:  # Low fitness indicates stress
            analysis['cortisol_spike'] = True
        
        # Check reasoning trace for specific failure-related keywords
        reasoning_trace = council_decision.get('reasoning_trace', [])
        for trace_item in reasoning_trace:
            if any(keyword in trace_item.lower() for keyword in 
                  ['reject', 'danger', 'unsafe', 'constraint violation', 'failure']):
                analysis['failure_detected'] = True
                break
        
        # Determine if response was preemptive (before actual damage occurred)
        # This would require comparing to some theoretical failure threshold
        # For now, we'll consider any rejection as potentially preemptive
        if analysis['kill_switch_triggered']:
            analysis['response_was_preemptive'] = True  # Simplified assumption
        
        # Determine if policy update is needed based on missed failures
        if not analysis['kill_switch_triggered'] and self._state_indicates_critical_failure(current_state):
            analysis['needs_policy_update'] = True
        
        return analysis
    
    def _state_indicates_critical_failure(self, state: Dict[str, float]) -> bool:
        """
        Determine if the current state indicates a critical failure that
        should have triggered a kill switch
        """
        critical_conditions = [
            state.get('spindle_load', 0) > 150,  # Way above normal
            state.get('temperature', 0) > 80,   # Dangerous temperature
            (state.get('vibration_x', 0) > 3.0 or state.get('vibration_y', 0) > 3.0),  # Severe vibration
            (state.get('spindle_load', 0) < 5 and 
             (state.get('vibration_x', 0) > 2.5 or state.get('vibration_y', 0) > 2.5))  # Phantom trauma
        ]
        
        return any(critical_conditions)
    
    def _determine_policy_updates(self, simulation_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Determine what policy updates are needed based on simulation results
        """
        updates_needed = []
        
        # If there were missed failures, suggest sensitivity adjustments
        if simulation_results['missed_failures'] > 0:
            updates_needed.append({
                'type': 'sensitivity_adjustment',
                'target': 'auditor_thresholds',
                'reason': f'{simulation_results["missed_failures"]} failures were not caught in time',
                'recommendation': 'Lower the detection thresholds for critical parameters'
            })
        
        # If too many kill switches were triggered unnecessarily
        if simulation_results['kill_switch_triggers'] > simulation_results['total_data_points'] * 0.1:  # More than 10%
            updates_needed.append({
                'type': 'sensitivity_adjustment',
                'target': 'auditor_thresholds',
                'reason': f'Too many kill switches ({simulation_results["kill_switch_triggers"]} out of {simulation_results["total_data_points"]})',
                'recommendation': 'Increase the detection thresholds to reduce false positives'
            })
        
        return updates_needed
    
    def consolidate_learning_from_batch(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Consolidate learning from multiple simulation runs to update dopamine policies
        """
        consolidated = {
            'total_simulations': len(batch_results),
            'average_failures_detected': sum([r['failures_detected'] for r in batch_results]) / len(batch_results) if batch_results else 0,
            'average_kill_switch_triggers': sum([r['kill_switch_triggers'] for r in batch_results]) / len(batch_results) if batch_results else 0,
            'average_cortisol_spikes': sum([r['cortisol_spikes'] for r in batch_results]) / len(batch_results) if batch_results else 0,
            'average_preemptive_responses': sum([r['preemptive_responses'] for r in batch_results]) / len(batch_results) if batch_results else 0,
            'average_missed_failures': sum([r['missed_failures'] for r in batch_results]) / len(batch_results) if batch_results else 0,
            'common_policy_updates': [],
            'learning_insights': []
        }
        
        # Identify common patterns in policy updates needed
        all_policy_updates = []
        for result in batch_results:
            all_policy_updates.extend(result.get('policy_updates_needed', []))
        
        # Group similar policy updates
        update_counts = {}
        for update in all_policy_updates:
            key = update['type'] + '_' + update['target']
            if key not in update_counts:
                update_counts[key] = {'count': 0, 'examples': []}
            update_counts[key]['count'] += 1
            update_counts[key]['examples'].append(update['reason'])
        
        for key, info in update_counts.items():
            consolidated['common_policy_updates'].append({
                'type_target': key,
                'frequency': info['count'],
                'examples': info['examples'][:3]  # Limit examples
            })
        
        # Generate learning insights
        if consolidated['average_missed_failures'] > 0:
            consolidated['learning_insights'].append(
                f"The system missed an average of {consolidated['average_missed_failures']:.2f} failures per simulation. "
                "Consider adjusting detection thresholds for better sensitivity."
            )
        
        if consolidated['average_preemptive_responses'] / max(consolidated['average_failures_detected'], 1) < 0.5:
            consolidated['learning_insights'].append(
                "Preemptive response rate is low. The system is reacting rather than anticipating failures."
            )
        
        return consolidated
    
    def update_dopamine_policy(self, consolidated_learning: Dict[str, Any], 
                             policy_file_path: str = "dopamine_policy.json"):
        """
        Update the dopamine policy based on consolidated learning from simulations
        """
        # Load existing policy or create new one
        policy_path = Path(policy_file_path)
        if policy_path.exists():
            with open(policy_path, 'r') as f:
                policy = json.load(f)
        else:
            policy = {
                'version': '1.0',
                'last_updated': datetime.now().isoformat(),
                'sensitivity_levels': {
                    'spindle_load': 0.85,  # Default threshold
                    'temperature': 60.0,
                    'vibration_x': 1.5,
                    'vibration_y': 1.5,
                    'coolant_flow': 0.5
                },
                'learning_history': []
            }
        
        # Update policy based on learning insights
        insights = consolidated_learning.get('learning_insights', [])
        for insight in insights:
            if 'sensitivity' in insight.lower():
                # Adjust sensitivity based on missed failures
                if consolidated_learning['average_missed_failures'] > 0:
                    # Increase sensitivity (lower thresholds) for critical parameters
                    for param in ['spindle_load', 'temperature', 'vibration_x', 'vibration_y']:
                        if param in policy['sensitivity_levels']:
                            policy['sensitivity_levels'][param] *= 0.95  # Decrease threshold by 5%
        
        # Add learning entry to history
        policy['learning_history'].append({
            'date': datetime.now().isoformat(),
            'simulation_count': consolidated_learning['total_simulations'],
            'missed_failures_avg': consolidated_learning['average_missed_failures'],
            'preemptive_response_rate': consolidated_learning['average_preemptive_responses'] / max(consolidated_learning['average_failures_detected'], 1),
            'policy_adjustments': [u['type_target'] for u in consolidated_learning.get('common_policy_updates', [])]
        })
        
        # Update version and timestamp
        policy['version'] = str(float(policy['version']) + 0.1)
        policy['last_updated'] = datetime.now().isoformat()
        
        # Save updated policy
        with open(policy_path, 'w') as f:
            json.dump(policy, f, indent=2)
        
        return policy


# Example usage and testing
if __name__ == "__main__":
    # This would typically be initialized with real Shadow Council instance
    # For testing purposes, we'll create a simplified version
    print("Dreamer module loaded successfully.")
    print("Ready to run Nightmare Training simulations with Shadow Council.")