"""
Nightmare Training - Main Orchestrator
Coordinates the Adversary and Dreamer components to run offline learning sessions
"""

import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime
import logging
from pathlib import Path
import json

from .adversary import Adversary, FailureInjectionConfig, FailureType
from .dreamer import Dreamer
from ..services.shadow_council import ShadowCouncil, DecisionPolicy, CreatorAgent, AuditorAgent
from ..repositories.telemetry_repository import TelemetryRepository


logger = logging.getLogger(__name__)


class NightmareTrainingOrchestrator:
    """
    Main orchestrator for Nightmare Training protocol.
    Coordinates the Adversary (failure injection) and Dreamer (simulation loop) components
    to run offline learning sessions during machine idle time.
    """
    
    def __init__(self, shadow_council: ShadowCouncil, telemetry_repo: TelemetryRepository):
        self.shadow_council = shadow_council
        self.telemetry_repo = telemetry_repo
        self.adversary = Adversary()
        self.dreamer = Dreamer(shadow_council)
        
    def run_nightmare_training_session(self, machine_id: int, duration_hours: float = 1.0,
                                     failure_probability: float = 0.7) -> Dict[str, Any]:
        """
        Run a complete nightmare training session for a specific machine
        
        Args:
            machine_id: ID of the machine to train on
            duration_hours: Duration of historical data to replay (in hours)
            failure_probability: Probability of injecting failures (0.0 to 1.0)
            
        Returns:
            Dictionary with training results and policy updates
        """
        logger.info(f"Starting Nightmare Training session for machine {machine_id}, "
                   f"duration: {duration_hours} hours")
        
        # 1. Load historical telemetry data (REM Cycle - Data Replay)
        historical_data = self._load_historical_telemetry(machine_id, duration_hours)
        
        if historical_data.empty:
            logger.warning(f"No historical data found for machine {machine_id}")
            return {
                'status': 'NO_DATA',
                'machine_id': machine_id,
                'session_start': datetime.utcnow().isoformat(),
                'session_end': datetime.utcnow().isoformat(),
                'results': {}
            }
        
        # 2. Generate failure scenarios to inject
        failure_configs = self._generate_failure_configs(len(historical_data), failure_probability)
        
        # 3. Inject failures into historical data (creating multiple scenarios)
        failure_scenarios = self.adversary.generate_random_failures(historical_data, num_scenarios=5)
        
        # 4. Run each scenario through the Dreamer (Simulation Loop)
        simulation_results = []
        for i, (scenario_data, applied_failures) in enumerate(failure_scenarios):
            logger.info(f"Running scenario {i+1}/{len(failure_scenarios)} with failures: {applied_failures}")
            
            # Run simulation for this scenario
            scenario_result = self.dreamer.run_simulation_loop(scenario_data)
            scenario_result['scenario_id'] = i
            scenario_result['applied_failures'] = applied_failures
            simulation_results.append(scenario_result)
        
        # 5. Consolidate learning from all scenarios
        consolidated_learning = self.dreamer.consolidate_learning_from_batch(simulation_results)
        
        # 6. Update dopamine policy based on consolidated learning
        policy_updates = self.dreamer.update_dopamine_policy(consolidated_learning)
        
        # 7. Prepare session results
        session_end = datetime.utcnow()
        session_results = {
            'status': 'COMPLETED',
            'machine_id': machine_id,
            'session_start': (session_end - pd.Timedelta(hours=duration_hours)).isoformat(),
            'session_end': session_end.isoformat(),
            'original_data_points': len(historical_data),
            'scenarios_generated': len(failure_scenarios),
            'simulation_results': simulation_results,
            'consolidated_learning': consolidated_learning,
            'policy_updates_applied': policy_updates,
            'summary': {
                'total_kill_switch_triggers': sum(r['kill_switch_triggers'] for r in simulation_results),
                'total_preemptive_responses': sum(r['preemptive_responses'] for r in simulation_results),
                'total_missed_failures': sum(r['missed_failures'] for r in simulation_results),
                'learning_opportunities': len([r for r in simulation_results if r['missed_failures'] > 0])
            }
        }
        
        logger.info(f"Nightmare Training session completed for machine {machine_id}. "
                   f"Triggers: {session_results['summary']['total_kill_switch_triggers']}, "
                   f"Preemptive: {session_results['summary']['total_preemptive_responses']}, "
                   f"Missed: {session_results['summary']['total_missed_failures']}")
        
        return session_results
    
    def _load_historical_telemetry(self, machine_id: int, duration_hours: float) -> pd.DataFrame:
        """
        Load historical telemetry data for nightmare training
        """
        try:
            # Convert duration_hours to minutes for repository query
            duration_minutes = int(duration_hours * 60)
            
            # Get recent telemetry data
            telemetry_records = self.telemetry_repo.get_recent_by_machine(machine_id, minutes=duration_minutes)
            
            if not telemetry_records:
                logger.warning(f"No telemetry records found for machine {machine_id} in last {duration_hours} hours")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for record in telemetry_records:
                record_dict = {
                    'timestamp': getattr(record, 'timestamp', None),
                    'machine_id': getattr(record, 'machine_id', machine_id),
                    'spindle_load': getattr(record, 'spindle_load', 0.0),
                    'vibration_x': getattr(record, 'vibration_x', 0.0),
                    'vibration_y': getattr(record, 'vibration_y', 0.0),
                    'temperature': getattr(record, 'temperature', 0.0),
                    'feed_rate': getattr(record, 'feed_rate', 0.0),
                    'rpm': getattr(record, 'rpm', 0.0),
                    'coolant_flow': getattr(record, 'coolant_flow', 0.0)
                }
                data.append(record_dict)
            
            df = pd.DataFrame(data)
            
            # Sort by timestamp to ensure chronological order
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df)} telemetry records for machine {machine_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical telemetry for machine {machine_id}: {e}")
            return pd.DataFrame()
    
    def _generate_failure_configs(self, data_length: int, failure_probability: float) -> List[FailureInjectionConfig]:
        """
        Generate failure injection configurations for nightmare training
        """
        configs = []
        
        # Define various failure types to inject
        failure_types = [
            FailureType.SPINDLE_LOAD_SPIKE,
            FailureType.THERMAL_RUNAWAY,
            FailureType.PHANTOM_TRAUMA,
            FailureType.VIBRATION_ANOMALY,
            FailureType.COOLANT_FAILURE,
            FailureType.TOOL_BREAKAGE
        ]
        
        # Create several failure injection points throughout the data
        num_failures = max(1, int(data_length * 0.1))  # ~10% of data points
        failure_indices = sorted([int(i * data_length / num_failures) for i in range(num_failures)])
        
        for idx in failure_indices:
            # Randomly select a failure type
            failure_type = failure_types[idx % len(failure_types)]
            
            # Random parameters for this failure
            severity = 0.5 + (idx % 50) / 100.0  # Vary severity
            duration = min(20, max(5, idx % 25))  # Vary duration
            start_index = max(0, min(idx, data_length - duration))
            
            config = FailureInjectionConfig(
                failure_type=failure_type,
                severity=severity,
                duration=duration,
                start_time_index=start_index,
                probability=failure_probability
            )
            configs.append(config)
        
        return configs
    
    def schedule_nightmare_training(self, machine_ids: List[int], 
                                  schedule_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Schedule nightmare training sessions for multiple machines
        
        Args:
            machine_ids: List of machine IDs to train
            schedule_config: Configuration for the training schedule
            
        Returns:
            Dictionary with scheduling results
        """
        results = {
            'scheduled_sessions': [],
            'failed_sessions': [],
            'schedule_config': schedule_config
        }
        
        for machine_id in machine_ids:
            try:
                # Determine if machine is currently idle (based on schedule or other criteria)
                is_idle = self._check_if_machine_idle(machine_id)
                
                if is_idle or schedule_config.get('force_during_operation', False):
                    # Run nightmare training session
                    session_result = self.run_nightmare_training_session(
                        machine_id=machine_id,
                        duration_hours=schedule_config.get('duration_hours', 1.0),
                        failure_probability=schedule_config.get('failure_probability', 0.7)
                    )
                    
                    results['scheduled_sessions'].append(session_result)
                    logger.info(f"Scheduled nightmare training for machine {machine_id}: {session_result['status']}")
                else:
                    logger.info(f"Skipping nightmare training for busy machine {machine_id}")
                    results['failed_sessions'].append({
                        'machine_id': machine_id,
                        'reason': 'Machine not idle'
                    })
                    
            except Exception as e:
                logger.error(f"Failed to schedule nightmare training for machine {machine_id}: {e}")
                results['failed_sessions'].append({
                    'machine_id': machine_id,
                    'error': str(e)
                })
        
        return results
    
    def _check_if_machine_idle(self, machine_id: int) -> bool:
        """
        Check if a machine is currently idle and suitable for nightmare training
        """
        # This is a simplified check - in practice, this would integrate with
        # machine status monitoring systems
        try:
            # Get most recent telemetry record
            recent_data = self.telemetry_repo.get_recent_by_machine(machine_id, minutes=5)
            
            if not recent_data:
                # No recent data might indicate idle state
                return True
            
            # Check if recent activity is minimal (customize based on your metrics)
            latest_record = recent_data[0]  # Most recent first
            spindle_load = getattr(latest_record, 'spindle_load', 0.0) or 0.0
            feed_rate = getattr(latest_record, 'feed_rate', 0.0) or 0.0
            
            # Consider machine idle if spindle load and feed rate are very low
            is_idle = spindle_load < 10.0 and feed_rate < 100.0
            return is_idle
            
        except Exception as e:
            logger.warning(f"Could not determine idle status for machine {machine_id}: {e}")
            # Default to allowing training if we can't determine status
            return True
    
    def get_training_statistics(self, machine_id: int = None) -> Dict[str, Any]:
        """
        Get statistics about nightmare training sessions
        """
        # This would typically query a database of training results
        # For now, returning placeholder data
        stats = {
            'total_sessions_completed': 0,
            'average_kill_switch_triggers': 0.0,
            'average_preemptive_responses': 0.0,
            'improvement_trend': 'neutral',  # positive, negative, neutral
            'last_session_date': None,
            'policy_update_frequency': 'daily'  # How often policies are updated
        }
        
        return stats
        """
        Get statistics about nightmare training sessions
        """
        # This would typically query a database of training results
        # For now, returning placeholder data
        stats = {
            'total_sessions_completed': 0,
            'average_kill_switch_triggers': 0.0,
            'average_preemptive_responses': 0.0,
            'improvement_trend': 'neutral',  # positive, negative, neutral
            'last_session_date': None,
            'policy_update_frequency': 'daily'  # How often policies are updated
        }
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    # This would typically be initialized with real components
    # For testing purposes, we'll outline the expected usage:
    print("Nightmare Training Orchestrator initialized.")
    print("Ready to coordinate Adversary and Dreamer components for offline learning.")