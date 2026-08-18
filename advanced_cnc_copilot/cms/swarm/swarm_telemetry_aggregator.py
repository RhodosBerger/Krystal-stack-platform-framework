from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import json
import asyncio
from dataclasses import dataclass

# Use standard redis instead of redis.asyncio for compatibility
import redis
from sqlalchemy.orm import Session

from ..models import Telemetry
from ..repositories.telemetry_repository import TelemetryRepository
from ..services.shadow_council import ShadowCouncil, DecisionPolicy, CreatorAgent, AuditorAgent, AccountantAgent
from ..services.dopamine_engine import DopamineEngine
from ..services.economics_engine import EconomicsEngine


@dataclass
class SwarmTraumaEvent:
    """Represents a critical event that should be shared across the fleet"""
    machine_id: int  # Changed from str to int to match database schema
    timestamp: datetime
    event_type: str  # 'vibration_spike', 'thermal_overload', 'tool_break', etc.
    severity_level: float  # 0.0-1.0
    parameters_at_event: Dict[str, Any]
    recovery_actions: List[str]
    physics_signature: Dict[str, Any]  # Unique signature of the physics involved


class SwarmTelemetryAggregator:
    """
    Implements Distributed Logic Injection for the Hive Mind Activation
    Shares critical limits (Mantinels) and trauma experiences across the fleet in real-time
    Demonstrates the concept where "one machine's failure is all machines' knowledge"
    """
    
    def __init__(self, 
                 redis_client: redis.Redis, 
                 telemetry_repo: TelemetryRepository,
                 shadow_council: ShadowCouncil,
                 dopamine_engine: DopamineEngine,
                 economics_engine: EconomicsEngine):
        self.redis = redis_client
        self.telemetry_repo = telemetry_repo
        self.shadow_council = shadow_council
        self.dopamine_engine = dopamine_engine
        self.economics_engine = economics_engine
        self.logger = logging.getLogger(__name__)
        
        # Channel for sharing critical events across the fleet
        self.trauma_broadcast_channel = "cnc.swarm.trauma.events"
        self.mantinel_update_channel = "cnc.swarm.mantinel.updates"
        self.physics_signature_channel = "cnc.swarm.physics.signatures"
    
    async def monitor_and_detect_critical_events(self, machine_id: int) -> None:  # Changed to int
        """
        Monitors telemetry data and detects critical events that should be shared with the fleet
        """
        while True:
            try:
                # Get the latest telemetry for this machine
                latest_telemetry = self.telemetry_repo.get_latest_by_machine(machine_id)
                
                if latest_telemetry:
                    # Check for critical conditions that constitute "trauma"
                    trauma_event = self._detect_trauma_conditions(latest_telemetry)
                    
                    if trauma_event:
                        # Share the trauma event across the fleet
                        await self._broadcast_trauma_event(trauma_event)
                        self.logger.info(f"Trauma event broadcasted from machine {machine_id}: {trauma_event.event_type}")
                
                # Sleep before next check (e.g., every 100ms for near real-time monitoring)
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in critical event monitoring: {e}")
                await asyncio.sleep(1)  # Longer sleep on error
    
    def _detect_trauma_conditions(self, telemetry: Telemetry) -> Optional[SwarmTraumaEvent]:
        """
        Detects if current telemetry constitutes a "trauma" event worth sharing with fleet
        """
        # Extract values safely
        spindle_load = getattr(telemetry, 'spindle_load', 0.0) or 0.0
        vibration_x = getattr(telemetry, 'vibration_x', 0.0) or 0.0
        temperature = getattr(telemetry, 'temperature', 0.0) or 0.0
        cortisol_level = getattr(telemetry, 'cortisol_level', 0.0) or 0.0
        dopamine_score = getattr(telemetry, 'dopamine_score', 0.0) or 0.0
        
        # Define critical thresholds for trauma events
        critical_vibration_threshold = 2.0  # High vibration indicating potential issues
        critical_temperature_threshold = 65.0  # High temperature indicating thermal stress
        critical_spindle_load_threshold = 95.0  # Near-maximum spindle load
        critical_cortisol_threshold = 0.8  # High stress level
        
        # Check for different types of trauma events
        if vibration_x > critical_vibration_threshold:
            return SwarmTraumaEvent(
                machine_id=telemetry.machine_id,
                timestamp=telemetry.timestamp,
                event_type='vibration_spike',
                severity_level=min(1.0, vibration_x / 3.0),
                parameters_at_event={
                    'spindle_load': float(spindle_load) if spindle_load is not None else 0.0,
                    'vibration_x': float(vibration_x) if vibration_x is not None else 0.0,
                    'temperature': float(temperature) if temperature is not None else 0.0,
                    'cortisol_level': float(cortisol_level) if cortisol_level is not None else 0.0,
                    'dopamine_score': float(dopamine_score) if dopamine_score is not None else 0.0
                },
                recovery_actions=[
                    'Reduce feed rate by 15%',
                    'Check tool condition',
                    'Verify material clamping'
                ],
                physics_signature={
                    'frequency_spectrum': self._calculate_frequency_signature(vibration_x),
                    'load_correlation': spindle_load / vibration_x if vibration_x > 0 else 0.0,
                    'thermal_factor': temperature
                }
            )
        
        elif temperature > critical_temperature_threshold:
            return SwarmTraumaEvent(
                machine_id=telemetry.machine_id,
                timestamp=telemetry.timestamp,
                event_type='thermal_overload',
                severity_level=min(1.0, temperature / 80.0),
                parameters_at_event={
                    'spindle_load': float(spindle_load) if spindle_load is not None else 0.0,
                    'vibration_x': float(vibration_x) if vibration_x is not None else 0.0,
                    'temperature': float(temperature) if temperature is not None else 0.0,
                    'cortisol_level': float(cortisol_level) if cortisol_level is not None else 0.0,
                    'dopamine_score': float(dopamine_score) if dopamine_score is not None else 0.0
                },
                recovery_actions=[
                    'Increase coolant flow',
                    'Reduce spindle speed by 20%',
                    'Check cutting parameters'
                ],
                physics_signature={
                    'heat_generation_rate': temperature,
                    'load_efficiency_ratio': spindle_load / temperature if temperature > 0 else 0.0,
                    'cooling_response_time': 0.0  # Would be calculated based on cooling changes
                }
            )
        
        elif spindle_load > critical_spindle_load_threshold:
            return SwarmTraumaEvent(
                machine_id=telemetry.machine_id,
                timestamp=telemetry.timestamp,
                event_type='high_load_stress',
                severity_level=min(1.0, spindle_load / 100.0),
                parameters_at_event={
                    'spindle_load': float(spindle_load) if spindle_load is not None else 0.0,
                    'vibration_x': float(vibration_x) if vibration_x is not None else 0.0,
                    'temperature': float(temperature) if temperature is not None else 0.0,
                    'cortisol_level': float(cortisol_level) if cortisol_level is not None else 0.0,
                    'dopamine_score': float(dopamine_score) if dopamine_score is not None else 0.0
                },
                recovery_actions=[
                    'Optimize feed rate vs speed ratio',
                    'Verify tool sharpness',
                    'Check material hardness'
                ],
                physics_signature={
                    'load_distribution': spindle_load,
                    'vibration_correlation': vibration_x / spindle_load if spindle_load > 0 else 0.0,
                    'thermal_generation': temperature
                }
            )
        
        elif cortisol_level > critical_cortisol_threshold and dopamine_score < 0.3:
            # High stress with low reward indicates problematic situation
            return SwarmTraumaEvent(
                machine_id=telemetry.machine_id,
                timestamp=telemetry.timestamp,
                event_type='neuro_stress_event',
                severity_level=float(cortisol_level) if cortisol_level is not None else 0.0,
                parameters_at_event={
                    'spindle_load': float(spindle_load) if spindle_load is not None else 0.0,
                    'vibration_x': float(vibration_x) if vibration_x is not None else 0.0,
                    'temperature': float(temperature) if temperature is not None else 0.0,
                    'cortisol_level': float(cortisol_level) if cortisol_level is not None else 0.0,
                    'dopamine_score': float(dopamine_score) if dopamine_score is not None else 0.0
                },
                recovery_actions=[
                    'Switch to ECONOMY mode',
                    'Review process parameters',
                    'Check for Phantom Trauma'
                ],
                physics_signature={
                    'stress_reward_imbalance': cortisol_level - dopamine_score,
                    'system_instability_factor': cortisol_level,
                    'efficiency_loss': 1.0 - dopamine_score
                }
            )
        
        # No critical event detected
        return None
    
    def _calculate_frequency_signature(self, vibration_value: float) -> Dict[str, Any]:
        """
        Calculate frequency signature for vibration analysis
        This would normally use FFT or other signal processing techniques
        """
        # Simplified signature calculation for demonstration
        return {
            'primary_frequency': vibration_value * 10,  # Simplified mapping
            'harmonic_content': vibration_value * 0.3,  # Harmonic content approximation
            'energy_distribution': vibration_value * 0.7  # Energy distribution approximation
        }
    
    async def _broadcast_trauma_event(self, trauma_event: SwarmTraumaEvent) -> None:
        """
        Broadcasts trauma event to all machines in the fleet via Redis pub/sub
        """
        try:
            serialized_event = {
                'machine_id': trauma_event.machine_id,
                'timestamp': trauma_event.timestamp.isoformat(),
                'event_type': trauma_event.event_type,
                'severity_level': trauma_event.severity_level,
                'parameters_at_event': trauma_event.parameters_at_event,
                'recovery_actions': trauma_event.recovery_actions,
                'physics_signature': trauma_event.physics_signature
            }
            
            # Publish to Redis channel for fleet-wide distribution
            self.redis.publish(self.trauma_broadcast_channel, json.dumps(serialized_event))
            
            # Also store in a persistent queue for machines that join later
            self.redis.lpush(f"persistent_trauma_events:{trauma_event.event_type}", 
                             json.dumps(serialized_event))
            
        except Exception as e:
            self.logger.error(f"Error broadcasting trauma event: {e}")
    
    async def subscribe_to_fleet_events(self, machine_id: int, callback_handler) -> None:  # Changed to int
        """
        Subscribe to fleet-wide events to receive shared trauma experiences
        """
        pubsub = self.redis.pubsub()
        pubsub.subscribe(self.trauma_broadcast_channel, 
                         self.mantinel_update_channel, 
                         self.physics_signature_channel)
        
        for message in pubsub.listen():
            if message['type'] == 'message':
                channel = message['channel'].decode('utf-8')
                data = json.loads(message['data'].decode('utf-8'))
                
                if channel == self.trauma_broadcast_channel:
                    # Process shared trauma event
                    await self._handle_shared_trauma_event(data, machine_id, callback_handler)
                elif channel == self.mantinel_update_channel:
                    # Process shared mantinel update
                    await self._handle_shared_mantinel_update(data, machine_id, callback_handler)
                elif channel == self.physics_signature_channel:
                    # Process shared physics signature
                    await self._handle_shared_physics_signature(data, machine_id, callback_handler)
    
    async def _handle_shared_trauma_event(self, event_data: Dict, target_machine_id: int, callback_handler) -> None:  # Changed to int
        """
        Handles a shared trauma event from another machine in the fleet
        Updates the local machine's "Fleet Hippocampus" with the shared experience
        """
        source_machine_id = event_data['machine_id']
        
        # Only process if the event is relevant to our current operations
        if await self._is_event_relevant_to_machine(event_data, target_machine_id):
            self.logger.info(f"Processing shared trauma from machine {source_machine_id} for machine {target_machine_id}")
            
            # Update the local dopamine/cortisol memory with shared trauma
            await self._update_local_memory_with_shared_experience(event_data, target_machine_id)
            
            # Trigger local shadow council to adjust parameters based on shared knowledge
            await self._adjust_parameters_based_on_shared_knowledge(event_data, target_machine_id)
            
            # Call the callback handler to update local systems
            await callback_handler('shared_trauma_event', event_data, target_machine_id)
    
    async def _handle_shared_mantinel_update(self, event_data: Dict, target_machine_id: int, callback_handler) -> None:  # Changed to int
        """
        Handles a shared mantinel update from another machine in the fleet
        """
        source_machine_id = event_data.get('source_machine', 'unknown')
        self.logger.info(f"Processing shared mantinel update from machine {source_machine_id} for machine {target_machine_id}")
        
        # Apply the logic update to the target machine
        logic_update = event_data.get('logic_update', {})
        await self._apply_mantinel_update(logic_update, target_machine_id)
        
        # Call the callback handler
        await callback_handler('shared_mantinel_update', event_data, target_machine_id)
    
    async def _handle_shared_physics_signature(self, event_data: Dict, target_machine_id: int, callback_handler) -> None:  # Changed to int
        """
        Handles a shared physics signature from another machine in the fleet
        """
        source_machine_id = event_data.get('machine_id', 'unknown')
        self.logger.info(f"Processing shared physics signature from machine {source_machine_id} for machine {target_machine_id}")
        
        # Update physics pattern recognition based on shared signature
        signature_data = event_data.get('signature_data', {})
        await self._update_physics_patterns(signature_data, target_machine_id)
        
        # Call the callback handler
        await callback_handler('shared_physics_signature', event_data, target_machine_id)
    
    async def _is_event_relevant_to_machine(self, event_data: Dict, machine_id: int) -> bool:  # Changed to int
        """
        Determines if a shared event is relevant to the current machine
        Uses physics signature matching and operational context
        """
        # Get current operational context for the target machine
        current_state = self.dopamine_engine.calculate_current_state(machine_id, {})
        
        # Extract physics signature from event
        event_signature = event_data.get('physics_signature', {})
        event_type = event_data.get('event_type', '')
        
        # For now, use a simplified relevance check
        # In practice, this would use more sophisticated pattern matching
        if event_type in ['vibration_spike', 'high_load_stress']:
            # Check if current parameters are similar to those that caused the trauma
            current_params = getattr(current_state, 'current_parameters', {})
            event_params = event_data.get('parameters_at_event', {})
            
            # Calculate similarity score between current and event parameters
            similarity_score = self._calculate_parameter_similarity(current_params, event_params)
            
            # If parameters are similar, the event is relevant
            return similarity_score > 0.6  # 60% similarity threshold
        
        return True  # Default to relevant if we can't determine otherwise
    
    def _calculate_parameter_similarity(self, params1: Dict, params2: Dict) -> float:
        """
        Calculate similarity between two sets of parameters
        """
        if not params1 or not params2:
            return 0.0
        
        # Calculate similarity for key parameters
        similarities = []
        
        for key in ['spindle_load', 'vibration_x', 'temperature']:
            if key in params1 and key in params2:
                val1 = params1[key]
                val2 = params2[key]
                
                # Calculate normalized similarity (0-1)
                max_val = max(val1, val2, 1.0)  # Avoid division by zero
                similarity = 1.0 - abs(val1 - val2) / max_val
                similarities.append(max(0.0, similarity))
        
        if similarities:
            return sum(similarities) / len(similarities)
        else:
            return 0.0
    
    async def _update_local_memory_with_shared_experience(self, event_data: Dict, machine_id: int) -> None:  # Changed to int
        """
        Updates the local machine's memory with shared trauma experience
        This is the "Fleet Hippocampus" functionality
        """
        # Update the local dopamine and cortisol memory based on shared experience
        event_type = event_data['event_type']
        severity = event_data['severity_level']
        
        # Log the shared experience in the local knowledge base
        self.logger.info(f"Machine {machine_id} learned from shared trauma event: {event_type} with severity {severity}")
    
    async def _adjust_parameters_based_on_shared_knowledge(self, event_data: Dict, machine_id: int) -> None:  # Changed to int
        """
        Adjusts machine parameters based on shared fleet knowledge
        Implements the "Quadratic Mantinel" update across the fleet
        """
        event_type = event_data['event_type']
        event_params = event_data['parameters_at_event']
        
        # Get current machine parameters
        latest_telemetry = self.telemetry_repo.get_latest_by_machine(machine_id)
        current_params = {}  # Initialize empty dict to prevent unbound variable error
        
        if latest_telemetry:
            current_params = {
                'spindle_load': float(getattr(latest_telemetry, 'spindle_load', 0.0) or 0.0),
                'vibration_x': float(getattr(latest_telemetry, 'vibration_x', 0.0) or 0.0),
                'temperature': float(getattr(latest_telemetry, 'temperature', 0.0) or 0.0),
                'cortisol_level': float(getattr(latest_telemetry, 'cortisol_level', 0.0) or 0.0),
                'dopamine_score': float(getattr(latest_telemetry, 'dopamine_score', 0.0) or 0.0)
            }
        else:
            # Default values if no telemetry available
            current_params = {
                'spindle_load': 50.0,
                'vibration_x': 0.5,
                'temperature': 35.0,
                'cortisol_level': 0.1,
                'dopamine_score': 0.5
            }
        
        # Apply conservative adjustments based on shared trauma
        if event_type == 'vibration_spike':
            # Reduce aggressiveness to avoid similar vibration issues
            if 'feed_rate' in current_params and current_params['feed_rate']:
                new_feed_rate = min(current_params['feed_rate'], event_params.get('feed_rate', current_params['feed_rate']) * 0.9)
                # Update the quadratic mantinel constraint locally
                self._update_quadratic_mantinel(machine_id, 'feed_rate', new_feed_rate)
        
        elif event_type == 'thermal_overload':
            # Adjust thermal parameters based on shared experience
            if 'spindle_load' in current_params and current_params['spindle_load']:
                new_spindle_load = min(current_params['spindle_load'], event_params.get('spindle_load', current_params['spindle_load']) * 0.85)
                self._update_thermal_constraint(machine_id, new_spindle_load)
    
    def _update_quadratic_mantinel(self, machine_id: int, parameter: str, new_limit: float) -> None:  # Changed to int
        """
        Updates the local Quadratic Mantinel based on shared fleet knowledge
        """
        # This would update the local physics constraints based on fleet-wide learning
        self.logger.info(f"Updating Quadratic Mantinel for machine {machine_id}, parameter {parameter} to {new_limit}")
        
        # In a real implementation, this would update the local constraint matrices
        # that govern how the machine responds to different conditions
    
    def _update_thermal_constraint(self, machine_id: int, new_spindle_load_limit: float) -> None:  # Changed to int
        """
        Updates thermal constraints based on shared knowledge
        """
        self.logger.info(f"Updating thermal constraint for machine {machine_id}, new spindle load limit: {new_spindle_load_limit}")
    
    async def inject_logic_updates(self, source_machine_id: int, target_machine_ids: List[int],  # Changed to int
                                   logic_update: Dict[str, Any]) -> None:
        """
        Injects critical logic updates (Mantinels) from one machine to others in the fleet
        This is the "Distributed Logic Injection" mechanism
        """
        update_payload = {
            'source_machine': source_machine_id,
            'timestamp': datetime.utcnow().isoformat(),
            'update_type': 'mantinel_update',
            'logic_update': logic_update,
            'target_machines': target_machine_ids
        }
        
        # Broadcast the logic update to the fleet
        self.redis.publish(self.mantinel_update_channel, json.dumps(update_payload))
        
        self.logger.info(f"Distributed logic injection from {source_machine_id} to {len(target_machine_ids)} machines")
    
    async def _apply_mantinel_update(self, logic_update: Dict[str, Any], machine_id: int) -> None:  # Changed to int
        """
        Applies a mantinel update to a specific machine
        """
        self.logger.info(f"Applying mantinel update to machine {machine_id}: {logic_update}")
        # In a real implementation, this would update the machine's local constraint system
    
    async def _update_physics_patterns(self, signature_data: Dict[str, Any], machine_id: int) -> None:  # Changed to int
        """
        Updates physics pattern recognition based on shared signatures
        """
        self.logger.info(f"Updating physics patterns for machine {machine_id} with signature: {signature_data}")
        # In a real implementation, this would update the machine's pattern recognition algorithms
    
    async def synchronize_physics_signatures(self, machine_id: int, signature_data: Dict[str, Any]) -> None:  # Changed to int
        """
        Synchronizes physics signatures across the fleet to improve pattern recognition
        """
        signature_payload = {
            'machine_id': machine_id,
            'timestamp': datetime.utcnow().isoformat(),
            'signature_type': 'physics_pattern',
            'signature_data': signature_data,
            'pattern_category': self._categorize_physics_pattern(signature_data)
        }
        
        # Share the physics signature with the fleet
        self.redis.publish(self.physics_signature_channel, json.dumps(signature_payload))
    
    def _categorize_physics_pattern(self, signature_data: Dict[str, Any]) -> str:
        """
        Categorizes physics patterns to enable better matching and learning
        """
        # Simplified categorization based on dominant signature elements
        energy_distribution = signature_data.get('energy_distribution', 0)
        harmonic_content = signature_data.get('harmonic_content', 0)
        heat_generation_rate = signature_data.get('heat_generation_rate', 0)
        
        if energy_distribution > 1.5:
            return 'high_energy_operation'
        elif harmonic_content > 1.0:
            return 'vibration_prone_operation'
        elif heat_generation_rate > 50:
            return 'thermal_stress_operation'
        else:
            return 'normal_operation'
    
    async def get_fleet_wide_insights(self, machine_id: int, current_operation: str) -> Dict[str, Any]:  # Changed to int
        """
        Retrieves fleet-wide insights relevant to the current operation
        """
        # Get recent shared events from the persistent queue
        recent_traumas_bytes = self.redis.lrange(f"persistent_trauma_events:*", 0, 100)
        
        # Decode bytes to strings and parse JSON
        recent_traumas = []
        for trauma in recent_traumas_bytes:
            if isinstance(trauma, bytes):
                trauma_str = trauma.decode('utf-8')
            else:
                trauma_str = trauma
            recent_traumas.append(json.loads(trauma_str))
        
        relevant_insights = []
        for trauma in recent_traumas:
            # Check if this trauma is relevant to our current operation
            if await self._is_event_relevant_to_machine(trauma, machine_id):
                relevant_insights.append(trauma)
        
        # Calculate fleet-wide statistics for relevant insights
        if relevant_insights:
            avg_severity = sum(i['severity_level'] for i in relevant_insights) / len(relevant_insights)
            common_recovery_actions = self._get_common_recovery_actions(relevant_insights)
            
            return {
                'relevant_shared_experiences': len(relevant_insights),
                'average_severity': avg_severity,
                'common_recovery_actions': common_recovery_actions,
                'recommended_parameter_adjustments': self._generate_parameter_recommendations(avg_severity, common_recovery_actions)
            }
        else:
            return {
                'relevant_shared_experiences': 0,
                'average_severity': 0.0,
                'common_recovery_actions': [],
                'recommended_parameter_adjustments': {}
            }
    
    def _get_common_recovery_actions(self, insights: List[Dict]) -> List[str]:
        """
        Extract common recovery actions from shared insights
        """
        action_counts = {}
        for insight in insights:
            for action in insight.get('recovery_actions', []):
                action_counts[action] = action_counts.get(action, 0) + 1
        
        # Return actions that appear in more than 30% of insights
        return [action for action, count in action_counts.items() 
                if count / len(insights) > 0.3]
    
    def _generate_parameter_recommendations(self, avg_severity: float, common_actions: List[str]) -> Dict[str, Any]:
        """
        Generate parameter recommendations based on fleet insights
        """
        recommendations = {}
        
        if avg_severity > 0.7:
            # High severity suggests conservative approach
            recommendations['feed_rate_adjustment'] = 0.85  # Reduce by 15%
            recommendations['spindle_speed_adjustment'] = 0.90  # Reduce by 10%
            recommendations['safety_margin_increase'] = 0.20  # Increase safety margins by 20%
        
        elif avg_severity > 0.4:
            # Moderate severity suggests cautious approach
            recommendations['feed_rate_adjustment'] = 0.95  # Reduce by 5%
            recommendations['safety_margin_increase'] = 0.10  # Increase safety margins by 10%
        
        return recommendations


# Example usage and initialization
if __name__ == "__main__":
    import asyncio
    
    # This would be instantiated with actual dependencies in the application
    async def example_usage():
        # Example of how the aggregator would be used in practice
        # This is a demonstration of the concept: "failure of one is knowledge of all"
        
        print("Swarm Telemetry Aggregator - Hive Mind Activation Module")
        print("This component enables distributed logic injection across the CNC fleet")
        print("One machine's failure becomes all machines' knowledge through real-time sharing")
        
        # The aggregator would be integrated into the main application lifecycle
        # to continuously monitor, detect, and share critical events across the fleet