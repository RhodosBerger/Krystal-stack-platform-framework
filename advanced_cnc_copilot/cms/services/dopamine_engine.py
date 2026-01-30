from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import math
from dataclasses import dataclass

# Removed direct import of Session since we're importing from models and repositories
from ..models import Telemetry
from ..repositories.telemetry_repository import TelemetryRepository

logger = logging.getLogger(__name__)


@dataclass
class NeuroState:
    """Data class for neuro-chemical state"""
    dopamine_level: float  # Reward/efficiency signal (0.0-1.0)
    cortisol_level: float  # Stress/risk signal (0.0-1.0)
    timestamp: datetime
    reasoning_trace: List[str]


class DopamineEngine:
    """
    Implements the Dopamine/Cortisol feedback system for Neuro-Safety
    Based on the 'Phantom Trauma' concept: distinguishes sensor drift from actual stress events
    Uses continuous gradients instead of binary error flags
    """
    
    def __init__(self, repository: TelemetryRepository):
        self.repository = repository
        self.decay_factor = 0.95  # How quickly dopamine/cortisol memories fade
        self.phantom_trauma_threshold = 0.3  # Threshold for detecting phantom trauma
        self.stress_decay_time = 30  # Minutes for stress to decay
        self.reward_decay_time = 60  # Minutes for reward to decay
    
    def calculate_current_state(self, machine_id: int, current_metrics: Dict) -> NeuroState:
        """
        Calculate the current neuro-chemical state based on telemetry data
        Implements continuous gradients instead of binary safety flags
        """
        # Get recent telemetry to establish context
        recent_data = self.repository.get_recent_by_machine(machine_id, minutes=10)
        
        # Calculate current dopamine (reward) level based on efficiency
        dopamine_level = self._calculate_dopamine_response(current_metrics)
        
        # Calculate current cortisol (stress) level based on risk factors
        cortisol_level = self._calculate_cortisol_response(current_metrics)
        
        # Apply memory decay to prevent permanent trauma responses
        dopamine_level = self._apply_memory_decay(dopamine_level, machine_id, 'dopamine')
        cortisol_level = self._apply_memory_decay(cortisol_level, machine_id, 'cortisol')
        
        # Create reasoning trace
        reasoning_trace = self._generate_reasoning_trace(current_metrics, dopamine_level, cortisol_level)
        
        return NeuroState(
            dopamine_level=dopamine_level,
            cortisol_level=cortisol_level,
            timestamp=datetime.utcnow(),
            reasoning_trace=reasoning_trace
        )
    
    def _calculate_dopamine_response(self, metrics: Dict) -> float:
        """
        Calculate dopamine response based on efficiency and positive outcomes
        Higher values indicate better performance/reward
        """
        # Calculate efficiency components
        spindle_efficiency = self._calculate_spindle_efficiency(metrics.get('spindle_load', 50.0))
        vibration_efficiency = self._calculate_vibration_efficiency(metrics.get('vibration_x', 0.5))
        temperature_efficiency = self._calculate_temperature_efficiency(metrics.get('temperature', 35.0))
        feed_efficiency = self._calculate_feed_efficiency(metrics.get('feed_rate', 1000.0))
        
        # Weighted average of efficiency components
        weights = [0.3, 0.25, 0.25, 0.2]  # Adjust weights as needed
        efficiency_score = (
            spindle_efficiency * weights[0] +
            vibration_efficiency * weights[1] +
            temperature_efficiency * weights[2] +
            feed_efficiency * weights[3]
        )
        
        return min(1.0, max(0.0, efficiency_score))
    
    def _calculate_cortisol_response(self, metrics: Dict) -> float:
        """
        Calculate cortisol response based on stress and risk factors
        Higher values indicate higher stress/danger
        """
        # Calculate stress components
        spindle_stress = self._calculate_spindle_stress(metrics.get('spindle_load', 50.0))
        vibration_stress = self._calculate_vibration_stress(metrics.get('vibration_x', 0.5))
        temperature_stress = self._calculate_temperature_stress(metrics.get('temperature', 35.0))
        tool_wear_stress = self._calculate_tool_wear_stress(metrics.get('tool_wear', 0.0))
        
        # Weighted average of stress components
        weights = [0.3, 0.3, 0.25, 0.15]  # Adjust weights as needed
        stress_score = (
            spindle_stress * weights[0] +
            vibration_stress * weights[1] +
            temperature_stress * weights[2] +
            tool_wear_stress * weights[3]
        )
        
        return min(1.0, max(0.0, stress_score))
    
    def _calculate_spindle_efficiency(self, spindle_load: float) -> float:
        """
        Calculate efficiency based on spindle load
        Optimal range is 70-85% for best efficiency
        """
        if 70 <= spindle_load <= 85:
            # Optimal range - full efficiency
            return 1.0
        elif 50 <= spindle_load <= 100:
            # Good range but not optimal - reduced efficiency
            distance_from_optimal = min(abs(spindle_load - 70), abs(spindle_load - 85))
            return max(0.3, 1.0 - (distance_from_optimal / 15.0))
        else:
            # Outside good range - low efficiency
            return max(0.0, 0.5 - abs(spindle_load - 50) / 100.0)
    
    def _calculate_vibration_efficiency(self, vibration: float) -> float:
        """
        Calculate efficiency based on vibration levels
        Lower vibration is better
        """
        # Inverse relationship: lower vibration = higher efficiency
        max_acceptable_vibration = 2.0
        efficiency = max(0.0, 1.0 - (vibration / max_acceptable_vibration))
        return efficiency
    
    def _calculate_temperature_efficiency(self, temperature: float) -> float:
        """
        Calculate efficiency based on temperature
        Optimal range is 35-45°C
        """
        if 35 <= temperature <= 45:
            return 1.0
        else:
            distance_from_optimal = min(abs(temperature - 35), abs(temperature - 45))
            return max(0.1, 1.0 - (distance_from_optimal / 15.0))
    
    def _calculate_feed_efficiency(self, feed_rate: float) -> float:
        """
        Calculate efficiency based on feed rate
        Optimal range depends on material and tooling
        """
        # For this example, assume optimal feed rate is 800-1200 mm/min
        if 800 <= feed_rate <= 1200:
            return 1.0
        else:
            distance_from_optimal = min(abs(feed_rate - 800), abs(feed_rate - 1200))
            return max(0.2, 1.0 - (distance_from_optimal / 800.0))
    
    def _calculate_spindle_stress(self, spindle_load: float) -> float:
        """
        Calculate stress based on spindle load
        High loads create high stress
        """
        if spindle_load > 95:
            # Dangerously high load
            return 1.0
        elif spindle_load > 85:
            # High load
            return 0.8
        elif spindle_load > 70:
            # Moderate load
            return 0.4
        else:
            # Low load
            return 0.1
    
    def _calculate_vibration_stress(self, vibration: float) -> float:
        """
        Calculate stress based on vibration levels
        High vibration creates high stress
        """
        max_dangerous_vibration = 3.0
        stress = min(1.0, (vibration / max_dangerous_vibration))
        return stress
    
    def _calculate_temperature_stress(self, temperature: float) -> float:
        """
        Calculate stress based on temperature
        High temperature creates high stress
        """
        if temperature > 70:
            return 1.0  # Critical stress
        elif temperature > 60:
            return 0.8  # High stress
        elif temperature > 50:
            return 0.5  # Moderate stress
        else:
            return max(0.0, (temperature - 30) / 20.0)  # Low stress, increasing with temperature
    
    def _calculate_tool_wear_stress(self, tool_wear: float) -> float:
        """
        Calculate stress based on tool wear
        High wear creates high stress
        """
        max_acceptable_wear = 0.8  # 80% wear
        stress = min(1.0, tool_wear / max_acceptable_wear)
        return stress
    
    def _apply_memory_decay(self, current_level: float, machine_id: int, chemical_type: str) -> float:
        """
        Apply decay to neuro-chemical levels over time
        Prevents permanent trauma responses while maintaining learning
        """
        # This would typically use the database to track historical levels
        # For now, we'll implement a simple decay model
        return current_level * self.decay_factor
    
    def _generate_reasoning_trace(self, metrics: Dict, dopamine: float, cortisol: float) -> List[str]:
        """
        Generate reasoning trace for decision transparency
        Implements the "Invisible Church" concept from Shadow Council
        """
        trace = [
            f"Dopamine Level: {dopamine:.3f} (Efficiency/Performance)",
            f"Cortisol Level: {cortisol:.3f} (Stress/Risk)",
            f"Neuro-State Balance: {'Reward-Focused' if dopamine > cortisol else 'Safety-Focused'}"
        ]
        
        if cortisol > 0.7:
            trace.append("⚠️ HIGH STRESS DETECTED: Recommend conservative operation")
        elif dopamine > 0.8 and cortisol < 0.3:
            trace.append("✅ OPTIMAL CONDITIONS: Aggressive optimization possible")
        elif abs(dopamine - cortisol) < 0.2:
            trace.append("⚖️ BALANCED STATE: Moderate approach recommended")
        
        return trace
    
    def detect_phantom_trauma(self, machine_id: int, current_metrics: Dict) -> Tuple[bool, str]:
        """
        Detect 'Phantom Trauma' - when system is overly sensitive to safe conditions
        Based on the 'Memory of Pain' concept where stress responses linger unnecessarily
        """
        # Get recent high-stress events to establish pattern
        recent_data = self.repository.get_recent_by_machine(machine_id, minutes=30)
        
        if not recent_data:
            return False, "No historical data for comparison"
        
        # Calculate current stress vs. recent average
        current_stress = self._calculate_cortisol_response(current_metrics)
        
        # Calculate average stress from recent data
        historical_stresses = []
        for record in recent_data:
            # Calculate stress from historical record
            hist_metrics = {
                'spindle_load': record.spindle_load or 0.0,
                'vibration_x': record.vibration_x or 0.0,
                'temperature': record.temperature or 35.0,
                'tool_wear': record.tool_wear or 0.0
            }
            hist_stress = self._calculate_cortisol_response(hist_metrics)
            historical_stresses.append(hist_stress)
        
        avg_historical_stress = sum(historical_stresses) / len(historical_stresses) if historical_stresses else 0.0
        
        # Detect phantom trauma: current conditions are safe but system shows high stress
        current_vibration = current_metrics.get('vibration_x', 0.5)
        current_temperature = current_metrics.get('temperature', 35.0)
        current_load = current_metrics.get('spindle_load', 50.0)
        
        # Determine if current physical conditions are actually safe
        physical_conditions_safe = (
            current_vibration < 1.0 and  # Low vibration
            current_temperature < 55 and  # Normal temperature
            current_load < 80  # Moderate load
        )
        
        # Phantom trauma exists when physical conditions are safe but stress is high
        if physical_conditions_safe and current_stress > 0.6 and current_stress > avg_historical_stress * 1.2:
            return True, f"Phantom Trauma Detected: Physical conditions safe (vib={current_vibration}, temp={current_temperature}, load={current_load}) but stress elevated ({current_stress:.3f}). System may be overly sensitive to safe conditions."
        
        return False, f"Normal response: Current stress ({current_stress:.3f}) consistent with physical conditions and historical patterns (avg={avg_historical_stress:.3f})"
    
    def get_process_recommendation(self, machine_id: int, current_metrics: Dict) -> Dict[str, Any]:
        """
        Generate process recommendations based on neuro-chemical state
        """
        neuro_state = self.calculate_current_state(machine_id, current_metrics)
        
        # Determine operational mode based on dopamine/cortisol balance
        if neuro_state.cortisol_level > 0.7:
            operational_mode = "ECONOMY_MODE"  # Conservative due to high stress
            suggested_action = "Reduce aggressiveness, prioritize safety"
        elif neuro_state.dopamine_level > 0.8 and neuro_state.cortisol_level < 0.3:
            operational_mode = "RUSH_MODE"  # Aggressive when safe and efficient
            suggested_action = "Increase feed rates, optimize for speed"
        else:
            operational_mode = "BALANCED_MODE"  # Moderate approach
            suggested_action = "Maintain current parameters, monitor closely"
        
        # Calculate confidence in recommendation
        confidence = 1.0 - abs(neuro_state.dopamine_level - neuro_state.cortisol_level)
        
        return {
            'suggested_action': suggested_action,
            'operational_mode': operational_mode,
            'dopamine_level': neuro_state.dopamine_level,
            'cortisol_level': neuro_state.cortisol_level,
            'confidence': confidence,
            'reasoning_trace': neuro_state.reasoning_trace,
            'phantom_trauma_risk': self._assess_phantom_trauma_risk(machine_id, current_metrics)
        }
    
    def _assess_phantom_trauma_risk(self, machine_id: int, current_metrics: Dict) -> float:
        """
        Assess the risk of phantom trauma based on system sensitivity
        """
        is_phantom, _ = self.detect_phantom_trauma(machine_id, current_metrics)
        return 1.0 if is_phantom else 0.0