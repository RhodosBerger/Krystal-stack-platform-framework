"""
Performance Preset Management for GAMESA Grid System
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List
import time


@dataclass
class PerformancePreset:
    """Performance preset with strategic weights and configuration"""
    name: str
    config: Dict[str, Any]
    weights: Dict[str, float] = field(default_factory=dict)
    effectiveness_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate weights after initialization"""
        if not self.weights:
            self.weights = self._calculate_weights()
    
    def _calculate_weights(self) -> Dict[str, float]:
        """Calculate strategic weights based on preset goals"""
        weights = {
            'performance': self.config.get('performance_weight', 0.4),
            'power': self.config.get('power_weight', 0.3),
            'thermal': self.config.get('thermal_weight', 0.3),
            'resource_balance': self.config.get('balance_weight', 0.2),
            'latency': self.config.get('latency_weight', 0.1)
        }
        return weights


# Define the available presets
PERFORMANCE_PRESET = PerformancePreset("performance", {
    'performance_weight': 0.6,
    'power_weight': 0.1,
    'thermal_weight': 0.2,
    'balance_weight': 0.1,
    'latency_weight': 0.5,
    'prioritize_gpu': True,
    'max_cpu_cores': 80,
    'memory_bandwidth_focus': True
})

POWER_PRESET = PerformancePreset("power", {
    'performance_weight': 0.2,
    'power_weight': 0.6,
    'thermal_weight': 0.1,
    'balance_weight': 0.4,
    'latency_weight': 0.1,
    'prioritize_power_efficient': True,
    'limit_gpu_usage': True,
    'memory_efficiency_focus': True
})

THERMAL_PRESET = PerformancePreset("thermal", {
    'performance_weight': 0.1,
    'power_weight': 0.2,
    'thermal_weight': 0.7,
    'balance_weight': 0.3,
    'latency_weight': 0.1,
    'thermal_prioritization': True,
    'cooling_aware': True,
    'temperature_limits': True
})

BALANCED_PRESET = PerformancePreset("balanced", {
    'performance_weight': 0.3,
    'power_weight': 0.3,
    'thermal_weight': 0.3,
    'balance_weight': 0.5,
    'latency_weight': 0.2,
    'balanced_approach': True
})


class PresetManager:
    """Manager for performance presets with adaptation capability"""
    
    def __init__(self):
        self.presets = {
            'performance': PERFORMANCE_PRESET,
            'power': POWER_PRESET,
            'thermal': THERMAL_PRESET,
            'balanced': BALANCED_PRESET
        }
        self.current_preset = 'balanced'  # Default preset
        self.adaptation_enabled = True
    
    def get_preset(self, name: str) -> PerformancePreset:
        """Get a performance preset by name"""
        return self.presets.get(name, self.presets['balanced'])
    
    def set_current_preset(self, name: str) -> bool:
        """Set the current performance preset"""
        if name in self.presets:
            self.current_preset = name
            return True
        return False
    
    def adapt_preset_based_on_feedback(self, preset_name: str, performance_feedback: Dict[str, Any]):
        """Adapt preset configuration based on performance outcomes"""
        preset = self.presets.get(preset_name)
        if not preset:
            return
        
        # Analyze recent performance data
        performance_metrics = performance_feedback.get('metrics', {})
        
        # Adjust weights based on performance gaps
        new_weights = preset.weights.copy()
        
        if performance_metrics.get('throughput', 0) < performance_feedback.get('target_throughput', 0):
            # Need more performance - increase performance weight
            new_weights['performance'] = min(1.0, new_weights['performance'] + 0.1)
            # Reduce other weights proportionally
            total_other = sum(v for k, v in new_weights.items() if k != 'performance')
            if total_other > 0:
                for k in new_weights:
                    if k != 'performance':
                        new_weights[k] = new_weights[k] * 0.9  # Reduce other weights
        elif performance_metrics.get('power_consumption', 0) > performance_feedback.get('max_power', float('inf')):
            # Need less power - increase power efficiency weight
            new_weights['power'] = min(1.0, new_weights['power'] + 0.1)
        elif performance_metrics.get('temperature', 0) > performance_feedback.get('max_temperature', float('inf')):
            # Need thermal management - increase thermal weight
            new_weights['thermal'] = min(1.0, new_weights['thermal'] + 0.1)
        
        # Update preset with new weights
        preset.weights = new_weights
        preset.effectiveness_history.append({
            'timestamp': time.time(),
            'weights': new_weights.copy(),
            'feedback': performance_feedback
        })
        
        print(f"Adapted preset {preset_name} with new weights: {new_weights}")
    
    def get_effectiveness_analysis(self, preset_name: str) -> Dict[str, Any]:
        """Get effectiveness analysis for a preset"""
        preset = self.presets.get(preset_name)
        if not preset:
            return {}
        
        history = preset.effectiveness_history
        if not history:
            return {'message': 'No effectiveness data available'}
        
        # Calculate average effectiveness metrics
        performance_scores = []
        power_scores = []
        thermal_scores = []
        
        for record in history[-10:]:  # Last 10 records
            if 'feedback' in record:
                feedback = record['feedback']
                if 'metrics' in feedback:
                    metrics = feedback['metrics']
                    if 'performance_score' in metrics:
                        performance_scores.append(metrics['performance_score'])
                    if 'power_score' in metrics:
                        power_scores.append(metrics['power_score'])
                    if 'thermal_score' in metrics:
                        thermal_scores.append(metrics['thermal_score'])
        
        analysis = {
            'preset_name': preset_name,
            'history_records': len(history),
            'average_performance': sum(performance_scores) / len(performance_scores) if performance_scores else 0,
            'average_power': sum(power_scores) / len(power_scores) if power_scores else 0,
            'average_thermal': sum(thermal_scores) / len(thermal_scores) if thermal_scores else 0,
            'last_updated': history[-1]['timestamp'] if history else None
        }
        
        return analysis


# Create global preset manager instance
preset_manager = PresetManager()