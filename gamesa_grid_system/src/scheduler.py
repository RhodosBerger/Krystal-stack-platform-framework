"""
Adaptive Scheduling System for GAMESA Grid System
"""
from typing import Any, Tuple, Optional
from .grid import MemoryGrid3D
from .guardian import GuardianCharacter, GuardianState
from .telemetry import HardwareTelemetry, ai_guided_placement
import time


class AdaptiveScheduler:
    """Adaptive scheduler using 3D grid and Guardian strategy"""
    
    def __init__(self, memory_grid: MemoryGrid3D):
        self.grid = memory_grid
        self.guardian = GuardianCharacter()
        self.telemetry = HardwareTelemetry()
        self.scheduling_presets = {}  # Will be set externally
        self.adaptation_history = []
    
    def schedule_operation(self, operation: Any) -> Tuple[bool, Optional[Tuple[int, int, int]]]:
        """Schedule operation using 3D grid and Guardian strategy"""
        # 1. Guardian observes current system state
        self.guardian.transition(GuardianState.OBSERVING)
        system_state = self._collect_system_state()
        
        # 2. Guardian evaluates potential grid positions
        self.guardian.transition(GuardianState.EVALUATING)
        position_scores = self._evaluate_grid_positions(operation, system_state)
        
        # 3. Guardian makes strategic decision
        self.guardian.transition(GuardianState.DECIDING)
        optimal_position = self.guardian.tic_tac_toe_decision(
            self.grid.grid, 
            operation
        )
        
        # 4. Guardian acts by placing operation
        self.guardian.transition(GuardianState.ACTING)
        success = self.grid.place_operation(operation, optimal_position)
        
        # 5. Guardian learns from outcome
        self.guardian.transition(GuardianState.LEARNING)
        self._learn_from_scheduling(operation, success, optimal_position)
        
        return success, optimal_position
    
    def _collect_system_state(self):
        """Collect current system state from telemetry"""
        return self.telemetry.collect_telemetry()
    
    def _evaluate_grid_positions(self, operation: Any, system_state: dict):
        """Evaluate all positions in the grid for the operation"""
        # For now, just return a basic evaluation
        # In a real system, this would be much more comprehensive
        return {
            'system_state': system_state,
            'grid_occupancy': self.grid.calculate_occupancy(),
            'available_positions': self._get_available_positions()
        }
    
    def _get_available_positions(self) -> list:
        """Get all available positions in the grid"""
        available = []
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                for z in range(self.grid.depth):
                    if self.grid.is_available(x, y, z):
                        available.append((x, y, z))
        return available
    
    def _get_ai_recommendation(self, operation: Any, system_state: dict, grid_state):
        """Get AI recommendation for operation placement"""
        return ai_guided_placement(operation, system_state, grid_state)
    
    def _learn_from_scheduling(self, operation: Any, success: bool, position: Tuple[int, int, int]):
        """Learn from scheduling outcome"""
        self.guardian.learn_from_outcome(position, success, operation)
        self.adaptation_history.append({
            'operation': operation,
            'position': position,
            'success': success,
            'timestamp': time.time()
        })
    
    def _combine_ai_and_tactical(self, ai_positions: list, tactical_positions: list, weights: dict):
        """Combine AI recommendations with tactical decisions"""
        # For now, use the first AI recommendation if available
        if ai_positions:
            return ai_positions[0]
        elif tactical_positions:
            return tactical_positions[0]
        else:
            # Default to center
            return (self.grid.width // 2, self.grid.height // 2, self.grid.depth // 2)
    
    def _get_tactical_positions(self) -> list:
        """Get positions from tactical analysis"""
        # For now, just return some strategic positions
        return [(3, 3, 3), (4, 4, 4), (5, 5, 5)]
    
    def _get_optimal_position_by_preset(self, operation: Any, preset: str) -> Tuple[int, int, int]:
        """Get optimal position based on selected performance preset"""
        system_state = self._collect_system_state()
        
        if preset == "performance":
            return self._find_throughput_optimal_position(operation, system_state)
        elif preset == "power":
            return self._find_power_optimal_position(operation, system_state)
        elif preset == "thermal":
            return self._find_thermal_optimal_position(operation, system_state)
        elif preset == "balanced":
            return self._find_balanced_position(operation, system_state)
        else:
            return self._find_intelligent_position(operation, system_state)
    
    def _find_throughput_optimal_position(self, operation: Any, system_state: dict) -> Tuple[int, int, int]:
        """Find position optimal for maximum throughput"""
        # Prioritize high-performance resource areas
        # This is a simplified implementation
        return self._find_best_position(operation, priority_factor=0.7)
    
    def _find_power_optimal_position(self, operation: Any, system_state: dict) -> Tuple[int, int, int]:
        """Find position optimal for power efficiency"""
        # Prioritize power-efficient resource areas
        return self._find_best_position(operation, power_factor=0.7)
    
    def _find_thermal_optimal_position(self, operation: Any, system_state: dict) -> Tuple[int, int, int]:
        """Find position optimal for thermal management"""
        # Prioritize areas that help with cooling
        return self._find_best_position(operation, thermal_factor=0.7)
    
    def _find_balanced_position(self, operation: Any, system_state: dict) -> Tuple[int, int, int]:
        """Find position that balances all factors"""
        return self._find_best_position(operation, balanced=True)
    
    def _find_intelligent_position(self, operation: Any, system_state: dict) -> Tuple[int, int, int]:
        """Find intelligent position using Guardian strategy"""
        return self.guardian.tic_tac_toe_decision(self.grid.grid, operation)
    
    def _find_best_position(self, operation: Any, priority_factor=0.5, 
                           power_factor=0.5, thermal_factor=0.5, balanced=False) -> Tuple[int, int, int]:
        """Find best position based on given factors"""
        if balanced:
            # For balanced approach, use Guardian's strategic decision
            return self.guardian.tic_tac_toe_decision(self.grid.grid, operation)
        
        # Otherwise, prioritize the given factor
        best_position = self.grid.find_best_position(operation)
        return best_position if best_position else (3, 3, 3)