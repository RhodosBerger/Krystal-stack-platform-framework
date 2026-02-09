"""
Anti-Fragile Marketplace - Survivor Ranking Algorithm
Implements the "Survivor Score" algorithm that ranks G-Code strategies by resilience
rather than speed, focusing on their ability to survive chaotic conditions.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import math
import uuid


@dataclass
class EnvironmentalStressTest:
    """Record of a G-Code strategy tested under specific environmental stresses"""
    strategy_id: str
    test_type: str  # 'vibration', 'thermal', 'load', 'speed', etc.
    stress_level: float  # Intensity of the stress (0.0 to 1.0)
    success: bool  # Whether the operation completed successfully under stress
    performance_metrics: Dict[str, float]  # Quality, accuracy, etc. under stress
    timestamp: datetime
    machine_id: str
    notes: str = ""


@dataclass
class SurvivorBadge:
    """Badge awarded to G-Code strategies that demonstrate resilience"""
    strategy_id: str
    strategy_name: str
    material: str
    operation_type: str
    survivor_score: float  # The Anti-Fragile Score
    stress_tests_completed: int
    stress_tests_passed: int
    environmental_resilience: Dict[str, float]  # Resilience by stress type
    complexity_factor: float
    badge_level: str  # 'Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond'
    awarded_at: datetime
    awarded_by: str
    validity_period: int  # Days before re-evaluation needed
    tags: List[str]  # Descriptive tags for the strategy


class SurvivorRankingSystem:
    """
    The Survivor Ranking System - Calculates Anti-Fragile Scores for G-Code strategies
    
    The core algorithm:
    Anti-Fragile Score = (Successes_in_High_Stress_Environments / Total_High_Stress_Attempts) × Complexity_Factor
    
    This shifts focus from "fastest cycle time" to "ability to maintain quality under chaos"
    """
    
    def __init__(self):
        self.stress_tests = []  # Records of all stress tests performed
        self.badges = {}  # Awarded survivor badges
        self.ranking_history = {}  # Historical scores for strategies
    
    def calculate_survivor_score(self, strategy_id: str, stress_test_results: List[EnvironmentalStressTest]) -> float:
        """
        Calculate the Anti-Fragile Score for a strategy based on stress test results.
        
        Args:
            strategy_id: Unique identifier for the G-Code strategy
            stress_test_results: List of environmental stress tests for this strategy
            
        Returns:
            Anti-Fragile Score (0.0 to 1.0, where 1.0 is perfectly anti-fragile)
        """
        if not stress_test_results:
            # No stress tests means score of 0 (untested strategies are not trusted)
            return 0.0
        
        # Calculate basic success rate under stress
        total_stress_tests = len(stress_test_results)
        successful_tests = sum(1 for test in stress_test_results if test.success)
        
        if total_stress_tests == 0:
            return 0.0
        
        base_success_rate = successful_tests / total_stress_tests
        
        # Calculate stress-weighted success rate (higher stress = more weight)
        weighted_success = 0.0
        total_weight = 0.0
        
        for test in stress_test_results:
            # Higher stress levels contribute more to the score
            weight = test.stress_level  # Could be adjusted with exponentiation
            weighted_success += (1.0 if test.success else 0.0) * weight
            total_weight += weight
        
        stress_weighted_score = weighted_success / total_weight if total_weight > 0 else 0.0
        
        # Calculate complexity factor based on the sophistication of the strategy
        # More complex strategies (with more parameters/variations) get higher complexity factor
        complexity_factor = self._calculate_complexity_factor(strategy_id, stress_test_results)
        
        # Final Anti-Fragile Score combines stress performance with complexity
        anti_fragile_score = min(1.0, stress_weighted_score * complexity_factor)
        
        return anti_fragile_score
    
    def _calculate_complexity_factor(self, strategy_id: str, stress_test_results: List[EnvironmentalStressTest]) -> float:
        """
        Calculate a complexity factor that rewards strategies that handle diverse conditions.
        
        Args:
            strategy_id: Strategy identifier
            stress_test_results: Test results for this strategy
            
        Returns:
            Complexity factor (typically 1.0 to 1.5)
        """
        if not stress_test_results:
            return 1.0
        
        # Count unique stress types encountered
        unique_stress_types = len(set(test.test_type for test in stress_test_results))
        
        # Count unique stress levels (more granular testing = higher complexity)
        unique_stress_levels = len(set(round(test.stress_level, 2) for test in stress_test_results))
        
        # Calculate diversity bonus based on variety of tests
        diversity_bonus = (unique_stress_types * 0.1) + (unique_stress_levels * 0.05)
        
        # Base complexity factor
        complexity_factor = 1.0 + min(diversity_bonus, 0.5)  # Cap at 1.5
        
        return complexity_factor
    
    def evaluate_strategy_resilience(self, strategy_id: str, strategy_name: str, 
                                   material: str, operation_type: str) -> SurvivorBadge:
        """
        Evaluate a strategy's resilience and award a Survivor Badge if appropriate.
        
        Args:
            strategy_id: Unique identifier for the G-Code strategy
            strategy_name: Human-readable name for the strategy
            material: Material the strategy is designed for
            operation_type: Type of operation (face_mill, drill, etc.)
            
        Returns:
            SurvivorBadge with calculated anti-fragile score and badge level
        """
        # Get all stress tests for this strategy
        strategy_tests = [test for test in self.stress_tests if test.strategy_id == strategy_id]
        
        # Calculate the anti-fragile score
        anti_fragile_score = self.calculate_survivor_score(strategy_id, strategy_tests)
        
        # Calculate stress test statistics
        total_tests = len(strategy_tests)
        passed_tests = sum(1 for test in strategy_tests if test.success)
        
        # Calculate environmental resilience by test type
        env_resilience = {}
        for test_type in set(test.test_type for test in strategy_tests):
            type_tests = [test for test in strategy_tests if test.test_type == test_type]
            if type_tests:
                type_successes = sum(1 for test in type_tests if test.success)
                env_resilience[test_type] = type_successes / len(type_tests)
        
        # Determine badge level based on score
        badge_level = self._determine_badge_level(anti_fragile_score)
        
        # Calculate complexity factor
        complexity_factor = self._calculate_complexity_factor(strategy_id, strategy_tests)
        
        # Create the survivor badge
        badge = SurvivorBadge(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            material=material,
            operation_type=operation_type,
            survivor_score=anti_fragile_score,
            stress_tests_completed=total_tests,
            stress_tests_passed=passed_tests,
            environmental_resilience=env_resilience,
            complexity_factor=complexity_factor,
            badge_level=badge_level,
            awarded_at=datetime.utcnow(),
            awarded_by="AntiFragile_Marketplace_Engine",
            validity_period=365,  # Valid for one year
            tags=self._generate_strategy_tags(anti_fragile_score, env_resilience)
        )
        
        # Store the badge
        self.badges[strategy_id] = badge
        
        return badge
    
    def _determine_badge_level(self, score: float) -> str:
        """
        Determine the badge level based on the anti-fragile score.
        
        Args:
            score: The calculated anti-fragile score (0.0 to 1.0)
            
        Returns:
            String representing the badge level
        """
        if score >= 0.95:
            return "Diamond"  # Nearly perfect resilience
        elif score >= 0.85:
            return "Platinum"  # Excellent resilience
        elif score >= 0.70:
            return "Gold"  # Good resilience
        elif score >= 0.50:
            return "Silver"  # Adequate resilience
        elif score > 0.0:
            return "Bronze"  # Basic resilience
        else:
            return "Unrated"  # No resilience data
    
    def _generate_strategy_tags(self, score: float, env_resilience: Dict[str, float]) -> List[str]:
        """
        Generate descriptive tags for a strategy based on its performance.
        
        Args:
            score: The anti-fragile score
            env_resilience: Environmental resilience by test type
            
        Returns:
            List of descriptive tags
        """
        tags = []
        
        if score >= 0.9:
            tags.append("ultra_resilient")
        elif score >= 0.7:
            tags.append("highly_resilient")
        elif score >= 0.5:
            tags.append("moderately_resilient")
        
        # Add tags based on environmental resilience
        for env_type, resilience in env_resilience.items():
            if resilience >= 0.9:
                tags.append(f"excellent_{env_type}_tolerance")
            elif resilience >= 0.7:
                tags.append(f"good_{env_type}_tolerance")
            elif resilience >= 0.5:
                tags.append(f"fair_{env_type}_tolerance")
        
        # Add general tags
        if len(env_resilience) >= 3:
            tags.append("multi_environment")
        if score > 0.0:
            tags.append("stress_validated")
        
        return tags if tags else ["unrated"]
    
    def record_stress_test(self, strategy_id: str, test_type: str, stress_level: float, 
                          success: bool, performance_metrics: Dict[str, float], 
                          machine_id: str, notes: str = "") -> EnvironmentalStressTest:
        """
        Record the results of a stress test performed on a G-Code strategy.
        
        Args:
            strategy_id: ID of the strategy being tested
            test_type: Type of stress test ('vibration', 'thermal', etc.)
            stress_level: Intensity of the stress (0.0 to 1.0)
            success: Whether the operation completed successfully
            performance_metrics: Quality metrics achieved under stress
            machine_id: Which machine performed the test
            notes: Additional observations
            
        Returns:
            The created EnvironmentalStressTest record
        """
        test_record = EnvironmentalStressTest(
            strategy_id=strategy_id,
            test_type=test_type,
            stress_level=stress_level,
            success=success,
            performance_metrics=performance_metrics,
            timestamp=datetime.utcnow(),
            machine_id=machine_id,
            notes=notes
        )
        
        self.stress_tests.append(test_record)
        
        # Log the test
        print(f"[SURVIVOR_RANKING] Stress test recorded for {strategy_id}: "
              f"{test_type} at {stress_level:.2f} intensity - "
              f"{'SUCCESS' if success else 'FAILURE'}")
        
        return test_record
    
    def get_top_survivors(self, limit: int = 10, material: Optional[str] = None,
                         operation_type: Optional[str] = None) -> List[SurvivorBadge]:
        """
        Get the top-performing strategies based on anti-fragile score.
        
        Args:
            limit: Maximum number of results to return
            material: Optional filter by material
            operation_type: Optional filter by operation type
            
        Returns:
            List of top SurvivorBadge objects sorted by score
        """
        # Get all badges and filter if needed
        badges_to_sort = list(self.badges.values())
        
        if material:
            badges_to_sort = [b for b in badges_to_sort if b.material.lower() == material.lower()]
        
        if operation_type:
            badges_to_sort = [b for b in badges_to_sort if b.operation_type.lower() == operation_type.lower()]
        
        # Sort by survivor score (descending)
        sorted_badges = sorted(badges_to_sort, key=lambda b: b.survivor_score, reverse=True)
        
        return sorted_badges[:limit]
    
    def get_strategy_performance_history(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get the complete performance history for a specific strategy.
        
        Args:
            strategy_id: ID of the strategy to look up
            
        Returns:
            Dictionary with complete performance history
        """
        strategy_tests = [test for test in self.stress_tests if test.strategy_id == strategy_id]
        
        if not strategy_tests:
            return {
                'strategy_id': strategy_id,
                'tests_performed': 0,
                'history': [],
                'current_score': 0.0,
                'badge': None
            }
        
        # Calculate rolling average of scores over time
        test_timeline = sorted(strategy_tests, key=lambda t: t.timestamp)
        
        # Calculate progressive score (how score has changed over time)
        progressive_scores = []
        cumulative_tests = []
        
        for i, test in enumerate(test_timeline):
            cumulative_tests.append(test)
            cumulative_score = self.calculate_survivor_score(strategy_id, cumulative_tests)
            progressive_scores.append({
                'test_index': i,
                'date': test.timestamp,
                'score': cumulative_score,
                'test_type': test.test_type,
                'stress_level': test.stress_level,
                'success': test.success
            })
        
        return {
            'strategy_id': strategy_id,
            'tests_performed': len(strategy_tests),
            'history': progressive_scores,
            'current_score': self.badges.get(strategy_id, 
                                           SurvivorBadge(
                                               strategy_id=strategy_id,
                                               strategy_name="Unknown",
                                               material="Unknown",
                                               operation_type="Unknown",
                                               survivor_score=0.0,
                                               stress_tests_completed=0,
                                               stress_tests_passed=0,
                                               environmental_resilience={},
                                               complexity_factor=1.0,
                                               badge_level="Unrated",
                                               awarded_at=datetime.min,
                                               awarded_by="System",
                                               validity_period=365,
                                               tags=["no_data"]
                                           )).survivor_score,
            'badge': self.badges.get(strategy_id)
        }
    
    def get_marketplace_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics for the Anti-Fragile Marketplace.
        
        Returns:
            Dictionary with marketplace statistics
        """
        total_strategies = len(self.badges)
        total_tests = len(self.stress_tests)
        
        # Count badges by level
        badge_counts = {
            'Diamond': 0,
            'Platinum': 0,
            'Gold': 0,
            'Silver': 0,
            'Bronze': 0,
            'Unrated': 0
        }
        
        for badge in self.badges.values():
            badge_counts[badge.badge_level] += 1
        
        # Calculate average scores by material and operation
        material_scores = {}
        operation_scores = {}
        
        for badge in self.badges.values():
            # Average by material
            if badge.material not in material_scores:
                material_scores[badge.material] = []
            material_scores[badge.material].append(badge.survivor_score)
            
            # Average by operation
            if badge.operation_type not in operation_scores:
                operation_scores[badge.operation_type] = []
            operation_scores[badge.operation_type].append(badge.survivor_score)
        
        # Calculate averages
        avg_material_scores = {mat: sum(scores)/len(scores) for mat, scores in material_scores.items()}
        avg_operation_scores = {op: sum(scores)/len(scores) for op, scores in operation_scores.items()}
        
        return {
            'total_strategies_ranked': total_strategies,
            'total_stress_tests_performed': total_tests,
            'badge_distribution': badge_counts,
            'average_survivor_score': sum(b.survivor_score for b in self.badges.values()) / total_strategies if total_strategies > 0 else 0.0,
            'top_materials_by_resilience': sorted(avg_material_scores.items(), key=lambda x: x[1], reverse=True)[:5],
            'top_operations_by_resilience': sorted(avg_operation_scores.items(), key=lambda x: x[1], reverse=True)[:5],
            'last_updated': datetime.utcnow().isoformat()
        }


# Example usage and testing
if __name__ == "__main__":
    print("Survivor Ranking System initialized successfully.")
    print("Ready to calculate Anti-Fragile Scores for G-Code strategies.")
    
    # Example usage would be:
    # ranking_system = SurvivorRankingSystem()
    # 
    # # Record some stress tests
    # ranking_system.record_stress_test(
    #     strategy_id="STRAT_INCONEL_FACE_MILL_001",
    #     test_type="vibration",
    #     stress_level=0.8,
    #     success=True,
    #     performance_metrics={"quality": 0.95, "accuracy": 0.002},
    #     machine_id="M001"
    # )
    # 
    # # Evaluate and award badge
    # badge = ranking_system.evaluate_strategy_resilience(
    #     strategy_id="STRAT_INCONEL_FACE_MILL_001",
    #     strategy_name="Inconel Face Mill Aggressive",
    #     material="Inconel-718",
    #     operation_type="face_mill"
    # )
    # 
    # print(f"Survivor Score: {badge.survivor_score:.3f}")
    # print(f"Badge Level: {badge.badge_level}")
    # print(f"Tags: {badge.tags}")