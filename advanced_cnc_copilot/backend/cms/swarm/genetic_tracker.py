"""
Anti-Fragile Marketplace - Genetic Tracker
Tracks the evolution of G-Code files as they mutate across the fleet, creating a "Genealogy of Code"
that shows how toolpaths evolve and improve through the collective intelligence of the swarm.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib
import uuid
import json
from enum import Enum


class MutationType(Enum):
    """Types of mutations that can occur to G-Code strategies"""
    PARAMETER_OPTIMIZATION = "parameter_optimization"
    GEOMETRIC_REFINEMENT = "geometric_refinement"
    FEED_RATE_ADJUSTMENT = "feed_rate_adjustment"
    RPM_ADJUSTMENT = "rpm_adjustment"
    PATH_MODIFICATION = "path_modification"
    MATERIAL_SPECIFIC_ADAPTATION = "material_specific_adaptation"
    ERROR_CORRECTION = "error_correction"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    SAFETY_ENHANCEMENT = "safety_enhancement"


@dataclass
class GCodeMutation:
    """Represents a single mutation to a G-Code strategy"""
    mutation_id: str
    parent_strategy_id: str
    mutated_strategy_id: str
    mutation_type: MutationType
    mutation_description: str
    parameters_changed: Dict[str, Any]  # Parameters that were modified
    improvement_metric: float  # Improvement score (positive) or degradation (negative)
    timestamp: datetime
    machine_id: str
    operator_id: Optional[str]  # Who made the change
    notes: str = ""
    fitness_before: float = 0.0
    fitness_after: float = 0.0


@dataclass
class GeneticLineage:
    """Represents the complete genetic history of a G-Code strategy"""
    lineage_root_id: str
    current_strategy_id: str
    generation_count: int
    mutation_history: List[GCodeMutation]
    survival_score: float  # Overall fitness of this genetic line
    last_improvement: datetime
    total_improvements: int
    total_mutations: int
    branching_factor: int  # How many variants spawned from this lineage
    material_lineage: str  # Material this lineage is optimized for
    operation_type: str  # Operation this lineage specializes in
    genetic_diversity: float  # Measure of how much this has changed from original


class GeneticTracker:
    """
    Genetic Tracker - Version control for G-Code mutations across the swarm
    
    Tracks how G-Code strategies evolve through:
    - Parameter optimizations
    - Geometric refinements
    - Material-specific adaptations
    - Error corrections
    - Performance improvements
    """
    
    def __init__(self):
        self.mutations = {}  # mutation_id -> GCodeMutation
        self.lineages = {}  # strategy_id -> GeneticLineage
        self.genealogy_graph = {}  # parent_id -> [child_ids]
        self.mutation_history = []  # Chronological record of all mutations
        self.genetic_diversity_cache = {}  # Cached diversity calculations
    
    def register_initial_strategy(self, strategy_id: str, material: str, 
                                operation_type: str, initial_parameters: Dict[str, Any]) -> GeneticLineage:
        """
        Register the initial version of a G-Code strategy in the genetic tracking system.
        
        Args:
            strategy_id: Unique identifier for the initial strategy
            material: Material the strategy is designed for
            operation_type: Type of operation (face_mill, drill, etc.)
            initial_parameters: Initial parameter set for the strategy
            
        Returns:
            GeneticLineage representing the root of this strategy's evolution tree
        """
        lineage = GeneticLineage(
            lineage_root_id=strategy_id,
            current_strategy_id=strategy_id,
            generation_count=0,
            mutation_history=[],
            survival_score=0.0,
            last_improvement=datetime.utcnow(),
            total_improvements=0,
            total_mutations=0,
            branching_factor=0,
            material_lineage=material,
            operation_type=operation_type,
            genetic_diversity=0.0
        )
        
        self.lineages[strategy_id] = lineage
        
        print(f"[GENETIC_TRACKER] Registered initial strategy: {strategy_id} "
              f"for {material} {operation_type}")
        
        return lineage
    
    def record_mutation(self, parent_strategy_id: str, mutation_type: MutationType,
                       mutation_description: str, parameters_changed: Dict[str, Any],
                       improvement_metric: float, machine_id: str,
                       fitness_before: float = 0.0, fitness_after: float = 0.0,
                       operator_id: Optional[str] = None, notes: str = "") -> GCodeMutation:
        """
        Record a mutation event where a G-Code strategy is modified.
        
        Args:
            parent_strategy_id: ID of the original strategy
            mutation_type: Type of mutation applied
            mutation_description: Description of the change
            parameters_changed: Dictionary of parameters that were modified
            improvement_metric: How much this mutation improved (positive) or degraded (negative) performance
            machine_id: Which machine made the modification
            fitness_before: Fitness score before mutation
            fitness_after: Fitness score after mutation
            operator_id: Optional ID of the operator who made the change
            notes: Additional notes about the mutation
            
        Returns:
            GCodeMutation record of the mutation event
        """
        # Generate a new strategy ID for the mutated version
        mutation_id = str(uuid.uuid4())
        mutated_strategy_id = f"{parent_strategy_id}_M{mutation_id[:8]}"
        
        # Create the mutation record
        mutation = GCodeMutation(
            mutation_id=mutation_id,
            parent_strategy_id=parent_strategy_id,
            mutated_strategy_id=mutated_strategy_id,
            mutation_type=mutation_type,
            mutation_description=mutation_description,
            parameters_changed=parameters_changed,
            improvement_metric=improvement_metric,
            timestamp=datetime.utcnow(),
            machine_id=machine_id,
            operator_id=operator_id,
            notes=notes,
            fitness_before=fitness_before,
            fitness_after=fitness_after
        )
        
        # Store the mutation
        self.mutations[mutation_id] = mutation
        self.mutation_history.append(mutation)
        
        # Update the parent's lineage to include this mutation
        if parent_strategy_id in self.lineages:
            parent_lineage = self.lineages[parent_strategy_id]
            parent_lineage.mutation_history.append(mutation)
            parent_lineage.total_mutations += 1
            parent_lineage.generation_count += 1
            
            if improvement_metric > 0:
                parent_lineage.total_improvements += 1
                parent_lineage.last_improvement = datetime.utcnow()
            
            # Update survival score (average improvement across lineage)
            total_improvement = sum(m.improvement_metric for m in parent_lineage.mutation_history)
            parent_lineage.survival_score = total_improvement / len(parent_lineage.mutation_history) if parent_lineage.mutation_history else 0.0
            
            # Update genetic diversity
            parent_lineage.genetic_diversity = self._calculate_genetic_diversity(parent_lineage)
        
        # Create a new lineage for the mutated strategy
        parent_lineage_data = self.lineages.get(parent_strategy_id)
        new_lineage = GeneticLineage(
            lineage_root_id=parent_lineage_data.lineage_root_id if parent_lineage_data else parent_strategy_id,
            current_strategy_id=mutated_strategy_id,
            generation_count=0,  # This is now the root of a new branch
            mutation_history=[],
            survival_score=fitness_after,
            last_improvement=datetime.utcnow(),
            total_improvements=1 if improvement_metric > 0 else 0,
            total_mutations=1,
            branching_factor=0,
            material_lineage=parent_lineage_data.material_lineage if parent_lineage_data else "",
            operation_type=parent_lineage_data.operation_type if parent_lineage_data else "",
            genetic_diversity=0.0
        )
        
        self.lineages[mutated_strategy_id] = new_lineage
        
        # Update the genealogy graph
        if parent_strategy_id not in self.genealogy_graph:
            self.genealogy_graph[parent_strategy_id] = []
        self.genealogy_graph[parent_strategy_id].append(mutated_strategy_id)
        
        # Update branching factor of parent
        if parent_strategy_id in self.lineages:
            self.lineages[parent_strategy_id].branching_factor = len(self.genealogy_graph[parent_strategy_id])
        
        print(f"[GENETIC_TRACKER] Mutation recorded: {mutation_type.value} on {parent_strategy_id} "
              f"-> {mutated_strategy_id} (Improvement: {improvement_metric:+.3f})")
        
        return mutation
    
    def _calculate_genetic_diversity(self, lineage: GeneticLineage) -> float:
        """Calculate the genetic diversity of a lineage compared to its root."""
        if not lineage.mutation_history:
            return 0.0
        
        # Simple diversity calculation based on number of mutations and their impact
        total_mutations = len(lineage.mutation_history)
        total_improvement = sum(abs(m.improvement_metric) for m in lineage.mutation_history)
        
        # Diversity is proportional to number of mutations and magnitude of changes
        diversity = (total_mutations * 0.1) + (total_improvement * 0.05)
        
        # Cap at 1.0 for maximum diversity
        return min(1.0, diversity)
    
    def get_genealogy_tree(self, root_strategy_id: str) -> Dict[str, Any]:
        """
        Get the complete genealogy tree starting from a root strategy.
        
        Args:
            root_strategy_id: The root strategy to build the tree from
            
        Returns:
            Dictionary representing the genealogy tree
        """
        def build_subtree(node_id: str) -> Dict[str, Any]:
            children = self.genealogy_graph.get(node_id, [])
            lineage = self.lineages.get(node_id)
            
            node_data = {
                'strategy_id': node_id,
                'lineage_data': lineage.__dict__ if lineage else None,
                'mutations_applied': [
                    {
                        'mutation_id': m.mutation_id,
                        'type': m.mutation_type.value,
                        'description': m.mutation_description,
                        'improvement': m.improvement_metric,
                        'timestamp': m.timestamp.isoformat(),
                        'machine_id': m.machine_id
                    } for m in lineage.mutation_history
                ] if lineage else [],
                'children': [build_subtree(child_id) for child_id in children]
            }
            
            return node_data
        
        return build_subtree(root_strategy_id)
    
    def get_evolution_path(self, strategy_id: str) -> List[GCodeMutation]:
        """
        Get the complete evolution path from the root to the given strategy.
        
        Args:
            strategy_id: The strategy to trace the evolution path for
            
        Returns:
            List of mutations in chronological order from root to the strategy
        """
        path = []
        current_id = strategy_id
        
        # Walk backwards from the strategy to the root
        while current_id:
            # Find the mutation that created this strategy
            found_parent = False
            for mutation in self.mutation_history:
                if mutation.mutated_strategy_id == current_id:
                    path.insert(0, mutation)  # Insert at beginning to maintain order
                    current_id = mutation.parent_strategy_id
                    found_parent = True
                    break
            
            if not found_parent:
                break
        
        return path
    
    def get_genetic_similarity(self, strategy1_id: str, strategy2_id: str) -> float:
        """
        Calculate genetic similarity between two strategies based on their shared ancestry
        and the number of divergent mutations.
        
        Args:
            strategy1_id: First strategy to compare
            strategy2_id: Second strategy to compare
            
        Returns:
            Similarity score between 0.0 (completely different) and 1.0 (identical)
        """
        path1 = self.get_evolution_path(strategy1_id)
        path2 = self.get_evolution_path(strategy2_id)
        
        if not path1 or not path2:
            return 0.0 if path1 or path2 else 1.0  # If one doesn't exist, they're different; if neither exist, they're same
        
        # Find common ancestor by comparing paths
        min_len = min(len(path1), len(path2))
        common_prefix_length = 0
        
        for i in range(min_len):
            if path1[i].parent_strategy_id == path2[i].parent_strategy_id:
                common_prefix_length += 1
            else:
                break
        
        # Calculate similarity based on shared history vs unique mutations
        total_unique_mutations = (len(path1) - common_prefix_length) + (len(path2) - common_prefix_length)
        total_possible_mutations = len(path1) + len(path2)
        
        if total_possible_mutations == 0:
            return 1.0
        
        similarity = 1.0 - (total_unique_mutations / total_possible_mutations)
        return max(0.0, similarity)
    
    def get_most_successful_lineage(self, material: Optional[str] = None, 
                                  operation_type: Optional[str] = None) -> Optional[GeneticLineage]:
        """
        Get the most successful genetic lineage based on survival score.
        
        Args:
            material: Optional filter by material
            operation_type: Optional filter by operation type
            
        Returns:
            The most successful GeneticLineage or None if no lineages match criteria
        """
        candidate_lineages = []
        
        for lineage in self.lineages.values():
            if (not material or lineage.material_lineage == material) and \
               (not operation_type or lineage.operation_type == operation_type):
                candidate_lineages.append(lineage)
        
        if not candidate_lineages:
            return None
        
        # Return the lineage with the highest survival score
        return max(candidate_lineages, key=lambda l: l.survival_score)
    
    def get_lineage_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about all genetic lineages in the system.
        
        Returns:
            Dictionary with lineage statistics
        """
        total_lineages = len(self.lineages)
        total_mutations = len(self.mutation_history)
        
        # Calculate stats by material
        material_stats = {}
        operation_stats = {}
        for lineage in self.lineages.values():
            # Material stats
            material = lineage.material_lineage
            if material not in material_stats:
                material_stats[material] = {
                    'count': 0,
                    'total_improvements': 0,
                    'total_mutations': 0,
                    'avg_survival_score': 0.0,
                    'avg_diversity': 0.0
                }
            mat_stat = material_stats[material]
            mat_stat['count'] += 1
            mat_stat['total_improvements'] += lineage.total_improvements
            mat_stat['total_mutations'] += lineage.total_mutations
            mat_stat['avg_survival_score'] += lineage.survival_score
            mat_stat['avg_diversity'] += lineage.genetic_diversity
            
            # Operation stats
            operation = lineage.operation_type
            if operation not in operation_stats:
                operation_stats[operation] = {
                    'count': 0,
                    'total_improvements': 0,
                    'total_mutations': 0,
                    'avg_survival_score': 0.0
                }
            op_stat = operation_stats[operation]
            op_stat['count'] += 1
            op_stat['total_improvements'] += lineage.total_improvements
            op_stat['total_mutations'] += lineage.total_mutations
            op_stat['avg_survival_score'] += lineage.survival_score
        
        # Calculate averages
        for mat, stats in material_stats.items():
            if stats['count'] > 0:
                stats['avg_survival_score'] /= stats['count']
                stats['avg_diversity'] /= stats['count']
        
        for op, stats in operation_stats.items():
            if stats['count'] > 0:
                stats['avg_survival_score'] /= stats['count']
        
        # Find most successful lineage
        most_successful = None
        if self.lineages:
            most_successful = max(self.lineages.values(), key=lambda l: l.survival_score)
        
        return {
            'total_lineages': total_lineages,
            'total_mutations_recorded': total_mutations,
            'total_improvements': sum(l.total_improvements for l in self.lineages.values()),
            'average_generation_count': sum(l.generation_count for l in self.lineages.values()) / total_lineages if total_lineages > 0 else 0,
            'material_statistics': material_stats,
            'operation_statistics': operation_stats,
            'most_successful_lineage': {
                'strategy_id': most_successful.current_strategy_id,
                'material': most_successful.material_lineage,
                'operation': most_successful.operation_type,
                'survival_score': most_successful.survival_score,
                'generations': most_successful.generation_count,
                'improvements': most_successful.total_improvements
            } if most_successful else None,
            'last_updated': datetime.utcnow().isoformat()
        }


# Example usage and testing
if __name__ == "__main__":
    print("Genetic Tracker initialized successfully.")
    print("Ready to track G-Code evolution across the fleet.")
    
    # Example usage would be:
    # tracker = GeneticTracker()
    # 
    # # Register an initial strategy
    # initial_lineage = tracker.register_initial_strategy(
    #     strategy_id="STRAT_ALUMINUM_FACE_MILL_001",
    #     material="Aluminum-6061",
    #     operation_type="face_mill",
    #     initial_parameters={"feed_rate": 2000, "rpm": 4000, "depth": 1.0}
    # )
    # 
    # # Record a mutation
    # mutation = tracker.record_mutation(
    #     parent_strategy_id="STRAT_ALUMINUM_FACE_MILL_001",
    #     mutation_type=MutationType.PARAMETER_OPTIMIZATION,
    #     mutation_description="Increased feed rate for better efficiency",
    #     parameters_changed={"feed_rate": 2200},
    #     improvement_metric=0.15,
    #     machine_id="M001",
    #     fitness_before=0.7,
    #     fitness_after=0.85
    # )
    # 
    # # Get evolution path
    # path = tracker.get_evolution_path(mutation.mutated_strategy_id)
    # print(f"Evolution path has {len(path)} mutations")
    # 
    # # Get genealogy tree
    # tree = tracker.get_genealogy_tree("STRAT_ALUMINUM_FACE_MILL_001")
    # print(f"Genealogy tree generated with {len(tree['children'])} direct descendants")
    # 
    # # Get statistics
    # stats = tracker.get_lineage_statistics()
    # print(f"Lineage statistics: {stats['total_lineages']} lineages tracked")