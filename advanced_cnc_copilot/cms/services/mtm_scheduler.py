from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from enum import Enum


class SpindleType(Enum):
    MAIN = "main"
    SUB = "sub"


@dataclass
class Feature:
    id: str
    name: str
    mass: float  # Complexity/effort measure
    dependencies: List[str]  # List of feature IDs this feature depends on
    spindle_preference: Optional[SpindleType] = None  # Preferred spindle, if any


class SpindleLoadBalancer:
    """
    Implements load balancing optimization for Multi-Tasking Machine (MTM) operations.
    Based on research: "Planning optimised multi-tasking operations"
    Minimizes the time difference between main spindle (S1) and sub-spindle (S2).
    """
    
    def __init__(self):
        self.fpm_matrix = {}  # Feature Precedence Matrix for dependencies
        self.time_cache = {}  # Cache for computed operation times
    
    def cluster_features(self, features_list: List[Feature]) -> Tuple[List[Feature], List[Feature], List[Feature]]:
        """
        Cluster features into three sets based on spindle compatibility:
        - S_minus_z: Features for main spindle (Main Spindle)
        - S_plus_z: Features for sub-spindle (Sub-Spindle)  
        - S_float: "Setup-free" features that can go to either
        """
        s_minus_z = []  # Main spindle features
        s_plus_z = []   # Sub spindle features
        s_float = []    # Flexible features
        
        for feature in features_list:
            if feature.spindle_preference == SpindleType.MAIN:
                s_minus_z.append(feature)
            elif feature.spindle_preference == SpindleType.SUB:
                s_plus_z.append(feature)
            else:
                # If no preference, assume it's a flexible feature
                s_float.append(feature)
        
        return s_minus_z, s_plus_z, s_float
    
    def greedy_assignment(self, s_float: List[Feature], s_minus_z: List[Feature], s_plus_z: List[Feature]) -> Tuple[List[Feature], List[Feature]]:
        """
        Greedy assignment of S_float features based on mass (gravitational planning concept)
        Assigns features to spindles to balance the load
        """
        # Sort S_float features by mass in descending order (greedy approach)
        sorted_float = sorted(s_float, key=lambda f: f.mass, reverse=True)
        
        # Calculate initial loads
        main_load = sum(f.mass for f in s_minus_z)
        sub_load = sum(f.mass for f in s_plus_z)
        
        main_spindle = s_minus_z.copy()
        sub_spindle = s_plus_z.copy()
        
        # Assign each floating feature to the spindle with lower current load
        for feature in sorted_float:
            if main_load <= sub_load:
                main_spindle.append(feature)
                main_load += feature.mass
            else:
                sub_spindle.append(feature)
                sub_load += feature.mass
        
        return main_spindle, sub_spindle
    
    def optimize_with_dependencies(self, features_list: List[Feature]) -> Tuple[List[Feature], List[Feature]]:
        """
        Optimizes assignment considering feature dependencies (FPM matrix)
        """
        # Cluster features first
        s_minus_z, s_plus_z, s_float = self.cluster_features(features_list)
        
        # Create dependency graph
        dependency_graph = self._build_dependency_graph(features_list)
        
        # Validate dependencies are respected
        if not self._validate_dependencies(s_minus_z + s_plus_z, dependency_graph):
            raise ValueError("Initial clustering violates dependencies")
        
        # Greedily assign floating features while respecting dependencies
        main_spindle, sub_spindle = self._assign_with_dependencies(
            s_float, s_minus_z, s_plus_z, dependency_graph
        )
        
        return main_spindle, sub_spindle
    
    def _build_dependency_graph(self, features_list: List[Feature]) -> Dict[str, Set[str]]:
        """
        Builds a dependency graph from the features
        """
        graph = {}
        for feature in features_list:
            graph[feature.id] = set(feature.dependencies)
        return graph
    
    def _validate_dependencies(self, assigned_features: List[Feature], dependency_graph: Dict[str, Set[str]]) -> bool:
        """
        Validates that dependencies are properly ordered in the assignment
        """
        feature_order = {f.id: idx for idx, f in enumerate(assigned_features)}
        
        for feature_id, dependencies in dependency_graph.items():
            if feature_id not in feature_order:
                continue  # Not in this spindle's assignment
            
            feature_idx = feature_order[feature_id]
            for dep_id in dependencies:
                if dep_id in feature_order:
                    dep_idx = feature_order[dep_id]
                    if dep_idx > feature_idx:  # Dependency comes after this feature
                        return False
        
        return True
    
    def _assign_with_dependencies(
        self, s_float: List[Feature], s_minus_z: List[Feature], s_plus_z: List[Feature], 
        dependency_graph: Dict[str, Set[str]]
    ) -> Tuple[List[Feature], List[Feature]]:
        """
        Assigns floating features while respecting dependencies
        """
        main_spindle = s_minus_z.copy()
        sub_spindle = s_plus_z.copy()
        
        # Calculate initial loads
        main_load = sum(f.mass for f in s_minus_z)
        sub_load = sum(f.mass for f in s_plus_z)
        
        # Process floating features in dependency order
        remaining_float = s_float.copy()
        assigned = []
        
        while remaining_float:
            # Find features whose dependencies are already satisfied
            ready_features = []
            for feature in remaining_float:
                deps_satisfied = True
                for dep_id in dependency_graph.get(feature.id, set()):
                    # Check if dependency is already assigned to either spindle
                    dep_assigned = any(f.id == dep_id for f in main_spindle + sub_spindle + assigned)
                    if not dep_assigned:
                        deps_satisfied = False
                        break
                
                if deps_satisfied:
                    ready_features.append(feature)
            
            if not ready_features:
                raise ValueError("Circular dependency detected in features")
            
            # Sort ready features by mass (gravitational planning)
            ready_features.sort(key=lambda f: f.mass, reverse=True)
            
            # Assign the largest ready feature to the lighter spindle
            feature_to_assign = ready_features[0]
            if main_load <= sub_load:
                main_spindle.append(feature_to_assign)
                main_load += feature_to_assign.mass
            else:
                sub_spindle.append(feature_to_assign)
                sub_load += feature_to_assign.mass
            
            assigned.append(feature_to_assign)
            remaining_float.remove(feature_to_assign)
        
        return main_spindle, sub_spindle
    
    def calculate_balance_metric(self, main_spindle: List[Feature], sub_spindle: List[Feature]) -> float:
        """
        Calculate the balance metric: f(x) = |Time(S1) - Time(S2)|
        """
        main_time = self._calculate_total_time(main_spindle)
        sub_time = self._calculate_total_time(sub_spindle)
        
        return abs(main_time - sub_time)
    
    def _calculate_total_time(self, features: List[Feature]) -> float:
        """
        Calculate total time for a set of features
        """
        return sum(f.mass for f in features)  # Simplified: mass as proxy for time