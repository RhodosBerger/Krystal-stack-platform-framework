#!/usr/bin/env python3
"""
Genetic Algorithms and Evolutionary Pipelines Framework

This module implements genetic algorithms, evolutionary practices,
and communication pipelines based on the existing architecture
for distributed data transfers and enhanced communication features.
"""

import numpy as np
import random
import time
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import uuid
from datetime import datetime
import logging
from collections import defaultdict, deque
import copy
from functools import partial
import asyncio
import queue
import pickle
from abc import ABC, abstractmethod


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EvolutionaryStage(Enum):
    """Stages in the evolutionary process."""
    INITIALIZATION = "initialization"
    SELECTION = "selection"
    CROSSOVER = "crossover"
    MUTATION = "mutation"
    EVALUATION = "evaluation"
    REPLACEMENT = "replacement"
    COMMUNICATION = "communication"


class CommunicationChannel(Enum):
    """Types of communication channels."""
    LOCAL_QUEUE = "local_queue"
    SHARED_MEMORY = "shared_memory"
    NETWORK_SOCKET = "network_socket"
    MESSAGE_BROKER = "message_broker"
    PIPELINE_STREAM = "pipeline_stream"


@dataclass
class GeneticIndividual:
    """Represents an individual in the genetic algorithm."""
    id: str
    genes: List[float]
    fitness: float = 0.0
    age: int = 0
    generation: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EvolutionaryParameters:
    """Parameters for the evolutionary algorithm."""
    population_size: int = 100
    mutation_rate: float = 0.01
    crossover_rate: float = 0.8
    elite_size: int = 10
    max_generations: int = 100
    convergence_threshold: float = 1e-6
    tournament_size: int = 5
    gene_length: int = 10


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""
    
    def __init__(self, name: str):
        self.name = name
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.is_running = False
        self.stage_id = f"STAGE_{name.upper()}_{uuid.uuid4().hex[:8].upper()}"
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process data in this stage."""
        pass
    
    def run(self):
        """Run the pipeline stage."""
        self.is_running = True
        while self.is_running:
            try:
                data = self.input_queue.get(timeout=0.1)
                result = self.process(data)
                self.output_queue.put(result)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in pipeline stage {self.name}: {e}")
                continue


class CommunicationPipeline:
    """Pipeline for communication between different evolutionary components."""
    
    def __init__(self, name: str):
        self.name = name
        self.stages: List[PipelineStage] = []
        self.communication_channel = CommunicationChannel.PIPELINE_STREAM
        self.pipeline_id = f"PIPE_{name.upper()}_{uuid.uuid4().hex[:8].upper()}"
        self.message_history = deque(maxlen=1000)
        self.lock = threading.RLock()
        
    def add_stage(self, stage: PipelineStage):
        """Add a stage to the communication pipeline."""
        self.stages.append(stage)
        logger.info(f"Added stage {stage.name} to pipeline {self.name}")
    
    def send_message(self, message: Any, destination: str = None):
        """Send a message through the pipeline."""
        with self.lock:
            message_data = {
                'id': str(uuid.uuid4()),
                'timestamp': time.time(),
                'source_pipeline': self.pipeline_id,
                'destination': destination,
                'content': message,
                'stage_count': len(self.stages)
            }
            
            self.message_history.append(message_data)
            
            # Process through stages
            current_data = message_data
            for stage in self.stages:
                current_data = stage.process(current_data)
            
            return current_data
    
    def connect_to_pipeline(self, other_pipeline: 'CommunicationPipeline'):
        """Connect this pipeline to another pipeline."""
        # In a real implementation, this would establish actual connections
        logger.info(f"Connected pipeline {self.name} to {other_pipeline.name}")


class EvolutionaryAlgorithm:
    """Base class for evolutionary algorithms."""
    
    def __init__(self, parameters: EvolutionaryParameters):
        self.parameters = parameters
        self.population: List[GeneticIndividual] = []
        self.generation = 0
        self.best_individual = None
        self.fitness_history = deque(maxlen=1000)
        self.evolutionary_log = deque(maxlen=1000)
        self.lock = threading.RLock()
        # Initialize population
        self.population = self.initialize_population()
        
    def initialize_population(self) -> List[GeneticIndividual]:
        """Initialize the population with random individuals."""
        population = []
        for i in range(self.parameters.population_size):
            genes = [random.random() for _ in range(self.parameters.gene_length)]
            individual = GeneticIndividual(
                id=f"IND_{self.generation:04d}_{i:04d}_{uuid.uuid4().hex[:6].upper()}",
                genes=genes,
                generation=self.generation,
                metadata={'created_at': time.time()}
            )
            population.append(individual)
        return population
    
    def evaluate_fitness(self, individual: GeneticIndividual) -> float:
        """Evaluate the fitness of an individual."""
        # Default fitness function - should be overridden
        # For demonstration, using a simple sphere function
        fitness = sum(gene ** 2 for gene in individual.genes)
        return 1.0 / (1.0 + fitness)  # Higher values are better
    
    def selection(self) -> List[GeneticIndividual]:
        """Select individuals for reproduction using tournament selection."""
        selected = []
        for _ in range(self.parameters.population_size - self.parameters.elite_size):
            tournament = random.sample(self.population, self.parameters.tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(copy.deepcopy(winner))
        return selected
    
    def crossover(self, parent1: GeneticIndividual, parent2: GeneticIndividual) -> Tuple[GeneticIndividual, GeneticIndividual]:
        """Perform crossover between two parents."""
        if random.random() > self.parameters.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        crossover_point = random.randint(1, len(parent1.genes) - 1)
        child1_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        child2_genes = parent2.genes[:crossover_point] + parent1.genes[crossover_point:]
        
        child1 = GeneticIndividual(
            id=f"IND_{self.generation:04d}_{uuid.uuid4().hex[:6].upper()}",
            genes=child1_genes,
            generation=self.generation,
            metadata={'created_at': time.time(), 'parent_ids': [parent1.id, parent2.id]}
        )
        
        child2 = GeneticIndividual(
            id=f"IND_{self.generation:04d}_{uuid.uuid4().hex[:6].upper()}",
            genes=child2_genes,
            generation=self.generation,
            metadata={'created_at': time.time(), 'parent_ids': [parent2.id, parent1.id]}
        )
        
        return child1, child2
    
    def mutation(self, individual: GeneticIndividual) -> GeneticIndividual:
        """Mutate an individual."""
        mutated_genes = []
        for gene in individual.genes:
            if random.random() < self.parameters.mutation_rate:
                # Add small random change
                mutated_gene = gene + random.gauss(0, 0.1)
                # Keep within bounds [0, 1]
                mutated_gene = max(0.0, min(1.0, mutated_gene))
                mutated_genes.append(mutated_gene)
            else:
                mutated_genes.append(gene)
        
        mutated_individual = GeneticIndividual(
            id=f"IND_MUT_{individual.id}",
            genes=mutated_genes,
            generation=individual.generation,
            metadata=copy.deepcopy(individual.metadata)
        )
        
        return mutated_individual
    
    def evolve_generation(self):
        """Evolve one generation."""
        with self.lock:
            # Evaluate fitness for all individuals
            for individual in self.population:
                individual.fitness = self.evaluate_fitness(individual)
            
            # Sort population by fitness (descending)
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Log best individual
            current_best = self.population[0]
            self.fitness_history.append({
                'generation': self.generation,
                'best_fitness': current_best.fitness,
                'average_fitness': sum(ind.fitness for ind in self.population) / len(self.population),
                'timestamp': time.time()
            })
            
            # Update best individual if needed
            if (self.best_individual is None or 
                current_best.fitness > self.best_individual.fitness):
                self.best_individual = copy.deepcopy(current_best)
            
            # Log evolutionary stage
            self.evolutionary_log.append({
                'stage': EvolutionaryStage.EVALUATION.value,
                'generation': self.generation,
                'population_size': len(self.population),
                'best_fitness': current_best.fitness,
                'timestamp': time.time()
            })
            
            # Select parents
            selected = self.selection()
            
            # Create next generation through crossover and mutation
            next_generation = []
            
            # Keep elite individuals
            next_generation.extend(self.population[:self.parameters.elite_size])
            
            # Generate offspring
            while len(next_generation) < self.parameters.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                next_generation.extend([child1, child2])
            
            # Trim to exact population size
            self.population = next_generation[:self.parameters.population_size]
            self.generation += 1


class GeneticAlgorithm(EvolutionaryAlgorithm):
    """Implementation of a genetic algorithm."""
    
    def __init__(self, parameters: EvolutionaryParameters, fitness_function: Callable = None):
        super().__init__(parameters)
        self.fitness_function = fitness_function or self.default_fitness_function
        self.algorithm_id = f"GA_{uuid.uuid4().hex[:8].upper()}"
        # Initialize population after setting fitness function
        self.population = self.initialize_population()

    def default_fitness_function(self, individual: GeneticIndividual) -> float:
        """Default fitness function."""
        # Default fitness function - sphere function for demonstration
        fitness = sum(gene ** 2 for gene in individual.genes)
        return 1.0 / (1.0 + fitness)

    def evaluate_fitness(self, individual: GeneticIndividual) -> float:
        """Evaluate fitness using the provided function."""
        return self.fitness_function(individual)


class EvolutionaryPipelineStage(PipelineStage):
    """Pipeline stage for evolutionary operations."""
    
    def __init__(self, name: str, evolutionary_algorithm: EvolutionaryAlgorithm):
        super().__init__(name)
        self.evolutionary_algorithm = evolutionary_algorithm
    
    def process(self, data: Any) -> Any:
        """Process evolutionary data through this stage."""
        if isinstance(data, dict) and 'evolutionary_data' in data:
            # Process evolutionary data
            evolutionary_data = data['evolutionary_data']
            # Apply evolutionary algorithm operations
            result = {
                'processed_by': self.stage_id,
                'original_data': data,
                'evolutionary_result': evolutionary_data,
                'timestamp': time.time()
            }
            return result
        else:
            return data


class DistributedEvolutionarySystem:
    """Distributed system for evolutionary algorithms with communication pipelines."""
    
    def __init__(self):
        self.pipelines: Dict[str, CommunicationPipeline] = {}
        self.evolutionary_algorithms: Dict[str, EvolutionaryAlgorithm] = {}
        self.communication_channels: Dict[str, Any] = {}
        self.system_id = f"DIST_SYS_{uuid.uuid4().hex[:8].upper()}"
        self.coordinator = None
        self.is_running = False
        self.lock = threading.RLock()
        
    def create_pipeline(self, name: str) -> CommunicationPipeline:
        """Create a new communication pipeline."""
        pipeline = CommunicationPipeline(name)
        self.pipelines[name] = pipeline
        logger.info(f"Created pipeline: {name}")
        return pipeline
    
    def create_algorithm(self, name: str, parameters: EvolutionaryParameters, 
                        algorithm_type: str = "genetic") -> EvolutionaryAlgorithm:
        """Create a new evolutionary algorithm."""
        if algorithm_type == "genetic":
            algorithm = GeneticAlgorithm(parameters)
        else:
            algorithm = EvolutionaryAlgorithm(parameters)
        
        algorithm.initialize_population()
        self.evolutionary_algorithms[name] = algorithm
        logger.info(f"Created algorithm: {name} ({algorithm_type})")
        return algorithm
    
    def connect_pipelines(self, pipeline1_name: str, pipeline2_name: str):
        """Connect two pipelines for communication."""
        if pipeline1_name in self.pipelines and pipeline2_name in self.pipelines:
            self.pipelines[pipeline1_name].connect_to_pipeline(self.pipelines[pipeline2_name])
            logger.info(f"Connected pipelines: {pipeline1_name} -> {pipeline2_name}")
    
    def distribute_workload(self, algorithm_name: str, pipeline_name: str, 
                          data: Any) -> Dict[str, Any]:
        """Distribute workload through the communication pipeline."""
        if algorithm_name not in self.evolutionary_algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not found")
        
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not found")
        
        algorithm = self.evolutionary_algorithms[algorithm_name]
        pipeline = self.pipelines[pipeline_name]
        
        # Prepare evolutionary data
        evolutionary_data = {
            'algorithm_id': algorithm.algorithm_id if hasattr(algorithm, 'algorithm_id') else 'N/A',
            'generation': algorithm.generation,
            'population_size': len(algorithm.population),
            'best_fitness': algorithm.best_individual.fitness if algorithm.best_individual else 0.0,
            'data': data
        }
        
        # Send through pipeline
        result = pipeline.send_message({
            'evolutionary_data': evolutionary_data,
            'timestamp': time.time(),
            'source_algorithm': algorithm_name
        })
        
        return result
    
    def synchronize_algorithms(self):
        """Synchronize different algorithms through communication."""
        with self.lock:
            # Share best individuals between algorithms
            best_individuals = {}
            for name, algorithm in self.evolutionary_algorithms.items():
                if algorithm.best_individual:
                    best_individuals[name] = algorithm.best_individual
            
            # Exchange information between algorithms
            for name, algorithm in self.evolutionary_algorithms.items():
                if best_individuals:
                    # Inject best individuals from other algorithms (if different)
                    for other_name, other_best in best_individuals.items():
                        if other_name != name and random.random() < 0.1:  # 10% chance
                            # Add best individual from other algorithm to current population
                            if len(algorithm.population) > 0:
                                # Replace worst individual
                                worst_idx = min(range(len(algorithm.population)), 
                                              key=lambda i: algorithm.population[i].fitness)
                                algorithm.population[worst_idx] = copy.deepcopy(other_best)
                                logger.info(f"Injected best individual from {other_name} into {name}")
    
    def run_evolutionary_cycle(self):
        """Run one cycle of all evolutionary algorithms."""
        with self.lock:
            # Evolve each algorithm
            for name, algorithm in self.evolutionary_algorithms.items():
                algorithm.evolve_generation()
                logger.debug(f"Evolved generation {algorithm.generation} for {name}")
            
            # Synchronize algorithms
            self.synchronize_algorithms()
    
    def start_system(self):
        """Start the distributed evolutionary system."""
        self.is_running = True
        logger.info(f"Started distributed evolutionary system: {self.system_id}")
    
    def stop_system(self):
        """Stop the distributed evolutionary system."""
        self.is_running = False
        logger.info(f"Stopped distributed evolutionary system: {self.system_id}")


class AdvancedEvolutionaryPractices:
    """Implementation of advanced evolutionary practices and techniques."""
    
    def __init__(self):
        self.practices_registry = {}
        self.register_default_practices()
    
    def register_default_practices(self):
        """Register default evolutionary practices."""
        practices = {
            'adaptive_mutation': self.adaptive_mutation_rate,
            'dynamic_crossover': self.dynamic_crossover_rate,
            'elitism_with_diversity': self.elitism_with_diversity_preservation,
            'island_model': self.island_model_evolution,
            'coevolution': self.coevolutionary_approach,
            'speciation': self.speciation_approach
        }
        
        for name, func in practices.items():
            self.practices_registry[name] = func
            logger.info(f"Registered evolutionary practice: {name}")
    
    def adaptive_mutation_rate(self, algorithm: EvolutionaryAlgorithm, 
                              fitness_variance: float) -> float:
        """Adaptively adjust mutation rate based on population fitness variance."""
        base_rate = algorithm.parameters.mutation_rate
        
        # Higher variance = lower mutation (exploitation)
        # Lower variance = higher mutation (exploration)
        if fitness_variance < 0.01:  # Low diversity
            return min(0.1, base_rate * 2)  # Increase mutation
        elif fitness_variance > 0.1:  # High diversity
            return max(0.001, base_rate * 0.5)  # Decrease mutation
        else:
            return base_rate
    
    def dynamic_crossover_rate(self, algorithm: EvolutionaryAlgorithm,
                             generation: int, max_generations: int) -> float:
        """Dynamically adjust crossover rate during evolution."""
        # Higher crossover early in evolution, lower later
        progress = generation / max_generations
        base_rate = algorithm.parameters.crossover_rate
        return base_rate * (1 - progress * 0.3)  # Gradually decrease
    
    def elitism_with_diversity_preservation(self, population: List[GeneticIndividual],
                                          elite_size: int) -> List[GeneticIndividual]:
        """Preserve elite individuals while maintaining diversity."""
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        # Select top elite individuals
        elite = sorted_pop[:elite_size]
        
        # Calculate diversity and ensure it's maintained
        diverse_elite = []
        for individual in elite:
            if not diverse_elite:
                diverse_elite.append(individual)
            else:
                # Check if this individual is diverse enough from existing elite
                is_diverse = True
                for existing in diverse_elite:
                    # Simple diversity check based on gene differences
                    avg_diff = np.mean([abs(a - b) for a, b in zip(individual.genes, existing.genes)])
                    if avg_diff < 0.1:  # Too similar
                        is_diverse = False
                        break
                
                if is_diverse:
                    diverse_elite.append(individual)
        
        # Fill up with random individuals if we don't have enough diverse elite
        if len(diverse_elite) < elite_size:
            remaining = [ind for ind in sorted_pop[elite_size:] 
                        if ind not in diverse_elite]
            diverse_elite.extend(random.sample(remaining, 
                                             elite_size - len(diverse_elite)))
        
        return diverse_elite[:elite_size]
    
    def island_model_evolution(self, islands: List[EvolutionaryAlgorithm],
                              migration_rate: float = 0.1):
        """Implement island model with migration between populations."""
        for i, island in enumerate(islands):
            # Evolve the island
            island.evolve_generation()
            
            # Perform migration between islands
            if random.random() < migration_rate and len(islands) > 1:
                # Select another island for migration
                other_island_idx = random.choice([j for j in range(len(islands)) if j != i])
                other_island = islands[other_island_idx]
                
                # Exchange best individuals
                if (island.best_individual and other_island.best_individual and
                    len(island.population) > 0 and len(other_island.population) > 0):
                    
                    # Replace worst individual in each island with best from other
                    worst_island_idx = min(range(len(island.population)), 
                                         key=lambda x: island.population[x].fitness)
                    worst_other_idx = min(range(len(other_island.population)), 
                                        key=lambda x: other_island.population[x].fitness)
                    
                    island.population[worst_island_idx] = copy.deepcopy(other_island.best_individual)
                    other_island.population[worst_other_idx] = copy.deepcopy(island.best_individual)
                    
                    logger.info(f"Migrated individuals between islands {i} and {other_island_idx}")
    
    def coevolutionary_approach(self, population_a: List[GeneticIndividual],
                               population_b: List[GeneticIndividual],
                               evaluate_interaction: Callable) -> Tuple[List[GeneticIndividual], List[GeneticIndividual]]:
        """Implement coevolutionary approach where two populations evolve together."""
        # Evaluate interactions between populations
        for individual_a in population_a:
            for individual_b in population_b:
                interaction_fitness = evaluate_interaction(individual_a, individual_b)
                # Update fitness based on interaction
                individual_a.fitness = max(individual_a.fitness, interaction_fitness)
                individual_b.fitness = max(individual_b.fitness, interaction_fitness)
        
        return population_a, population_b
    
    def speciation_approach(self, population: List[GeneticIndividual],
                           similarity_threshold: float = 0.2) -> Dict[str, List[GeneticIndividual]]:
        """Group individuals into species based on similarity."""
        species = {}
        species_id = 0
        
        for individual in population:
            assigned = False
            for spec_id, spec_members in species.items():
                # Calculate similarity to species representative
                rep = spec_members[0]  # Use first member as representative
                similarity = 1 - np.mean([abs(a - b) for a, b in zip(individual.genes, rep.genes)])
                
                if similarity > similarity_threshold:
                    species[spec_id].append(individual)
                    assigned = True
                    break
            
            if not assigned:
                species[f"SPEC_{species_id}"] = [individual]
                species_id += 1
        
        return species


def demo_genetic_algorithms_pipelines():
    """Demonstrate the genetic algorithms and communication pipelines."""
    print("=" * 80)
    print("GENETIC ALGORITHMS AND EVOLUTIONARY PIPELINES DEMONSTRATION")
    print("=" * 80)
    
    # Create distributed evolutionary system
    dist_system = DistributedEvolutionarySystem()
    dist_system.start_system()
    print(f"[OK] Created distributed evolutionary system: {dist_system.system_id}")
    
    # Create evolutionary parameters
    params = EvolutionaryParameters(
        population_size=50,
        mutation_rate=0.02,
        crossover_rate=0.8,
        elite_size=5,
        max_generations=20,
        gene_length=10
    )
    
    # Create algorithms
    ga1 = dist_system.create_algorithm("genetic_optimization_1", params, "genetic")
    ga2 = dist_system.create_algorithm("genetic_optimization_2", params, "genetic")
    
    print(f"[OK] Created 2 genetic algorithms")
    
    # Create communication pipelines
    pipeline1 = dist_system.create_pipeline("optimization_pipeline_1")
    pipeline2 = dist_system.create_pipeline("optimization_pipeline_2")
    
    # Add stages to pipelines
    stage1 = EvolutionaryPipelineStage("evolution_stage_1", ga1)
    stage2 = EvolutionaryPipelineStage("evolution_stage_2", ga2)
    
    pipeline1.add_stage(stage1)
    pipeline2.add_stage(stage2)
    
    print(f"[OK] Created 2 communication pipelines with evolutionary stages")
    
    # Connect pipelines
    dist_system.connect_pipelines("optimization_pipeline_1", "optimization_pipeline_2")
    
    # Create advanced evolutionary practices
    practices = AdvancedEvolutionaryPractices()
    print(f"[OK] Created advanced evolutionary practices system")
    
    # Run evolutionary cycles
    print(f"\n--- Evolutionary Cycles Demo ---")
    for generation in range(5):
        print(f"Running generation {generation + 1}/5...")
        
        # Run evolutionary cycle
        dist_system.run_evolutionary_cycle()
        
        # Check best fitness so far
        best_ga1 = ga1.best_individual.fitness if ga1.best_individual else 0
        best_ga2 = ga2.best_individual.fitness if ga2.best_individual else 0
        
        print(f"  GA1 Best Fitness: {best_ga1:.6f}")
        print(f"  GA2 Best Fitness: {best_ga2:.6f}")
        
        time.sleep(0.1)  # Small delay for demonstration
    
    # Demonstrate advanced practices
    print(f"\n--- Advanced Evolutionary Practices Demo ---")
    
    # Adaptive mutation demonstration
    fitness_variance = np.var([ind.fitness for ind in ga1.population if ind.fitness > 0])
    new_mutation_rate = practices.adaptive_mutation_rate(ga1, fitness_variance)
    print(f"  Fitness variance: {fitness_variance:.6f}")
    print(f"  Adaptive mutation rate: {new_mutation_rate:.4f} (was {ga1.parameters.mutation_rate:.4f})")
    
    # Dynamic crossover demonstration
    new_crossover_rate = practices.dynamic_crossover_rate(ga1, ga1.generation, params.max_generations)
    print(f"  Dynamic crossover rate: {new_crossover_rate:.4f} (was {ga1.parameters.crossover_rate:.4f})")
    
    # Speciation demonstration
    species = practices.speciation_approach(ga1.population[:20])  # Use subset
    print(f"  Speciated population into {len(species)} species")
    
    # Demonstrate communication through pipelines
    print(f"\n--- Communication Pipeline Demo ---")
    
    test_data = {
        "optimization_request": "find_best_parameters",
        "target_fitness": 0.95,
        "constraints": {"min_value": 0.0, "max_value": 1.0}
    }
    
    result1 = dist_system.distribute_workload("genetic_optimization_1", "optimization_pipeline_1", test_data)
    result2 = dist_system.distribute_workload("genetic_optimization_2", "optimization_pipeline_2", test_data)
    
    print(f"  Pipeline 1 result: {result1['processed_by'] if 'processed_by' in result1 else 'N/A'}")
    print(f"  Pipeline 2 result: {result2['processed_by'] if 'processed_by' in result2 else 'N/A'}")
    
    # Show final results
    print(f"\n--- Final Results ---")
    final_best_ga1 = ga1.best_individual
    final_best_ga2 = ga2.best_individual
    
    if final_best_ga1:
        print(f"  GA1 Final Best Individual: Fitness={final_best_ga1.fitness:.6f}, Genes={final_best_ga1.genes[:3]}...")
    
    if final_best_ga2:
        print(f"  GA2 Final Best Individual: Fitness={final_best_ga2.fitness:.6f}, Genes={final_best_ga2.genes[:3]}...")
    
    # Show pipeline statistics
    print(f"\n--- Pipeline Statistics ---")
    for name, pipeline in dist_system.pipelines.items():
        print(f"  {name}: {len(pipeline.stages)} stages, {len(pipeline.message_history)} messages")
    
    # Stop the system
    dist_system.stop_system()
    print(f"\n[OK] Distributed system stopped")
    
    print(f"\n" + "=" * 80)
    print("GENETIC ALGORITHMS AND EVOLUTIONARY PIPELINES DEMONSTRATION COMPLETE")
    print("The system demonstrates:")
    print("- Distributed evolutionary algorithms with communication")
    print("- Genetic algorithms with fitness evaluation")
    print("- Communication pipelines for data transfer")
    print("- Advanced evolutionary practices (adaptive mutation, speciation, etc.)")
    print("- Pipeline-based processing and synchronization")
    print("- Real-time evolution and optimization")
    print("=" * 80)


if __name__ == "__main__":
    demo_genetic_algorithms_pipelines()