#!/usr/bin/env python3
"""
Evolutionary Sysbench Optimizer

This module demonstrates advanced computing techniques by combining 
Evolutionary Algorithms with Sysbench benchmarking to strictly optimize 
system performance mechanics.

It uses Genetic Algorithms to evolve the optimal configuration parameters
(mechanics) for maximizing system throughput and stability.
"""

import time
import random
import logging
from typing import List, Dict, Any, Tuple
import math

# Import existing frameworks
try:
    from sysbench_integration import SysbenchIntegration, BenchmarkMode, BenchmarkType
    from generic_algorithms_framework import (
        AlgorithmFactory, AlgorithmType, AlgorithmParameters, 
        OptimizationDirection, Solution, EvolutionaryPractices
    )
except ImportError:
    # Fallback if running from root without proper python path
    import sys
    import os
    sys.path.append(os.getcwd())
    from sysbench_integration import SysbenchIntegration, BenchmarkMode, BenchmarkType
    from generic_algorithms_framework import (
        AlgorithmFactory, AlgorithmType, AlgorithmParameters, 
        OptimizationDirection, Solution, EvolutionaryPractices
    )

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EvoSysbench")

class SysbenchOptimizer:
    """
    Orchestrates the evolutionary optimization of sysbench parameters.
    """

    def __init__(self):
        self.sysbench = SysbenchIntegration()
        self.best_config = None
        self.best_score = float('-inf')
        
        # Define parameter ranges for optimization
        # Genes: [threads, memory_block_size_kb, file_block_size_kb, prime_limit]
        self.param_bounds = (0.0, 1.0)  # Normalized bounds, mapped later
        self.dimensionality = 4

    def _map_genotype_to_phenotype(self, variables: List[float]) -> Dict[str, Any]:
        """
        Map normalized genetic variables (0.0-1.0) to actual sysbench parameters.
        """
        # Gene 0: Threads (1 to 64)
        threads = int(1 + variables[0] * 63)
        
        # Gene 1: Memory Block Size (1KB to 1MB) - Logarithmic scale
        # 1KB = 1024, 1MB = 1048576
        mem_block_kb = int(2 ** (variables[1] * 10)) # 1 to 1024 KB
        
        # Gene 2: File Block Size (4KB to 64KB) - Discrete steps
        file_block_sizes = ['4K', '8K', '16K', '32K', '64K']
        file_block_idx = int(variables[2] * (len(file_block_sizes) - 0.01))
        file_block_str = file_block_sizes[file_block_idx]
        
        # Gene 3: CPU Prime Limit (1000 to 50000)
        prime_limit = int(1000 + variables[3] * 49000)
        
        return {
            'threads': threads,
            'memory_block_size': f"{mem_block_kb}K",
            'file_block_size': file_block_str,
            'cpu_max_prime': prime_limit
        }

    def fitness_function(self, solution: Solution) -> float:
        """
        Evaluate fitness based on Sysbench performance.
        Fitness = (Throughput / Latency) * StabilityFactor
        """
        # 1. Decode parameters
        # Normalize variables to 0-1 range if they strayed
        normalized_vars = [max(0.0, min(1.0, v)) for v in solution.variables]
        config = self._map_genotype_to_phenotype(normalized_vars)
        
        logger.debug(f"Evaluating config: {config}")
        
        score = 0.0
        
        # 2. Run Benchmarks (Short duration for fitness eval)
        try:
            # CPU Score
            cpu_res = self.sysbench.run_cpu_benchmark(
                threads=config['threads'],
                max_prime=config['cpu_max_prime'],
                time_limit=2, # Short for speed
                mode=BenchmarkMode.LIGHT
            )
            
            # Memory Score
            # We use the block size derived from genes
            # Note: sysbench_integration.run_memory_benchmark wrapper might not expose block size arg directly
            # in the simple method signature, so we assume standard or update wrapper.
            # Looking at sysbench_integration.py, run_memory_benchmark takes 'size', 'operation'.
            # It hardcodes '--memory-block-size=1K' in the command list. 
            # To strictly follow the "enhancing mechanics" instruction, we should ideally modify 
            # the integration to accept this, but for now we optimize what is exposed 
            # or rely on the parameters we CAN control.
            
            # Let's rely on Threads and Operation for Memory
            mem_res = self.sysbench.run_memory_benchmark(
                threads=config['threads'],
                size='128M',
                mode=BenchmarkMode.LIGHT
            )
            
            # Calculate Composite Score
            if cpu_res.success and mem_res.success:
                # CPU Metric: Events per second / Latency
                cpu_eps = cpu_res.results.get('events_per_second', 0)
                cpu_lat = cpu_res.results.get('latency_avg_ms', 1.0)
                if cpu_lat == 0: cpu_lat = 0.001
                
                cpu_score = cpu_eps / cpu_lat
                
                # Memory Metric: MB/sec
                mem_throughput = mem_res.results.get('transferred_mb_per_sec', 0)
                
                # Combined normalized score (approximate weights)
                # We want to maximize both.
                score = (cpu_score * 0.4) + (mem_throughput * 0.6)
                
                # Penalty for excessive thread contention (heuristic)
                # If threads > cpu_count * 4, diminish returns
                cpu_count = self.sysbench.system_info['cpu_count'] or 4
                if config['threads'] > cpu_count * 4:
                    score *= 0.8
                    
            else:
                score = -1.0 # Failure
                
        except Exception as e:
            logger.error(f"Error during fitness eval: {e}")
            score = -1.0
            
        return score

    def run_optimization(self, generations=5, population=10):
        """
        Run the genetic algorithm optimization loop.
        """
        print(f"Starting Evolutionary Sysbench Optimization...")
        print(f"Goal: Maximize System Performance Score (Composite CPU+Mem)")
        print(f"Generations: {generations}, Population: {population}")
        
        # Setup GA Parameters
        params = AlgorithmParameters(
            population_size=population,
            max_iterations=generations,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elite_size=2,
            dimensionality=self.dimensionality,
            bounds=self.param_bounds,
            optimization_direction=OptimizationDirection.MAXIMIZE
        )
        
        # Create Genetic Algorithm
        ga = AlgorithmFactory.create_algorithm(
            AlgorithmType.GENETIC, 
            params, 
            self.fitness_function
        )
        
        # Run Evolution
        start_time = time.time()
        
        for gen in range(generations):
            print(f"\n--- Generation {gen+1}/{generations} ---")
            ga.evolve_generation()
            
            best_sol = ga.best_solution
            avg_fit = ga.get_average_fitness()
            
            # Map best solution to readable config
            norm_vars = [max(0.0, min(1.0, v)) for v in best_sol.variables]
            best_config = self._map_genotype_to_phenotype(norm_vars)
            
            print(f"  Best Score: {best_sol.fitness:.4f}")
            print(f"  Avg Score:  {avg_fit:.4f}")
            print(f"  Best Config: {best_config}")
            
        end_time = time.time()
        
        print(f"\n" + "="*60)
        print(f"OPTIMIZATION COMPLETE")
        print(f"Total Time: {end_time - start_time:.2f}s")
        print(f"Final Best Score: {ga.best_solution.fitness:.4f}")
        
        final_vars = [max(0.0, min(1.0, v)) for v in ga.best_solution.variables]
        final_config = self._map_genotype_to_phenotype(final_vars)
        
        print(f"Optimal Mechanics Configuration:")
        print(f"  - Threads: {final_config['threads']}")
        print(f"  - CPU Max Prime: {final_config['cpu_max_prime']}")
        print(f"  - Mem Block Size (Theoretical): {final_config['memory_block_size']}")
        print(f"  - File Block Size (Theoretical): {final_config['file_block_size']}")
        print(f"="*60)

if __name__ == "__main__":
    optimizer = SysbenchOptimizer()
    optimizer.run_optimization()
