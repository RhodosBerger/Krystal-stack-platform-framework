import random
import copy
import logging
from typing import Dict, List
from backend.integrations.grid_interface import GridInterface

class EvolutionaryOptimizer:
    """
    The 'Discovery Engine'. 
    Uses Genetic Algorithms to explore the parameter space and find 
    'Functionalities you don't know about' (Novel Optimizations).
    """
    def __init__(self):
        self.grid = GridInterface()
        self.logger = logging.getLogger("EvoOptimizer")
        self.population_size = 10
        self.generations = 5

    def evolve_profile(self, profile: Dict) -> Dict:
        """
        Takes a base profile and evolves it to maximize efficiency/stability.
        Returns the 'Best Mutant'.
        """
        self.logger.info("Starting Evolutionary Discovery...")
        
        # 1. Initialize Population (Mutants)
        population = [self._mutate(profile) for _ in range(self.population_size)]
        
        for gen in range(self.generations):
            # 2. Evaluate Fitness via Simulation
            fitness_scores = []
            for mutant in population:
                score = self._evaluate_fitness(mutant)
                fitness_scores.append((score, mutant))
            
            # 3. Selection (Survival of the Fittest)
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            top_performers = [x[1] for x in fitness_scores[:int(self.population_size / 2)]]
            
            self.logger.info(f"Generation {gen}: Best Score = {fitness_scores[0][0]}")
            
            # 4. Reproduction (Breeding/Mutation)
            new_population = top_performers[:]
            while len(new_population) < self.population_size:
                parent = random.choice(top_performers)
                child = self._mutate(parent)
                new_population.append(child)
            
            population = new_population

        best_mutant = fitness_scores[0][1]
        return best_mutant

    def _mutate(self, profile: Dict) -> Dict:
        """
        Randomly alters profile parameters (Feed Rate, Strategy).
        """
        mutant = copy.deepcopy(profile)
        for segment in mutant.get("segments", []):
            # Mutation: Random Feed Rate Adjustment
            if random.random() < 0.3:
                feed = segment.get("optimized_feed", 1000)
                # Try something wild?
                new_feed = feed * random.uniform(0.8, 1.5) 
                segment["optimized_feed"] = int(new_feed)
                
            # Mutation: Strategy Switch (Novelty)
            if random.random() < 0.1:
                segment["strategy"] = random.choice(["trochoidal", "adaptive", "plunge"])
                
        return mutant

    def _evaluate_fitness(self, profile: Dict) -> float:
        """
        Simulates the profile and assigns a score.
        Score = Speed - Risk - Energy
        """
        total_time = 0
        risk_penalty = 0
        
        for segment in profile.get("segments", []):
            # Simulation
            status = self.grid.simulate_segment(segment)
            if status == "COLLISION":
                return -1000.0 # Instant Death
            elif status == "RISK":
                risk_penalty += 50
                
            # Efficiency
            feed = segment.get("optimized_feed", 1000)
            dist = 10 # Mock distance
            time_taken = dist / feed
            total_time += time_taken
            
        # Fitness Function
        score = (1.0 / total_time) * 1000 - risk_penalty
        return score
