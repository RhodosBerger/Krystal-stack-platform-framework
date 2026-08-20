import random
import uuid
import time
import math
from typing import Dict, List, Callable, Any
from dataclasses import dataclass

# --- The Evolution Engine ---

@dataclass
class LogicGene:
    """A single unit of 'Condition-Action' logic that can evolve."""
    id: str
    condition_formula: str  # e.g. "cpu > 0.8"
    action_type: str        # e.g. "THROTTLE", "BOOST", "MIGRATE"
    weight: float           # "Dopamine" ranking (Success rate)
    generation: int

class ConditionalLogicGenerator:
    """
    Simulates the 'Evolutionary' aspect. 
    It generates random logic, tests it against 'Virtual' scenarios,
    and keeps the best ones (Survival of the Fittest).
    """
    def __init__(self):
        self.gene_pool: List[LogicGene] = []
        self.generation_count = 0
        self.dopamine_history = []

    def seed_population(self, size=10):
        """Creates the initial random 'Primordial Soup' of logic."""
        actions = ["BOOST_THREAD", "CLEAR_CACHE", "MIGRATE_TO_VULKAN", "SLEEP"]
        conditions = ["cpu > 0.9", "ram_usage > 0.8", "temp > 70", "fps < 30"]
        
        for _ in range(size):
            gene = LogicGene(
                id=f"GENE-{uuid.uuid4().hex[:4]}",
                condition_formula=random.choice(conditions),
                action_type=random.choice(actions),
                weight=1.0, # Neutral start
                generation=0
            )
            self.gene_pool.append(gene)

    def evaluate_fitness(self, gene: LogicGene, scenario: Dict[str, float]) -> float:
        """
        Runs the logic against a scenario. 
        Returns a 'Dopamine' score (Fitness).
        """
        # Parse condition (Safety wrapper in real life needed)
        # Here we mock the evaluation context
        ctx = scenario
        try:
            # Simple eval for demo: "cpu > 0.9" -> 0.95 > 0.9 -> True
            condition_met = eval(gene.condition_formula, {}, ctx)
        except:
            condition_met = False

        if condition_met:
            # Did the action help? (Mock simulation)
            if gene.action_type == "BOOST_THREAD" and ctx['fps'] < 60:
                return 1.5 # Good! Dopamine Hit.
            elif gene.action_type == "CLEAR_CACHE" and ctx['ram_usage'] > 0.9:
                return 2.0 # Great!
            elif gene.action_type == "SLEEP" and ctx['cpu'] > 0.9:
                return -1.0 # Bad! Don't sleep under load.
            else:
                return 0.1 # Neutral
        else:
            return 0.0 # Not applicable

    def evolve(self, scenarios: List[Dict[str, float]]):
        """
        The Core Cycle:
        1. Evaluate all genes.
        2. Kill the weak (Low Weight).
        3. Mutate the strong (Create new Generation).
        """
        self.generation_count += 1
        print(f"--- Evolution Cycle {self.generation_count} ---")
        
        # 1. Evaluation
        for gene in self.gene_pool:
            score = 0
            for scen in scenarios:
                score += self.evaluate_fitness(gene, scen)
            
            # Apply Dopamine Learning Rate
            gene.weight = (gene.weight * 0.8) + (score * 0.2)
        
        # 2. Selection (Keep top 50%)
        self.gene_pool.sort(key=lambda x: x.weight, reverse=True)
        survivors = self.gene_pool[:len(self.gene_pool)//2]
        
        # 3. Mutation / Reproduction
        new_genes = []
        for parent in survivors:
            # Create a child
            child = LogicGene(
                id=f"GENE-{uuid.uuid4().hex[:4]}",
                condition_formula=parent.condition_formula, # Inherit condition
                action_type=parent.action_type,             # Inherit action
                weight=parent.weight,
                generation=self.generation_count
            )
            
            # Mutation: 20% chance to change condition or action
            if random.random() < 0.2:
                conditions = ["cpu > 0.5", "ram_usage > 0.5", "temp > 50", "fps < 60"]
                child.condition_formula = random.choice(conditions)
                child.id += "-MUT"
                
            new_genes.append(child)
        
        self.gene_pool = survivors + new_genes
        print(f"  > Gene Pool Size: {len(self.gene_pool)}")
        print(f"  > Best Logic: {self.gene_pool[0].condition_formula} -> {self.gene_pool[0].action_type} (Weight: {self.gene_pool[0].weight:.2f})")

# --- Testing the Evolution ---
if __name__ == "__main__":
    gen_sys = ConditionalLogicGenerator()
    gen_sys.seed_population(20)
    
    # Simulate a "Blender Rendering" Scenario
    scenarios = [
        {"cpu": 0.95, "ram_usage": 0.4, "temp": 75, "fps": 10},
        {"cpu": 0.20, "ram_usage": 0.95, "temp": 40, "fps": 60},
        {"cpu": 0.50, "ram_usage": 0.5, "temp": 50, "fps": 30},
    ]
    
    for i in range(5):
        gen_sys.evolve(scenarios)
        time.sleep(0.1)
