import random
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Candidate:
    threads: int
    streams: int
    score: float = 0.0


class EvolutionaryTuner:
    def __init__(self, population_size: int, mutation_rate: float) -> None:
        self.population_size = max(2, population_size)
        self.mutation_rate = max(0.0, min(1.0, mutation_rate))
        self.population: List[Candidate] = [
            Candidate(threads=random.choice([2, 4, 8, 16]), streams=random.choice([1, 2, 3]))
            for _ in range(self.population_size)
        ]

    def step(self, reward_signal: float) -> Dict[str, int]:
        for c in self.population:
            c.score = 0.8 * c.score + 0.2 * (reward_signal - 0.05 * c.threads + 0.03 * c.streams)

        self.population.sort(key=lambda c: c.score, reverse=True)
        elites = self.population[: max(1, self.population_size // 3)]
        new_population: List[Candidate] = elites.copy()

        while len(new_population) < self.population_size:
            p = random.choice(elites)
            child = Candidate(threads=p.threads, streams=p.streams, score=p.score)
            if random.random() < self.mutation_rate:
                child.threads = max(1, min(32, child.threads + random.choice([-2, -1, 1, 2])))
            if random.random() < self.mutation_rate:
                child.streams = max(1, min(8, child.streams + random.choice([-1, 1])))
            new_population.append(child)

        self.population = new_population
        best = self.population[0]
        return {"threads": best.threads, "streams": best.streams}

