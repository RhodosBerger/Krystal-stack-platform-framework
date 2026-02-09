from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    cycles: int = 10
    interval_seconds: float = 0.5
    population_size: int = 8
    mutation_rate: float = 0.2
    grid_x: int = 6
    grid_y: int = 6
    grid_z: int = 4

