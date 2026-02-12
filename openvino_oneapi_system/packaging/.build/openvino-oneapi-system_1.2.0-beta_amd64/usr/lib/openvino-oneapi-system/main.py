import argparse

from ovo.config import RuntimeConfig
from ovo.orchestrator import OVOOrchestrator


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="OpenVINO + oneAPI performance system")
    parser.add_argument("--cycles", type=int, default=10, help="Number of main cycles")
    parser.add_argument("--interval", type=float, default=0.5, help="Cycle interval in seconds")
    parser.add_argument("--population", type=int, default=8, help="Evolution population size")
    parser.add_argument("--mutation", type=float, default=0.2, help="Evolution mutation rate")
    parser.add_argument("--grid-x", type=int, default=6, help="3D grid size X")
    parser.add_argument("--grid-y", type=int, default=6, help="3D grid size Y")
    parser.add_argument("--grid-z", type=int, default=4, help="3D grid size Z")
    args = parser.parse_args()

    return RuntimeConfig(
        cycles=args.cycles,
        interval_seconds=args.interval,
        population_size=args.population,
        mutation_rate=args.mutation,
        grid_x=args.grid_x,
        grid_y=args.grid_y,
        grid_z=args.grid_z,
    )


def main() -> None:
    config = parse_args()
    orchestrator = OVOOrchestrator(config)
    orchestrator.run()


if __name__ == "__main__":
    main()

