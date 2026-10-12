"""
main.py
Entry point for the GAMESA/KrystalStack framework.
"""

import argparse

from src.python.organism import IndustrialOrganism


def main():
    parser = argparse.ArgumentParser(description="GAMESA/KrystalStack - Industrial Organism Framework")
    parser.add_argument(
        "--mode",
        type=str,
        default="production",
        help="Mode: production, simulation, or demo",
    )

    args = parser.parse_args()

    print("Initializing KrystalStack Foundation Kernel...")
    print("Copyright (c) 2026 The KrystalStack Foundation")
    print(f"Mode: {args.mode.upper()}")

    # Instantiate the Organism
    organism = IndustrialOrganism(name="Krystal-Prime")

    # Begin Life Cycle
    organism.awaken()


if __name__ == "__main__":
    main()
