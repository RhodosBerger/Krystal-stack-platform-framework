#!/usr/bin/env python3
# ==============================================================================
# KRYSTAL-STACK: DEMO BENCHMARK RUNNER FOR NATIVE PULSE LOGGER (MODULE E-07)
# ==============================================================================
# Executes micro-benchmark across H1, H2, and H3 storage strategies,
# evaluates ObserverOverheadRatio vs tau_cache_transit (11.97 µs),
# validates constitutional invariant VITAL_MAX_HP == 6,
# and prints the formal E-07 ExperimentReceipt.
# ==============================================================================

import sys
import os

# Add workspace root to import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from krystal_kernel.pulse_reader import (
    PulseSequenceLogger,
    ExperimentE07Benchmark,
    VITAL_MAX_HP,
)


def main():
    print("==============================================================================")
    print("       KRYSTAL-STACK: NATIVE PULSE SEQUENCE LOGGER EXPERIMENT (E-07)          ")
    print("==============================================================================")
    print(f"[*] Constitutional Invariant: VITAL_MAX_HP == {VITAL_MAX_HP}")

    logger = PulseSequenceLogger(capacity=65536)
    print(f"[*] Execution Plane:          {'Native C11/Rust hot-path' if logger.is_native else 'Python Emulation'}")
    print(f"[*] Calibrated Counter Freq:  {logger.tsc_freq_hz / 1e9:.3f} GHz")
    print("[*] Running 100,000 iterations across H1, H2, and H3 storage strategies...")

    benchmark_data = ExperimentE07Benchmark.run_benchmark(logger, iterations=100_000)
    receipt = ExperimentE07Benchmark.generate_receipt(benchmark_data)

    print("\n" + receipt)


if __name__ == "__main__":
    main()
