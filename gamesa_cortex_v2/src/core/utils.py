import time

class PreciseTimer:
    """
    Gamesa Cortex V2: Monotonic High-Resolution Timer.
    Wraps CLOCK_MONOTONIC_RAW (via time.perf_counter_ns).
    """
    def __init__(self):
        self.start_ns = time.perf_counter_ns()
        
    def elapsed_ms(self) -> float:
        return (time.perf_counter_ns() - self.start_ns) / 1_000_000.0
