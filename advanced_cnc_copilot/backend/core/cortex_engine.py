import logging
from typing import Dict, Any, Generator

# Sub-Engines
try:
    from backend.cms.services.dopamine_engine import DopamineEngine
except ImportError:
    class DopamineEngine: # Mock
        pass

from backend.core.parallel_streamer import ParallelStreamer
from backend.core.evolutionary_optimizer import EvolutionaryOptimizer
from backend.core.hex_logger import HexTraceLogger

class CortexEngine:
    """
    The 'Unified Engine' that orchestrates the entire 'Open Mechanic' architecture.
    Integrates:
    - Parallel Execution (Streamer)
    - Discovery (Evolutionary Optimizer)
    - Tracing (Hex Logger)
    - Biological Logic (Dopamine Engine)
    """
    def __init__(self):
        self.logger = logging.getLogger("CortexEngine")
        
        # 1. Initialize Core Engines
        self.dopamine = DopamineEngine()
        self.streamer = ParallelStreamer(self.dopamine)
        self.optimizer = EvolutionaryOptimizer()
        self.tracer = HexTraceLogger()
        
        self.logger.info("Cortex Engine Initialized. Architecture Extended.")

    def execute_job(self, profile: Dict[str, Any], optimize: bool = True) -> Generator[Dict, None, None]:
        """
        Executes a manufacturing job.
        """
        self.logger.info(f"Received Job. Optimize={optimize}")
        
        # 2. Optimization Phase (Discovery)
        final_profile = profile
        if optimize:
            self.logger.info("Running Evolutionary Optimizer...")
            final_profile = self.optimizer.evolve_profile(profile)
            self.logger.info("Optimization Complete. Best Mutant Selected.")

        # 3. Execution Phase (Parallel Streamer)
        for event in self.streamer.execute_profile(final_profile):
            # Pass through events
            yield event

    def run_nightly_discovery(self, profiles: list[Dict]):
        """
        Runs the optimizer on a batch of profiles to find new efficiencies overnight.
        The 'Best Functionalities' you don't know about yet.
        """
        results = []
        for p in profiles:
            best = self.optimizer.evolve_profile(p)
            results.append(best)
        return results
