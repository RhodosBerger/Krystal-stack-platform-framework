"""
Grid Memory Controller: Imaginary Memory Cache Extension
--------------------------------------------------------
Synthesizes "Numbers as Waves" concept where the 3D Grid acts as a 
high-performance cache for an infinite "Imaginary" memory space.
"""

import math
import time
import random
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List, Any
import logging

# Configure logging for "Precision Telemetry"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [TELEMETRY] - %(message)s')
logger = logging.getLogger("GridController")

@dataclass
class WaveCoordinate:
    """Represents a point in the Imaginary Wave Space."""
    x: int
    y: int
    z: int
    t: float = 0.0  # Time dimension

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.x, self.y, self.z)

@dataclass
class TelemetryLog:
    """Precise mirror of internal state."""
    tick: int
    coordinate: Tuple[int, int, int]
    wave_amplitude: float
    cache_hit: bool
    operation: str
    thread_id: str

class ImaginaryMemory:
    """
    The backing store. Infinite memory space where values are defined 
    by wave functions rather than physical bits until observed.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.base_frequency = 0.1

    def fetch(self, coord: WaveCoordinate) -> float:
        """
        'Imagines' a value based on wave interference at the coordinate.
        Numbers are waves.
        """
        # Constructive/Destructive interference logic
        wave_x = math.sin(coord.x * self.base_frequency + coord.t)
        wave_y = math.cos(coord.y * self.base_frequency + coord.t)
        wave_z = math.sin(coord.z * self.base_frequency + self.seed)
        
        # Complex interference
        value = (wave_x * wave_y) + wave_z
        return value

class GridCache:
    """
    The 3D Grid Cache Extension.
    Solidifies imaginary values into quantized, accessible memory.
    """
    def __init__(self, size: int = 8):
        self.size = size
        # 3D Grid: Map[(x,y,z) -> Value]
        self.cache: Dict[Tuple[int, int, int], float] = {}
        self.access_history: Dict[Tuple[int, int, int], int] = {}
        self.telemetry_stream: List[TelemetryLog] = []
        self.tick_counter = 0

    def is_cached(self, coord: Tuple[int, int, int]) -> bool:
        return coord in self.cache

    def read(self, coord: Tuple[int, int, int]) -> Optional[float]:
        if coord in self.cache:
            self.access_history[coord] = self.tick_counter
            return self.cache[coord]
        return None

    def write(self, coord: Tuple[int, int, int], value: float):
        # Eviction logic (Simple LRU if full)
        if len(self.cache) >= self.size**3:
            self._evict_gravitational()
        
        self.cache[coord] = value
        self.access_history[coord] = self.tick_counter

    def _evict_gravitational(self):
        """
        Evicts items with the lowest 'gravitational pull' (least recently used 
        and furthest from center).
        """
        center = self.size / 2.0
        
        def gravity_score(pos):
            # Distance from center
            dist = math.sqrt((pos[0]-center)**2 + (pos[1]-center)**2 + (pos[2]-center)**2)
            # Recency
            last_access = self.access_history.get(pos, 0)
            age = self.tick_counter - last_access
            
            # High gravity = Close to center AND recently used
            # We want to evict LOW gravity
            return (1.0 / (dist + 0.1)) * (1.0 / (age + 1.0))

        # Find entry with minimum gravity
        if self.cache:
            victim = min(self.cache.keys(), key=gravity_score)
            del self.cache[victim]
            del self.access_history[victim]

    def log_telemetry(self, log: TelemetryLog):
        self.telemetry_stream.append(log)
        # Precision Mirroring: Log immediately
        logger.info(f"T:{log.tick} | OP:{log.operation} | POS:{log.coordinate} | AMP:{log.wave_amplitude:.4f} | HIT:{log.cache_hit}")

class GridMemoryController:
    """
    Controller that manages the flow between Imaginary Memory and the Grid Cache.
    Implements the "Dependency Builder" pattern for fetching.
    """
    def __init__(self):
        self.imaginary = ImaginaryMemory()
        self.cache = GridCache(size=8)
        self.tick = 0

    def access_memory(self, x: int, y: int, z: int, thread_id: str = "MAIN") -> float:
        """
        Standard entry point.
        1. Checks Cache (Grid).
        2. If Miss: Fetches from Imaginary (Wave function).
        3. Updates Telemetry.
        """
        self.tick += 1
        self.cache.tick_counter = self.tick
        coord = (x, y, z)
        wave_coord = WaveCoordinate(x, y, z, t=self.tick * 0.1)
        
        cached_val = self.cache.read(coord)
        
        if cached_val is not None:
            # Hit
            log = TelemetryLog(
                tick=self.tick,
                coordinate=coord,
                wave_amplitude=cached_val,
                cache_hit=True,
                operation="READ_HIT",
                thread_id=thread_id
            )
            self.cache.log_telemetry(log)
            return cached_val
        else:
            # Miss - Synthesize from Imagination
            value = self.imaginary.fetch(wave_coord)
            
            # Quantize for Grid Storage (as requested in prompt "quantized render parameters")
            # Quantizing to 4 decimal places for 'rendering' precision
            quantized_value = round(value, 4)
            
            self.cache.write(coord, quantized_value)
            
            log = TelemetryLog(
                tick=self.tick,
                coordinate=coord,
                wave_amplitude=quantized_value,
                cache_hit=False,
                operation="IMAGINE_FETCH",
                thread_id=thread_id
            )
            self.cache.log_telemetry(log)
            return quantized_value

    def boot_sequence(self):
        """
        Simulates the 'Fresh Boot' gravitational loading.
        Preloads the 'center' of the grid with imaginary potential.
        """
        logger.info("INITIATING GRAVITATIONAL BOOT SEQUENCE...")
        center = 4
        # Spiral out from center
        for r in range(3):
            for x in range(center-r, center+r+1):
                for y in range(center-r, center+r+1):
                    for z in range(center-r, center+r+1):
                        self.access_memory(x, y, z, thread_id="BOOT_LOADER")
        logger.info("BOOT SEQUENCE COMPLETE. GRID POTENTIAL STABILIZED.")

def main():
    """
    Demonstration of the system.
    """
    controller = GridMemoryController()
    
    # 1. Boot
    controller.boot_sequence()
    
    # 2. Random Access Pattern (Simulating multithreaded input)
    logger.info("STARTING RANDOM ACCESS PATTERN...")
    for i in range(10):
        # Generate 'Gravitational' pull towards active sectors
        tx = random.randint(2, 6)
        ty = random.randint(2, 6)
        tz = random.randint(2, 6)
        controller.access_memory(tx, ty, tz, thread_id=f"THREAD_{i%3}")
        time.sleep(0.01)

if __name__ == "__main__":
    main()
