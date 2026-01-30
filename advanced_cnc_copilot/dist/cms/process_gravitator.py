"""
Process Gravitator
KrystalStack Core Module

PARADIGM: Process Scheduling via Gravitational Physics
Processes have Mass (priority) and are pulled to CPU cores

METAPHOR: Heavy objects (critical processes) have stronger
gravitational pull and always win access to resources
"""

import numpy as np
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
import threading
import time


@dataclass
class GravitationalProcess:
    """
    Process with gravitational properties
    
    Attributes:
        process_id: Unique identifier
        mass: Priority (higher = more important)
        velocity: Execution speed hint
        core_affinity: Preferred CPU core (-1 = any)
        callback: Function to execute
    """
    process_id: str
    mass: float  # Priority weight
    velocity: float  # Speed hint (not used for scheduling, metadata)
    core_affinity: int  # -1 = any core, 0+ = specific core
    callback: Callable


class ProcessGravitator:
    """
    Gravitational process scheduler
    
    Instead of traditional priority scheduling,
    uses gravitational attraction to determine which
    process runs next.
    
    PHYSICS:
    - Processes with higher mass have stronger "pull"
    - Distance to CPU core affects force
    - Gravitational constant determines sensitivity
    
    RESULT:
    - Deterministic priority without starvation
    - Intuitive mental model (heavy = important)
    - Adaptive to system state
    """
    
    GRAVITATIONAL_CONSTANT = 6.674e-11  # Or adjust for your needs
    
    def __init__(self, num_cores: int = 4):
        """
        Initialize process gravitator
        
        Args:
            num_cores: Number of CPU cores
        """
        self.num_cores = num_cores
        self.processes: List[GravitationalProcess] = []
        
        # Core positions in "space" (arbitrary coordinate system)
        self.core_positions = np.array([
            [i, 0, 0] for i in range(num_cores)
        ], dtype=np.float64)
        
        # Process positions (updated dynamically)
        self.process_positions: Dict[str, np.ndarray] = {}
        
        # Execution lock
        self.lock = threading.Lock()
    
    def add_process(self, process: GravitationalProcess):
        """
        Add process to scheduler
        
        Args:
            process: Process to add
        """
        with self.lock:
            self.processes.append(process)
            
            # Initial position (random or based on affinity)
            if process.core_affinity >= 0 and process.core_affinity < self.num_cores:
                # Near preferred core
                self.process_positions[process.process_id] = (
                    self.core_positions[process.core_affinity] + 
                    np.random.randn(3) * 0.1
                )
            else:
                # Random position
                self.process_positions[process.process_id] = np.random.randn(3)
    
    def remove_process(self, process_id: str):
        """Remove process from scheduler"""
        with self.lock:
            self.processes = [p for p in self.processes if p.process_id != process_id]
            if process_id in self.process_positions:
                del self.process_positions[process_id]
    
    def calculate_gravitational_force(self, 
                                     process: GravitationalProcess,
                                     core_idx: int) -> float:
        """
        Calculate gravitational force between process and core
        
        F = G * (m1 * m2) / r²
        
        Args:
            process: Process
            core_idx: Core index
        
        Returns:
            Gravitational force magnitude
        """
        if process.process_id not in self.process_positions:
            return 0.0
        
        process_pos = self.process_positions[process.process_id]
        core_pos = self.core_positions[core_idx]
        
        # Distance
        distance = np.linalg.norm(process_pos - core_pos)
        if distance < 0.1:  # Prevent singularity
            distance = 0.1
        
        # Gravitational force (assume core has mass = 1.0)
        force = self.GRAVITATIONAL_CONSTANT * (process.mass * 1.0) / (distance ** 2)
        
        return force
    
    def schedule_next(self) -> Optional[GravitationalProcess]:
        """
        Schedule next process to run
        
        Calculates gravitational forces for all processes
        to all cores, selects process with strongest pull
        
        Returns:
            Process to execute (or None)
        """
        with self.lock:
            if not self.processes:
                return None
            
            # Calculate forces for all process-core pairs
            max_force = -1
            selected_process = None
            selected_core = -1
            
            for process in self.processes:
                for core_idx in range(self.num_cores):
                    # Skip if process has affinity and this isn't it
                    if (process.core_affinity >= 0 and 
                        process.core_affinity != core_idx):
                        continue
                    
                    force = self.calculate_gravitational_force(process, core_idx)
                    
                    if force > max_force:
                        max_force = force
                        selected_process = process
                        selected_core = core_idx
            
            return selected_process
    
    def execute_next(self):
        """
        Execute next scheduled process
        
        Returns when process completes
        """
        process = self.schedule_next()
        
        if process is None:
            return
        
        # Execute callback
        try:
            process.callback()
        except Exception as e:
            print(f"Error executing process {process.process_id}: {e}")
    
    def run_loop(self, duration_seconds: Optional[float] = None):
        """
        Run scheduling loop
        
        Args:
            duration_seconds: Run for this long (None = forever)
        """
        start_time = time.time()
        
        while True:
            # Check duration
            if duration_seconds and (time.time() - start_time) > duration_seconds:
                break
            
            # Execute next process
            self.execute_next()
            
            # Small sleep to prevent busy loop
            time.sleep(0.001)  # 1ms


# Example usage
def example_gravitational_scheduling():
    """
    Example: Schedule processes with different priorities
    """
    gravitator = ProcessGravitator(num_cores=4)
    
    # E-Stop monitor (critical - heavy mass)
    def e_stop_monitor():
        print("🚨 E-Stop monitor running (CRITICAL)")
    
    gravitator.add_process(GravitationalProcess(
        process_id='e_stop',
        mass=1000.0,  # Very heavy = highest priority
        velocity=1.0,
        core_affinity=0,  # Pin to core 0
        callback=e_stop_monitor
    ))
    
    # Chatter detection (medium priority)
    def chatter_detection():
        print("📊 Chatter detection running (MEDIUM)")
    
    gravitator.add_process(GravitationalProcess(
        process_id='chatter',
        mass=100.0,  # Medium mass
        velocity=1.0,
        core_affinity=-1,  # Any core
        callback=chatter_detection
    ))
    
    # Data logging (low priority)
    def data_logging():
        print("📝 Data logging running (LOW)")
    
    gravitator.add_process(GravitationalProcess(
        process_id='logging',
        mass=1.0,  # Light = lowest priority
        velocity=1.0,
        core_affinity=-1,
        callback=data_logging
    ))
    
    # Run for 1 second
    print("Starting gravitational scheduler...")
    gravitator.run_loop(duration_seconds=1.0)
    print("Done!")


if __name__ == "__main__":
    example_gravitational_scheduling()
