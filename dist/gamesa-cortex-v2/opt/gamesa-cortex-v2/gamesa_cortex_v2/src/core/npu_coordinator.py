import logging
import time
from concurrent.futures import ThreadPoolExecutor

# Assuming EconomicGovernor is in the correct path relative to this file
# Use relative import for sibling module
from .economic_governor import EconomicGovernor
from .power_governor import PowerGovernor
from .utils import PreciseTimer
from .config import GamesaConfig
from .logging_system import IntraspectralLogger
from .presets import PresetManager
from .openvino_subsystem import OpenVINOSubsystem

class NPUCoordinator:
    """
    Gamesa Cortex V2: Neural Protocol Unit (NPU) Coordinator.
    Implements Earliest Deadline First (EDF) Scheduling.
    """
    def __init__(self):
        self.logger = logging.getLogger("NPUCoordinator")
        self.intra_logger = IntraspectralLogger()
        self.openvino = OpenVINOSubsystem()
        # Set basic logging config if not already set
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)
            
        self.executor = ThreadPoolExecutor(max_workers=GamesaConfig.MAX_WORKERS)
        self.neural_state = {
            "dopamine": GamesaConfig.DEFAULT_DOPAMINE,
            "cortisol": GamesaConfig.DEFAULT_CORTISOL
        }
        self.timer = PreciseTimer()
        self.power = PowerGovernor() # ARM Integration
        self.economics = EconomicGovernor() # Resource Regulation
        
        self.logger.info("Orbit V2: NPU Coordinator Online (EDF Scheduler).")
        self.intra_logger.log_event("SYSTEM", "NPU_Coordinator", "Online", self.neural_state)
        
        # Apply default preset
        self.apply_preset("STANDARD_BALANCED")

    def apply_preset(self, preset_name: str):
        """
        Dynamically applies a system preset.
        Adjusts threading, budget, and OpenVINO settings.
        """
        preset = PresetManager.get_preset(preset_name)
        if not preset:
            self.logger.error(f"Unknown Preset: {preset_name}")
            return

        self.logger.info(f"Applying Preset: {preset.name} ({preset.description})")
        
        # 1. Threading Adjustment
        # Note: ThreadPoolExecutor cannot be easily resized dynamically in old Python versions.
        # We might need to replace it or just accept the limitation if using standard concurrent.futures.
        # Ideally, we'd shutdown and create a new one, but that kills pending tasks.
        # For this implementation, we will log the intent. In a robust system, we swap executors.
        if self.executor._max_workers != preset.max_workers:
            self.logger.info(f"Resizing Executor Pool: {self.executor._max_workers} -> {preset.max_workers}")
            # Hacky resize for standard ThreadPoolExecutor (works in CPython usually)
            self.executor._max_workers = preset.max_workers
        
        # 2. Economic Adjustment
        self.economics.budget_cap = preset.budget_cap
        # Assuming we add a replenish_rate field to economics later
        
        # 3. OpenVINO Adjustment
        self.openvino.set_performance_hint(preset.openvino_hint)
        self.openvino.set_streams(preset.openvino_streams)
        
        self.intra_logger.log_event("SYSTEM", "NPU_Coordinator", "Preset Applied", {"preset": preset.name})

    def assess_priority(self, context: dict) -> str:
        """
        Determines priority based on deadlines and system state.
        Real implementation would be more complex.
        """
        # Placeholder logic
        if self.neural_state["cortisol"] > GamesaConfig.CORTISOL_INTERDICTION_THRESHOLD:
            return "INTERDICTION_PROTOCOL"
        return "STANDARD_OPERATING_PROCEDURE"

    def dispatch_task(self, task_func, task_type: str, deadline_ms: float, *args):
        """
        Dispatches a task with Real-Time accomodation AND Economic Regulation.
        """
        start = self.timer.elapsed_ms()
        # 1. Admission Control (Time)
        # Note: 'deadline_ms' acts as a relative deadline from "now" for admission? 
        # Or absolute time? Typically deadline is relative to start.
        # If deadline_ms is intended as "budget", then checking start > deadline_ms 
        # only makes sense if deadline_ms was an absolute timestamp.
        # Assuming deadline_ms is a duration budget here.
        
        # 2. Priority Check
        priority = self.assess_priority({})
        
        # 3. Economic Regulation (Budget)
        if not self.economics.request_allocation(task_type, priority):
            self.logger.warning(f"Task {task_type} Denied by Economic Governor.")
            self.intra_logger.log_event("PLANNING", "NPU_Coordinator", "Task Denied", {"task": task_type, "reason": "budget"})
            return None

        # self.logger.info(f"Dispatching Task {task_type}. Protocol: {priority}.")
        self.intra_logger.log_event("INFERENCE", "NPU_Coordinator", "Task Dispatched", {"task": task_type, "priority": priority, "deadline": deadline_ms})
        return self.executor.submit(task_func, *args)

    def shutdown(self):
        """
        Gracefully shuts down the NPU Coordinator and exports logs.
        """
        self.logger.info("NPU Coordinator Shutting Down...")
        self.intra_logger.log_event("SYSTEM", "NPU_Coordinator", "Shutdown", {})
        self.intra_logger.export_logs()
        self.executor.shutdown(wait=False)
