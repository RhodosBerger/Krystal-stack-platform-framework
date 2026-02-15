import ctypes
import os
import logging

class RustBridge:
    """
    Gamesa Cortex V2: Rust Bridge.
    Interfacing Python Control Plane with Rust Compute Plane.
    """
    def __init__(self):
        self.logger = logging.getLogger("RustBridge")
        self.lib = None
        self._load_library()

    def _load_library(self):
        # Path where 'cargo build --release' would output the .so file
        lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                                "../../rust_planner/target/release/librust_planner.so"))
        try:
            if os.path.exists(lib_path):
                self.lib = ctypes.CDLL(lib_path)
                self.lib.plan_path.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
                self.lib.plan_path.restype = ctypes.c_int
                self.logger.info("Rust Planner Library Loaded Successfully.")
            else:
                self.logger.warning(f"Rust Library not found at {lib_path}. Running in Fallback Mode.")
        except Exception as e:
            self.logger.error(f"Failed to load Rust Library: {e}")

    def plan_path(self, start, goal):
        """
        Calls Rust 'plan_path' or falls back to Python.
        """
        if self.lib:
            result = self.lib.plan_path(start[0], start[1], goal[0], goal[1])
            return f"RUST_PATH_COST_{result}"
        else:
            # Fallback (Python Logic)
            dist = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            return f"PYTHON_PATH_COST_{dist}"

    def optimize_schedule(self, tasks_count):
        if self.lib:
            return self.lib.optimize_schedule(tasks_count)
        return tasks_count * 10 # Python is slower (simulated)
