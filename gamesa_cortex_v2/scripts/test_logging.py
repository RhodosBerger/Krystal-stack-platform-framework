import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from gamesa_cortex_v2.src.core.logging_system import IntraspectralLogger

def test_logger():
    print("Initializing Intraspectral Logger...", flush=True)
    logger = IntraspectralLogger(log_dir="test_logs")
    
    print("Logging events...", flush=True)
    logger.log_event("SYSTEM", "TestScript", "Startup", {"version": "1.0"})
    time.sleep(0.1)
    logger.log_event("PLANNING", "Planner", "PathFound", {"cost": 42, "nodes": 150})
    time.sleep(0.1)
    logger.log_event("SAFETY", "PowerMonitor", "VoltageCheck", {"v_cpu": 1.25, "temp": 45.0})
    
    print("Exporting logs...", flush=True)
    logger.export_logs("test_output.json")
    
    if os.path.exists("test_logs/test_output.json"):
        print("Log file created successfully: test_logs/test_output.json", flush=True)
        with open("test_logs/test_output.json", "r") as f:
            print("Log Content Preview:", flush=True)
            print(f.read(), flush=True)
    else:
        print("Error: Log file not found!", flush=True)

if __name__ == "__main__":
    test_logger()
