from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
import uvicorn
import threading

from .npu_coordinator import NPUCoordinator
from .presets import PresetManager

# --- Application Setup ---
app = FastAPI(
    title="Gamesa Cortex V2 API",
    description="REST Interface for the Neural Protocol Unit Coordinator",
    version="0.1.0"
)

# Global Coordinator Instance (Singleton Pattern)
coordinator = NPUCoordinator()

# --- Pydantic Models ---
class PresetRequest(BaseModel):
    name: str

class TaskRequest(BaseModel):
    task_type: str
    workload: int
    deadline_ms: float
    priority: str = "STANDARD"

class SystemStatus(BaseModel):
    neural_state: Dict[str, float]
    budget_cap: int
    active_threads: int
    openvino_hint: str
    preset: str = "UNKNOWN"

# --- Endpoints ---

@app.get("/", tags=["System"])
async def root():
    return {"message": "Gamesa Cortex V2 API Online"}

@app.get("/system/status", response_model=SystemStatus, tags=["System"])
async def get_system_status():
    """
    Returns the current telemetry and configuration of the NPU Coordinator.
    """
    return {
        "neural_state": coordinator.neural_state,
        "budget_cap": coordinator.economics.budget_cap,
        "active_threads": coordinator.executor._max_workers,
        "openvino_hint": coordinator.openvino.current_hint,
    }

@app.post("/system/preset/{name}", tags=["Configuration"])
async def set_system_preset(name: str):
    """
    Dynamically switches the system preset (e.g., IDLE_ECO, HIGH_PERFORMANCE).
    """
    preset = PresetManager.get_preset(name)
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset '{name}' not found.")
    
    coordinator.apply_preset(name)
    return {"status": "success", "message": f"Applied preset: {name}"}

@app.get("/openvino/devices", tags=["OpenVINO"])
async def get_openvino_devices():
    """
    Lists available hardware accelerators for AI Inference.
    """
    devices = coordinator.openvino.get_available_devices()
    return {"devices": devices}

@app.post("/tasks/dispatch", tags=["Tasks"])
async def dispatch_task(task: TaskRequest, background_tasks: BackgroundTasks):
    """
    Submits a task to the NPU Coordinator.
    """
    # Define a simple wrapper to execute the task
    def dummy_task_executor(workload):
        import time
        time.sleep(workload / 1000.0)
        return "Task Completed"

    future = coordinator.dispatch_task(
        dummy_task_executor, 
        task.task_type, 
        task.deadline_ms, 
        task.workload
    )
    
    if future is None:
        raise HTTPException(status_code=503, detail="Task denied by Economic Governor")
        
    return {"status": "dispatched", "task": task.task_type}

# --- Launcher ---
def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_api()
