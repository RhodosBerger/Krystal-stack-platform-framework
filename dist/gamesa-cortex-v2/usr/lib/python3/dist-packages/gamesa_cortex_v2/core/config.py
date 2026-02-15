import os

class GamesaConfig:
    """
    Centralized configuration for Gamesa Cortex V2.
    """
    # NPU Coordinator
    DEFAULT_DOPAMINE = float(os.getenv("GAMESA_DEFAULT_DOPAMINE", 0.5))
    DEFAULT_CORTISOL = float(os.getenv("GAMESA_DEFAULT_CORTISOL", 0.1))
    MAX_WORKERS = int(os.getenv("GAMESA_MAX_WORKERS", 8))
    
    # Economic Governor
    INITIAL_BUDGET_CREDITS = int(os.getenv("GAMESA_INITIAL_BUDGET", 1000))
    COST_MODEL = {
        "NATIVE_EXECUTION": 1,
        "AVX_EMULATION": 10,
        "MESH_TESSELLATION": 50,
        "AI_INFERENCE": 20,
        "DEFAULT": 5
    }
    
    # Thresholds
    CORTISOL_INTERDICTION_THRESHOLD = 0.8
    DOPAMINE_OPTIMIZATION_THRESHOLD = 0.7
