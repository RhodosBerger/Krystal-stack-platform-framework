"""
Krystal Stack Platform Framework - Composable Open-Source Mesh Projects
========================================================================
Package: projects/
Constitutional Invariant: VITAL_MAX_HP == 6

Exports the 5 functional blueprints synthesized from the Composable Mesh architecture:
  1. SovereignGameCompanionBridge (Godot 4 + llama.cpp + Whisper.cpp)
  2. ZeroStutterDccBridge (Blender + Ollama + EpistemicCalorimeter)
  3. SelfTuningWorkstationEngine (Linux Kernel + eBPF + Z3 + VBlankInjector)
  4. VerifiedRoboticsTwinBridge (ROS 2 + Gazebo + BifurcatedConflictRing)
  5. LowLatencyIdeMeshCoordinator (VS Code/Zed + rustc + vLLM + pytest)
"""

from projects.sovereign_game_companion import (
    SovereignGameCompanionBridge,
    GameRenderFrameTelemetry,
)
from projects.zero_stutter_dcc_copilot import (
    ZeroStutterDccBridge,
    DccTaskResult,
)
from projects.self_tuning_workstation import (
    SelfTuningWorkstationEngine,
    HardwareTuningHypothesis,
    TuningEvaluationReport,
)
from projects.verified_robotics_twin import (
    VerifiedRoboticsTwinBridge,
    SensorObservation,
    RobotNavigationCommand,
)
from projects.low_latency_ide_mesh import (
    LowLatencyIdeMeshCoordinator,
    EditorFrameQoS,
    CodeAssistantSuggestion,
)

__all__ = [
    "SovereignGameCompanionBridge",
    "GameRenderFrameTelemetry",
    "ZeroStutterDccBridge",
    "DccTaskResult",
    "SelfTuningWorkstationEngine",
    "HardwareTuningHypothesis",
    "TuningEvaluationReport",
    "VerifiedRoboticsTwinBridge",
    "SensorObservation",
    "RobotNavigationCommand",
    "LowLatencyIdeMeshCoordinator",
    "EditorFrameQoS",
    "CodeAssistantSuggestion",
]
