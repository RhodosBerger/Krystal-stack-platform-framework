"""
Generation Pipeline Orchestrator ⚡
Responsibility:
1. Manage multi-stage generation workflows.
2. Chain transformations: Input -> Transform -> Emit -> Seal.
"""
from typing import List, Dict, Any, Callable
from backend.platform.entity import PlatformEntity, EntityStatus
import uuid

class PipelineStage:
    def __init__(self, name: str, processor: Callable[[Any], Any]):
        self.id = f"STAGE-{uuid.uuid4().hex[:4].upper()}"
        self.name = name
        self.processor = processor

    def execute(self, input_data: Any) -> Any:
        return self.processor(input_data)

class GenerationPipeline:
    def __init__(self, name: str):
        self.id = f"PIPE-{uuid.uuid4().hex[:6].upper()}"
        self.name = name
        self.stages: List[PipelineStage] = []
        self.execution_log: List[Dict[str, Any]] = []

    def add_stage(self, stage: PipelineStage):
        self.stages.append(stage)

    def run(self, entity: PlatformEntity, initial_data: Any) -> Dict[str, Any]:
        """
        Executes all stages in sequence, feeding output -> input.
        """
        entity.transition(EntityStatus.IN_PIPELINE)
        current_data = initial_data

        for stage in self.stages:
            self.execution_log.append({
                "stage_id": stage.id,
                "stage_name": stage.name,
                "input_preview": str(current_data)[:100]
            })
            current_data = stage.execute(current_data)

        entity.transition(EntityStatus.COMPLETED)
        return {
            "pipeline_id": self.id,
            "entity_id": entity.id,
            "final_output": current_data,
            "execution_log": self.execution_log
        }

# --- Built-in Stage Processors ---
def gcode_formatter(data: str) -> str:
    """Wraps raw data in G-Code header/footer."""
    return f"%\n(GENERATED_BY_PLATFORM)\nG21 G90\n{data}\nM30\n%"

def cortex_logger(data: str) -> str:
    """Simulates logging to Cortex Memory Membrane."""
    print(f"[CORTEX_MIRROR] Pipeline data logged: {len(data)} chars")
    return data

# Global default pipeline instance
default_pipeline = GenerationPipeline("DefaultGenPipeline")
default_pipeline.add_stage(PipelineStage("GCodeFormat", gcode_formatter))
default_pipeline.add_stage(PipelineStage("CortexLog", cortex_logger))
