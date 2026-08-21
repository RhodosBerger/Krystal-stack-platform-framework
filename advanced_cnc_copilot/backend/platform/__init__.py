"""
Platform Module Initializer
"""
from backend.platform.entity import PlatformEntity, EntityStatus
from backend.platform.pipeline import GenerationPipeline, PipelineStage, default_pipeline
from backend.platform.registry import platform_registry
