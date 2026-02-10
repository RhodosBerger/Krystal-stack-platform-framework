"""
Platform Module Initializer
"""
from backend.system_platform.entity import PlatformEntity, EntityStatus
from backend.system_platform.pipeline import GenerationPipeline, PipelineStage, default_pipeline
from backend.system_platform.registry import platform_registry
