"""
CNC Intelligence Framework (CIF)
Inspired by OpenVINO and oneAPI

Unified AI inference framework for manufacturing with:
- Hardware abstraction (CPU, GPU, FPGA, Edge TPU)
- Model optimization (quantization, pruning, fusion)
- Asynchronous inference
- Real-time performance
"""

from .cif_core import CIFCore, CIFModel, CompiledModel, InferRequest
from .cif_model_converter import CIFModelConverter
from .cif_pipeline import Pipeline, PipelineStage

__version__ = "1.0.0"

__all__ = [
    'CIFCore',
    'CIFModel',
    'CompiledModel',
    'InferRequest',
    'CIFModelConverter',
    'Pipeline',
    'PipelineStage',
]
