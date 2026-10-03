"""
Krystal-Lang: Topological Bytecode & Queue-Partitioned Geometric Language
========================================================================
Compiles procedural code into topological manifolds, partitions work
into parallel execution queues, and renders running programs as 3D shapes.
"""

from krystal_lang.ast_nodes import (
    ProgramNode, QueueDefinition, GeometricShapeDecl,
    TransformOp, StreamPipeline, QuantumLoop
)
from krystal_lang.compiler import KrystalCompiler, KrystalLexer
from krystal_lang.bytecode_to_shape import BytecodeShapeTranspiler
from krystal_lang.virtual_machine import TopologicalVM

__all__ = [
    "ProgramNode",
    "QueueDefinition",
    "GeometricShapeDecl",
    "TransformOp",
    "StreamPipeline",
    "QuantumLoop",
    "KrystalCompiler",
    "KrystalLexer",
    "BytecodeShapeTranspiler",
    "TopologicalVM"
]
