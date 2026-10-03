"""
Krystal-Lang: Abstract Syntax Tree & Language Primitives
========================================================
Represents instructions, spatial queue partitions, and geometric
transformations for topological computing.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union

@dataclass
class ASTNode:
    pass

@dataclass
class LiteralNode(ASTNode):
    val_type: str  # 'INT', 'FLOAT', 'VEC3', 'STRING'
    value: Any

@dataclass
class IdentifierNode(ASTNode):
    name: str

@dataclass
class QueueDefinition(ASTNode):
    name: str
    priority: int = 1
    capacity: int = 256
    spatial_affinity: Tuple[float, float, float] = (0.0, 0.0, 0.0)

@dataclass
class GeometricShapeDecl(ASTNode):
    shape_type: str  # 'SPHERE', 'BOX', 'CYLINDER', 'TORUS', 'GYROID', 'IFS_FRACTAL'
    params: Dict[str, Any]
    csg_op: str = "UNION"  # 'UNION', 'INTERSECTION', 'SUBTRACTION', 'SMOOTH_MIN'
    smoothness: float = 0.2

@dataclass
class TransformOp(ASTNode):
    op_type: str  # 'TRANSLATE', 'ROTATE', 'MIRROR_DN', 'TWIST', 'DISPLACE'
    params: Dict[str, Any]

@dataclass
class StreamPipeline(ASTNode):
    source_queue: str
    target_queue: str
    transforms: List[TransformOp] = field(default_factory=list)
    action: str = "PASS"  # 'PASS', 'EVAL_SDF', 'RAYMARCH', 'FILTER', 'SINK'

@dataclass
class QuantumLoop(ASTNode):
    iterations: int
    symmetry_fold: int = 6
    body: List[ASTNode] = field(default_factory=list)

@dataclass
class ProgramNode(ASTNode):
    name: str
    queues: List[QueueDefinition] = field(default_factory=list)
    shapes: List[GeometricShapeDecl] = field(default_factory=list)
    pipelines: List[StreamPipeline] = field(default_factory=list)
    loops: List[QuantumLoop] = field(default_factory=list)
