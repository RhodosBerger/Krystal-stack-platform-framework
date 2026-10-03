"""
Krystal-Lang: Compiler & Queue Partitioner
==========================================
Lexes, parses, and compiles Krystal-Lang source into topological bytecode
and partitioned parallel work queues.
"""

import re
import math
from typing import Dict, Any, List, Optional, Tuple
from krystal_lang.ast_nodes import (
    ProgramNode, QueueDefinition, GeometricShapeDecl,
    TransformOp, StreamPipeline, QuantumLoop
)

class KrystalLexer:
    """Tokenizes Krystal-Lang source code."""
    TOKEN_SPEC = [
        ("COMMENT",     r"#.*"),
        ("NUMBER",      r"-?\d+(\.\d+)?"),
        ("STRING",      r'"[^"]*"'),
        ("VEC3",        r"\[\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*\]"),
        ("KEYWORD",     r"\b(module|queue|shape|pipeline|loop|from|to|transform|action|type|priority|capacity|spatial|iterations|scale|csg|rate|folds|amp|freq|thickness)\b"),
        ("IDENTIFIER",  r"[a-zA-Z_][a-zA-Z0-9_]*"),
        ("LBRACE",      r"\{"),
        ("RBRACE",      r"\}"),
        ("LPAREN",      r"\("),
        ("RPAREN",      r"\)"),
        ("COLON",       r":"),
        ("COMMA",       r","),
        ("WS",          r"\s+"),
    ]

    @classmethod
    def tokenize(cls, code: str) -> List[Tuple[str, str, int]]:
        tokens = []
        line_num = 1
        regex = "|".join(f"(?P<{name}>{pattern})" for name, pattern in cls.TOKEN_SPEC)
        for match in re.finditer(regex, code):
            kind = match.lastgroup
            val = match.group()
            if kind == "COMMENT" or kind == "WS":
                line_num += val.count("\n")
                continue
            tokens.append((kind, val, line_num))
        return tokens

class KrystalCompiler:
    """Compiles Krystal-Lang tokens into an AST, partitioned queue graph, and topological bytecode."""
    def __init__(self):
        self.tokens: List[Tuple[str, str, int]] = []
        self.pos: int = 0

    def compile(self, source_code: str) -> Dict[str, Any]:
        self.tokens = KrystalLexer.tokenize(source_code)
        self.pos = 0
        program = self._parse_program()
        bytecode = self._emit_topological_bytecode(program)
        queue_partitions = self._partition_queues(program)

        return {
            "module_name": program.name,
            "program_ast": program,
            "bytecode": bytecode,
            "queue_partitions": queue_partitions,
            "queues_count": len(program.queues),
            "shapes_count": len(program.shapes),
            "pipelines_count": len(program.pipelines),
            "bytecode_instructions_count": len(bytecode)
        }

    def _peek(self) -> Optional[Tuple[str, str, int]]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _next(self) -> Optional[Tuple[str, str, int]]:
        tok = self._peek()
        if tok:
            self.pos += 1
        return tok

    def _match(self, kind: str, val: Optional[str] = None) -> bool:
        tok = self._peek()
        if not tok:
            return False
        if tok[0] == kind:
            if val is None or tok[1].lower() == val.lower():
                self.pos += 1
                return True
        return False

    def _parse_program(self) -> ProgramNode:
        prog_name = "UntitledModule"
        if self._match("KEYWORD", "module"):
            tok = self._next()
            if tok and tok[0] == "IDENTIFIER":
                prog_name = tok[1]

        program = ProgramNode(name=prog_name)

        while self.pos < len(self.tokens):
            tok = self._peek()
            if not tok:
                break
            if tok[1].lower() == "queue":
                program.queues.append(self._parse_queue())
            elif tok[1].lower() == "shape":
                program.shapes.append(self._parse_shape())
            elif tok[1].lower() == "pipeline":
                program.pipelines.append(self._parse_pipeline())
            elif tok[1].lower() == "loop":
                program.loops.append(self._parse_loop())
            else:
                self.pos += 1  # Skip unrecognized token gracefully

        return program

    def _parse_queue(self) -> QueueDefinition:
        self._next()  # consume 'queue'
        name = "q_anon"
        tok = self._next()
        if tok and tok[0] == "IDENTIFIER":
            name = tok[1]

        q = QueueDefinition(name=name)
        if self._match("LBRACE"):
            while not self._match("RBRACE") and self.pos < len(self.tokens):
                key_tok = self._next()
                if not key_tok: break
                key = key_tok[1].lower()
                self._match("COLON")
                val_tok = self._next()
                if not val_tok: break
                val_str = val_tok[1]

                if key == "priority":
                    q.priority = int(float(val_str))
                elif key == "capacity":
                    q.capacity = int(float(val_str))
                elif key == "spatial":
                    # Parse [x, y, z]
                    coords = [float(x.strip()) for x in val_str.strip("[]").split(",") if x.strip()]
                    if len(coords) == 3:
                        q.spatial_affinity = (coords[0], coords[1], coords[2])

                self._match("COMMA")
        return q

    def _parse_shape(self) -> GeometricShapeDecl:
        self._next()  # consume 'shape'
        tok = self._next()
        s_type = "SPHERE"
        params = {}
        csg_op = "UNION"
        smoothness = 0.2

        if self._match("LBRACE"):
            while not self._match("RBRACE") and self.pos < len(self.tokens):
                key_tok = self._next()
                if not key_tok: break
                key = key_tok[1].lower()
                self._match("COLON")
                val_tok = self._next()
                if not val_tok: break
                val_str = val_tok[1]

                if key == "type":
                    s_type = val_str.upper()
                elif key == "scale":
                    params["scale"] = float(val_str)
                elif key == "iterations":
                    params["iterations"] = int(float(val_str))
                elif key == "frequency":
                    params["frequency"] = float(val_str)
                elif key == "thickness":
                    params["thickness"] = float(val_str)
                elif key == "csg":
                    csg_op = val_str.upper()
                    if self._match("LPAREN"):
                        num_tok = self._next()
                        if num_tok: smoothness = float(num_tok[1])
                        self._match("RPAREN")

                self._match("COMMA")

        return GeometricShapeDecl(shape_type=s_type, params=params, csg_op=csg_op, smoothness=smoothness)

    def _parse_pipeline(self) -> StreamPipeline:
        self._next()  # consume 'pipeline'
        self._next()  # consume name
        pipe = StreamPipeline(source_queue="in", target_queue="out")

        if self._match("LBRACE"):
            while not self._match("RBRACE") and self.pos < len(self.tokens):
                key_tok = self._next()
                if not key_tok: break
                key = key_tok[1].lower()
                self._match("COLON")
                val_tok = self._next()
                if not val_tok: break

                if key == "from":
                    pipe.source_queue = val_tok[1]
                elif key == "to":
                    pipe.target_queue = val_tok[1]
                elif key == "action":
                    pipe.action = val_tok[1].upper()
                elif key == "transform":
                    t_name = val_tok[1].upper()
                    t_params = {}
                    if self._match("LPAREN"):
                        while not self._match("RPAREN") and self.pos < len(self.tokens):
                            pk = self._next()
                            if not pk or pk[0] == "RPAREN": break
                            self._match("COLON")
                            pv = self._next()
                            if pv:
                                try:
                                    t_params[pk[1]] = float(pv[1])
                                except ValueError:
                                    t_params[pk[1]] = pv[1]
                            self._match("COMMA")
                    pipe.transforms.append(TransformOp(op_type=t_name, params=t_params))

                self._match("COMMA")
        return pipe

    def _parse_loop(self) -> QuantumLoop:
        self._next()  # consume 'loop'
        self._next()  # consume loop name
        iters = 4
        fold = 6
        if self._match("LPAREN"):
            while not self._match("RPAREN") and self.pos < len(self.tokens):
                pk = self._next()
                if not pk or pk[0] == "RPAREN": break
                self._match("COLON")
                pv = self._next()
                if pv and pk[1] == "iterations": iters = int(float(pv[1]))
                elif pv and pk[1] == "fold": fold = int(float(pv[1]))
                self._match("COMMA")

        loop = QuantumLoop(iterations=iters, symmetry_fold=fold)
        if self._match("LBRACE"):
            while not self._match("RBRACE") and self.pos < len(self.tokens):
                self.pos += 1
        return loop

    def _partition_queues(self, program: ProgramNode) -> List[Dict[str, Any]]:
        """
        Decomposes pipelines into topological execution stages.
        Pipelines with no direct dependencies are assigned to the same parallel stage.
        """
        stages: List[Dict[str, Any]] = []
        stage_idx = 0
        assigned_queues = set()

        # Stage 0: Ingestion Queues
        in_queues = [q for q in program.queues if not any(p.target_queue == q.name for p in program.pipelines)]
        if in_queues:
            stages.append({
                "stage": stage_idx,
                "type": "INGESTION_PARALLEL",
                "queues": [q.name for q in in_queues],
                "concurrency": len(in_queues)
            })
            stage_idx += 1
            assigned_queues.update(q.name for q in in_queues)

        # Stage 1: Pipeline Transformers
        for pipe in program.pipelines:
            stages.append({
                "stage": stage_idx,
                "type": "PIPELINE_TRANSFORM",
                "source": pipe.source_queue,
                "target": pipe.target_queue,
                "transforms": [t.op_type for t in pipe.transforms],
                "action": pipe.action
            })
            stage_idx += 1

        # Stage 2: Sinks / Storage
        out_queues = [q for q in program.queues if not any(p.source_queue == q.name for p in program.pipelines)]
        if out_queues:
            stages.append({
                "stage": stage_idx,
                "type": "OUTPUT_SINKS",
                "queues": [q.name for q in out_queues],
                "concurrency": len(out_queues)
            })

        return stages

    def _emit_topological_bytecode(self, program: ProgramNode) -> List[Dict[str, Any]]:
        """
        Translates AST nodes into topological bytecode instructions.
        Each instruction encodes memory allocations as spatial coordinates and shapes.
        """
        bc = []
        pc = 0

        # 1. Allocate Queues in metric space
        for q in program.queues:
            bc.append({
                "pc": pc,
                "op": "OP_ALLOC_QUEUE",
                "name": q.name,
                "priority": q.priority,
                "capacity": q.capacity,
                "spatial_coord": list(q.spatial_affinity)
            })
            pc += 1

        # 2. Instantiate Geometric Manifold Shapes
        for shape in program.shapes:
            bc.append({
                "pc": pc,
                "op": "OP_EMIT_SHAPE",
                "shape_type": shape.shape_type,
                "params": shape.params,
                "csg_op": shape.csg_op,
                "smoothness": shape.smoothness
            })
            pc += 1

        # 3. Stream Pipelines & Morphisms
        for pipe in program.pipelines:
            for t in pipe.transforms:
                bc.append({
                    "pc": pc,
                    "op": "OP_GEOM_TRANSFORM",
                    "transform": t.op_type,
                    "params": t.params,
                    "stream_source": pipe.source_queue
                })
                pc += 1

            bc.append({
                "pc": pc,
                "op": "OP_QUEUE_DISPATCH",
                "from_queue": pipe.source_queue,
                "to_queue": pipe.target_queue,
                "action": pipe.action
            })
            pc += 1

        # 4. Quantum Loops (Symmetry fold unrolling)
        for loop in program.loops:
            bc.append({
                "pc": pc,
                "op": "OP_SYMMETRY_FOLD",
                "iterations": loop.iterations,
                "dihedral_fold": loop.symmetry_fold
            })
            pc += 1

        return bc
