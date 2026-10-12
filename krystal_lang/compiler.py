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
    TransformOp, StreamPipeline, QuantumLoop,
    ConditionExpr, ConditionalBranch, ConditionalLadder,
    WhatIfScenarioNode, WhatIfBlockNode
)

class KrystalLexer:
    """Tokenizes Krystal-Lang source code."""
    TOKEN_SPEC = [
        ("COMMENT",     r"#.*"),
        ("NUMBER",      r"-?\d+(\.\d+)?"),
        ("STRING",      r'"[^"]*"'),
        ("VEC3",        r"\[\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*\]"),
        ("KEYWORD",     r"\b(module|queue|shape|pipeline|loop|if|elif|else|what_if|scenario|fallback|criterion|hypothesis|when|cost|utility|on|from|to|transform|action|type|priority|capacity|spatial|iterations|scale|csg|rate|folds|amp|freq|thickness)\b"),
        ("COMP_OP",     r"(>=|<=|==|!=|>|<)"),
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
            elif tok[1].lower() == "if":
                program.conditionals.append(self._parse_conditional_ladder())
            elif tok[1].lower() == "what_if":
                program.what_ifs.append(self._parse_what_if_block())
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

    def _parse_condition_expr(self, text: str) -> ConditionExpr:
        m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(>=|<=|==|!=|>|<)\s*(.+)", text.strip())
        if m:
            field = m.group(1)
            op = m.group(2)
            val_str = m.group(3).strip()
            try:
                val = float(val_str)
            except ValueError:
                if val_str.lower() == "true": val = True
                elif val_str.lower() == "false": val = False
                else: val = val_str.strip('"\'')
            return ConditionExpr(field=field, op=op, value=val)
        return ConditionExpr(field="status", op="==", value=text.strip())

    def _parse_conditional_ladder(self) -> ConditionalLadder:
        self._next()  # consume 'if'
        ladder_name = "CondLadder"
        tok = self._next()
        if tok and tok[0] == "IDENTIFIER":
            ladder_name = tok[1]

        src_queue = "InStream"
        peek = self._peek()
        if peek and peek[1].lower() == "on":
            self._next()
            src_tok = self._next()
            if src_tok: src_queue = src_tok[1]

        if_branch = ConditionalBranch()
        if self._match("LBRACE"):
            while not self._match("RBRACE") and self.pos < len(self.tokens):
                kt = self._next()
                if not kt: break
                key = kt[1].lower()
                self._match("COLON")
                vt = self._next()
                if not vt: break
                if key == "when":
                    cond_str = vt[1]
                    while self.pos < len(self.tokens) and self._peek() and self._peek()[0] in ("COMP_OP", "NUMBER", "STRING", "IDENTIFIER"):
                        cond_str += " " + self._next()[1]
                    if_branch.condition = self._parse_condition_expr(cond_str)
                elif key == "to":
                    if_branch.target_queue = vt[1]
                elif key == "action":
                    if_branch.action = vt[1].upper()
                self._match("COMMA")

        elif_branches = []
        else_branch = None

        while self.pos < len(self.tokens):
            pk = self._peek()
            if not pk: break
            if pk[1].lower() == "elif":
                self._next()  # consume 'elif'
                elif_b = ConditionalBranch()
                if self._match("LBRACE"):
                    while not self._match("RBRACE") and self.pos < len(self.tokens):
                        kt = self._next()
                        if not kt: break
                        key = kt[1].lower()
                        self._match("COLON")
                        vt = self._next()
                        if not vt: break
                        if key == "when":
                            cond_str = vt[1]
                            while self.pos < len(self.tokens) and self._peek() and self._peek()[0] in ("COMP_OP", "NUMBER", "STRING", "IDENTIFIER"):
                                cond_str += " " + self._next()[1]
                            elif_b.condition = self._parse_condition_expr(cond_str)
                        elif key == "to":
                            elif_b.target_queue = vt[1]
                        elif key == "action":
                            elif_b.action = vt[1].upper()
                        self._match("COMMA")
                elif_branches.append(elif_b)
            elif pk[1].lower() == "else":
                self._next()  # consume 'else'
                else_b = ConditionalBranch(condition=None)
                if self._match("LBRACE"):
                    while not self._match("RBRACE") and self.pos < len(self.tokens):
                        kt = self._next()
                        if not kt: break
                        key = kt[1].lower()
                        self._match("COLON")
                        vt = self._next()
                        if not vt: break
                        if key == "to":
                            else_b.target_queue = vt[1]
                        elif key == "action":
                            else_b.action = vt[1].upper()
                        self._match("COMMA")
                else_branch = else_b
                break
            else:
                break

        return ConditionalLadder(
            name=ladder_name,
            source_queue=src_queue,
            if_branch=if_branch,
            elif_branches=elif_branches,
            else_branch=else_branch
        )

    def _parse_what_if_block(self) -> WhatIfBlockNode:
        self._next()  # consume 'what_if'
        block_name = "WhatIfBlock"
        tok = self._next()
        if tok and tok[0] == "IDENTIFIER":
            block_name = tok[1]

        src_queue = "InboundStream"
        peek = self._peek()
        if peek and peek[1].lower() == "on":
            self._next()
            src_tok = self._next()
            if src_tok: src_queue = src_tok[1]

        scenarios = []
        fallback_q = "FallbackSink"
        criterion = "MAX_UTILITY"

        if self._match("LBRACE"):
            while not self._match("RBRACE") and self.pos < len(self.tokens):
                pk = self._peek()
                if not pk: break
                if pk[1].lower() == "scenario":
                    self._next()  # consume 'scenario'
                    sc_name = "Scenario"
                    sct = self._next()
                    if sct and sct[0] == "IDENTIFIER":
                        sc_name = sct[1]
                    hyp = sc_name
                    target_q = "OutQueue"
                    action = "SPECULATE"
                    cost = 0.1
                    utility = 0.8
                    if self._match("LBRACE"):
                        while not self._match("RBRACE") and self.pos < len(self.tokens):
                            kt = self._next()
                            if not kt: break
                            key = kt[1].lower()
                            self._match("COLON")
                            vt = self._next()
                            if not vt: break
                            if key == "hypothesis": hyp = vt[1].strip('"\'')
                            elif key == "to": target_q = vt[1]
                            elif key == "action": action = vt[1].upper()
                            elif key in ("cost", "cost_penalty"): cost = float(vt[1])
                            elif key in ("utility", "expected_utility"): utility = float(vt[1])
                            self._match("COMMA")
                    scenarios.append(WhatIfScenarioNode(
                        name=sc_name, hypothesis=hyp, target_queue=target_q,
                        action=action, cost_penalty=cost, expected_utility=utility
                    ))
                elif pk[1].lower() == "fallback":
                    self._next()
                    self._match("COLON")
                    ft = self._next()
                    if ft: fallback_q = ft[1]
                    self._match("COMMA")
                elif pk[1].lower() == "criterion":
                    self._next()
                    self._match("COLON")
                    ct = self._next()
                    if ct: criterion = ct[1].upper()
                    self._match("COMMA")
                else:
                    self.pos += 1

        return WhatIfBlockNode(
            name=block_name,
            source_queue=src_queue,
            scenarios=scenarios,
            fallback_queue=fallback_q,
            commit_criterion=criterion
        )


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

        # 5. Conditional Ladders (if / elif / else)
        for cond in program.conditionals:
            bc.append({
                "pc": pc,
                "op": "OP_BRANCH_IF",
                "ladder_name": cond.name,
                "source_queue": cond.source_queue,
                "condition": {
                    "field": cond.if_branch.condition.field if cond.if_branch.condition else "default",
                    "op": cond.if_branch.condition.op if cond.if_branch.condition else "==",
                    "val": cond.if_branch.condition.value if cond.if_branch.condition else True
                },
                "target_queue": cond.if_branch.target_queue,
                "action": cond.if_branch.action,
                "csg_hyperplane": [1.0, 0.0, 0.0, 0.0]
            })
            pc += 1

            for i, elif_b in enumerate(cond.elif_branches):
                bc.append({
                    "pc": pc,
                    "op": "OP_BRANCH_ELIF",
                    "ladder_name": cond.name,
                    "index": i,
                    "source_queue": cond.source_queue,
                    "condition": {
                        "field": elif_b.condition.field if elif_b.condition else "default",
                        "op": elif_b.condition.op if elif_b.condition else "==",
                        "val": elif_b.condition.value if elif_b.condition else True
                    },
                    "target_queue": elif_b.target_queue,
                    "action": elif_b.action,
                    "csg_hyperplane": [0.0, 1.0, 0.0, float(i + 1) * 0.5]
                })
                pc += 1

            if cond.else_branch:
                bc.append({
                    "pc": pc,
                    "op": "OP_BRANCH_ELSE",
                    "ladder_name": cond.name,
                    "source_queue": cond.source_queue,
                    "target_queue": cond.else_branch.target_queue,
                    "action": cond.else_branch.action,
                    "csg_hyperplane": [-1.0, 0.0, 0.0, 0.0]
                })
                pc += 1

        # 6. What-If Speculative Blocks
        for wib in program.what_ifs:
            sc_list = []
            for sc in wib.scenarios:
                sc_list.append({
                    "name": sc.name,
                    "hypothesis": sc.hypothesis,
                    "target_queue": sc.target_queue,
                    "action": sc.action,
                    "cost_penalty": sc.cost_penalty,
                    "expected_utility": sc.expected_utility
                })
            bc.append({
                "pc": pc,
                "op": "OP_WHAT_IF_SPECULATE",
                "block_name": wib.name,
                "source_queue": wib.source_queue,
                "fallback_queue": wib.fallback_queue,
                "criterion": wib.commit_criterion,
                "scenarios": sc_list,
                "radial_bifurcation_fold": max(3, len(sc_list))
            })
            pc += 1

        return bc
