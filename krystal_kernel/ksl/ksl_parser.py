"""
==============================================================================
KSL (KRYSTAL SYSTEMS LANGUAGE) — RECURSIVE DESCENT PARSER
==============================================================================
Module: krystal_kernel/ksl/ksl_parser.py
==============================================================================
"""

from __future__ import annotations

from typing import List, Optional, Any
from .ksl_lexer import Token, KslSyntaxError
from .ksl_ast import (
    ProgramNode, DeviceGroupDecl, ConstraintDecl, StateDecl, StateField,
    TransitionDecl, RingDecl, FunctionDecl, Parameter, TypeAnnotation,
    Statement, LetStmt, ZeroAllocBlock, CandidatesBlock, ActionCandidate,
    CommitStmt, ValidateBlock, Expression, IdentifierExpr, LiteralExpr,
    BinaryExpr, ObserveExpr, PredictExpr, AllocExpr
)


class KslParser:
    """Parses a token stream into a KSL AST."""

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        return self.tokens[self.pos]

    def peek(self, offset: int = 1) -> Token:
        idx = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[idx]

    def check(self, expected_type: str, expected_value: Optional[str] = None) -> bool:
        tok = self.current()
        if tok.type != expected_type:
            return False
        if expected_value is not None and tok.value != expected_value:
            return False
        return True

    def match(self, expected_type: str, expected_value: Optional[str] = None) -> bool:
        if self.check(expected_type, expected_value):
            self.pos += 1
            return True
        return False

    def expect(self, expected_type: str, expected_value: Optional[str] = None) -> Token:
        tok = self.current()
        if not self.check(expected_type, expected_value):
            val_str = f" '{expected_value}'" if expected_value else ""
            raise KslSyntaxError(
                f"Expected {expected_type}{val_str}, got '{tok.value}' ({tok.type})",
                tok.line, tok.column
            )
        self.pos += 1
        return tok

    def parse_program(self) -> ProgramNode:
        prog = ProgramNode()

        while not self.check("EOF"):
            tok = self.current()
            if self.match("KEYWORD", "device_group"):
                prog.device_groups.append(self.parse_device_group())
            elif self.match("KEYWORD", "constraint"):
                prog.constraints.append(self.parse_constraint())
            elif self.match("KEYWORD", "state"):
                prog.states.append(self.parse_state())
            elif self.match("KEYWORD", "transition"):
                prog.transitions.append(self.parse_transition())
            elif self.match("KEYWORD", "ring"):
                prog.rings.append(self.parse_ring())
            elif self.match("KEYWORD", "fn"):
                prog.functions.append(self.parse_function())
            else:
                raise KslSyntaxError(f"Unexpected top-level declaration: '{tok.value}'", tok.line, tok.column)

        return prog

    def parse_device_group(self) -> DeviceGroupDecl:
        name_tok = self.expect("IDENTIFIER")
        self.expect("PUNCTUATION", "{")
        devices: List[str] = []
        while not self.check("PUNCTUATION", "}") and not self.check("EOF"):
            dev_tok = self.expect("IDENTIFIER")
            self.expect("PUNCTUATION", ";")
            devices.append(dev_tok.value)
        self.expect("PUNCTUATION", "}")
        return DeviceGroupDecl(name=name_tok.value, devices=devices)

    def parse_constraint(self) -> ConstraintDecl:
        name_tok = self.expect("IDENTIFIER")
        self.expect("PUNCTUATION", "{")
        # e.g. state.thermal < 0.78 or thermal < 0.78
        field_parts = []
        first_tok = self.current()
        if first_tok.type in {"IDENTIFIER", "KEYWORD"}:
            field_parts.append(first_tok.value)
            self.pos += 1
        else:
            raise KslSyntaxError(f"Expected field identifier in constraint, got '{first_tok.value}'", first_tok.line, first_tok.column)

        while self.match("PUNCTUATION", "."):
            next_tok = self.current()
            if next_tok.type in {"IDENTIFIER", "KEYWORD"}:
                field_parts.append(next_tok.value)
                self.pos += 1
            else:
                raise KslSyntaxError(f"Expected field identifier after '.', got '{next_tok.value}'", next_tok.line, next_tok.column)

        field_name = ".".join(field_parts)

        op_tok = self.expect("OPERATOR")
        num_tok = self.expect("NUMBER")
        self.match("PUNCTUATION", ";")
        self.expect("PUNCTUATION", "}")

        # Parse limit number
        raw_val = num_tok.value.rstrip("msusBKiMB")
        val = float(raw_val)
        return ConstraintDecl(name=name_tok.value, target_field=field_name, operator=op_tok.value, limit_value=val)

    def parse_state(self) -> StateDecl:
        name_tok = self.expect("IDENTIFIER")
        self.expect("PUNCTUATION", "{")
        fields: List[StateField] = []
        while not self.check("PUNCTUATION", "}") and not self.check("EOF"):
            f_tok = self.current()
            if f_tok.type not in {"IDENTIFIER", "KEYWORD"}:
                raise KslSyntaxError(f"Expected field name in state, got '{f_tok.value}'", f_tok.line, f_tok.column)
            self.pos += 1
            f_name = f_tok.value
            self.expect("PUNCTUATION", ":")
            type_tok = self.current()
            if type_tok.type not in {"IDENTIFIER", "KEYWORD"}:
                raise KslSyntaxError(f"Expected field type in state, got '{type_tok.value}'", type_tok.line, type_tok.column)
            self.pos += 1
            f_type = type_tok.value
            self.match("PUNCTUATION", ",")
            fields.append(StateField(name=f_name, field_type=f_type))
        self.expect("PUNCTUATION", "}")
        return StateDecl(name=name_tok.value, fields=fields)

    def parse_transition(self) -> TransitionDecl:
        name_tok = self.expect("IDENTIFIER")
        self.expect("PUNCTUATION", "{")
        # before: State; action: Action; predict: State; after: State;
        before_state = ""
        action_type = ""
        predict_state = ""
        after_state = ""

        while not self.check("PUNCTUATION", "}") and not self.check("EOF"):
            key = self.expect("IDENTIFIER").value
            self.expect("PUNCTUATION", ":")
            val_type = self.parse_type().to_string()
            self.expect("PUNCTUATION", ";")
            if key == "before":
                before_state = val_type
            elif key == "action":
                action_type = val_type
            elif key == "predict":
                predict_state = val_type
            elif key == "after":
                after_state = val_type

        self.expect("PUNCTUATION", "}")
        return TransitionDecl(
            name=name_tok.value,
            before_state=before_state,
            action_type=action_type,
            predict_state=predict_state,
            after_state=after_state
        )

    def parse_ring(self) -> RingDecl:
        self.expect("OPERATOR", "<")
        elem_type = self.expect("IDENTIFIER").value
        self.expect("PUNCTUATION", ",")
        cap_tok = self.expect("NUMBER")
        self.expect("PUNCTUATION", ",")
        frame_tok = self.expect("NUMBER")
        self.expect("OPERATOR", ">")
        name_tok = self.expect("IDENTIFIER")
        self.expect("PUNCTUATION", ";")

        capacity = int(cap_tok.value)
        frame_bytes = int(frame_tok.value.rstrip("B"))
        return RingDecl(name=name_tok.value, element_type=elem_type, capacity=capacity, frame_size_bytes=frame_bytes)

    def parse_type(self) -> TypeAnnotation:
        tok = self.current()
        self.pos += 1

        epistemic = None
        if tok.value in {"Measured", "Predicted", "Validated", "Derived"}:
            epistemic = tok.value
            self.expect("OPERATOR", "<")
            inner = self.parse_type()
            self.expect("OPERATOR", ">")
            inner.epistemic_wrapper = epistemic
            return inner

        generics: List[TypeAnnotation] = []
        if self.match("OPERATOR", "<"):
            generics.append(self.parse_type())
            while self.match("PUNCTUATION", ","):
                generics.append(self.parse_type())
            self.expect("OPERATOR", ">")

        return TypeAnnotation(name=tok.value, generics=generics)

    def parse_function(self) -> FunctionDecl:
        name_tok = self.expect("IDENTIFIER")
        self.expect("PUNCTUATION", "(")
        params: List[Parameter] = []
        if not self.check("PUNCTUATION", ")"):
            p_name = self.expect("IDENTIFIER").value
            self.expect("PUNCTUATION", ":")
            p_type = self.parse_type()
            params.append(Parameter(name=p_name, param_type=p_type))
            while self.match("PUNCTUATION", ","):
                p_name = self.expect("IDENTIFIER").value
                self.expect("PUNCTUATION", ":")
                p_type = self.parse_type()
                params.append(Parameter(name=p_name, param_type=p_type))
        self.expect("PUNCTUATION", ")")

        # Optional clauses: latency <= 20ms, on auto(...), requires ..., effect ...
        latency_bound = None
        placement = None
        req_constraints: List[str] = []
        effects: List[str] = []

        while True:
            if self.match("KEYWORD", "latency") or self.match("KEYWORD", "deadline"):
                self.expect("OPERATOR", "<=")
                num_tok = self.expect("NUMBER").value
                # e.g. 20ms -> 20.0
                latency_bound = float(num_tok.rstrip("msus"))
            elif self.match("KEYWORD", "on"):
                # e.g. auto(compute_cluster) or gpu0
                place_id = self.expect("IDENTIFIER").value
                if self.match("PUNCTUATION", "("):
                    arg = self.expect("IDENTIFIER").value
                    self.expect("PUNCTUATION", ")")
                    placement = f"{place_id}({arg})"
                else:
                    placement = place_id
            elif self.match("KEYWORD", "requires"):
                req_constraints.append(self.expect("IDENTIFIER").value)
            elif self.match("KEYWORD", "effect"):
                eff = self.expect("IDENTIFIER").value
                if self.match("PUNCTUATION", "("):
                    eff_arg = self.expect("IDENTIFIER").value
                    self.expect("PUNCTUATION", ")")
                    eff = f"{eff}({eff_arg})"
                effects.append(eff)
            else:
                break

        self.expect("PUNCTUATION", "{")
        body = self.parse_statements_until("}")
        self.expect("PUNCTUATION", "}")

        return FunctionDecl(
            name=name_tok.value,
            parameters=params,
            return_type=None,
            latency_bound_ms=latency_bound,
            placement=placement,
            required_constraints=req_constraints,
            effects=effects,
            body=body
        )

    def parse_statements_until(self, end_punct: str) -> List[Statement]:
        stmts: List[Statement] = []
        while not self.check("PUNCTUATION", end_punct) and not self.check("EOF"):
            stmts.append(self.parse_statement())
        return stmts

    def parse_statement(self) -> Statement:
        if self.match("KEYWORD", "zeroalloc"):
            label_tok = self.expect("IDENTIFIER")
            self.expect("PUNCTUATION", "{")
            body = self.parse_statements_until("}")
            self.expect("PUNCTUATION", "}")
            return ZeroAllocBlock(label=label_tok.value, body=body)

        if self.match("KEYWORD", "let"):
            name_tok = self.expect("IDENTIFIER")
            type_ann = None
            if self.match("PUNCTUATION", ":"):
                type_ann = self.parse_type()
            self.expect("OPERATOR", "=")
            init_expr = self.parse_expression()
            self.expect("PUNCTUATION", ";")
            return LetStmt(name=name_tok.value, type_annotation=type_ann, initializer=init_expr)

        if self.match("KEYWORD", "candidates"):
            self.expect("PUNCTUATION", "{")
            actions: List[ActionCandidate] = []
            while not self.check("PUNCTUATION", "}") and not self.check("EOF"):
                if self.match("KEYWORD", "hold"):
                    self.expect("PUNCTUATION", ";")
                    actions.append(ActionCandidate(action_kind="HOLD"))
                elif self.match("KEYWORD", "route"):
                    dest = self.expect("IDENTIFIER").value
                    self.expect("PUNCTUATION", ";")
                    actions.append(ActionCandidate(action_kind="ROUTE", target=dest))
                elif self.match("KEYWORD", "batch_size") or self.match("IDENTIFIER", "batch_size"):
                    op = self.expect("OPERATOR").value
                    delta = int(self.expect("NUMBER").value)
                    self.expect("PUNCTUATION", ";")
                    actions.append(ActionCandidate(action_kind="BATCH_CHANGE", delta=-delta if op == "-=" else delta))
                elif self.match("KEYWORD", "workers") or self.match("IDENTIFIER", "workers"):
                    op = self.expect("OPERATOR").value
                    delta = int(self.expect("NUMBER").value)
                    self.expect("PUNCTUATION", ";")
                    actions.append(ActionCandidate(action_kind="WORKERS_CHANGE", delta=-delta if op == "-=" else delta))
                elif self.match("KEYWORD", "performance") or self.match("IDENTIFIER", "performance"):
                    op = self.expect("OPERATOR").value
                    delta = int(self.expect("NUMBER").value)
                    self.expect("PUNCTUATION", ";")
                    actions.append(ActionCandidate(action_kind="PERFORMANCE", delta=-delta if op == "-=" else delta))
                else:
                    tok = self.current()
                    raise KslSyntaxError(f"Unexpected candidate action: '{tok.value}'", tok.line, tok.column)
            self.expect("PUNCTUATION", "}")
            return CandidatesBlock(actions=actions)

        if self.match("KEYWORD", "commit"):
            plan_expr = self.parse_expression()
            self.expect("PUNCTUATION", ";")
            return CommitStmt(plan_expr=plan_expr)

        if self.match("KEYWORD", "validate"):
            target = "transition"
            if self.match("KEYWORD", "transition") or self.match("IDENTIFIER", "transition"):
                target = "transition"
            self.expect("PUNCTUATION", "{")
            conds: List[BinaryExpr] = []
            while not self.check("PUNCTUATION", "}") and not self.check("EOF"):
                cond_expr = self.parse_expression()
                self.expect("PUNCTUATION", ";")
                if isinstance(cond_expr, BinaryExpr):
                    conds.append(cond_expr)
                else:
                    # e.g. output.valid is parsed as output.valid == true
                    conds.append(BinaryExpr(left=cond_expr, operator="==", right=LiteralExpr(value=True, literal_type="bool")))
            self.expect("PUNCTUATION", "}")
            return ValidateBlock(target=target, conditions=conds)

        # Standalone expression statement
        expr = self.parse_expression()
        self.expect("PUNCTUATION", ";")
        return LetStmt(name="_", type_annotation=None, initializer=expr)

    def parse_expression(self) -> Expression:
        if self.match("KEYWORD", "observe"):
            state_id = self.expect("IDENTIFIER").value
            return ObserveExpr(state_name=state_id)

        if self.match("KEYWORD", "predict"):
            cand_id = self.expect("KEYWORD", "candidates").value if self.check("KEYWORD", "candidates") else self.expect("IDENTIFIER").value
            objectives: List[str] = []
            req_constraints: List[str] = []
            while True:
                if self.match("KEYWORD", "minimize"):
                    tok = self.current()
                    if tok.type in {"IDENTIFIER", "KEYWORD"}:
                        self.pos += 1
                        objectives.append(f"minimize {tok.value}")
                    else:
                        raise KslSyntaxError(f"Expected objective name after 'minimize', got '{tok.value}'", tok.line, tok.column)
                elif self.match("KEYWORD", "maximize"):
                    tok = self.current()
                    if tok.type in {"IDENTIFIER", "KEYWORD"}:
                        self.pos += 1
                        objectives.append(f"maximize {tok.value}")
                    else:
                        raise KslSyntaxError(f"Expected objective name after 'maximize', got '{tok.value}'", tok.line, tok.column)
                elif self.match("KEYWORD", "requires"):
                    req_constraints.append(self.expect("IDENTIFIER").value)
                else:
                    break
            return PredictExpr(candidate_ref=cand_id, objectives=objectives, required_constraints=req_constraints)

        if self.match("KEYWORD", "malloc") or self.match("KEYWORD", "new"):
            # Heap allocation attempt, consume argument if present e.g. malloc(1024)
            if self.match("PUNCTUATION", "("):
                while not self.check("PUNCTUATION", ")") and not self.check("EOF"):
                    self.pos += 1
                self.expect("PUNCTUATION", ")")
            return AllocExpr(alloc_type="heap_malloc")

        return self.parse_binary_expression()

    def parse_binary_expression(self) -> Expression:
        left = self.parse_primary()
        if self.check("OPERATOR") and self.current().value in {"<", "<=", ">", ">=", "==", "!=", "+", "-", "*", "/"}:
            op = self.expect("OPERATOR").value
            right = self.parse_primary()
            return BinaryExpr(left=left, operator=op, right=right)
        return left

    def parse_primary(self) -> Expression:
        tok = self.current()

        if self.match("KEYWORD", "true"):
            return LiteralExpr(value=True, literal_type="bool")
        if self.match("KEYWORD", "false"):
            return LiteralExpr(value=False, literal_type="bool")

        if self.match("NUMBER"):
            raw = tok.value.rstrip("msusBKiMB")
            val = float(raw) if "." in raw else int(raw)
            return LiteralExpr(value=val, literal_type="float" if isinstance(val, float) else "int")

        if self.match("STRING"):
            return LiteralExpr(value=tok.value, literal_type="string")

        if tok.type in {"IDENTIFIER", "KEYWORD"} and tok.value not in {"true", "false", "observe", "predict", "malloc", "new"}:
            self.pos += 1
            id_name = tok.value
            while self.match("PUNCTUATION", "."):
                sub_tok = self.current()
                if sub_tok.type in {"IDENTIFIER", "KEYWORD"}:
                    self.pos += 1
                    id_name = f"{id_name}.{sub_tok.value}"
                else:
                    raise KslSyntaxError(f"Expected identifier after '.', got '{sub_tok.value}'", sub_tok.line, sub_tok.column)
            return IdentifierExpr(name=id_name)

        if self.match("PUNCTUATION", "("):
            expr = self.parse_expression()
            self.expect("PUNCTUATION", ")")
            return expr

        raise KslSyntaxError(f"Unexpected token in expression: '{tok.value}'", tok.line, tok.column)
