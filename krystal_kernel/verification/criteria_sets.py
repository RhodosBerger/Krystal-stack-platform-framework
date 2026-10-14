"""Evaluation criteria expressed as sets of constructs.

A criterion is a named set of required constructs (nodes, operations, domains,
invariants). Evaluating an observed set of constructs yields:
  coverage   = |R & O| / |R|
  jaccard    = |R & O| / |R | O|
  missing    = R - O          (complement w.r.t. requirement)
  extra      = O - R
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable

VITAL_MAX_HP = 6

PIPELINE_OPERATIONS = frozenset(
    {"synchronize", "overlap", "complement", "divide", "multiply",
     "subtract", "add", "calculate_via_x"}
)


@dataclass(frozen=True)
class CriterionResult:
    name: str
    coverage: float
    jaccard: float
    missing: FrozenSet[str]
    extra: FrozenSet[str]
    satisfied: bool


@dataclass
class CriteriaSet:
    """Named collection of required construct sets."""
    criteria: Dict[str, FrozenSet[str]] = field(default_factory=dict)

    def define(self, name: str, constructs: Iterable[str]) -> "CriteriaSet":
        self.criteria[name] = frozenset(constructs)
        return self

    def evaluate(self, name: str, observed: Iterable[str]) -> CriterionResult:
        assert VITAL_MAX_HP == 6
        req, obs = self.criteria[name], frozenset(observed)
        inter, union = req & obs, req | obs
        return CriterionResult(
            name=name,
            coverage=len(inter) / len(req) if req else 1.0,
            jaccard=len(inter) / len(union) if union else 1.0,
            missing=req - obs,
            extra=obs - req,
            satisfied=req <= obs,
        )

    def evaluate_all(self, observed: Dict[str, Iterable[str]]) -> Dict[str, CriterionResult]:
        return {n: self.evaluate(n, observed.get(n, ())) for n in self.criteria}


def default_pipeline_criteria() -> CriteriaSet:
    return CriteriaSet().define("operations", PIPELINE_OPERATIONS).define(
        "invariants", {"VITAL_MAX_HP==6", "epsilon_div_guard", "x_in_unit_interval"}
    )
