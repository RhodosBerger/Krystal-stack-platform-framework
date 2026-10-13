#!/usr/bin/env python3
"""
==============================================================================
KARAF: ALGORITHM TEMPLATE PROTOCOL
==============================================================================
Module: krystal_kernel/karaf/algorithm_template.py
Description:
    Canonical declarative template specification representing runtime algorithms:
      - Declarative identity, epistemic evidence requirements, inputs/outputs
      - Objectives, constraints, prediction targets, falsification experiments
      - Enforces execution.mode == SHADOW and authority.direct_actuation == false
==============================================================================
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP


@dataclass
class TemplateIdentity:
    id: str
    version: str
    family: str
    description: str = ""


@dataclass
class TemplateEpistemic:
    status: str
    evidence_required: List[str]
    vital_max_hp: int = VITAL_MAX_HP


@dataclass
class TemplateObjective:
    maximize: List[str] = field(default_factory=list)
    minimize: List[str] = field(default_factory=list)


@dataclass
class AlgorithmTemplate:
    """Canonical declarative contract representing a KARAF runtime algorithm."""
    identity: TemplateIdentity
    epistemic: TemplateEpistemic
    inputs: List[Dict[str, str]] = field(default_factory=list)
    derived_metrics: List[str] = field(default_factory=list)
    search_space: Dict[str, Any] = field(default_factory=dict)
    objective: TemplateObjective = field(default_factory=TemplateObjective)
    constraints: List[str] = field(default_factory=list)
    prediction: Dict[str, Any] = field(default_factory=dict)
    falsification: Dict[str, Any] = field(default_factory=dict)
    execution_mode: str = "SHADOW"
    telemetry_emit: List[str] = field(default_factory=list)
    hypothesis_update: bool = True
    direct_actuation: bool = False
    applicable_when: Dict[str, Any] = field(default_factory=dict)
    raw_config: Dict[str, Any] = field(default_factory=dict)
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"
        assert self.direct_actuation is False, "Templates must never claim direct actuation authority"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlgorithmTemplate":
        cfg = data.get("algorithm_template", data)
        ident_d = cfg.get("identity", {})
        ident = TemplateIdentity(
            id=ident_d.get("id", "unknown_template"),
            version=ident_d.get("version", "1.0.0"),
            family=ident_d.get("family", "GENERAL"),
            description=ident_d.get("description", "")
        )

        epist_d = cfg.get("epistemic", {})
        epist = TemplateEpistemic(
            status=epist_d.get("status", "HYPOTHESIS"),
            evidence_required=epist_d.get("evidence_required", []),
            vital_max_hp=epist_d.get("vital_max_hp", VITAL_MAX_HP)
        )

        obj_d = cfg.get("objective", {})
        obj = TemplateObjective(
            maximize=obj_d.get("maximize", []),
            minimize=obj_d.get("minimize", [])
        )

        exec_d = cfg.get("execution", {})
        auth_d = cfg.get("authority", {})
        learn_d = cfg.get("learning", {})

        return cls(
            identity=ident,
            epistemic=epist,
            inputs=cfg.get("inputs", []),
            derived_metrics=cfg.get("derived_metrics", []),
            search_space=cfg.get("search_space", {}),
            objective=obj,
            constraints=cfg.get("constraints", []),
            prediction=cfg.get("prediction", {}),
            falsification=cfg.get("falsification", {}),
            execution_mode=exec_d.get("mode", "SHADOW"),
            telemetry_emit=cfg.get("telemetry", {}).get("emit", []),
            hypothesis_update=learn_d.get("hypothesis_update", True),
            direct_actuation=auth_d.get("direct_actuation", False),
            applicable_when=cfg.get("applicable_when", {}),
            raw_config=cfg,
            vital_max_hp=VITAL_MAX_HP
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": {
                "id": self.identity.id,
                "version": self.identity.version,
                "family": self.identity.family,
                "description": self.identity.description
            },
            "epistemic": {
                "status": self.epistemic.status,
                "evidence_required": self.epistemic.evidence_required,
                "vital_max_hp": self.epistemic.vital_max_hp
            },
            "inputs": self.inputs,
            "derived_metrics": self.derived_metrics,
            "search_space": self.search_space,
            "objective": {
                "maximize": self.objective.maximize,
                "minimize": self.objective.minimize
            },
            "constraints": self.constraints,
            "prediction": self.prediction,
            "falsification": self.falsification,
            "execution": {"mode": self.execution_mode},
            "telemetry": {"emit": self.telemetry_emit},
            "learning": {"hypothesis_update": self.hypothesis_update},
            "authority": {"direct_actuation": self.direct_actuation},
            "vital_max_hp": self.vital_max_hp
        }
