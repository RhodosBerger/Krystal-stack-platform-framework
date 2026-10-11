#!/usr/bin/env python3
"""
KRYSTAL-STACK // ONTOLOGY COMPILER CORE
=============================================================================
Converts proposed system mechanisms into operational mathematical ontologies.

Enforces:
  1. Every entity must define: state, observable, hidden, control variables,
     constraints, dependencies, invariants, and failure states.
  2. Directed relationship graph G = (V, E) with explicit mathematical expressions,
     confidence, measurement sources, sampling rates, delays, and falsification tests.
  3. Rejection of circular explanations that lack independent observable variables.
  4. Marking unobservable/non-falsifiable concepts as NON_OPERATIONAL.
  5. Generation of canonical artifacts:
     - ontology.json
     - dependency_graph.json
     - invariants.json
     - unknowns.json
=============================================================================
"""

import os
import sys
import json
import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.epistemic_research_engine import EpistemicStatus


@dataclass
class OntologyEntity:
    entity_id: str
    name: str
    description: str
    state_variables: List[str] = field(default_factory=list)
    observable_variables: List[str] = field(default_factory=list)
    hidden_variables: List[str] = field(default_factory=list)
    control_variables: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    invariant: str = ""
    failure_states: List[str] = field(default_factory=list)
    is_operational: bool = True
    epistemic_status: EpistemicStatus = EpistemicStatus.HYPOTHESIS

    def validate_operational_soundness(self) -> Tuple[bool, List[str]]:
        """Verifies if entity has measurable and falsifiable consequences."""
        issues = []
        if not self.observable_variables and not self.control_variables:
            issues.append("Entity has no observable or control variables. Cannot be directly verified.")
            self.is_operational = False
        if not self.invariant:
            issues.append("Entity lacks formal invariant declaration.")
        return (len(issues) == 0, issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "description": self.description,
            "state_variables": self.state_variables,
            "observable_variables": self.observable_variables,
            "hidden_variables": self.hidden_variables,
            "control_variables": self.control_variables,
            "constraints": self.constraints,
            "dependencies": self.dependencies,
            "invariant": self.invariant,
            "failure_states": self.failure_states,
            "is_operational": self.is_operational,
            "operational_status": "OPERATIONAL" if self.is_operational else "NON_OPERATIONAL",
            "epistemic_status": self.epistemic_status.value
        }


@dataclass
class OntologyRelation:
    source_entity_id: str
    target_entity_id: str
    relation_type: str               # e.g., CAUSAL, CONSTRAINED_BY, REGULATES, DERIVES
    mathematical_expression: str     # e.g., "T_spindle = T_amb + (P_spindle * R_th) * (1 - e^(-t/tau))"
    confidence: float                # [0.0 - 1.0]
    measurement_source: str          # e.g., "/sys/devices/virtual/thermal/thermal_zone0/temp"
    sampling_frequency_hz: float     # e.g., 50.0
    expected_delay_ms: float         # e.g., 15.0
    falsification_test: str          # e.g., "Inject step load +20%; if delta_T < 2C within 5s, falsified."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_entity_id,
            "target": self.target_entity_id,
            "relation_type": self.relation_type,
            "mathematical_expression": self.mathematical_expression,
            "confidence": self.confidence,
            "measurement_source": self.measurement_source,
            "sampling_frequency_hz": self.sampling_frequency_hz,
            "expected_delay_ms": self.expected_delay_ms,
            "falsification_test": self.falsification_test
        }


class CircularDependencyError(Exception):
    """Raised when an unobservable circular loop is discovered in the ontology."""
    pass


class OntologyCompiler:
    """Compiles conceptual mechanisms into operational ontologies and verified graphs."""

    def __init__(self):
        self.entities: Dict[str, OntologyEntity] = {}
        self.relations: List[OntologyRelation] = []

    def register_entity(self, entity: OntologyEntity) -> OntologyEntity:
        valid, issues = entity.validate_operational_soundness()
        if not valid:
            entity.is_operational = False
        self.entities[entity.entity_id] = entity
        return entity

    def add_relation(self, relation: OntologyRelation):
        if relation.source_entity_id not in self.entities:
            raise KeyError(f"Source entity {relation.source_entity_id} not registered in ontology")
        if relation.target_entity_id not in self.entities:
            raise KeyError(f"Target entity {relation.target_entity_id} not registered in ontology")
        
        self.relations.append(relation)

    def detect_circular_unobservable_cycles(self) -> List[List[str]]:
        """
        Finds circular cycles in the dependency graph.
        If a cycle has NO independent observable variable, it constitutes a non-operational tautology.
        """
        adj: Dict[str, List[str]] = {eid: [] for eid in self.entities}
        for rel in self.relations:
            adj[rel.source_entity_id].append(rel.target_entity_id)

        cycles: List[List[str]] = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])

            rec_stack.remove(node)
            path.pop()

        for entity_id in self.entities:
            if entity_id not in visited:
                dfs(entity_id, [])

        # Check which cycles lack observable variables
        tautological_cycles = []
        for cyc in cycles:
            observables_in_cycle = sum(
                len(self.entities[eid].observable_variables)
                for eid in cyc[:-1] if eid in self.entities
            )
            if observables_in_cycle == 0:
                tautological_cycles.append(cyc)

        return tautological_cycles

    def compile(self) -> Dict[str, Any]:
        """
        Executes full validation and compilation of the ontology.
        Rejects tautologies and produces structured artifact bundles.
        """
        tautological_cycles = self.detect_circular_unobservable_cycles()
        if tautological_cycles:
            cycle_desc = " -> ".join(tautological_cycles[0])
            raise CircularDependencyError(
                f"Ontology compilation rejected: Circular explanation without independent observables detected: {cycle_desc}"
            )

        # Build artifacts
        ontology_json = {
            "compilation_timestamp": time.time(),
            "version": "2.0.0-RESEARCH",
            "entities": {k: v.to_dict() for k, v in self.entities.items()}
        }

        dependency_graph_json = {
            "directed_graph": {
                "nodes": [
                    {"id": eid, "name": ent.name, "operational": ent.is_operational}
                    for eid, ent in self.entities.items()
                ],
                "edges": [rel.to_dict() for rel in self.relations]
            }
        }

        invariants_json = {
            "invariants": [
                {
                    "entity_id": eid,
                    "invariant_expression": ent.invariant,
                    "failure_states": ent.failure_states
                }
                for eid, ent in self.entities.items() if ent.invariant
            ]
        }

        unknowns_json = {
            "hidden_variables": [
                {
                    "entity_id": eid,
                    "variable_name": var,
                    "requires_estimator": True
                }
                for eid, ent in self.entities.items()
                for var in ent.hidden_variables
            ],
            "non_operational_entities": [
                eid for eid, ent in self.entities.items() if not ent.is_operational
            ]
        }

        return {
            "ontology.json": ontology_json,
            "dependency_graph.json": dependency_graph_json,
            "invariants.json": invariants_json,
            "unknowns.json": unknowns_json
        }


def build_default_krystal_kernel_ontology() -> OntologyCompiler:
    """Pre-builds the canonical research ontology for Krystal-Stack Linux/CNC Kernel."""
    compiler = OntologyCompiler()

    # 1. CPU & Cgroups Governor
    compiler.register_entity(OntologyEntity(
        entity_id="E_CGROUPS_CPU",
        name="Cgroups v2 CPU Slice & Throttling Controller",
        description="Controls CPU quota, affinity masks, and detects unannounced kernel throttling.",
        state_variables=["cpu_usage_usec", "nr_periods", "nr_throttled"],
        observable_variables=["cpu.stat:nr_throttled", "cpu.stat:throttled_usec", "context_switches_sec"],
        hidden_variables=["thread_migration_overhead_ns", "l3_cache_line_pollution"],
        control_variables=["cpu.max", "cpu.weight", "allowed_cores_mask"],
        constraints=["cpu.weight >= 100", "nr_throttled == 0 in HighPrecision mode"],
        dependencies=[],
        invariant="nr_throttled == 0 OR Alert(DriftEvent)",
        failure_states=["SILENT_CGROUP_THROTTLING", "SCHEDULING_L2_CACHE_THRASHING"]
    ))

    # 2. Spindle & Thermal Governor
    compiler.register_entity(OntologyEntity(
        entity_id="E_SPINDLE_THERMAL",
        name="Spindle Electrical & Thermal Metabolism Governor",
        description="Regulates cutting tool current, resistance, and temperature under Loewen-Shaw physics.",
        state_variables=["winding_temperature_celsius", "effective_resistance_ohms", "phase_current_amps"],
        observable_variables=["current_sensor_amps", "thermocouple_celsius", "coolant_pressure_mpa"],
        hidden_variables=["thermal_contact_conductance", "internal_coil_hotspot_delta"],
        control_variables=["duty_cycle_pwm", "override_feed_rate_pct", "coolant_pump_state"],
        constraints=["phase_current_amps <= 10.0", "winding_temperature_celsius <= 75.0"],
        dependencies=["E_CGROUPS_CPU"],
        invariant="winding_temperature_celsius <= 75.0 AND phase_current_amps <= 10.0",
        failure_states=["THERMAL_RUNAWAY", "OHMIC_BREAKDOWN_SHORT_CIRCUIT"]
    ))

    # 3. Memory & Page Cache Governor
    compiler.register_entity(OntologyEntity(
        entity_id="E_MEMORY_SLAB",
        name="Zero-Allocation Memory & Slab Buffer",
        description="Supervises ringbuffers, avoiding page faults and anonymous swap allocations.",
        state_variables=["memory_current_bytes", "anon_bytes", "file_bytes"],
        observable_variables=["memory.current", "memory.high", "memory.swap.current", "major_page_faults"],
        hidden_variables=["fragmentation_index", "tlb_miss_penalty_cycles"],
        control_variables=["memory.high", "memory.max", "ringbuffer_prealloc_mb"],
        constraints=["memory.swap.current == 0", "memory_current_bytes <= memory.high"],
        dependencies=["E_CGROUPS_CPU"],
        invariant="memory.swap.current == 0 (Zero swap page-in allowed)",
        failure_states=["SWAP_INCURRED_LATENCY_SPIKE", "OOM_KILLER_INVOCATION"]
    ))

    # Relationships
    compiler.add_relation(OntologyRelation(
        source_entity_id="E_CGROUPS_CPU",
        target_entity_id="E_SPINDLE_THERMAL",
        relation_type="REGULATES",
        mathematical_expression="PID_update_interval_ms = f(cpu_utilization_pct, nr_throttled)",
        confidence=0.98,
        measurement_source="/proc/stat & /sys/fs/cgroup/krystal.slice/cpu.stat",
        sampling_frequency_hz=100.0,
        expected_delay_ms=2.0,
        falsification_test="Throttle CPU artificially by 50%; if thermal sampling loop drops below 50Hz, relation supported."
    ))

    compiler.add_relation(OntologyRelation(
        source_entity_id="E_MEMORY_SLAB",
        target_entity_id="E_CGROUPS_CPU",
        relation_type="CONSTRAINED_BY",
        mathematical_expression="major_faults_penalty_us = major_page_faults * 4500.0",
        confidence=0.95,
        measurement_source="/proc/vmstat",
        sampling_frequency_hz=10.0,
        expected_delay_ms=5.0,
        falsification_test="Drop pagecache; if p95 latency does not spike by >10ms within 100ms, relation falsified."
    ))

    return compiler
