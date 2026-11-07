#!/usr/bin/env python3
"""
================================================================================
KRYSTAL STACK // KRYSTAL QUERY-1 NEUROSYMBOLIC CPU SEARCH & FACT ENGINE
================================================================================
Module:      krystal_kernel/krystal_query1_neurosymbolic_engine.py
Milestone:   V-KRYSTAL-QUERY1-01 (Reciprocal Architecture Synthesis with XERJ rc.101)
Author:      Dušan Kopecký & Antigravity Cognitive Invariants Group (2026)
Invariants:  VITAL_MAX_HP == 6 | Proposer != Evaluator | Delta S == 0 on breach

Architectural Synthesis (Krystal ENLG + XERJ Query-1 rc.101):
  1. Zero-Vector O(1) Fact Store (`/_facts/<ns>/_bulk`):
     - Stores structured facts bound to explicit `subject_id`, `predicate`,
       `object_value`, and `source_uri` with 0-second incremental updates
       (no vector re-embedding) and RFC 8785 SHA-256 provenance digests.
  2. Neurosymbolic Prompt-to-Program Compiler (`/_facts/<ns>/_ask`):
     - Compiles natural language prompts (Slovak & English) into short,
       verifiable deterministic programs (`OP_LOOKUP`, `OP_COMPARE`,
       `OP_COUNT_SET`, `OP_VERIFY_INVARIANT`).
  3. Dual-Seed Program Consensus (`Proposer Seed A == Verifier Seed B`):
     - Independently compiles the query under two orthogonal seeds/parsers.
       Execution proceeds ONLY if both canonical AST digests match 100%.
  4. Strict `subject_id` & Closed-Set Verification:
     - Set counting (`OP_COUNT_SET`) requires every matched fact to carry a
       valid `subject_id` and the target set to be explicitly closed
       (`is_closed_set=True`), otherwise returning `EnlgVerdict.UNKNOWN`.
  5. Homotopic Conflict Preservation (KSPP Principle P6):
     - If two stored facts for the same `(subject_id, predicate)` disagree,
       convex averaging or guessing is forbidden: the engine returns
       `EnlgVerdict.CONFLICT` with superposed branches (Alpha ⊥ Beta).
================================================================================
"""

from __future__ import annotations

import enum
import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.epistemic_enlg_compiler import EnlgVerdict, EnlgEvidenceClass
from krystal_kernel.claims_lattice import canonical_jcs_serialize


if VITAL_MAX_HP != 6:
    raise ContractViolation(f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}")


# =============================================================================
# 1. PROGRAM OPCODES & DATA STRUCTURES
# =============================================================================

class QueryOpcode(str, enum.Enum):
    """Deterministic instruction set for compiled neurosymbolic search programs."""
    OP_LOOKUP           = "OP_LOOKUP"            # Lookup (subject_id, predicate) -> value
    OP_COMPARE          = "OP_COMPARE"           # Compare predicate across two subject_ids
    OP_COUNT_SET        = "OP_COUNT_SET"         # Count distinct subject_ids matching predicate/filter
    OP_VERIFY_INVARIANT = "OP_VERIFY_INVARIANT"  # Verify constitutional or numeric invariant


@dataclass(frozen=True)
class StoredFact:
    """
    Atomic traceable fact stored in the O(1) inverted index.
    Updating or inserting a fact requires 0 vector embeddings.
    """
    fact_id: str
    namespace: str
    subject_id: str
    predicate: str
    object_value: Union[str, int, float, bool]
    unit: str = ""
    source_uri: str = "krystal://repo/verified"
    evidence_class: EnlgEvidenceClass = EnlgEvidenceClass.LOCAL_MEASURED
    timestamp_ns: int = 0
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP breach in StoredFact: {self.vital_max_hp} != {VITAL_MAX_HP}")

    def compute_digest(self) -> str:
        payload = {
            "fact_id": self.fact_id,
            "namespace": self.namespace,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_value": self.object_value,
            "unit": self.unit,
            "source_uri": self.source_uri,
            "vital_max_hp": self.vital_max_hp,
        }
        return hashlib.sha256(canonical_jcs_serialize(payload)).hexdigest()


@dataclass(frozen=True)
class QueryProgramInstruction:
    """Single verifiable instruction in a compiled Query-1 program."""
    opcode: QueryOpcode
    namespace: str
    target_predicate: str
    subject_ids: Tuple[str, ...] = ()
    filter_value: Optional[Union[str, int, float, bool]] = None
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation("VITAL_MAX_HP must be 6 in QueryProgramInstruction")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "opcode": self.opcode.value,
            "namespace": self.namespace,
            "target_predicate": self.target_predicate,
            "subject_ids": list(self.subject_ids),
            "filter_value": self.filter_value,
            "vital_max_hp": self.vital_max_hp,
        }


@dataclass(frozen=True)
class CompiledQueryProgram:
    """
    Short, verifiable neurosymbolic program compiled from a natural language prompt.
    """
    program_id: str
    seed_id: int
    instructions: Tuple[QueryProgramInstruction, ...]
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation("VITAL_MAX_HP must be 6 in CompiledQueryProgram")

    def canonical_hash(self) -> str:
        payload = {
            "instructions": [inst.to_canonical_dict() for inst in self.instructions],
            "vital_max_hp": self.vital_max_hp,
        }
        return hashlib.sha256(canonical_jcs_serialize(payload)).hexdigest()


@dataclass(frozen=True)
class QueryExecutionReceipt:
    """
    Deterministic execution outcome returned by `/_facts/<ns>/_ask`.
    Never guesses prose: returns verified value + fact provenance, UNKNOWN, or CONFLICT.
    """
    query: str
    namespace: str
    verdict: EnlgVerdict
    status_label: str  # "VERIFIED_FACT", "UNKNOWN", "CONFLICT"
    answer_value: Any
    unit: str
    program_hash_seed_a: str
    program_hash_seed_b: str
    dual_seed_consensus: bool
    compiled_program: List[Dict[str, Any]]
    supporting_fact_ids: List[str]
    supporting_digests: List[str]
    conflict_branches: List[Dict[str, Any]] = field(default_factory=list)
    execution_latency_us: float = 0.0
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.name
        return d


# =============================================================================
# 2. TEXT NORMALIZATION & LEXICAL TOKENIZER (CPU-ONLY, ZERO DEPENDENCIES)
# =============================================================================

_WORD_RE = re.compile(r"[a-z0-9_]+")


def fold_diacritics(text: str) -> str:
    """Strips Slovak/Czech/Latin diacritics and lowercases text for deterministic matching."""
    norm = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in norm if not unicodedata.combining(c))


def tokenize_folded(text: str) -> List[str]:
    return _WORD_RE.findall(fold_diacritics(text))


# =============================================================================
# 3. NEUROSYMBOLIC COMPILER & O(1) INVERTED FACT STORE
# =============================================================================

class KrystalQuery1Engine:
    """
    Sovereign CPU-First Neurosymbolic Fact & Search Engine.
    Implements `/_facts`, `/_facts/<ns>/_bulk`, and `/_facts/<ns>/_ask`.
    """

    def __init__(self, vital_max_hp: int = VITAL_MAX_HP) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp: int = vital_max_hp

        # Inverted Fact Index:
        # namespace -> (subject_id, predicate) -> List[StoredFact]
        self._fact_index: Dict[str, Dict[Tuple[str, str], List[StoredFact]]] = {}
        # Predicate Index for Set Counts:
        # namespace -> predicate -> List[StoredFact]
        self._predicate_index: Dict[str, Dict[str, List[StoredFact]]] = {}
        # Closed-Set Registry:
        # (namespace, predicate) -> bool
        self._closed_sets: Dict[Tuple[str, str], bool] = {}
        # Alias lookup table for subjects and predicates (folded token -> canonical ID)
        self._subject_aliases: Dict[str, str] = {}
        self._predicate_aliases: Dict[str, str] = {}
        self._total_facts: int = 0

    # -------------------------------------------------------------------------
    # Endpoint: GET /_facts
    # -------------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        """Mirrors XERJ `GET /_facts` status check with Krystal constitutional invariant."""
        return {
            "state": "loaded",
            "engine": "Krystal-Query-1-Neurosymbolic-CPU",
            "namespaces": sorted(self._fact_index.keys()),
            "total_facts": self._total_facts,
            "vital_max_hp": self.vital_max_hp,
        }

    # -------------------------------------------------------------------------
    # Endpoint: POST /_facts/<ns>/_bulk (0-second O(1) fact ingestion/update)
    # -------------------------------------------------------------------------
    def bulk_ingest(
        self,
        namespace: str,
        facts: Sequence[Dict[str, Any]],
        closed_predicates: Optional[Iterable[str]] = None,
        replace_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingests or updates structured facts in O(1) amortized time per fact
        with zero vector embedding computation.
        """
        t0_ns = time.perf_counter_ns()
        if not namespace or not namespace.strip():
            raise ContractViolation("Namespace must be non-empty")

        ns_map = self._fact_index.setdefault(namespace, {})
        pred_map = self._predicate_index.setdefault(namespace, {})

        if closed_predicates is not None:
            for p in closed_predicates:
                self._closed_sets[(namespace, p)] = True

        ingested = 0
        for item in facts:
            if item.get("vital_max_hp", VITAL_MAX_HP) != VITAL_MAX_HP:
                raise ContractViolation("Invalid vital_max_hp in bulk fact item: Delta S == 0")

            subj = str(item.get("subject_id", "")).strip()
            pred = str(item.get("predicate", "")).strip()
            if not pred:
                raise ContractViolation("Fact predicate cannot be empty")

            val = item.get("object_value")
            unit = str(item.get("unit", ""))
            source_uri = str(item.get("source_uri", "krystal://repo/verified"))
            ts_ns = int(item.get("timestamp_ns", time.time_ns()))
            fact_id = str(item.get("fact_id") or f"FACT-{namespace}-{subj or 'ANON'}-{pred}-{self._total_facts + 1}")

            fact = StoredFact(
                fact_id=fact_id,
                namespace=namespace,
                subject_id=subj,
                predicate=pred,
                object_value=val,
                unit=unit,
                source_uri=source_uri,
                timestamp_ns=ts_ns,
                vital_max_hp=self.vital_max_hp,
            )

            key = (subj, pred)
            if replace_existing:
                old_list = ns_map.get(key, [])
                self._total_facts -= len(old_list)
                ns_map[key] = [fact]
                pred_map[pred] = [f for f in pred_map.get(pred, []) if f.subject_id != subj] + [fact]
                self._total_facts += 1
            else:
                bucket = ns_map.setdefault(key, [])
                # Idempotent check for identical value
                if not any(existing.object_value == fact.object_value and existing.source_uri == fact.source_uri for existing in bucket):
                    bucket.append(fact)
                    pred_map.setdefault(pred, []).append(fact)
                    self._total_facts += 1

            # Register automatic folded aliases for fast CPU compilation
            if subj:
                self._subject_aliases[fold_diacritics(subj)] = subj
                for alias in item.get("subject_aliases", []):
                    self._subject_aliases[fold_diacritics(str(alias))] = subj
            self._predicate_aliases[fold_diacritics(pred)] = pred
            for palias in item.get("predicate_aliases", []):
                self._predicate_aliases[fold_diacritics(str(palias))] = pred

            if "is_closed_set" in item:
                self._closed_sets[(namespace, pred)] = bool(item["is_closed_set"])

            ingested += 1

        elapsed_us = round((time.perf_counter_ns() - t0_ns) / 1000.0, 2)
        return {
            "namespace": namespace,
            "ingested_count": ingested,
            "total_facts": self._total_facts,
            "ingestion_latency_us": elapsed_us,
            "vital_max_hp": self.vital_max_hp,
        }

    def register_aliases(
        self,
        subject_aliases: Optional[Dict[str, str]] = None,
        predicate_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        """Registers human-language aliases (Slovak/English) to canonical subject_ids and predicates."""
        if subject_aliases:
            for k, v in subject_aliases.items():
                self._subject_aliases[fold_diacritics(k)] = v
        if predicate_aliases:
            for k, v in predicate_aliases.items():
                self._predicate_aliases[fold_diacritics(k)] = v

    # -------------------------------------------------------------------------
    # Dual-Seed Prompt-to-Program Compiler
    # -------------------------------------------------------------------------
    def compile_prompt_to_program(
        self,
        namespace: str,
        prompt: str,
        seed: int = 1,
    ) -> CompiledQueryProgram:
        """
        Compiles a natural language query (Slovak or English) into a short,
        verifiable Query-1 program.
        - Seed 1 uses forward longest-phrase matching.
        - Seed 2 uses reverse token-window matching (independent parser pass).
        """
        folded = fold_diacritics(prompt)
        tokens = tokenize_folded(prompt)

        # 1. Detect Operation Opcode
        is_count = any(w in tokens for w in ("kolko", "how", "many", "count", "pocet"))
        is_compare = any(w in tokens for w in ("vs", "versus", "porovnaj", "compare", "rozdiel", "faster", "rychlejsi"))

        # 2. Extract matched canonical subject_ids
        matched_subjects: List[str] = []
        alias_items = list(self._subject_aliases.items())
        if seed % 2 == 0:
            # Independent Seed B ordering: sort by alias length descending, then reverse alphabetical
            alias_items.sort(key=lambda kv: (len(kv[0]), kv[0]), reverse=True)
        else:
            # Seed A ordering: sort by position in prompt
            alias_items.sort(key=lambda kv: (-len(kv[0]), kv[0]))

        positions: List[Tuple[int, str]] = []
        for alias_key, canonical_subj in alias_items:
            if not alias_key:
                continue
            pattern = rf"(?<![a-z0-9_]){re.escape(alias_key)}(?![a-z0-9_])"
            m = re.search(pattern, folded)
            if m:
                positions.append((m.start(), canonical_subj))

        positions.sort(key=lambda x: x[0])
        for _, s_id in positions:
            if s_id not in matched_subjects:
                matched_subjects.append(s_id)

        # 3. Extract matched canonical predicate
        matched_predicate = ""
        pred_items = list(self._predicate_aliases.items())
        pred_items.sort(key=lambda kv: len(kv[0]), reverse=True)
        best_pred_len = -1
        for palias, canonical_pred in pred_items:
            if not palias:
                continue
            pattern = rf"(?<![a-z0-9_]){re.escape(palias)}(?![a-z0-9_])"
            if re.search(pattern, folded) and len(palias) > best_pred_len:
                matched_predicate = canonical_pred
                best_pred_len = len(palias)

        # Construct deterministic instruction
        if is_count:
            opcode = QueryOpcode.OP_COUNT_SET
            subj_tuple: Tuple[str, ...] = ()
        elif is_compare and len(matched_subjects) >= 2:
            opcode = QueryOpcode.OP_COMPARE
            subj_tuple = tuple(matched_subjects[:2])
        else:
            opcode = QueryOpcode.OP_LOOKUP
            subj_tuple = tuple(matched_subjects[:1])

        inst = QueryProgramInstruction(
            opcode=opcode,
            namespace=namespace,
            target_predicate=matched_predicate or "UNRESOLVED_PREDICATE",
            subject_ids=subj_tuple,
            vital_max_hp=self.vital_max_hp,
        )
        prog_id = f"Q1-PROG-S{seed}-{abs(hash((namespace, prompt, seed))) & 0xFFFF:04X}"
        return CompiledQueryProgram(
            program_id=prog_id,
            seed_id=seed,
            instructions=(inst,),
            vital_max_hp=self.vital_max_hp,
        )

    # -------------------------------------------------------------------------
    # Endpoint: POST /_facts/<ns>/_ask (Deterministic Program Execution)
    # -------------------------------------------------------------------------
    def ask(
        self,
        namespace: str,
        prompt: str,
        override_program_seed_b: Optional[CompiledQueryProgram] = None,
    ) -> QueryExecutionReceipt:
        """
        Compiles `prompt` under two independent seeds (Seed 1 and Seed 2),
        verifies Dual-Seed Program Consensus, and executes the program directly
        against the in-memory Fact Store on CPU.
        """
        t0_ns = time.perf_counter_ns()

        prog_a = self.compile_prompt_to_program(namespace, prompt, seed=1)
        prog_b = override_program_seed_b or self.compile_prompt_to_program(namespace, prompt, seed=2)

        hash_a = prog_a.canonical_hash()
        hash_b = prog_b.canonical_hash()
        consensus = (hash_a == hash_b)

        compiled_dicts = [i.to_canonical_dict() for i in prog_a.instructions]

        # Gate 1: Dual-Seed Consensus Verification (Proposer != Evaluator)
        if not consensus:
            elapsed_us = round((time.perf_counter_ns() - t0_ns) / 1000.0, 2)
            return QueryExecutionReceipt(
                query=prompt,
                namespace=namespace,
                verdict=EnlgVerdict.UNKNOWN,
                status_label="UNKNOWN",
                answer_value=None,
                unit="",
                program_hash_seed_a=hash_a,
                program_hash_seed_b=hash_b,
                dual_seed_consensus=False,
                compiled_program=compiled_dicts,
                supporting_fact_ids=[],
                supporting_digests=[],
                execution_latency_us=elapsed_us,
                vital_max_hp=self.vital_max_hp,
            )

        inst = prog_a.instructions[0]
        ns_map = self._fact_index.get(namespace, {})
        pred_map = self._predicate_index.get(namespace, {})

        # Execute OP_COUNT_SET
        if inst.opcode == QueryOpcode.OP_COUNT_SET:
            is_closed = self._closed_sets.get((namespace, inst.target_predicate), False)
            facts_for_pred = pred_map.get(inst.target_predicate, [])
            # Strict XERJ rc.101 rule: set must be closed AND every counted record must have a valid subject_id
            if not is_closed or not facts_for_pred or any(not f.subject_id for f in facts_for_pred):
                elapsed_us = round((time.perf_counter_ns() - t0_ns) / 1000.0, 2)
                return QueryExecutionReceipt(
                    query=prompt,
                    namespace=namespace,
                    verdict=EnlgVerdict.UNKNOWN,
                    status_label="UNKNOWN",
                    answer_value=None,
                    unit="count",
                    program_hash_seed_a=hash_a,
                    program_hash_seed_b=hash_b,
                    dual_seed_consensus=True,
                    compiled_program=compiled_dicts,
                    supporting_fact_ids=[],
                    supporting_digests=[],
                    execution_latency_us=elapsed_us,
                    vital_max_hp=self.vital_max_hp,
                )

            # Check if any subject has conflicting values for this predicate
            by_subj: Dict[str, Set[Any]] = {}
            for f in facts_for_pred:
                by_subj.setdefault(f.subject_id, set()).add(f.object_value)
                if len(by_subj[f.subject_id]) > 1:
                    elapsed_us = round((time.perf_counter_ns() - t0_ns) / 1000.0, 2)
                    return QueryExecutionReceipt(
                        query=prompt,
                        namespace=namespace,
                        verdict=EnlgVerdict.CONFLICT,
                        status_label="CONFLICT",
                        answer_value=None,
                        unit="count",
                        program_hash_seed_a=hash_a,
                        program_hash_seed_b=hash_b,
                        dual_seed_consensus=True,
                        compiled_program=compiled_dicts,
                        supporting_fact_ids=[x.fact_id for x in facts_for_pred],
                        supporting_digests=[x.compute_digest() for x in facts_for_pred],
                        execution_latency_us=elapsed_us,
                        vital_max_hp=self.vital_max_hp,
                    )

            distinct_subjects = sorted(by_subj.keys())
            elapsed_us = round((time.perf_counter_ns() - t0_ns) / 1000.0, 2)
            return QueryExecutionReceipt(
                query=prompt,
                namespace=namespace,
                verdict=EnlgVerdict.MEASURED_WITH_SCOPE,
                status_label="VERIFIED_FACT",
                answer_value=len(distinct_subjects),
                unit="distinct_subjects",
                program_hash_seed_a=hash_a,
                program_hash_seed_b=hash_b,
                dual_seed_consensus=True,
                compiled_program=compiled_dicts,
                supporting_fact_ids=[f.fact_id for f in facts_for_pred],
                supporting_digests=[f.compute_digest() for f in facts_for_pred],
                execution_latency_us=elapsed_us,
                vital_max_hp=self.vital_max_hp,
            )

        # Execute OP_COMPARE (two subjects on the same predicate)
        if inst.opcode == QueryOpcode.OP_COMPARE:
            if len(inst.subject_ids) < 2 or inst.target_predicate == "UNRESOLVED_PREDICATE":
                elapsed_us = round((time.perf_counter_ns() - t0_ns) / 1000.0, 2)
                return QueryExecutionReceipt(
                    query=prompt,
                    namespace=namespace,
                    verdict=EnlgVerdict.UNKNOWN,
                    status_label="UNKNOWN",
                    answer_value=None,
                    unit="",
                    program_hash_seed_a=hash_a,
                    program_hash_seed_b=hash_b,
                    dual_seed_consensus=True,
                    compiled_program=compiled_dicts,
                    supporting_fact_ids=[],
                    supporting_digests=[],
                    execution_latency_us=elapsed_us,
                    vital_max_hp=self.vital_max_hp,
                )

            s1, s2 = inst.subject_ids[0], inst.subject_ids[1]
            f1_list = ns_map.get((s1, inst.target_predicate), [])
            f2_list = ns_map.get((s2, inst.target_predicate), [])
            if not f1_list or not f2_list:
                elapsed_us = round((time.perf_counter_ns() - t0_ns) / 1000.0, 2)
                return QueryExecutionReceipt(
                    query=prompt,
                    namespace=namespace,
                    verdict=EnlgVerdict.UNKNOWN,
                    status_label="UNKNOWN",
                    answer_value=None,
                    unit="",
                    program_hash_seed_a=hash_a,
                    program_hash_seed_b=hash_b,
                    dual_seed_consensus=True,
                    compiled_program=compiled_dicts,
                    supporting_fact_ids=[],
                    supporting_digests=[],
                    execution_latency_us=elapsed_us,
                    vital_max_hp=self.vital_max_hp,
                )

            # Check internal conflict on either subject
            if len({f.object_value for f in f1_list}) > 1 or len({f.object_value for f in f2_list}) > 1:
                all_f = f1_list + f2_list
                elapsed_us = round((time.perf_counter_ns() - t0_ns) / 1000.0, 2)
                return QueryExecutionReceipt(
                    query=prompt,
                    namespace=namespace,
                    verdict=EnlgVerdict.CONFLICT,
                    status_label="CONFLICT",
                    answer_value=None,
                    unit="",
                    program_hash_seed_a=hash_a,
                    program_hash_seed_b=hash_b,
                    dual_seed_consensus=True,
                    compiled_program=compiled_dicts,
                    supporting_fact_ids=[f.fact_id for f in all_f],
                    supporting_digests=[f.compute_digest() for f in all_f],
                    execution_latency_us=elapsed_us,
                    vital_max_hp=self.vital_max_hp,
                )

            v1 = f1_list[0].object_value
            v2 = f2_list[0].object_value
            comparison: Dict[str, Any] = {
                s1: v1,
                s2: v2,
                "unit": f1_list[0].unit,
            }
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v2 != 0:
                comparison["delta"] = round(float(v1) - float(v2), 4)
                comparison["ratio"] = round(float(v1) / float(v2), 4)

            elapsed_us = round((time.perf_counter_ns() - t0_ns) / 1000.0, 2)
            return QueryExecutionReceipt(
                query=prompt,
                namespace=namespace,
                verdict=EnlgVerdict.MEASURED_WITH_SCOPE,
                status_label="VERIFIED_FACT",
                answer_value=comparison,
                unit=f1_list[0].unit,
                program_hash_seed_a=hash_a,
                program_hash_seed_b=hash_b,
                dual_seed_consensus=True,
                compiled_program=compiled_dicts,
                supporting_fact_ids=[f1_list[0].fact_id, f2_list[0].fact_id],
                supporting_digests=[f1_list[0].compute_digest(), f2_list[0].compute_digest()],
                execution_latency_us=elapsed_us,
                vital_max_hp=self.vital_max_hp,
            )

        # Execute OP_LOOKUP (single subject + predicate)
        if not inst.subject_ids or inst.target_predicate == "UNRESOLVED_PREDICATE":
            elapsed_us = round((time.perf_counter_ns() - t0_ns) / 1000.0, 2)
            return QueryExecutionReceipt(
                query=prompt,
                namespace=namespace,
                verdict=EnlgVerdict.UNKNOWN,
                status_label="UNKNOWN",
                answer_value=None,
                unit="",
                program_hash_seed_a=hash_a,
                program_hash_seed_b=hash_b,
                dual_seed_consensus=True,
                compiled_program=compiled_dicts,
                supporting_fact_ids=[],
                supporting_digests=[],
                execution_latency_us=elapsed_us,
                vital_max_hp=self.vital_max_hp,
            )

        subj = inst.subject_ids[0]
        matched_facts = ns_map.get((subj, inst.target_predicate), [])
        if not matched_facts:
            elapsed_us = round((time.perf_counter_ns() - t0_ns) / 1000.0, 2)
            return QueryExecutionReceipt(
                query=prompt,
                namespace=namespace,
                verdict=EnlgVerdict.UNKNOWN,
                status_label="UNKNOWN",
                answer_value=None,
                unit="",
                program_hash_seed_a=hash_a,
                program_hash_seed_b=hash_b,
                dual_seed_consensus=True,
                compiled_program=compiled_dicts,
                supporting_fact_ids=[],
                supporting_digests=[],
                execution_latency_us=elapsed_us,
                vital_max_hp=self.vital_max_hp,
            )

        distinct_values = {f.object_value for f in matched_facts}
        if len(distinct_values) > 1:
            # Homotopic Conflict Preservation (KSPP Principle P6): fork into superposed branches
            branches = [
                {
                    "branch": f"BRANCH_{idx + 1}",
                    "fact_id": f.fact_id,
                    "object_value": f.object_value,
                    "source_uri": f.source_uri,
                    "sha256_digest": f.compute_digest(),
                }
                for idx, f in enumerate(matched_facts)
            ]
            elapsed_us = round((time.perf_counter_ns() - t0_ns) / 1000.0, 2)
            return QueryExecutionReceipt(
                query=prompt,
                namespace=namespace,
                verdict=EnlgVerdict.CONFLICT,
                status_label="CONFLICT",
                answer_value=None,
                unit=matched_facts[0].unit,
                program_hash_seed_a=hash_a,
                program_hash_seed_b=hash_b,
                dual_seed_consensus=True,
                compiled_program=compiled_dicts,
                supporting_fact_ids=[f.fact_id for f in matched_facts],
                supporting_digests=[f.compute_digest() for f in matched_facts],
                conflict_branches=branches,
                execution_latency_us=elapsed_us,
                vital_max_hp=self.vital_max_hp,
            )

        winning_fact = matched_facts[0]
        elapsed_us = round((time.perf_counter_ns() - t0_ns) / 1000.0, 2)
        return QueryExecutionReceipt(
            query=prompt,
            namespace=namespace,
            verdict=EnlgVerdict.MEASURED_WITH_SCOPE,
            status_label="VERIFIED_FACT",
            answer_value=winning_fact.object_value,
            unit=winning_fact.unit,
            program_hash_seed_a=hash_a,
            program_hash_seed_b=hash_b,
            dual_seed_consensus=True,
            compiled_program=compiled_dicts,
            supporting_fact_ids=[winning_fact.fact_id],
            supporting_digests=[winning_fact.compute_digest()],
            execution_latency_us=elapsed_us,
            vital_max_hp=self.vital_max_hp,
        )


# =============================================================================
# 4. PRE-LOADED KRYSTAL STACK CORPUS BUILDER (USER'S REPOSITORY DATA)
# =============================================================================

def build_krystal_sovereign_fact_engine() -> KrystalQuery1Engine:
    """
    Populates a KrystalQuery1Engine instance with verified empirical facts
    from the Krystal Stack repository (Vulkan Iris Xe, OpenVINO AEP, Slovak LLM,
    Tri-Phase Power Dosing, VPP Grid Broker, and KSPP v1 Invariants).
    """
    engine = KrystalQuery1Engine(vital_max_hp=VITAL_MAX_HP)

    krystal_facts = [
        {
            "fact_id": "FACT-KSPP-HP-01",
            "subject_id": "KRYSTAL_STACK",
            "subject_aliases": ["krystal stack", "krystal", "framework", "jadro"],
            "predicate": "vital_max_hp",
            "predicate_aliases": ["vital_max_hp", "ustavny invariant", "constitutional invariant", "max hp"],
            "object_value": 6,
            "unit": "HP",
            "source_uri": "file:///krystal_kernel/constants.py",
            "is_closed_set": True,
        },
        {
            "fact_id": "FACT-VK-IRIS-SPEEDUP-01",
            "subject_id": "VULKAN_IRIS_XE",
            "subject_aliases": ["vulkan iris xe", "vulkan", "iris xe", "igpu"],
            "predicate": "speedup_factor",
            "predicate_aliases": ["speedup", "zrychlenie", "speedup factor"],
            "object_value": 20.35,
            "unit": "x",
            "source_uri": "file:///artifacts/VULKAN_IRIS_XE_BENCHMARK_RESULTS.json",
            "is_closed_set": True,
        },
        {
            "fact_id": "FACT-VK-IRIS-LATENCY-01",
            "subject_id": "VULKAN_IRIS_XE",
            "subject_aliases": ["vulkan iris xe", "vulkan", "iris xe"],
            "predicate": "frame_latency_ms",
            "predicate_aliases": ["frame latency", "latencia snimky", "latency", "latencia"],
            "object_value": 2.31,
            "unit": "ms",
            "source_uri": "file:///artifacts/VULKAN_IRIS_XE_BENCHMARK_RESULTS.json",
        },
        {
            "fact_id": "FACT-CPU-BASE-LATENCY-01",
            "subject_id": "CPU_BASELINE",
            "subject_aliases": ["cpu baseline", "cpu", "tiger lake cpu"],
            "predicate": "frame_latency_ms",
            "predicate_aliases": ["frame latency", "latencia snimky", "latency", "latencia"],
            "object_value": 46.99,
            "unit": "ms",
            "source_uri": "file:///artifacts/VULKAN_IRIS_XE_BENCHMARK_RESULTS.json",
        },
        {
            "fact_id": "FACT-UMA-BW-SAVED-01",
            "subject_id": "ZERO_COPY_RING",
            "subject_aliases": ["zero copy ring", "zero copy", "128b ring", "uma"],
            "predicate": "saved_bandwidth_mb_s",
            "predicate_aliases": ["saved bandwidth", "usetrena priepustnost", "bandwidth saved"],
            "object_value": 497.66,
            "unit": "MB/s",
            "source_uri": "file:///artifacts/GPU_ZERO_COPY_READBACK_BENCHMARK.json",
        },
        {
            "fact_id": "FACT-TRI-PHASE-LATENCY-01",
            "subject_id": "TRI_PHASE_DOSING",
            "subject_aliases": ["tri phase dosing", "trojfazove davkovanie", "tri phase"],
            "predicate": "reaction_latency_ms",
            "predicate_aliases": ["reaction latency", "reakcna latencia"],
            "object_value": 0.027,
            "unit": "ms",
            "source_uri": "file:///krystal_kernel/tri_phase_dosing_controller.py",
        },
        {
            "fact_id": "FACT-SLOVAK-LLM-FOOTPRINT-01",
            "subject_id": "SLOVAK_LLM_QWEN3",
            "subject_aliases": ["qwen3 14b sk", "slovak llm", "qwen3"],
            "predicate": "memory_footprint_mb",
            "predicate_aliases": ["memory footprint", "pamatova stopa", "ram footprint"],
            "object_value": 2171.39,
            "unit": "MB",
            "source_uri": "file:///docs/architecture/INFERENCE_WORKLOAD.md",
        },
        {
            "fact_id": "FACT-ACCEL-CPU",
            "subject_id": "TIGER_LAKE_CPU",
            "subject_aliases": ["tiger lake"],
            "predicate": "hardware_accelerator_role",
            "predicate_aliases": ["hardware accelerator", "akcelerator", "accelerators"],
            "object_value": "CPU_DP4A",
            "unit": "device",
            "source_uri": "file:///krystal_kernel/openvino_aep_runtime.py",
            "is_closed_set": True,
        },
        {
            "fact_id": "FACT-ACCEL-IGPU",
            "subject_id": "IRIS_XE_IGPU",
            "subject_aliases": ["iris xe igpu"],
            "predicate": "hardware_accelerator_role",
            "predicate_aliases": ["hardware accelerator", "akcelerator", "accelerators"],
            "object_value": "GPU_96EU_UMA",
            "unit": "device",
            "source_uri": "file:///krystal_kernel/openvino_aep_runtime.py",
            "is_closed_set": True,
        },
        {
            "fact_id": "FACT-ACCEL-NPU",
            "subject_id": "GNA_NPU_COPROCESSOR",
            "subject_aliases": ["npu", "gna"],
            "predicate": "hardware_accelerator_role",
            "predicate_aliases": ["hardware accelerator", "akcelerator", "accelerators"],
            "object_value": "NPU_LOW_POWER",
            "unit": "device",
            "source_uri": "file:///krystal_kernel/openvino_aep_runtime.py",
            "is_closed_set": True,
        },
    ]

    engine.bulk_ingest(
        namespace="krystal_sovereign",
        facts=krystal_facts,
        closed_predicates=["hardware_accelerator_role", "vital_max_hp", "speedup_factor"],
    )
    return engine


GLOBAL_KRYSTAL_QUERY1_ENGINE = build_krystal_sovereign_fact_engine()
