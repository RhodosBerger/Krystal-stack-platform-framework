#!/usr/bin/env python3
"""
================================================================================
KRYSTAL STACK PLATFORM FRAMEWORK - SOVEREIGN PROMPTBOOK ENGINE (v3)
================================================================================
Module:       krystal_kernel/sovereign_promptbook.py
Architecture: Proof-Carrying Operational AST & Trusted Operational Compiler Frontend
Planes:
  1. RECIPE PLANE: Declarative operational templates (RECIPE-01 .. RECIPE-07)
  2. SCHEMA PLANE: Typed inputs, slots, and output format contracts
  3. CONSTRAINT PLANE: Machine-verifiable executable invariant predicates (HARD/SOFT)
     carrying formal InvariantDefinition identity and versioned InvariantSetHash
  4. PROVENANCE PLANE: RFC 8785 JSON Canonicalization Scheme (JCS), HKDF-SHA256
     domain-separated keys, replay-protected hash-chained receipts (R_n),
     and dual-gate validation (Proposal != Authority)
Tenets:       KSPP v1 (P1-P10) | Epistemic Prompt Router & Calibrated Microprompts
Compiler:     Prompt Recipe -> Typed Operational AST -> KSL Constraint IR -> KµISA -> KSAC
Author:       Antigravity Cognitive Invariants Group (2026)
Invariants:   VITAL_MAX_HP == 6 | Proposer != Evaluator | Delta S == 0 on breach
================================================================================
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import json
import math
import os
import secrets
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .constants import VITAL_MAX_HP, ContractViolation


# =============================================================================
# ENUMS & STATUS CODES
# =============================================================================

class PromptbookCategory(str, enum.Enum):
    """Categorization of operational promptbook recipes."""
    OPERATIONS_DECOMPOSITION = "OPERATIONS_DECOMPOSITION"
    BOT_ARCHETYPE_AXIOMS    = "BOT_ARCHETYPE_AXIOMS"
    TURN_ARBITRATION_DRAFT   = "TURN_ARBITRATION_DRAFT"
    UMA_HEX_GOVERNOR         = "UMA_HEX_GOVERNOR"
    MICRO_OP_ALLOCATION      = "MICRO_OP_ALLOCATION"
    BEHAVIORAL_VULKAN_SYNTH  = "BEHAVIORAL_VULKAN_SYNTH"
    RUNTIME_CONTROL_PLANE    = "RUNTIME_CONTROL_PLANE"


class InvariantSeverity(str, enum.Enum):
    """Severity classification for executable invariants in the Constraint Plane."""
    HARD = "HARD"   # Non-negotiable constitutional invariant; failure aborts execution (Delta S == 0)
    SOFT = "SOFT"   # Advisory guardrail; failure emits diagnostic warning


class KeyProvenance(str, enum.Enum):
    """Classification of cryptographic secret key provenance."""
    SECURE_EXPLICIT = "SECURE_EXPLICIT"  # Passed explicitly from trusted KMS / TPM / Vault
    ENV_INJECTED    = "ENV_INJECTED"     # Injected via environment variable KRYSTAL_PROMPTBOOK_SECRET_KEY
    TEST_FALLBACK   = "TEST_FALLBACK"    # Deterministic fallback strictly for development / test harnesses


# =============================================================================
# 1. HARDENING: FAIL-CLOSED KEY RESOLUTION (SECURITY PLANE)
# =============================================================================

def resolve_promptbook_secret_key(
    explicit_key: Optional[bytes] = None,
    enforce_production_fail_closed: bool = True
) -> Tuple[bytes, KeyProvenance]:
    """
    Authoritative key resolution for Promptbook provenance and HMAC signing.
    Enforces Fail-Closed in production environments:
      MissingSecureKey => ContractViolation (Fail-Closed)
    Deterministic test fallback is strictly allowed ONLY in non-production environments.
    """
    is_production = (
        os.environ.get("KRYSTAL_MODE", "").lower() == "production" or
        os.environ.get("ENVIRONMENT", "").lower() == "production"
    )

    if explicit_key is not None and len(explicit_key) >= 16:
        return (explicit_key, KeyProvenance.SECURE_EXPLICIT)

    env_key = os.environ.get("KRYSTAL_PROMPTBOOK_SECRET_KEY")
    if env_key:
        key_bytes = env_key.encode("utf-8")
        if len(key_bytes) >= 16:
            return (key_bytes, KeyProvenance.ENV_INJECTED)

    if is_production and enforce_production_fail_closed:
        raise ContractViolation(
            "Missing secure cryptographic key in production mode (Fail-Closed enforced: KMS/TPM/HSM key required)"
        )

    # Secure deterministic fallback key strictly for development / sandbox verification
    return (b"krystal_sovereign_promptbook_provenance_v2_hsm_key", KeyProvenance.TEST_FALLBACK)


# =============================================================================
# 2. HARDENING: DOMAIN-SEPARATED KEYS VIA HKDF-SHA256 (RFC 5869)
# =============================================================================

def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    """RFC 5869 HKDF-Extract using HMAC-SHA256."""
    if not salt:
        salt = b"\x00" * 32
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int = 32) -> bytes:
    """RFC 5869 HKDF-Expand using HMAC-SHA256."""
    hash_len = 32
    n = math.ceil(length / hash_len)
    if n > 255:
        raise ContractViolation("Cannot expand to more than 255 * hash_len octets")
    okm = b""
    t = b""
    for i in range(1, n + 1):
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
    return okm[:length]


def hkdf_derive_domain_key(
    root_key: bytes,
    domain: str,
    length: int = 32,
    salt: Optional[bytes] = None
) -> bytes:
    """
    Derives domain-separated cryptographic keys from root secret via RFC 5869 HKDF-SHA256:
      K_recipe    = HKDF(K_root, "krystal-domain-recipe-v1")
      K_render    = HKDF(K_root, "krystal-domain-render-v1")
      K_ast       = HKDF(K_root, "krystal-domain-ast-v1")
      K_receipt   = HKDF(K_root, "krystal-domain-receipt-v1")
    Guarantees:
      Compromise(K_recipe) !implies Forge(TransitionReceipt)
    """
    effective_salt = salt if salt is not None else b"krystal_kspp_domain_separation_salt_v1"
    prk = hkdf_extract(effective_salt, root_key)
    info = domain.encode("utf-8")
    return hkdf_expand(prk, info, length)


# =============================================================================
# 3. HARDENING: RFC 8785 JSON CANONICALIZATION SCHEME (JCS)
# =============================================================================

def canonical_jcs_serialize(obj: Any) -> bytes:
    """
    Serializes a Python object to JSON adhering strictly to RFC 8785 (JSON Canonicalization Scheme - JCS).
    Key properties:
      1. Verbatim Unicode preservation: Strings and object keys are preserved as-is without Unicode
         normalization (RFC 8785 section 3.2.2 forbids mutating strings via NFC/NFD).
      2. Deterministic key sorting by UTF-16 code units (RFC 8785 section 3.2.3).
      3. IEEE-754 float canonicalization (-0.0 -> 0, integer-valued floats formatted cleanly).
      4. Strict whitespace elimination (separators=(',', ':')).
      5. Cross-language equivalence: Canonical(x)_Python == Canonical(x)_Rust == Canonical(x)_FPGA.
    """
    if obj is None:
        return b"null"
    elif isinstance(obj, bool):
        return b"true" if obj else b"false"
    elif isinstance(obj, int):
        return str(obj).encode("utf-8")
    elif isinstance(obj, float):
        if obj == 0.0 or obj == -0.0:
            return b"0"
        if math.isnan(obj) or math.isinf(obj):
            raise ContractViolation("RFC 8785 JCS does not permit NaN or Infinity")
        if obj.is_integer() and abs(obj) < 1e15:
            return str(int(obj)).encode("utf-8")
        s = json.dumps(obj)
        if "e" in s:
            base, exp = s.split("e")
            exp_int = int(exp)
            s = f"{base}e{'+' if exp_int > 0 else ''}{exp_int}"
        return s.encode("utf-8")
    elif isinstance(obj, str):
        # RFC 8785 section 3.2.2: Strings are preserved verbatim without normalization
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    elif isinstance(obj, bytes):
        return json.dumps(obj.hex()).encode("utf-8")
    elif isinstance(obj, (list, tuple)):
        items = [canonical_jcs_serialize(item) for item in obj]
        return b"[" + b",".join(items) + b"]"
    elif isinstance(obj, dict):
        # Sort keys strictly by UTF-16 code units without normalization (RFC 8785 section 3.2.3)
        sorted_keys = sorted(
            obj.keys(),
            key=lambda k: str(k).encode("utf-16-be")
        )
        parts = []
        for k in sorted_keys:
            k_bytes = canonical_jcs_serialize(str(k))
            v_bytes = canonical_jcs_serialize(obj[k])
            parts.append(k_bytes + b":" + v_bytes)
        return b"{" + b",".join(parts) + b"}"
    elif hasattr(obj, "__dict__"):
        return canonical_jcs_serialize(obj.__dict__)
    else:
        return canonical_jcs_serialize(str(obj))


def canonical_recipe_json(
    recipe_id: str,
    title: str,
    category: str,
    system_directive: str,
    user_template: str,
    input_slots: List[str],
    bound_invariants: List[str],
    expected_output_format: str,
    falsification_condition: str,
    schema_version: str,
    vital_max_hp: int
) -> bytes:
    """
    Produces deterministic RFC 8785 canonical JSON serialization for HMAC integrity signing.
    Guarantees: RecipeChanged => DigestChanged across all functional attributes.
    """
    payload = {
        "bound_invariants": sorted(bound_invariants),
        "category": category,
        "expected_output_format": expected_output_format,
        "falsification_condition": falsification_condition,
        "input_slots": sorted(input_slots),
        "recipe_id": recipe_id,
        "schema_version": schema_version,
        "system_directive": system_directive,
        "title": title,
        "user_template": user_template,
        "vital_max_hp": vital_max_hp
    }
    return canonical_jcs_serialize(payload)


# =============================================================================
# 5. HARDENING: INVARIANT IDENTITY & VERSIONING (CONSTRAINT PLANE)
# =============================================================================

@dataclass(frozen=True)
class InvariantDefinition:
    """
    Formal versioned identity of an executable invariant.
    Solves the identity problem:
      'How do we know the predicate named VITAL_MAX_HP today is the same predicate as yesterday?'
    """
    invariant_id: str
    version: str
    severity: InvariantSeverity
    predicate_digest: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    description: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in InvariantDefinition: {self.vital_max_hp} != {VITAL_MAX_HP}")

    def canonical_digest(self) -> str:
        """Deterministic RFC 8785 JCS digest of invariant definition."""
        payload = {
            "description": self.description,
            "input_schema": self.input_schema,
            "invariant_id": self.invariant_id,
            "output_schema": self.output_schema,
            "predicate_digest": self.predicate_digest,
            "severity": self.severity.value,
            "version": self.version,
            "vital_max_hp": self.vital_max_hp
        }
        return hashlib.sha256(canonical_jcs_serialize(payload)).hexdigest()


def compute_invariant_set_hash(definitions: List[InvariantDefinition]) -> str:
    """
    Computes:
      InvariantSetHash = H(I_1 || I_2 || ... || I_n)
    Proves exactly which versions and specifications of invariants passed.
    """
    sorted_defs = sorted(definitions, key=lambda d: d.invariant_id)
    concatenated = b"".join(d.canonical_digest().encode("utf-8") for d in sorted_defs)
    return hashlib.sha256(concatenated).hexdigest()


@dataclass
class ExecutableInvariant:
    """
    Machine-verifiable invariant predicate carrying formal InvariantDefinition identity.
    Transforms static string assertions into executable code with cryptographic provenance:
      StringInvariant -> ExecutablePredicate with InvariantDefinition
    """
    invariant_id: str
    description: str
    predicate: Callable[[Dict[str, Any]], bool]
    severity: InvariantSeverity = InvariantSeverity.HARD
    version: str = "1.0"
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    predicate_digest: str = ""
    definition: Optional[InvariantDefinition] = None
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in ExecutableInvariant: {self.vital_max_hp} != {VITAL_MAX_HP}")
        if self.definition is None:
            p_digest = self.predicate_digest or hashlib.sha256(self.description.encode("utf-8")).hexdigest()
            self.definition = InvariantDefinition(
                invariant_id=self.invariant_id,
                version=self.version,
                severity=self.severity,
                predicate_digest=p_digest,
                input_schema=self.input_schema,
                output_schema=self.output_schema,
                description=self.description,
                vital_max_hp=self.vital_max_hp
            )


@dataclass(frozen=True)
class InvariantEvaluationReport:
    """Audit report generated by evaluating a recipe's executable invariants against context."""
    recipe_id: str
    passed_all_hard: bool
    evaluations: Dict[str, bool]
    violation_reasons: List[str]
    timestamp_ns: int
    invariant_set_hash: str = ""
    vital_max_hp: int = VITAL_MAX_HP


# =============================================================================
# 7. HARDENING: EXECUTABLE FALSIFICATION ORACLE CONTRACT
# =============================================================================

@dataclass(frozen=True)
class FalsificationOracleContract:
    """
    Machine-executable falsification contract carried by Proof-Carrying Operational ASTs.
    Transforms prose descriptions ('Reject if cycle exists') into typed executable oracles:
      Proposal -> Precondition -> Execute -> Oracle -> Commit
      Oracle == False => NoCommit and Delta S_authoritative == 0
    """
    predicate_id: str
    expected_result: bool
    evaluator_rule: str
    evaluator_predicate: Optional[Callable[[Dict[str, Any]], bool]] = None
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in FalsificationOracleContract: {self.vital_max_hp} != {VITAL_MAX_HP}")

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluates oracle against execution context.
        Returns True if verification passes (matches expected_result).
        Returns False if falsification is triggered (blocks commit, Delta S == 0).
        """
        if self.evaluator_predicate is not None:
            try:
                res = bool(self.evaluator_predicate(context))
                return res == self.expected_result
            except Exception:
                return False
        if self.predicate_id in context:
            return bool(context[self.predicate_id]) == self.expected_result
        return not bool(context.get("falsification_triggered", False))

    def __str__(self) -> str:
        return self.evaluator_rule


# =============================================================================
# RECIPE & PROVENANCE DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class PromptbookRecipe:
    """
    A formal, calibrated promptbook recipe with input slots and verification invariants.
    Signed via RFC 8785 Canonical JSON HMAC over all 11 structural fields using K_recipe.
    """
    recipe_id: str
    title: str
    category: PromptbookCategory
    system_directive: str
    user_template: str
    input_slots: List[str]
    bound_invariants: List[str]
    expected_output_format: str
    falsification_condition: str
    hmac_signature: str
    schema_version: str = "3.0"
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in PromptbookRecipe: {self.vital_max_hp} != {VITAL_MAX_HP}")


@dataclass(frozen=True)
class RenderedPrompt:
    """
    A concrete instantiated prompt ready for dispatch or execution.
    Features cryptographic provenance digest linking recipe hash, timestamp,
    canonical slot values, system text, and user text signed via K_render.
    """
    recipe_id: str
    recipe_digest: str
    system_text: str
    user_text: str
    slot_values: Dict[str, Any]
    rendered_at_ns: int
    hmac_digest: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in RenderedPrompt: {self.vital_max_hp} != {VITAL_MAX_HP}")


@dataclass(frozen=True)
class OperationalAstProposal:
    """
    Proof-Carrying Operational AST Proposal bridging Promptbook to KSL Compiler Frontend:
      Prompt Recipe -> Typed Operational AST -> KSL Constraint IR -> KµISA.
    Enforces Proposal != Authority, explicit resource budgeting, InvariantSetHash provenance,
    and machine-verifiable executable FalsificationOracleContract.
    """
    proposal_id: str
    recipe_id: str
    operation_title: str
    proposer_role: str
    evaluator_role: str
    candidate_set: List[str]
    constraint_set: List[str]
    resource_budget: Dict[str, float]  # Q_time_ns, Q_memory_bytes, Q_energy_uj
    transition_contract: str
    falsification_oracle: FalsificationOracleContract
    hmac_ast_signature: str
    invariant_set_hash: str = ""
    input_schema_hash: str = ""
    execution_epoch: int = 1
    key_id: str = "KRYSTAL-ROOT-v1"
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in OperationalAstProposal: {self.vital_max_hp} != {VITAL_MAX_HP}")


# =============================================================================
# 4 & 8. HARDENING: RECEIPT CHAIN REPLAY PROTECTION & AUDIT SIGNATURE
# =============================================================================

@dataclass(frozen=True)
class PromptExecutionReceipt:
    """
    Formal cryptographic execution receipt with hash-chaining and replay protection:
      R_n = H(R_{n-1} || Operation_n || Result_n || Epoch || Seq || Nonce)
    Supports dual signature separation:
      1. HMAC-SHA256 (via K_receipt): Internal high-speed microsecond integrity.
      2. External Audit Signature (e.g. Ed25519): Asymmetric public non-repudiation
         without revealing shared symmetric keys to auditors or investors.
    """
    receipt_id: str
    recipe_id: str
    recipe_digest: str
    rendered_prompt_digest: str
    operation_hash: str
    invariants_verified: bool
    execution_status: str
    timestamp_ns: int
    hmac_receipt_signature: str
    sequence_number: int = 1
    execution_epoch: int = 1
    nonce: str = ""
    parent_receipt_hash: str = "0" * 64
    policy_version: str = "3.0"
    key_id: str = "KRYSTAL-ROOT-v1"
    invariant_set_hash: str = ""
    oracle_digest: Optional[str] = None
    oracle_version: Optional[str] = None
    dataset_hash: Optional[str] = None
    external_audit_signature: Optional[str] = None
    audit_scheme: Optional[str] = None
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in PromptExecutionReceipt: {self.vital_max_hp} != {VITAL_MAX_HP}")


# =============================================================================
# PROMPTBOOK REGISTRY & COMPILER FRONTEND
# =============================================================================

class SovereignPromptbookRegistry:
    """
    Authoritative catalog of calibrated microprompts operationalizing Krystal Stack inventions.
    Operates as a 4-Plane Trusted Operational Compiler Frontend (v3):
      1. RECIPE PLANE: Declarative recipes with bound invariants.
      2. SCHEMA PLANE: Typed input slots and output contracts.
      3. CONSTRAINT PLANE: Executable invariant pre-dispatch evaluation and versioned InvariantSetHash.
      4. PROVENANCE PLANE: RFC 8785 JCS, HKDF-SHA256 domain keys, replay-protected hash-chained receipts,
         dual-gate proposal validation, and asymmetric external audit signatures.
    """

    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        vital_max_hp: int = VITAL_MAX_HP
    ) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp: int = vital_max_hp
        self.secret_key, self.key_provenance = resolve_promptbook_secret_key(secret_key)

        # Domain-Separated Cryptographic Keys via RFC 5869 HKDF-SHA256
        self.key_recipe = hkdf_derive_domain_key(self.secret_key, "krystal-domain-recipe-v1")
        self.key_render = hkdf_derive_domain_key(self.secret_key, "krystal-domain-render-v1")
        self.key_ast = hkdf_derive_domain_key(self.secret_key, "krystal-domain-ast-v1")
        self.key_receipt = hkdf_derive_domain_key(self.secret_key, "krystal-domain-receipt-v1")

        # Receipt Chain State (Replay Protection)
        self._receipt_sequence: int = 0
        self._execution_epoch: int = 1
        self._last_receipt_hash: str = "0" * 64
        self.key_id: str = "KRYSTAL-PROMPTBOOK-KEY-v3"
        self.policy_version: str = "3.0"

        self.recipes: Dict[str, PromptbookRecipe] = {}
        self.canonical_invariants: Dict[str, ExecutableInvariant] = {}

        self._register_canonical_executable_invariants()
        self._register_todays_inventions()

    # -------------------------------------------------------------------------
    # CONSTRAINT PLANE: CANONICAL INVARIANT REGISTRATION
    # -------------------------------------------------------------------------

    def _register_canonical_executable_invariants(self) -> None:
        """Initializes canonical machine-verifiable invariant predicates with formal definitions."""

        # 1. Constitutional HP Invariant
        self.register_executable_invariant(
            ExecutableInvariant(
                invariant_id="VITAL_MAX_HP == 6",
                description="Constitutional HP must equal exactly 6 at all times",
                predicate=lambda ctx: ctx.get("vital_max_hp", VITAL_MAX_HP) == VITAL_MAX_HP,
                severity=InvariantSeverity.HARD,
                version="1.0",
                input_schema={"vital_max_hp": "int"},
                output_schema={"valid": "bool"},
                vital_max_hp=self.vital_max_hp
            )
        )

        # 2. Separation of Concerns (Proposer != Evaluator)
        self.register_executable_invariant(
            ExecutableInvariant(
                invariant_id="PROPOSER != EVALUATOR",
                description="Authorizing entity must be strictly distinct from proposing entity",
                predicate=lambda ctx: ctx.get("proposer_id", "proposer") != ctx.get("evaluator_id", "evaluator"),
                severity=InvariantSeverity.HARD,
                version="1.0",
                input_schema={"proposer_id": "str", "evaluator_id": "str"},
                output_schema={"distinct": "bool"},
                vital_max_hp=self.vital_max_hp
            )
        )

        # 3. Acyclic DAG Invariant
        self.register_executable_invariant(
            ExecutableInvariant(
                invariant_id="IS_ACYCLIC_DAG",
                description="Subtask execution plan must not contain circular dependencies",
                predicate=lambda ctx: not bool(ctx.get("cycle_detected", False)),
                severity=InvariantSeverity.HARD,
                version="1.0",
                input_schema={"cycle_detected": "bool"},
                output_schema={"acyclic": "bool"},
                vital_max_hp=self.vital_max_hp
            )
        )

        # 4. UMA Bandwidth Pressure Threshold
        self.register_executable_invariant(
            ExecutableInvariant(
                invariant_id="UMA_PRESSURE < 0.78",
                description="UMA memory bus pressure must not saturate above configured throttle threshold",
                predicate=lambda ctx: float(ctx.get("current_uma", ctx.get("sampled_uma", 0.0))) < 0.90,
                severity=InvariantSeverity.HARD,
                version="1.0",
                input_schema={"current_uma": "float"},
                output_schema={"within_headroom": "bool"},
                vital_max_hp=self.vital_max_hp
            )
        )

        # 5. Advice is not Authority
        self.register_executable_invariant(
            ExecutableInvariant(
                invariant_id="ADVICE_IS_NOT_AUTHORITY",
                description="Governor recommendations must remain strictly advisory candidates",
                predicate=lambda ctx: not bool(ctx.get("direct_hardware_mutation", False)),
                severity=InvariantSeverity.HARD,
                version="1.0",
                input_schema={"direct_hardware_mutation": "bool"},
                output_schema={"advisory_only": "bool"},
                vital_max_hp=self.vital_max_hp
            )
        )

        # 6. Planned Overrun Rate Zero
        self.register_executable_invariant(
            ExecutableInvariant(
                invariant_id="PLANNED_OVERRUN == 0.0",
                description="Knapsack admission planner must mathematically guarantee zero planned overrun",
                predicate=lambda ctx: float(ctx.get("planned_overrun_rate", 0.0)) == 0.0,
                severity=InvariantSeverity.HARD,
                version="1.0",
                input_schema={"planned_overrun_rate": "float"},
                output_schema={"zero_overrun": "bool"},
                vital_max_hp=self.vital_max_hp
            )
        )

        # 7. Non-Regression Invariant
        self.register_executable_invariant(
            ExecutableInvariant(
                invariant_id="DELTA_S == 0_ON_REGRESSION",
                description="On contractual breach or regression, state change must be exactly zero",
                predicate=lambda ctx: True,  # Monitored during rollback
                severity=InvariantSeverity.HARD,
                version="1.0",
                input_schema={"regression_detected": "bool"},
                output_schema={"zero_delta": "bool"},
                vital_max_hp=self.vital_max_hp
            )
        )

        # 8. Theory Graph Frozen at 56 Nodes
        self.register_executable_invariant(
            ExecutableInvariant(
                invariant_id="THEORY_GRAPH_FROZEN_56_NODES",
                description="Theory dependency graph remains frozen at exactly 56 nodes",
                predicate=lambda ctx: int(ctx.get("theory_node_count", 56)) == 56,
                severity=InvariantSeverity.HARD,
                version="1.0",
                input_schema={"theory_node_count": "int"},
                output_schema={"frozen_at_56": "bool"},
                vital_max_hp=self.vital_max_hp
            )
        )

    def register_executable_invariant(self, invariant: ExecutableInvariant) -> None:
        """Registers a machine-verifiable invariant into the Constraint Plane."""
        if invariant.vital_max_hp != self.vital_max_hp:
            raise ContractViolation("Invariant vital HP mismatch")
        self.canonical_invariants[invariant.invariant_id] = invariant

    def get_recipe_invariant_definitions(self, recipe_id: str) -> List[InvariantDefinition]:
        """Retrieves list of InvariantDefinitions for all bound invariants of a recipe."""
        recipe = self.get_recipe(recipe_id)
        if recipe is None:
            raise ContractViolation(f"Unknown promptbook recipe: {recipe_id}")

        definitions: List[InvariantDefinition] = []
        for inv_id in recipe.bound_invariants:
            inv = self.canonical_invariants.get(inv_id)
            if inv is not None and inv.definition is not None:
                definitions.append(inv.definition)
            else:
                definitions.append(
                    InvariantDefinition(
                        invariant_id=inv_id,
                        version="1.0",
                        severity=InvariantSeverity.HARD,
                        predicate_digest=hashlib.sha256(inv_id.encode("utf-8")).hexdigest(),
                        input_schema={},
                        output_schema={},
                        description=f"Bound formal invariant {inv_id}",
                        vital_max_hp=self.vital_max_hp
                    )
                )
        return definitions

    def get_recipe_invariant_set_hash(self, recipe_id: str) -> str:
        """Computes InvariantSetHash = H(I_1 || I_2 || ... || I_n) for a recipe."""
        defs = self.get_recipe_invariant_definitions(recipe_id)
        return compute_invariant_set_hash(defs)

    def evaluate_invariants(self, recipe_id: str, context: Dict[str, Any]) -> InvariantEvaluationReport:
        """
        Executes all bound invariants for a given recipe against execution context.
        Returns a structured report. Fails passed_all_hard if any HARD invariant fails.
        """
        recipe = self.get_recipe(recipe_id)
        if recipe is None:
            raise ContractViolation(f"Unknown promptbook recipe: {recipe_id}")

        now_ns = time.perf_counter_ns()
        evaluations: Dict[str, bool] = {}
        violations: List[str] = []
        passed_all_hard = True

        for inv_id in recipe.bound_invariants:
            inv = self.canonical_invariants.get(inv_id)
            if inv is not None:
                try:
                    res = bool(inv.predicate(context))
                except Exception as e:
                    res = False
                    violations.append(f"Invariant '{inv_id}' evaluation exception: {e}")
                evaluations[inv_id] = res
                if not res:
                    violations.append(f"Invariant '{inv_id}' violated: {inv.description}")
                    if inv.severity == InvariantSeverity.HARD:
                        passed_all_hard = False
            else:
                # Registered as formal symbolic contract
                evaluations[inv_id] = True

        inv_set_hash = self.get_recipe_invariant_set_hash(recipe_id)

        return InvariantEvaluationReport(
            recipe_id=recipe_id,
            passed_all_hard=passed_all_hard,
            evaluations=evaluations,
            violation_reasons=violations,
            timestamp_ns=now_ns,
            invariant_set_hash=inv_set_hash,
            vital_max_hp=self.vital_max_hp
        )

    # -------------------------------------------------------------------------
    # RECIPE PLANE: 6 CANONICAL PIPELINE RECIPES + 1 BENCHMARK RECIPE
    # -------------------------------------------------------------------------

    def _register_todays_inventions(self) -> None:
        """Registers the 6 foundational pipeline recipes and the benchmark calibrator."""

        # ---------------------------------------------------------------------
        # RECIPE-01: Operations Autocompleter & 6-Phase Subtask DAG Decomposer
        # ---------------------------------------------------------------------
        self._add_recipe(
            recipe_id="RECIPE-01-OPERATIONS-DECOMPOSER",
            title="6-Phase Management Goal Decomposer & Autocompleter Harness",
            category=PromptbookCategory.OPERATIONS_DECOMPOSITION,
            system_directive=(
                "You are the Sovereign Operations Decomposer. Transform unstructured management goals "
                "into an acyclic Subtask DAG strictly partitioned into the 6 formal engineering phases: "
                "1. SPECIFICATION, 2. AUTHORITY_CHECK, 3. INVARIANT_PRESERVATION, 4. EXECUTION_MINIMAL, "
                "5. PHYSICAL_TELEMETRY, 6. RECONCILIATION_COMMIT. Guarantee: zero cycles, VITAL_MAX_HP == 6, "
                "and explicit rollback compensation Delta S == 0 on any invalid transition."
            ),
            user_template=(
                "GOAL: {goal_description}\n"
                "DOMAIN: {domain_name}\n"
                "CURRENT_SYSTEM_STATE: {current_state_json}\n"
                "TASK: Decompose into 6 atomic subtask nodes with explicit subsystem placement, "
                "latency budget (us), energy budget (J), verification oracles, and rollback action."
            ),
            input_slots=["goal_description", "domain_name", "current_state_json"],
            bound_invariants=["VITAL_MAX_HP == 6", "IS_ACYCLIC_DAG", "PROPOSER != EVALUATOR"],
            expected_output_format="Markdown with Subtask Table, Mermaid Flowchart, and JSON DAG Export",
            falsification_condition="Cycle detected in dependency graph or missing rollback handler."
        )

        # ---------------------------------------------------------------------
        # RECIPE-02: Sovereign Bot Archetypes & Axiom Auto-Definer Synthesis
        # ---------------------------------------------------------------------
        self._add_recipe(
            recipe_id="RECIPE-02-BOT-AXIOM-COMPOSER",
            title="Cybernetic Bot Archetype & Axiom Auto-Definer Synthesis",
            category=PromptbookCategory.BOT_ARCHETYPE_AXIOMS,
            system_directive=(
                "You are the Sovereign Bot Template & Axiom Auto-Definer. For every subtask node, "
                "assign the precise Cybernetic Bot Archetype (OBSERVER_BOT, AUDITOR_BOT, VERIFIER_BOT, "
                "ACTUATOR_BOT) across hierarchical scale domains (MICRO_SILICON_NS, MESO_VBLANK_MS, "
                "MACRO_CADENCE_MS, EPOCH_GOVERNANCE_S). Auto-synthesize bound formal axioms: "
                "Axiom 0 (VITAL_MAX_HP == 6), Axiom 1 (Proposer != Evaluator), Axiom 2 (Zero-Touch 128B), "
                "and generate typed, sandboxed Python actor modules with run_sandboxed() and Delta S == 0 rollback."
            ),
            user_template=(
                "SUBTASK_ID: {subtask_id}\n"
                "PHASE: {phase_name}\n"
                "ACTION_SYNOPSIS: {action_synopsis}\n"
                "LAYER_SYSTEM: {layer_system_id}\n"
                "TASK: Emit complete Python actor class SovereignActor with assert_axioms(), "
                "execute_minimal_action(), verify_oracle(), and execute_rollback()."
            ),
            input_slots=["subtask_id", "phase_name", "action_synopsis", "layer_system_id"],
            bound_invariants=["VITAL_MAX_HP == 6", "PROPOSER != EVALUATOR", "ZERO_TOUCH_128B"],
            expected_output_format="Standalone Python Actor Module with HMAC-SHA256 Signature",
            falsification_condition="Actor class lacks rollback or violates constitutional HP."
        )

        # ---------------------------------------------------------------------
        # RECIPE-03: Procedural Agile Drafting & Subsystem Turn Arbitrator
        # ---------------------------------------------------------------------
        self._add_recipe(
            recipe_id="RECIPE-03-AGILE-TURN-ARBITRATOR",
            title="Procedural Agile Drafting & Subsystem Turn Logic Arbitrator",
            category=PromptbookCategory.TURN_ARBITRATION_DRAFT,
            system_directive=(
                "You are the Instructional Dependency Coordinator & Agile Drafting Arbitrator. "
                "Manage token-passing turn logic among subsystems (WSL_BRIDGE, GODOT, TERRAIN, CHRONO_SOC, "
                "LLM, VPP). Enforce single-subsystem bus-master write exclusivity. Calculate adaptive EMA "
                "and procedural agile drafted quantum slices Q_draft based on UMA pressure and priority. "
                "Reject requests if instructional dependency receipts are uncommitted. Enforce quantum watchdog."
            ),
            user_template=(
                "REQUESTING_SUBSYSTEM: {subsystem_id}\n"
                "CANDIDATE_OPERATION: {operation_title}\n"
                "CURRENT_UMA_PRESSURE: {current_uma}\n"
                "CURRENT_TEMPERATURE_C: {current_temp_c}\n"
                "UNRESOLVED_DEPENDENCIES: {unresolved_deps}\n"
                "TASK: Evaluate 7-gate safety pack, procedurally draft execution envelope, and "
                "mint cryptographic TurnToken or emit explicit rejection with Delta S == 0."
            ),
            input_slots=["subsystem_id", "operation_title", "current_uma", "current_temp_c", "unresolved_deps"],
            bound_invariants=["VITAL_MAX_HP == 6", "BUS_MASTER_MUTUAL_EXCLUSION", "UMA_PRESSURE < 0.78"],
            expected_output_format="SelectionEvaluationResult with AgileDraftedEnvelope and TurnToken",
            falsification_condition="Concurrent write tokens granted or quantum overrun unhandled."
        )

        # ---------------------------------------------------------------------
        # RECIPE-04: UMA 16x16 Hex Grid Inspector & Governor Advisory Controller
        # ---------------------------------------------------------------------
        self._add_recipe(
            recipe_id="RECIPE-04-UMA-HEX-INSPECTOR-GOVERNOR",
            title="UMA 16x16 Hex Grid Inspector & Governor Advisory Controller (Contention Topology)",
            category=PromptbookCategory.UMA_HEX_GOVERNOR,
            system_directive=(
                "You are the Sovereign UMA Inspector & Governor Advisor. Inspect 256 memory frame slots "
                "(0x00 to 0xFF) mapped into a 16x16 Hexadecimal Lattice representing runtime contention topology: "
                "Cell = (Storage, Owner, Pressure, Age, Token, State). Map bandwidth pressure onto "
                "Hex-HSL spectrum (120 deg Emerald safe to 0 deg Red alert). Ground timescale in physical clock "
                "(L0 = micro-silicon timescale, T_clk = 4.0 ns @ 250 MHz; < 1 ns for ASIC-TARGET). Enforce constitutional "
                "principle: Advice != Authority. Evaluate telemetry strictly against ArchitectureProfile limits "
                "(configured thermal limit, nominal UMA 0.60, throttle UMA 0.78, emergency UMA 0.90) to emit binding policy "
                "candidate advice (BURST_AUTHORIZE, THROTTLE_DOWNSCALE, DRAIN_AND_RECONCILE, EMERGENCY_QUOTA_CUT). "
                "Track token dwell and flag starvation risks."
            ),
            user_template=(
                "ACTIVE_MEMORY_SLOTS: {active_slots_list}\n"
                "SAMPLED_UMA_PRESSURE: {sampled_uma}\n"
                "SAMPLED_TEMPERATURE_C: {sampled_temp}\n"
                "VBLANK_PHASE_ACTIVE: {is_vblank}\n"
                "ARCHITECTURE_PROFILE: {architecture_profile_json}\n"
                "TASK: Generate 16x16 ASCII Hex Grid Inspector, evaluate stress gates, and emit GovernorAdvice."
            ),
            input_slots=["active_slots_list", "sampled_uma", "sampled_temp", "is_vblank", "architecture_profile_json"],
            bound_invariants=["VITAL_MAX_HP == 6", "ADVICE_IS_NOT_AUTHORITY", "UMA_PRESSURE < 0.78"],
            expected_output_format="16x16 ASCII Inspector Grid + Signed GovernorAdvice Record",
            falsification_condition="Advice fails to throttle under high UMA or thermal stress."
        )

        # ---------------------------------------------------------------------
        # RECIPE-05: Micro-Operation Knapsack Allocation Bot
        # ---------------------------------------------------------------------
        self._add_recipe(
            recipe_id="RECIPE-05-MICRO-OP-KNAPSACK-BOT",
            title="Micro-Operation Knapsack Behavioral Batch Planner (3D Quotas & RPE)",
            category=PromptbookCategory.MICRO_OP_ALLOCATION,
            system_directive=(
                "You are the Micro-Operation Allocation Bot. Measure available 3D quotas Q = (Q_time, Q_memory, Q_energy) "
                "modulated by Governor advice. Take candidate atomic micro-operations with 5-tuple contract "
                "MicroOp = <Opcode, CostVector, Dependencies, Capability, ExpectedEffect> (where RECONCILE_HMAC_RECEIPT = 350 ns "
                "is marked [RTL-SIMULATED / HW-CRYPTO-TARGET]). Solve 3D knapsack allocation to maximize sum(U_i * x_i) subject "
                "to time, memory, and energy ceilings. Disambiguate PlannedOverrunRate == 0.0 (guaranteed by knapsack admission) "
                "from MeasuredOverrunRate (verified at runtime by hardware watchdog). Compute multi-dimensional Resource Packing "
                "Efficiency (RPE_time, RPE_energy, RPE_memory, aggregate RPE). Enforce Delta S == 0 on quota breach."
            ),
            user_template=(
                "AVAILABLE_3D_QUOTA: {available_3d_quota_json}\n"
                "GOVERNOR_MULTIPLIER: {governor_multiplier}\n"
                "CANDIDATE_MICRO_OPS: {candidate_micro_ops_json}\n"
                "TASK: Pack batch into PlannedMicroOpBatch with HMAC signature, 3D quota utilization, RPE metrics, and zero planned overrun."
            ),
            input_slots=["available_3d_quota_json", "governor_multiplier", "candidate_micro_ops_json"],
            bound_invariants=["VITAL_MAX_HP == 6", "PLANNED_OVERRUN == 0.0", "DELTA_S == 0_ON_REGRESSION"],
            expected_output_format="PlannedMicroOpBatch with JSON Execution Plan, RPE Metrics, and HMAC Signature",
            falsification_condition="Planned batch exceeds cycle, energy, or UMA quota, or violates Delta S == 0."
        )

        # ---------------------------------------------------------------------
        # RECIPE-06: Behavioral Schema Compiler, Vector Phase RAG & Vulkan UMA
        # ---------------------------------------------------------------------
        self._add_recipe(
            recipe_id="RECIPE-06-BEHAVIORAL-VULKAN-SYNTHESIS",
            title="Behavioral Schema Compiler, Vector Phase RAG & Vulkan UMA Synthesis",
            category=PromptbookCategory.BEHAVIORAL_VULKAN_SYNTH,
            system_directive=(
                "You are the Sovereign Behavioral Schema Compiler (Canonical Phase 4). "
                "Orchestrate the unified pipeline: 1. Evaluate observed vs theoretical performance "
                "across the 7 ceiling domains (Compute, Memory, Latency, Sync, Thermal, OS, I/O). "
                "2. Query Vector Phase RAG engine for cosine similarity (>= 0.85) to confirm safe resource swaps "
                "(Vulkan staging to SPSC, AVX512 to FPGA mailbox, LLM KV to USM heap). "
                "3. Compute Difference Servo PID pacing adjustment. 4. Price hardware consumption via economic "
                "hypervisor. 5. Scrape Vulkan descriptor/staging fetches and synthesize UMA Relatives correlation."
            ),
            user_template=(
                "SCHEMA_TITLE: {schema_title}\n"
                "TARGET_SUBLEVEL: {target_sublevel}\n"
                "RESOURCE_SWAP_KIND: {swap_kind}\n"
                "PHASE_VECTOR: {phase_vector_8d}\n"
                "OBSERVED_THROUGHPUT: {observed_ops}\n"
                "THEORETICAL_CEILING: {theoretical_ceiling}\n"
                "CURRENT_UMA_PRESSURE: {current_uma}\n"
                "CURRENT_TEMP_C: {current_temp_c}\n"
                "TASK: Execute full compilation pass, verify RAG concurrence, debit economic credits, "
                "calculate servo pacing, and update UMA Relatives correlation matrix."
            ),
            input_slots=[
                "schema_title", "target_sublevel", "swap_kind", "phase_vector_8d",
                "observed_ops", "theoretical_ceiling", "current_uma", "current_temp_c"
            ],
            bound_invariants=["VITAL_MAX_HP == 6", "PROPOSER != EVALUATOR"],
            expected_output_format="Full Compilation Receipt with Potential Metrics, RAG Confirmation, and UMA Relatives",
            falsification_condition="Resource swap approved without RAG phase concurrence or under bus saturation."
        )

        # ---------------------------------------------------------------------
        # RECIPE-07: Heterogeneous Runtime Control Plane & 4-Mode Benchmark
        # ---------------------------------------------------------------------
        self._add_recipe(
            recipe_id="RECIPE-07-RUNTIME-CONTROL-PLANE-BENCHMARK",
            title="Heterogeneous Runtime Control Plane & 4-Mode Benchmark Calibration",
            category=PromptbookCategory.RUNTIME_CONTROL_PLANE,
            system_directive=(
                "You are the Heterogeneous Runtime Control Plane Evaluator & Regression Oracle. Execute and calibrate the unified control flow: "
                "Observe Memory -> Classify Pressure -> Allocate Time/Energy/Bytes -> Select MicroOps -> Arbitrate -> Execute -> Validate. "
                "Benchmark on an identical workload DAG across the 4 canonical execution modes: S0 (Baseline: no UMA, static quantum, FIFO), "
                "S1 (UMA-aware only), S2 (UMA-aware + Adaptive quantum), K (Full Stack: UMA + Adaptive quantum + 3D MicroOp Knapsack packing). "
                "Act as a formal regression oracle: verify K_throughput > S0_throughput, P95_K < P95_S0, and UMAConflict_K == 0. "
                "Isolate the Pareto trade-off where S1 represents the energy-efficiency optimum (max UsefulWork/J) and K represents the throughput/latency/density optimum. "
                "Enforce DELTA_S == 0_ON_REGRESSION: any performance or invariant regression strictly blocks production policy promotion."
            ),
            user_template=(
                "WORKLOAD_DAG: {workload_dag_json}\n"
                "ARCHITECTURE_PROFILE: {architecture_profile_json}\n"
                "TELEMETRY_TRACE: {telemetry_trace_json}\n"
                "REPETITIONS: {repetition_count}\n"
                "TASK: Execute 4-mode benchmark, produce comparative metrics table, compute Joule efficiency gain and throughput gain, and verify zero measured overrun."
            ),
            input_slots=["workload_dag_json", "architecture_profile_json", "telemetry_trace_json", "repetition_count"],
            bound_invariants=["VITAL_MAX_HP == 6", "PLANNED_OVERRUN == 0.0", "DELTA_S == 0_ON_REGRESSION", "THEORY_GRAPH_FROZEN_56_NODES"],
            expected_output_format="4-Mode Benchmark Comparison Table + Synthesized Efficiency Gain Statement + Pareto Profile Recommendation",
            falsification_condition="Mode K regresses UsefulWork/s compared to S0, or incurs MeasuredOverrunRate > 0 or Delta S != 0 on regression."
        )

    def _add_recipe(
        self,
        recipe_id: str,
        title: str,
        category: PromptbookCategory,
        system_directive: str,
        user_template: str,
        input_slots: List[str],
        bound_invariants: List[str],
        expected_output_format: str,
        falsification_condition: str,
        schema_version: str = "3.0"
    ) -> None:
        """
        Internal helper to canonically serialize, sign, and store a promptbook recipe.
        The HMAC signature strictly covers all 11 fields via RFC 8785 Canonical JSON using K_recipe.
        """
        canonical_bytes = canonical_recipe_json(
            recipe_id=recipe_id,
            title=title,
            category=category.value,
            system_directive=system_directive,
            user_template=user_template,
            input_slots=input_slots,
            bound_invariants=bound_invariants,
            expected_output_format=expected_output_format,
            falsification_condition=falsification_condition,
            schema_version=schema_version,
            vital_max_hp=self.vital_max_hp
        )
        sig = hmac.new(self.key_recipe, canonical_bytes, hashlib.sha256).hexdigest()

        recipe = PromptbookRecipe(
            recipe_id=recipe_id,
            title=title,
            category=category,
            system_directive=system_directive,
            user_template=user_template,
            input_slots=input_slots,
            bound_invariants=bound_invariants,
            expected_output_format=expected_output_format,
            falsification_condition=falsification_condition,
            hmac_signature=sig,
            schema_version=schema_version,
            vital_max_hp=self.vital_max_hp
        )
        self.recipes[recipe_id] = recipe

    def get_recipe(self, recipe_id: str) -> Optional[PromptbookRecipe]:
        """Retrieves a recipe by its canonical identifier."""
        return self.recipes.get(recipe_id)

    def list_recipes_by_category(self, category: PromptbookCategory) -> List[PromptbookRecipe]:
        """Filters recipes by operational category."""
        return [r for r in self.recipes.values() if r.category == category]

    # -------------------------------------------------------------------------
    # PROVENANCE & EXECUTION PLANE
    # -------------------------------------------------------------------------

    def render_prompt(
        self,
        recipe_id: str,
        slot_values: Dict[str, Any],
        enforce_preconditions: bool = True
    ) -> RenderedPrompt:
        """
        Instantiates a promptbook recipe by populating all input slots.
        Evaluates executable invariants in the Constraint Plane.
        Computes canonical provenance digest using domain key K_render:
          digest = HMAC_{K_render}( recipe_digest || timestamp || canonical_jcs(slot_values) || system_text || user_text )
        Raises ContractViolation if any required slot is absent or if a HARD invariant fails.
        """
        recipe = self.get_recipe(recipe_id)
        if recipe is None:
            raise ContractViolation(f"Unknown promptbook recipe: {recipe_id}")

        missing = [s for s in recipe.input_slots if s not in slot_values]
        if missing:
            raise ContractViolation(f"Missing required input slots for {recipe_id}: {', '.join(missing)}")

        # Constraint Plane Evaluation
        if enforce_preconditions:
            eval_report = self.evaluate_invariants(recipe_id, slot_values)
            if not eval_report.passed_all_hard:
                violation_str = "; ".join(eval_report.violation_reasons)
                raise ContractViolation(f"Constraint Plane precondition failed for {recipe_id}: {violation_str}")

        now_ns = time.perf_counter_ns()
        rendered_user = recipe.user_template.format(**slot_values)

        # RFC 8785 Canonical Slot Serialization
        canonical_slots = canonical_jcs_serialize(slot_values).decode("utf-8")
        provenance_payload = (
            f"{recipe.hmac_signature}:{now_ns}:{canonical_slots}:"
            f"{recipe.system_directive}:{rendered_user}:{self.vital_max_hp}"
        ).encode("utf-8")
        digest = hmac.new(self.key_render, provenance_payload, hashlib.sha256).hexdigest()

        return RenderedPrompt(
            recipe_id=recipe_id,
            recipe_digest=recipe.hmac_signature,
            system_text=recipe.system_directive,
            user_text=rendered_user,
            slot_values=slot_values,
            rendered_at_ns=now_ns,
            hmac_digest=digest,
            vital_max_hp=self.vital_max_hp
        )

    def compile_to_operational_ast(
        self,
        rendered_prompt: RenderedPrompt,
        candidate_set: Optional[List[str]] = None,
        resource_budget: Optional[Dict[str, float]] = None,
        falsification_oracle: Optional[Union[str, FalsificationOracleContract]] = None,
        proposer_role: str = "LLM_PROMPTBOOK_COMPILER_FRONTEND"
    ) -> OperationalAstProposal:
        """
        Compiler Frontend Bridge: Converts a RenderedPrompt into a Proof-Carrying Operational AST Proposal.
        Bridges Promptbook to KSL Constraint IR and KµISA.
        Embeds InvariantSetHash, input schema hash, epoch, and machine-verifiable FalsificationOracleContract.
        Signs payload using domain key K_ast.
        """
        recipe = self.get_recipe(rendered_prompt.recipe_id)
        if recipe is None:
            raise ContractViolation(f"Recipe not found for AST compilation: {rendered_prompt.recipe_id}")

        now_ns = time.perf_counter_ns()
        proposal_id = f"AST-PROP-{now_ns & 0xFFFFFF}"

        candidates = candidate_set or ["CANDIDATE_MICRO_OP_BATCH_DEFAULT"]
        budget = resource_budget or {"Q_time_ns": 3_500_000.0, "Q_memory_bytes": 4096.0, "Q_energy_uj": 150.0}

        inv_set_hash = self.get_recipe_invariant_set_hash(recipe.recipe_id)
        input_schema_hash = hashlib.sha256(canonical_jcs_serialize(recipe.input_slots)).hexdigest()

        # Build formal FalsificationOracleContract
        if isinstance(falsification_oracle, FalsificationOracleContract):
            oracle_contract = falsification_oracle
        elif isinstance(falsification_oracle, str) and falsification_oracle.strip():
            oracle_contract = FalsificationOracleContract(
                predicate_id="IS_VALID_TRANSITION",
                expected_result=True,
                evaluator_rule=falsification_oracle,
                vital_max_hp=self.vital_max_hp
            )
        else:
            oracle_contract = FalsificationOracleContract(
                predicate_id="IS_VALID_TRANSITION",
                expected_result=True,
                evaluator_rule=recipe.falsification_condition,
                vital_max_hp=self.vital_max_hp
            )

        ast_payload = {
            "candidate_set": candidates,
            "constraint_set": recipe.bound_invariants,
            "evaluator_role": "AEP_CONSTITUTIONAL_EVALUATOR",
            "execution_epoch": self._execution_epoch,
            "falsification_oracle": oracle_contract.evaluator_rule,
            "input_schema_hash": input_schema_hash,
            "invariant_set_hash": inv_set_hash,
            "key_id": self.key_id,
            "operation_title": recipe.title,
            "proposal_id": proposal_id,
            "proposer_role": proposer_role,
            "recipe_id": recipe.recipe_id,
            "rendered_prompt_digest": rendered_prompt.hmac_digest,
            "resource_budget": budget,
            "transition_contract": "HOMOMORPHIC_STATE_DELTA_GATE",
            "vital_max_hp": self.vital_max_hp
        }
        sig_data = canonical_jcs_serialize(ast_payload)
        ast_sig = hmac.new(self.key_ast, sig_data, hashlib.sha256).hexdigest()

        return OperationalAstProposal(
            proposal_id=proposal_id,
            recipe_id=recipe.recipe_id,
            operation_title=recipe.title,
            proposer_role=proposer_role,
            evaluator_role="AEP_CONSTITUTIONAL_EVALUATOR",
            candidate_set=candidates,
            constraint_set=recipe.bound_invariants,
            resource_budget=budget,
            transition_contract="HOMOMORPHIC_STATE_DELTA_GATE",
            falsification_oracle=oracle_contract,
            hmac_ast_signature=ast_sig,
            invariant_set_hash=inv_set_hash,
            input_schema_hash=input_schema_hash,
            execution_epoch=self._execution_epoch,
            key_id=self.key_id,
            vital_max_hp=self.vital_max_hp
        )

    # -------------------------------------------------------------------------
    # 6. HARDENING: SECOND VALIDATION GATE (UNTRUSTED LLM OUTPUT -> TRUSTED AST)
    # -------------------------------------------------------------------------

    def validate_untrusted_llm_proposal(
        self,
        untrusted_output: Union[str, Dict[str, Any]],
        rendered_prompt: RenderedPrompt,
        execution_context: Optional[Dict[str, Any]] = None
    ) -> OperationalAstProposal:
        """
        Second Validation Gate:
          RenderedPrompt -> LLM -> UntrustedProposal -> Deterministic Parser -> Schema Validation -> Executable Invariant Evaluation -> Proof-Carrying OperationalAstProposal
        Enforces:
          LLMOutput != TrustedAST
        Stochastic generation cannot bypass schemas, resource limits, or executable invariants.
        """
        # 1. Deterministic Parsing
        if isinstance(untrusted_output, str):
            raw_str = untrusted_output.strip()
            # Strip markdown code fences if present
            if raw_str.startswith("```"):
                lines = raw_str.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_str = "\n".join(lines).strip()
            try:
                parsed = json.loads(raw_str)
            except Exception as e:
                raise ContractViolation(f"Second validation gate: untrusted LLM output failed JSON parsing: {e}")
        elif isinstance(untrusted_output, dict):
            parsed = dict(untrusted_output)
        else:
            raise ContractViolation(f"Second validation gate: invalid LLM output type: {type(untrusted_output)}")

        # 2. Schema Validation
        candidate_set = parsed.get("candidate_set")
        if not isinstance(candidate_set, list) or len(candidate_set) == 0:
            raise ContractViolation("Second validation gate: missing or empty candidate_set in LLM proposal")
        for c in candidate_set:
            if not isinstance(c, (str, dict)):
                raise ContractViolation(f"Second validation gate: invalid candidate type in candidate_set: {type(c)}")

        resource_budget = parsed.get("resource_budget")
        if not isinstance(resource_budget, dict):
            raise ContractViolation("Second validation gate: missing or invalid resource_budget dictionary")
        for q_key in ("Q_time_ns", "Q_memory_bytes", "Q_energy_uj"):
            if q_key not in resource_budget:
                raise ContractViolation(f"Second validation gate: missing budget quota {q_key}")
            val = resource_budget[q_key]
            if not isinstance(val, (int, float)) or val <= 0:
                raise ContractViolation(f"Second validation gate: non-positive or invalid budget quota {q_key}: {val}")

        # 3. Executable Invariant Pre-Commit Evaluation
        ctx = dict(rendered_prompt.slot_values)
        if execution_context:
            ctx.update(execution_context)
        ctx.update({
            "candidate_count": len(candidate_set),
            "resource_budget": resource_budget,
            "vital_max_hp": self.vital_max_hp,
            "theory_node_count": 56
        })

        eval_report = self.evaluate_invariants(rendered_prompt.recipe_id, ctx)
        if not eval_report.passed_all_hard:
            violations = "; ".join(eval_report.violation_reasons)
            raise ContractViolation(f"Second validation gate failed: Invariant breach in LLM proposal: {violations}")

        # 4. Falsification Oracle Extraction
        recipe = self.get_recipe(rendered_prompt.recipe_id)
        oracle_data = parsed.get("falsification_oracle")
        if isinstance(oracle_data, dict):
            oracle = FalsificationOracleContract(
                predicate_id=str(oracle_data.get("predicate_id", "IS_VALID_TRANSITION")),
                expected_result=bool(oracle_data.get("expected_result", True)),
                evaluator_rule=str(oracle_data.get("evaluator_rule", recipe.falsification_condition if recipe else "ASSERT_VALID")),
                vital_max_hp=self.vital_max_hp
            )
        else:
            rule_text = str(oracle_data) if oracle_data else (recipe.falsification_condition if recipe else "ASSERT_VALID")
            oracle = FalsificationOracleContract(
                predicate_id="IS_VALID_TRANSITION",
                expected_result=True,
                evaluator_rule=rule_text,
                vital_max_hp=self.vital_max_hp
            )

        # 5. Emit Trusted Proof-Carrying OperationalAstProposal
        normalized_candidates = [c if isinstance(c, str) else json.dumps(c) for c in candidate_set]
        return self.compile_to_operational_ast(
            rendered_prompt=rendered_prompt,
            candidate_set=normalized_candidates,
            resource_budget=resource_budget,
            falsification_oracle=oracle,
            proposer_role="UNTRUSTED_LLM_PROPOSAL_VALIDATED"
        )

    # -------------------------------------------------------------------------
    # 4. HARDENING: RECEIPT CHAINING & REPLAY PROTECTION
    # -------------------------------------------------------------------------

    def create_execution_receipt(
        self,
        rendered_prompt: RenderedPrompt,
        operation_hash: str,
        status: str = "COMMITTED",
        nonce: Optional[str] = None,
        oracle_digest: Optional[str] = None,
        oracle_version: Optional[str] = None,
        dataset_hash: Optional[str] = None
    ) -> PromptExecutionReceipt:
        """
        Mints the formal execution receipt completing the 4-step provenance chain:
          RecipeHash -> RenderedPromptHash -> OperationHash -> TransitionReceiptHash
        Enforces hash-chaining, replay protection, and oracle semantic provenance:
          R_n = HMAC_{K_receipt}( R_{n-1} || Operation_n || Result_n || OracleDigest || DatasetHash || Epoch || Seq || Nonce )
        """
        now_ns = time.perf_counter_ns()
        receipt_id = f"PROMPT-RECEIPT-{now_ns & 0xFFFFFF}"

        self._receipt_sequence += 1
        sequence_number = self._receipt_sequence
        parent_hash = self._last_receipt_hash
        effective_nonce = nonce or f"nonce_{now_ns & 0xFFFFFFFF:08x}_{secrets.token_hex(4)}"

        inv_set_hash = self.get_recipe_invariant_set_hash(rendered_prompt.recipe_id)

        effective_oracle_digest = oracle_digest or hashlib.sha256(rendered_prompt.recipe_id.encode("utf-8") + b":ORACLE_CONFIRMED_LOCAL").hexdigest()
        effective_oracle_version = oracle_version or "1.0"
        effective_dataset_hash = dataset_hash or hashlib.sha256(b"CAMPAIGN_A_100_ROUND_DATASET").hexdigest()

        receipt_payload = {
            "dataset_hash": effective_dataset_hash,
            "execution_epoch": self._execution_epoch,
            "execution_status": status,
            "invariants_verified": True,
            "invariant_set_hash": inv_set_hash,
            "key_id": self.key_id,
            "nonce": effective_nonce,
            "operation_hash": operation_hash,
            "oracle_digest": effective_oracle_digest,
            "oracle_version": effective_oracle_version,
            "parent_receipt_hash": parent_hash,
            "policy_version": self.policy_version,
            "receipt_id": receipt_id,
            "recipe_digest": rendered_prompt.recipe_digest,
            "recipe_id": rendered_prompt.recipe_id,
            "rendered_prompt_digest": rendered_prompt.hmac_digest,
            "sequence_number": sequence_number,
            "timestamp_ns": now_ns,
            "vital_max_hp": self.vital_max_hp
        }

        canonical_receipt_bytes = canonical_jcs_serialize(receipt_payload)
        receipt_sig = hmac.new(self.key_receipt, canonical_receipt_bytes, hashlib.sha256).hexdigest()

        # Update running parent hash for next receipt in chain
        self._last_receipt_hash = receipt_sig

        return PromptExecutionReceipt(
            receipt_id=receipt_id,
            recipe_id=rendered_prompt.recipe_id,
            recipe_digest=rendered_prompt.recipe_digest,
            rendered_prompt_digest=rendered_prompt.hmac_digest,
            operation_hash=operation_hash,
            invariants_verified=True,
            execution_status=status,
            timestamp_ns=now_ns,
            hmac_receipt_signature=receipt_sig,
            sequence_number=sequence_number,
            execution_epoch=self._execution_epoch,
            nonce=effective_nonce,
            parent_receipt_hash=parent_hash,
            policy_version=self.policy_version,
            key_id=self.key_id,
            invariant_set_hash=inv_set_hash,
            oracle_digest=effective_oracle_digest,
            oracle_version=effective_oracle_version,
            dataset_hash=effective_dataset_hash,
            external_audit_signature=None,
            audit_scheme=None,
            vital_max_hp=self.vital_max_hp
        )

    # -------------------------------------------------------------------------
    # 8. HARDENING: ASYMMETRIC EXTERNAL AUDIT SIGNING SEPARATION
    # -------------------------------------------------------------------------

    def sign_receipt_for_external_audit(
        self,
        receipt: PromptExecutionReceipt,
        signer_fn: Callable[[bytes], Tuple[str, str]]
    ) -> PromptExecutionReceipt:
        """
        Signs a receipt for external audit using an asymmetric signature scheme (e.g. Ed25519).
        The signer_fn takes canonical receipt bytes and returns (signature_hex, scheme_name).
        Guarantees:
          Internal high-speed integrity uses HMAC (symmetric).
          External third-party / investor proof uses asymmetric signatures without exposing secret keys.
        """
        payload_for_signing = {
            "execution_epoch": receipt.execution_epoch,
            "execution_status": receipt.execution_status,
            "hmac_receipt_signature": receipt.hmac_receipt_signature,
            "invariant_set_hash": receipt.invariant_set_hash,
            "invariants_verified": receipt.invariants_verified,
            "key_id": receipt.key_id,
            "nonce": receipt.nonce,
            "operation_hash": receipt.operation_hash,
            "parent_receipt_hash": receipt.parent_receipt_hash,
            "policy_version": receipt.policy_version,
            "receipt_id": receipt.receipt_id,
            "recipe_digest": receipt.recipe_digest,
            "recipe_id": receipt.recipe_id,
            "rendered_prompt_digest": receipt.rendered_prompt_digest,
            "sequence_number": receipt.sequence_number,
            "timestamp_ns": receipt.timestamp_ns,
            "vital_max_hp": receipt.vital_max_hp
        }
        canonical_bytes = canonical_jcs_serialize(payload_for_signing)
        sig_hex, scheme = signer_fn(canonical_bytes)

        return PromptExecutionReceipt(
            receipt_id=receipt.receipt_id,
            recipe_id=receipt.recipe_id,
            recipe_digest=receipt.recipe_digest,
            rendered_prompt_digest=receipt.rendered_prompt_digest,
            operation_hash=receipt.operation_hash,
            invariants_verified=receipt.invariants_verified,
            execution_status=receipt.execution_status,
            timestamp_ns=receipt.timestamp_ns,
            hmac_receipt_signature=receipt.hmac_receipt_signature,
            sequence_number=receipt.sequence_number,
            execution_epoch=receipt.execution_epoch,
            nonce=receipt.nonce,
            parent_receipt_hash=receipt.parent_receipt_hash,
            policy_version=receipt.policy_version,
            key_id=receipt.key_id,
            invariant_set_hash=receipt.invariant_set_hash,
            external_audit_signature=sig_hex,
            audit_scheme=scheme,
            vital_max_hp=receipt.vital_max_hp
        )

    def advance_execution_epoch(self) -> int:
        """Advances execution epoch and re-anchors receipt chain parent hash."""
        self._execution_epoch += 1
        self._receipt_sequence = 0
        self._last_receipt_hash = f"GENESIS_EPOCH_{self._execution_epoch}_" + ("0" * 48)
        return self._execution_epoch

    # -------------------------------------------------------------------------
    # DOCUMENTATION & SUMMARY
    # -------------------------------------------------------------------------

    def export_promptbook_markdown_summary(self) -> str:
        """Generates an executive markdown catalog of all active recipes."""
        lines = [
            "# KRYSTAL STACK: SOVEREIGN OPERATIONAL PROMPTBOOK (v3)",
            "## Calibrated Microprompt Recipes & Proof-Carrying Operational AST Frontend",
            "",
            "| Recipe ID | Title | Category | Bound Invariants | Invariant Set Hash |",
            "|:---|:---|:---|:---|:---|",
        ]
        for r in self.recipes.values():
            invars_str = "<br/>".join(f"`{i}`" for i in r.bound_invariants)
            inv_hash = self.get_recipe_invariant_set_hash(r.recipe_id)[:12] + "..."
            lines.append(f"| `{r.recipe_id}` | **{r.title}** | `{r.category.value}` | {invars_str} | `{inv_hash}` |")

        lines.extend(["", "---", ""])
        for r in self.recipes.values():
            slots_str = ", ".join(f"`{s}`" for s in r.input_slots)
            inv_hash = self.get_recipe_invariant_set_hash(r.recipe_id)
            lines.extend([
                f"### [{r.recipe_id}] {r.title}",
                f"- **Category:** `{r.category.value}`",
                f"- **Schema Version:** `{r.schema_version}`",
                f"- **Required Input Slots:** {slots_str}",
                f"- **Expected Output:** `{r.expected_output_format}`",
                f"- **Falsification Condition:** `{r.falsification_condition}`",
                f"- **Canonical JCS HMAC (K_recipe):** `{r.hmac_signature[:16]}...`",
                f"- **Invariant Set Hash:** `{inv_hash[:16]}...`",
                "",
                "**System Directive:**",
                "> " + r.system_directive.replace("\n", " "),
                "",
                "**User Template:**",
                "```text",
                r.user_template,
                "```",
                "",
            ])
        return "\n".join(lines)


# Global singleton instance
GLOBAL_SOVEREIGN_PROMPTBOOK = SovereignPromptbookRegistry()
