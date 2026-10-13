#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: CANONICAL SEMANTIC STATUS PROTOCOL v1.0 (SSOT)
==============================================================================
Module: krystal_kernel/verification/semantic_protocol.py
Description:
    Single Source of Truth (SSOT) for the 16-bit Status Code Matrix,
    Orthogonal Namespace Taxonomies, and Cross-Language ABI.

    Guarantees:
      - 15 Distinct Semantic Families (Upper Nibble 0x0 to 0xE, 0xF Reserved).
      - Category Separation: "Veto" is an authoritative verdict, NOT an epistemic state.
      - Disambiguated Quarantines: EPI_QUARANTINED, AUTH_QUARANTINED, PDX_QUARANTINED, MEM_QUARANTINED.
      - Deterministic dual-gate authority decomposition: f(SafetyGate, EvidenceGate) -> AuthorityState.
      - Automated transpilation to C ABI header, Rust module, JSON wire, and CSS classes.

    Constitutional Invariant:
      assert VITAL_MAX_HP == 6
==============================================================================
"""

import os
from enum import IntEnum, unique
from typing import Dict, Any, Optional, Tuple, Set

VITAL_MAX_HP: int = 6

# --- Bitmask Layout ---
FAMILY_SHIFT = 12
FAMILY_MASK = 0xF000

SUBKIND_SHIFT = 8
SUBKIND_MASK = 0x0F00

VALUE_MASK = 0x00FF


# ==============================================================================
# 1. STATUS CODE FAMILIES (Upper 4 bits: 0x0 to 0xF)
# ==============================================================================
@unique
class StatusCodeFamily(IntEnum):
    """The 16-bit Status Code Family (Top Nibble)."""
    SYS  = 0x0  # System & Base Lifecycle
    EPI  = 0x1  # Epistemic Truth & Lattice (What is known/unknown)
    TMP  = 0x2  # Temporal Lease & Freshness (TTL, monotonic latches)
    AUTH = 0x3  # Authority & Governance Gates (Verdicts, permissions)
    INV  = 0x4  # Constitutional Invariants & Safety Gate (VITAL_MAX_HP == 6)
    ACT  = 0x5  # Actuation Plane (Physical mutations & hardware dispatch)
    TXN  = 0x6  # Transactional Lifecycle (2-Phase Commit, compensation)
    OBS  = 0x7  # Physical Sensor Observation (Raw silicon telemetry)
    DER  = 0x8  # Mathematically Derived Signals (Headroom, energy deltas)
    PRED = 0x9  # Probabilistic Model Predictions & Forecast Horizons
    MEM  = 0xA  # Native Memory Fabric & Lifetimes (Arenas, ring buffers)
    HW   = 0xB  # Physical Silicon State (Thermal throttling, RAPL clamps)
    PDX  = 0xC  # Epistemic Paradoxes & Pathological Resonances
    MRK  = 0xD  # Merkle Hash Trees & Cryptographic DAG Integrity
    POL  = 0xE  # Dynamic Policies & Contextual Thresholds
    RESERVED_FUTURE = 0xF  # Strictly Reserved for Future Architectural Horizon


# ==============================================================================
# 2. CANONICAL ENUMS WITH 16-BIT NATIVE REPRESENTATIONS
# ==============================================================================

@unique
class EpistemicState(IntEnum):
    """0x1--- : Epistemic Evidence & Truth Lattice (What the system knows)."""
    EPI_UNKNOWN       = 0x1001  # Clock drift, uncalibrated domain, or missing origin
    EPI_OBSERVED      = 0x1002  # Directly sampled from physical or runtime sensors
    EPI_DERIVED       = 0x1003  # Mathematically computed from verified primary signals
    EPI_PREDICTED     = 0x1004  # Probabilistic / counterfactual model prediction
    EPI_CALIBRATED    = 0x1005  # Empirically adjusted by prediction error ledger
    EPI_VALIDATED     = 0x1006  # Formally verified against safety and evidence gates
    EPI_FALSIFIED     = 0x1007  # Empirically refuted by evidence probe
    EPI_CONTRADICTORY = 0x1008  # Disagreement detected across independent sensor channels
    EPI_QUARANTINED   = 0x1009  # Sensor stream frozen due to anomalous divergence or fault


@unique
class TemporalState(IntEnum):
    """0x2--- : Temporal Lease & Freshness Dynamics (How long it is valid)."""
    TMP_FRESH               = 0x2001  # Well within sampling window
    TMP_LEASED              = 0x2002  # Valid only within nanosecond TTL lease
    TMP_STALE               = 0x2003  # Exceeded sampling interval but within grace period
    TMP_EXPIRED             = 0x2004  # Exceeded valid_until; historical evidence ONLY
    TMP_MONOTONIC_LATCH     = 0x2005  # Write-once True within epoch; never resets to False
    TMP_BISTABLE_HYSTERETIC = 0x2006  # Requires upper/lower hysteresis margin to flip
    TMP_SHADOW_HYPOTHETICAL = 0x2007  # Exists strictly in shadow runtime; zero-mutation guaranteed


@unique
class AuthorityState(IntEnum):
    """0x3--- : Authoritative Governance Verdicts (What authority permits)."""
    AUTH_COMMIT_APPROVED = 0x3001  # Action permitted to proceed to actuation/commit
    AUTH_DEFERRED        = 0x3002  # Evidence insufficient; micro-probe sampling triggered
    AUTH_QUARANTINED     = 0x3003  # Action frozen by authority due to suspicious risk profile
    AUTH_VETOED          = 0x3004  # Explicitly rejected by governance gate (pre-actuation NOOP)
    AUTH_SHADOW_ONLY     = 0x3005  # Permitted exclusively in non-actuating counterfactual shadow


@unique
class SafetyState(IntEnum):
    """0x41-- : Binary Safety Filter (Sub-family under Invariant/Safety)."""
    SAFE_PASS   = 0x4101  # All hard physical and safety constraints satisfied
    SAFE_BREACH = 0x4102  # Violation of physical limit (temperature, power, memory bound)


@unique
class InvariantState(IntEnum):
    """0x40-- : Constitutional Invariants & Vitality (VITAL_MAX_HP == 6)."""
    INV_DEPLETED_ZERO  = 0x4000  # 0/6 HP: Catastrophic depletion / safety freeze
    INV_CRITICAL_LOCK  = 0x4001  # 1/6 HP: Critical threshold; high-risk actions vetoed
    INV_DEGRADED_WARN  = 0x4003  # 3/6 HP: Degraded vitality; telemetry frequency doubled
    INV_NOMINAL_6HP    = 0x4006  # 6/6 HP: Optimal constitutional equilibrium
    INV_BREACH_TRIPPED = 0x400E  # Attempted insertion into bounded list exceeding capacity 6


@unique
class ActuationState(IntEnum):
    """0x5--- : Actuation & Physical Execution Plane."""
    ACT_IDLE         = 0x5001  # No physical mutation in flight
    ACT_DISPATCHED   = 0x5002  # Directive sent to cgroups/QMP/Hyper-V/RAPL
    ACT_EXECUTED     = 0x5003  # Hardware confirmed completion of directive
    ACT_REVERTED     = 0x5004  # Direct compensation/rollback completed successfully


@unique
class TransactionState(IntEnum):
    """0x6--- : Transactional Lifecycle (2-Phase Commit & Rollback)."""
    TXN_IDLE         = 0x6001  # Ready for candidate proposal
    TXN_PREPARED     = 0x6002  # Candidate action assembled and locked
    TXN_VERIFIED     = 0x6003  # Proof bundle passed dual-gate validation
    TXN_COMMITTED    = 0x6004  # Atomic commit verified
    TXN_ABORTED      = 0x6005  # Rejected before mutation
    TXN_COMPENSATING = 0x6006  # Executing compensating inverse action
    TXN_ROLLED_BACK  = 0x6007  # State restored to pre-transaction StateVersion


@unique
class ObservationState(IntEnum):
    """0x7--- : Physical Raw Silicon Observations."""
    OBS_RAW_SAMPLE        = 0x7001  # Direct silicon register reading
    OBS_CALIBRATED_SAMPLE = 0x7002  # Corrected for known sensor bias/temperature curve
    OBS_DRIFT_DETECTED    = 0x7003  # Sampling noise exceeding sensor sigma bounds


@unique
class DerivedState(IntEnum):
    """0x8--- : Mathematically Derived Signals."""
    DER_COMPUTED_DELTA     = 0x8001  # First derivative delta (e.g. dT/dt, dE/dt)
    DER_PRESSURE_SCORE     = 0x8002  # Multi-dimensional pressure metric in [0, 1]
    DER_THERMAL_HEADROOM   = 0x8003  # Margin to TjMax (e.g. 100C - current_temp)
    DER_INTERFERENCE_INDEX = 0x8004  # Resource contention index across neighbors


@unique
class PredictionState(IntEnum):
    """0x9--- : Probabilistic Model Forecasts."""
    PRED_POINT_ESTIMATE       = 0x9001  # Single model point prediction
    PRED_CONFIDENCE_BOUND     = 0x9002  # Upper/lower credible interval [y_low, y_high]
    PRED_BREACH_WARNING       = 0x9003  # Forecast predicts threshold breach within horizon
    PRED_COUNTERFACTUAL_DELTA = 0x9004  # Expected delta in shadow runtime


@unique
class MemoryState(IntEnum):
    """0xA--- : Native Memory Fabric & Lifetimes."""
    MEM_ARENA_ALLOC    = 0xA001  # Arena allocated buffer
    MEM_ZERO_COPY_VIEW = 0xA002  # Borrowed view over ring buffer (zero allocation)
    MEM_QUARANTINED    = 0xA003  # Memory arena retired due to corruption or leak


@unique
class HardwareState(IntEnum):
    """0xB--- : Physical Silicon Hardware Execution Status."""
    HW_NOMINAL           = 0xB001  # Operating within design envelope
    HW_THROTTLED_THERMAL = 0xB002  # CPU/GPU PROCHOT / thermal throttling engaged
    HW_RAPL_CLAMPED      = 0xB003  # Intel RAPL / AMD power budget clamped


@unique
class ParadoxState(IntEnum):
    """0xC--- : Systemic Paradoxes & Pathological Resonance States."""
    PDX_DORMANT     = 0xC001  # Nominal equilibrium; zero paradox alarms
    PDX_RESONATING  = 0xC002  # Closed feedback loop detected (e.g. SP-11, SP-12)
    PDX_QUARANTINED = 0xC003  # Feedback loop isolated from actuation plane
    PDX_AUTO_HEALED = 0xC004  # Remediated by autonomous healing circuit


@unique
class MerkleState(IntEnum):
    """0xD--- : Merkle Hash Trees & Cryptographic DAG Integrity."""
    MRK_ROOT_VALID    = 0xD001  # Cryptographic root verified against evidence DAG
    MRK_LEAF_STALE    = 0xD002  # Individual leaf expired relative to current StateVersion
    MRK_HASH_MISMATCH = 0xD003  # Cryptographic integrity verification failed


@unique
class PolicyState(IntEnum):
    """0xE--- : Dynamic Contextual Policies."""
    POL_ACTIVE       = 0xE001  # Policy actively governing current decisions
    POL_CALIBRATING  = 0xE002  # Policy in online calibration via PredictionErrorLedger
    POL_SUPERSEDED   = 0xE003  # Replaced by newer PolicyVersion
    POL_REJECTED     = 0xE004  # Proposed policy failed offline simulation benchmark
    POL_EXPERIMENTAL = 0xE005  # Active only in shadow runtime


# ==============================================================================
# 3. REGISTRY MAPS AND WIRE TOKENS
# ==============================================================================

ALL_ENUM_CLASSES = [
    EpistemicState,
    TemporalState,
    AuthorityState,
    SafetyState,
    InvariantState,
    ActuationState,
    TransactionState,
    ObservationState,
    DerivedState,
    PredictionState,
    MemoryState,
    HardwareState,
    ParadoxState,
    MerkleState,
    PolicyState,
]

CODE_TO_ENUM_MAP: Dict[int, Any] = {}
NAME_TO_CODE_MAP: Dict[str, int] = {}
CODE_TO_WIRE_MAP: Dict[int, str] = {}
WIRE_TO_CODE_MAP: Dict[str, int] = {}

for enum_cls in ALL_ENUM_CLASSES:
    for member in enum_cls:
        val = int(member.value)
        name = member.name
        CODE_TO_ENUM_MAP[val] = member
        NAME_TO_CODE_MAP[name] = val

        # Derive canonical wire token, e.g. EPI_VALIDATED -> "epi:validated"
        parts = name.split("_", 1)
        if len(parts) == 2:
            prefix, rest = parts
            wire = f"{prefix.lower()}:{rest.lower()}"
        else:
            wire = name.lower()
        CODE_TO_WIRE_MAP[val] = wire
        WIRE_TO_CODE_MAP[wire] = val


# ==============================================================================
# 4. PROTOCOL METHODS & VALIDATORS
# ==============================================================================

def get_status_family(code: int) -> StatusCodeFamily:
    """Extracts the high-order nibble family from a 16-bit status code."""
    nibble = (code & FAMILY_MASK) >> FAMILY_SHIFT
    return StatusCodeFamily(nibble)


def to_wire_token(code: int) -> str:
    """Converts a 16-bit status code into its canonical lowercase wire token."""
    if code in CODE_TO_WIRE_MAP:
        return CODE_TO_WIRE_MAP[code]
    return f"raw:0x{code:04x}"


def from_wire_token(wire: str) -> int:
    """Parses a canonical wire token into its 16-bit status code integer."""
    if wire in WIRE_TO_CODE_MAP:
        return WIRE_TO_CODE_MAP[wire]
    if wire.startswith("raw:0x"):
        return int(wire[6:], 16)
    raise KeyError(f"Unknown wire token: '{wire}'")


def assert_orthogonal_category(code: int, expected_family: StatusCodeFamily) -> None:
    """
    Guarantees no category crossing.
    Raises TypeError if someone attempts to pass a status code belonging to
    another domain (e.g. AUTH_VETOED into an Epistemic slot).
    """
    actual = get_status_family(code)
    if actual != expected_family:
        raise TypeError(
            f"Category error: expected family {expected_family.name} (0x{expected_family.value:X}---), "
            f"but received status 0x{code:04X} belonging to family {actual.name} (0x{actual.value:X}---)"
        )


def evaluate_dual_gate(safety: SafetyState, evidence: EpistemicState) -> AuthorityState:
    """
    Canonical Epistemic Dual-Gate Authority Decomposition:
      AuthorityVerdict = f(SafetyGate, EvidenceGate)

    Mathematical invariant:
      Safety is a hard binary invariant filter; Evidence is a qualitative epistemic lattice.
    """
    assert_orthogonal_category(int(safety), StatusCodeFamily.INV)
    assert_orthogonal_category(int(evidence), StatusCodeFamily.EPI)

    if safety == SafetyState.SAFE_BREACH:
        # Pre-actuation veto: safety breaches NEVER proceed to actuation
        return AuthorityState.AUTH_VETOED

    if evidence == EpistemicState.EPI_VALIDATED:
        return AuthorityState.AUTH_COMMIT_APPROVED

    if evidence in (EpistemicState.EPI_CONTRADICTORY, EpistemicState.EPI_QUARANTINED):
        return AuthorityState.AUTH_QUARANTINED

    # Incomplete or predictive evidence triggers micro-probe deferral
    return AuthorityState.AUTH_DEFERRED


# ==============================================================================
# 5. CROSS-LANGUAGE CODE GENERATORS (C ABI, RUST, CSS)
# ==============================================================================

def generate_c_header() -> str:
    """Generates the C/C++ ABI Header (krystal_status_abi.h)."""
    lines = [
        "/* ============================================================================",
        " * KRYSTAL-STACK: CANONICAL 16-BIT STATUS ABI HEADER",
        " * Generated automatically from krystal_kernel.verification.semantic_protocol",
        " * Constitutional Invariant: VITAL_MAX_HP == 6",
        " * ============================================================================ */",
        "",
        "#ifndef KRYSTAL_STATUS_ABI_H",
        "#define KRYSTAL_STATUS_ABI_H",
        "",
        "#include <stdint.h>",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        "typedef uint16_t krystal_status_t;",
        "",
        "#define KRYSTAL_VITAL_MAX_HP 6",
        "",
        "/* Status Code Family Masks */",
        "#define KRYSTAL_FAMILY_MASK   0xF000",
        "#define KRYSTAL_SUBKIND_MASK  0x0F00",
        "#define KRYSTAL_VALUE_MASK    0x00FF",
        "",
        "#define KRYSTAL_GET_FAMILY(status)  (((status) & KRYSTAL_FAMILY_MASK) >> 12)",
        "#define KRYSTAL_GET_SUBKIND(status) (((status) & KRYSTAL_SUBKIND_MASK) >> 8)",
        "#define KRYSTAL_GET_VALUE(status)   ((status) & KRYSTAL_VALUE_MASK)",
        "",
        "/* Status Code Constants */"
    ]

    for enum_cls in ALL_ENUM_CLASSES:
        lines.append(f"\n/* {enum_cls.__doc__} */")
        for member in enum_cls:
            c_name = f"KRYSTAL_{member.name}"
            lines.append(f"#define {c_name:<36} 0x{member.value:04X}")

    lines.extend([
        "",
        "#ifdef __cplusplus",
        "}",
        "#endif",
        "",
        "#endif /* KRYSTAL_STATUS_ABI_H */",
        ""
    ])
    return "\n".join(lines)


def generate_rust_module() -> str:
    """Generates the Rust FFI/ABI module (src/rust-bot/src/epistemic_abi.rs)."""
    lines = [
        "// ============================================================================",
        "// KRYSTAL-STACK: CANONICAL RUST EPISTEMIC ABI",
        "// Generated automatically from krystal_kernel.verification.semantic_protocol",
        "// Constitutional Invariant: VITAL_MAX_HP == 6",
        "// ============================================================================",
        "",
        "#![allow(dead_code)]",
        "",
        "pub const VITAL_MAX_HP: usize = 6;",
        "",
        "pub const FAMILY_MASK: u16 = 0xF000;",
        "pub const SUBKIND_MASK: u16 = 0x0F00;",
        "pub const VALUE_MASK: u16 = 0x00FF;",
        "",
        "#[inline]",
        "pub const fn get_family(status: u16) -> u8 {",
        "    ((status & FAMILY_MASK) >> 12) as u8",
        "}",
        ""
    ]

    for enum_cls in ALL_ENUM_CLASSES:
        rust_name = enum_cls.__name__
        lines.append(f"#[repr(u16)]")
        lines.append(f"#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]")
        lines.append(f"pub enum {rust_name} {{")
        for member in enum_cls:
            parts = member.name.split("_")
            camel = "".join(p.capitalize() for p in parts)
            lines.append(f"    {camel:<26} = 0x{member.value:04X},")
        lines.append("}\n")

    return "\n".join(lines)


def generate_css_badge_classes() -> str:
    """Generates CSS utility classes for the Krystal UI Kit."""
    lines = [
        "/* ============================================================================",
        " * KRYSTAL-STACK LIVING UI KIT: CANONICAL STATUS BADGE TOKENS",
        " * Generated from krystal_kernel.verification.semantic_protocol",
        " * ============================================================================ */",
        ""
    ]

    color_schemes = {
        "epi": { "bg": "rgba(56, 189, 248, 0.15)", "color": "#38bdf8", "border": "rgba(56, 189, 248, 0.35)" },
        "auth": { "bg": "rgba(168, 85, 247, 0.15)", "color": "#c084fc", "border": "rgba(168, 85, 247, 0.35)" },
        "inv": { "bg": "rgba(244, 63, 94, 0.15)", "color": "#fb7185", "border": "rgba(244, 63, 94, 0.35)" },
        "safe": { "bg": "rgba(16, 185, 129, 0.15)", "color": "#34d399", "border": "rgba(16, 185, 129, 0.35)" },
        "der": { "bg": "rgba(251, 191, 36, 0.15)", "color": "#fcd34d", "border": "rgba(251, 191, 36, 0.35)" },
        "pred": { "bg": "rgba(236, 72, 153, 0.15)", "color": "#f472b6", "border": "rgba(236, 72, 153, 0.35)" },
        "pdx": { "bg": "rgba(245, 158, 11, 0.18)", "color": "#fbbf24", "border": "rgba(245, 158, 11, 0.4)" },
    }

    for code, wire in CODE_TO_WIRE_MAP.items():
        prefix = wire.split(":")[0]
        class_name = f"badge-{wire.replace(':', '-')}"
        scheme = color_schemes.get(prefix, { "bg": "rgba(255, 255, 255, 0.08)", "color": "#e2e8f0", "border": "rgba(255, 255, 255, 0.2)" })
        lines.append(f".{class_name} {{")
        lines.append(f"    background: {scheme['bg']};")
        lines.append(f"    color: {scheme['color']};")
        lines.append(f"    border: 1px solid {scheme['border']};")
        lines.append(f"    border-radius: var(--radius-xs, 4px);")
        lines.append(f"    padding: 2px 6px;")
        lines.append(f"    font-family: var(--font-code, monospace);")
        lines.append(f"    font-size: 10px;")
        lines.append(f"    font-weight: 600;")
        lines.append(f"    display: inline-flex;")
        lines.append(f"    align-items: center;")
        lines.append("}\n")

    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Krystal Semantic Status Protocol v1.0 [OK] (VITAL_MAX_HP={VITAL_MAX_HP})")
    print(f"Registered Enum Members: {len(CODE_TO_ENUM_MAP)}")
    print(f"Sample Status: EPI_VALIDATED -> 0x{EpistemicState.EPI_VALIDATED:04X} -> '{to_wire_token(EpistemicState.EPI_VALIDATED)}'")
