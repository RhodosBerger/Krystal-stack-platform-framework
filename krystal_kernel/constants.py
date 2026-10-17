"""
Krystal Stack Platform Framework - Global Constitutional Constants and Types
=============================================================================
Module: krystal_kernel/constants.py
Milestone: V-KSPP-PRINCIPLES-01
Reference: docs/architecture/KRYSTAL_SOVEREIGN_PRINCIPLE_PACK.md

Core invariants, binary layout parameters, bitmasks, and thresholds governing
all 10 Sovereign Principles in KSPP v1.
"""

from enum import IntEnum, IntFlag
from typing import Final

# Princíp 10: Ústavný invariant prežitia
VITAL_MAX_HP: Final[int] = 6

# Princíp 2: Konštanty 128-bajtového Zero-Copy rámca
FRAME_MAGIC: Final[int] = 0x4B525953  # "KRYS" v little-endian
FRAME_VERSION: Final[int] = 2
FRAME_SIZE_BYTES: Final[int] = 128
CACHE_LINE_BYTES: Final[int] = 64
AUTHENTICATED_PREFIX_BYTES: Final[int] = 96  # Bajty 0..95 kryté HMAC-SHA256

# Princíp 3 & 8: Hardvérové a časové prahy
UMA_SATURATION_THRESHOLD: Final[float] = 0.78
HYSTERESIS_DWELL_FRAMES: Final[int] = 12
HYSTERESIS_DOWNSCALE_RATIO: Final[float] = 0.90
HYSTERESIS_UPSCALE_RATIO: Final[float] = 0.70
ARRHENIUS_MAX_TEMP_C: Final[float] = 75.0


class CapabilityScope(IntFlag):
    """Princíp 4: Striktná separácia právomocí (Authority Separation)."""
    NONE          = 0
    CAN_OBSERVE   = 1 << 0
    CAN_PROPOSE   = 1 << 1
    CAN_VERIFY    = 1 << 2
    CAN_AUTHORIZE = 1 << 3
    CAN_EXECUTE   = 1 << 4


class FrameFlags(IntFlag):
    """Bitové príznaky na offsete 16..19 v 128B rámci."""
    NONE                        = 0
    VBLANK_ACTIVE               = 1 << 0  # Práve prebieha VBlank tieň
    UMA_THROTTLE_ACTIVE         = 1 << 1  # Zbernica >= 78 %, aktivovaný útlm
    HYSTERESIS_HELD             = 1 << 2  # Upscale blokovaný Schmitt-Triggerom
    BIFURCATED_SUPERPOSITION    = 1 << 3  # Rámec patrí do nerozriešeného konfliktu
    PARASITIC_DECOUPLED         = 1 << 4  # Hypotéza odpojená kalorimetrom
    FULL_HMAC_VERIFIED          = 1 << 5  # 96B kryptografický tag platný


from typing import Any, Final, Optional


class ContractViolation(ValueError, RuntimeError):
    """Výnimka vyvolaná pri porušení ktoréhokoľvek z 10 princípov KSPP."""

    def __init__(
        self,
        message: str,
        violation_code_or_contract_id: Optional[str] = None,
        violation_code: Optional[str] = None,
        contract_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if violation_code is not None:
            v_code = violation_code
            c_id = contract_id or violation_code_or_contract_id
        elif violation_code_or_contract_id is not None:
            if violation_code_or_contract_id.startswith("ERR_"):
                v_code = violation_code_or_contract_id
                c_id = contract_id
            else:
                c_id = violation_code_or_contract_id
                v_code = kwargs.get("violation_code") or "ERR_CONTRACT_VIOLATION"
        else:
            v_code = kwargs.get("violation_code") or "ERR_CONTRACT_VIOLATION"
            c_id = contract_id or kwargs.get("contract_id")

        self.message = message
        self.violation_code = v_code
        self.contract_id = c_id

        ctx = f" [Contract: {self.contract_id}]" if self.contract_id else ""
        formatted = f"[{self.violation_code}]{ctx} {self.message}" if not message.startswith("[") else message
        super().__init__(formatted)

    def __str__(self) -> str:
        ctx = f" [Contract: {self.contract_id}]" if self.contract_id else ""
        return f"[{self.violation_code}]{ctx} {self.message}"
