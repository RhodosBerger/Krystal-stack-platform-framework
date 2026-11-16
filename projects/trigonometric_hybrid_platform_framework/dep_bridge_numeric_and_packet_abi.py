"""Dependency Bridge Module 1: Unified Numeric Sanitization, Constitutional Constants,
128 B KPKT Packet ABI, and 32 B .kopt Optical Record Contract.

Decouples consumer modules from internal implementation details while enforcing
`VITAL_MAX_HP == 6`, `UMA < 0.78`, and `ContractViolation` under `python -O`.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Sequence

from krystal_kernel.constants import UMA_SATURATION_THRESHOLD, VITAL_MAX_HP, ContractViolation
from krystal_kernel.zero_copy_contracts import canonicalize_float32

from .arena_cache_optical_sockets_and_consumption import read_optical_binary, synthesize_and_write_optical_binary
from .cluster_pid_gemma_finetune_and_mesh_ml import pack_intranet_128b_packet, unpack_intranet_128b_packet


def sanitize_vector_f32(values: Sequence[float]) -> list[float]:
    """Sanitize a numeric sequence to finite IEEE-754 float32 without subnormals."""
    if VITAL_MAX_HP != 6:
        raise ContractViolation("VITAL_MAX_HP != 6 in ABI bridge", violation_code="ERR_ABI_HP")
    out: list[float] = []
    for v in values:
        fv = float(v)
        if not math.isfinite(fv):
            raise ContractViolation(f"Non-finite value in vector: {v!r}", violation_code="ERR_ABI_NONFINITE")
        out.append(canonicalize_float32(fv))
    return out


def emit_governed_128b_event(
    pkt_type: int,
    src_pid: int,
    dst_pid: int,
    uma_utilization: float,
    tag: str,
) -> Dict[str, Any]:
    """Pack, SHA-256 sign, and immediately verify a 128 B KPKT dependency-bridge event."""
    u = sanitize_vector_f32([uma_utilization])[0]
    raw = pack_intranet_128b_packet(pkt_type, src_pid, dst_pid, u, tag)
    if len(raw) != 128:
        raise ContractViolation("KPKT frame length != 128 B", violation_code="ERR_ABI_128B")
    decoded = unpack_intranet_128b_packet(raw)
    decoded["shadow_fallback_active"] = u >= UMA_SATURATION_THRESHOLD
    decoded["frame_sha256_hex"] = hashlib.sha256(raw).hexdigest()
    return decoded


def verify_abi_and_optical_dependencies() -> Dict[str, Any]:
    """Health-check ABI packet framing and .kopt optical binary integrity."""
    synthesize_and_write_optical_binary()
    kopt = read_optical_binary()
    evt = emit_governed_128b_event(3, 16777217, 17104902, 0.35, "ABI_BRIDGE_READY")
    return {
        "bridge_id": "DEP_BRIDGE_01_NUMERIC_AND_PACKET_ABI",
        "vital_max_hp": VITAL_MAX_HP,
        "kopt_verified": kopt["sha256_verified"],
        "packet_verified": evt["sha256_verified"],
        "frame_sha256_hex": evt["frame_sha256_hex"],
    }
