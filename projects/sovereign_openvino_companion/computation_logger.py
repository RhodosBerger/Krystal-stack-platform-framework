"""
KRYSTAL-STACK // SOVEREIGN OPENVINO COMPANION — Cryptographic Computation Logging System
========================================================================================
Implements a tamper-evident, hash-chained multi-channel computation ledger
governed by Sovereign Principle P9 (Full-Frame Provenance) and P10 (VITAL_MAX_HP == 6).

Each logged computation record normalizes floats via canonical IEEE-754 float32
representation (rejecting NaN/Inf with ContractViolation), serializes deterministically
using JCS canonical JSON, chains `prev_entry_sha256` -> `entry_sha256`, and records
captured reasoning / anomalous telemetry pattern tags across the 10-Stratum stack.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from krystal_kernel.claims_lattice import canonical_jcs_serialize
from krystal_kernel.constants import ContractViolation, VITAL_MAX_HP
from krystal_kernel.zero_copy_contracts import canonicalize_float32

if VITAL_MAX_HP != 6:
    raise ContractViolation(
        f"Constitutional invariant breach: VITAL_MAX_HP={VITAL_MAX_HP} != 6",
        violation_code="ERR_CONSTITUTIONAL_INVARIANT_BREACH",
    )

VALID_COMPUTATION_CHANNELS: Tuple[str, ...] = (
    "system",
    "telemetry",
    "planning",
    "policy",
    "inference",
    "companion_vblank",
    "gnosis_audit",
)

REASONING_PATTERN_CATALOG: Dict[str, str] = {
    "UMA_BANDWIDTH_PRESSURE_SPIKE": "bus_pressure >= 0.78 triggers immediate CPU fallback and K_spec=8->1 compression",
    "HYSTERETIC_DWELL_HOLD": "Schmitt-trigger 12-frame dwell blocks premature GPU upscale oscillation (Sovereign P8)",
    "MICRO_TENSOR_CPU_BYPASS": "payload < 16384 B routes to CPU VNNI avoiding 403 us iGPU queue dispatch latency",
    "THERMAL_PARASITE_DECOUPLED": "Delta_tau <= 0 epistemic entropy gain triggers instant P5 hypothesis quarantine",
    "HOMOTOPIC_CONFLICT_FORK": "Contradictory evidence (E_A perp E_B) forks alpha perp beta without convex averaging (P6)",
    "AUTHORITY_SEPARATION_QUARANTINE": "PROPOSER != VERIFIER or degraded/failed AepReceipt holds state delta at zero",
}

DEFAULT_LOG_DIR: Path = (
    Path(__file__).resolve().parent.parent.parent / "artifacts" / "logs" / "sovereign_openvino_companion"
)


def _canonicalize_value(value: Any) -> Any:
    """Recursively normalizes floats via canonicalize_float32 so JCS hashes remain bit-exact."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractViolation(
                f"Non-finite float rejected in computation log: {value}",
                violation_code="ERR_LOG_NON_FINITE_FLOAT",
            )
        return round(canonicalize_float32(value), 7)
    if isinstance(value, dict):
        return {str(k): _canonicalize_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(v) for v in value]
    return value


@dataclass(frozen=True)
class ComputationLogEntry:
    seq_index: int
    timestamp_ns: int
    channel: str
    operation: str
    stratum: str
    duration_ms: float
    epistemic_status: str
    detected_patterns: List[str]
    payload: Dict[str, Any]
    prev_entry_sha256: str
    entry_sha256: str
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ComputationLogSystem:
    """
    Append-only cryptographic JSONL computation ledger across 7 telemetry channels.
    Zero bare asserts: raises ContractViolation on invariant, float, or chain breach.
    """

    def __init__(self, log_dir: Optional[Path] = None, reset_ledger: bool = False) -> None:
        if VITAL_MAX_HP != 6:
            raise ContractViolation("VITAL_MAX_HP must equal 6 in ComputationLogSystem")

        self.log_dir = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.log_dir / "computation_ledger.jsonl"
        self.summary_path = self.log_dir / "computation_summary.json"

        self.entries: List[ComputationLogEntry] = []
        self._head_sha256: str = "0" * 64
        self._seq_index: int = 0

        if reset_ledger and self.ledger_path.exists():
            self.ledger_path.unlink()
        elif self.ledger_path.exists():
            self._hydrate_existing_ledger()

    def _hydrate_existing_ledger(self) -> None:
        lines = [ln for ln in self.ledger_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for raw_line in lines:
            rec = json.loads(raw_line)
            entry = ComputationLogEntry(
                seq_index=int(rec["seq_index"]),
                timestamp_ns=int(rec["timestamp_ns"]),
                channel=str(rec["channel"]),
                operation=str(rec["operation"]),
                stratum=str(rec["stratum"]),
                duration_ms=float(rec["duration_ms"]),
                epistemic_status=str(rec["epistemic_status"]),
                detected_patterns=list(rec.get("detected_patterns", [])),
                payload=dict(rec.get("payload", {})),
                prev_entry_sha256=str(rec["prev_entry_sha256"]),
                entry_sha256=str(rec["entry_sha256"]),
                vital_max_hp=int(rec.get("vital_max_hp", VITAL_MAX_HP)),
            )
            self.entries.append(entry)
            self._head_sha256 = entry.entry_sha256
            self._seq_index = entry.seq_index + 1

    @staticmethod
    def _compute_entry_hash(unsigned_record: Dict[str, Any]) -> str:
        return hashlib.sha256(canonical_jcs_serialize(unsigned_record)).hexdigest()

    @staticmethod
    def classify_reasoning_patterns(payload: Dict[str, Any]) -> List[str]:
        """Detects anomalous & strategic reasoning patterns directly from telemetry payload fields."""
        patterns: List[str] = []
        uma = payload.get("uma_saturation")
        if isinstance(uma, (int, float)) and uma >= 0.78:
            patterns.append("UMA_BANDWIDTH_PRESSURE_SPIKE")
        if payload.get("hysteresis_held") is True:
            patterns.append("HYSTERETIC_DWELL_HOLD")
        payload_bytes = payload.get("payload_bytes")
        if isinstance(payload_bytes, int) and payload_bytes < 16384 and payload.get("selected_device") == "CPU":
            patterns.append("MICRO_TENSOR_CPU_BYPASS")
        if payload.get("thermal_parasite_decoupled") is True:
            patterns.append("THERMAL_PARASITE_DECOUPLED")
        if payload.get("contradictory_evidence") is True:
            patterns.append("HOMOTOPIC_CONFLICT_FORK")
        if payload.get("quarantined_delta_zero") is True:
            patterns.append("AUTHORITY_SEPARATION_QUARANTINE")
        return patterns

    def log_event(
        self,
        channel: str,
        operation: str,
        stratum: str,
        duration_ms: float,
        payload: Dict[str, Any],
        epistemic_status: str = "LOCAL-MEASURED",
        detected_patterns: Optional[List[str]] = None,
    ) -> ComputationLogEntry:
        if VITAL_MAX_HP != 6:
            raise ContractViolation("Constitutional violation: VITAL_MAX_HP != 6")
        if channel not in VALID_COMPUTATION_CHANNELS:
            raise ContractViolation(
                f"Invalid log channel '{channel}'; expected one of {VALID_COMPUTATION_CHANNELS}",
                violation_code="ERR_LOG_INVALID_CHANNEL",
            )
        if not math.isfinite(duration_ms) or duration_ms < 0.0:
            raise ContractViolation(
                f"Invalid duration_ms '{duration_ms}'; must be finite and >= 0.0",
                violation_code="ERR_LOG_INVALID_DURATION",
            )

        clean_payload = _canonicalize_value(payload)
        clean_duration = _canonicalize_value(float(duration_ms))
        auto_patterns = self.classify_reasoning_patterns(clean_payload) if isinstance(clean_payload, dict) else []
        merged_patterns = sorted(set(auto_patterns).union(detected_patterns or []))

        seq = self._seq_index
        ts_ns = time.time_ns()
        prev_hash = self._head_sha256

        unsigned_record: Dict[str, Any] = {
            "seq_index": seq,
            "timestamp_ns": ts_ns,
            "channel": channel,
            "operation": str(operation),
            "stratum": str(stratum),
            "duration_ms": clean_duration,
            "epistemic_status": str(epistemic_status),
            "detected_patterns": merged_patterns,
            "payload": clean_payload,
            "prev_entry_sha256": prev_hash,
            "vital_max_hp": VITAL_MAX_HP,
        }
        entry_hash = self._compute_entry_hash(unsigned_record)

        entry = ComputationLogEntry(
            seq_index=seq,
            timestamp_ns=ts_ns,
            channel=channel,
            operation=str(operation),
            stratum=str(stratum),
            duration_ms=float(clean_duration),
            epistemic_status=str(epistemic_status),
            detected_patterns=merged_patterns,
            payload=clean_payload,
            prev_entry_sha256=prev_hash,
            entry_sha256=entry_hash,
            vital_max_hp=VITAL_MAX_HP,
        )

        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")

        self.entries.append(entry)
        self._head_sha256 = entry_hash
        self._seq_index += 1
        return entry

    def verify_chain_integrity(self, ledger_path: Optional[Path] = None) -> Tuple[bool, Dict[str, Any]]:
        target = Path(ledger_path) if ledger_path is not None else self.ledger_path
        if not target.exists():
            return False, {
                "chain_valid": False,
                "reason": "LEDGER_FILE_MISSING",
                "entries_checked": 0,
                "ledger_path": str(target),
            }

        expected_prev = "0" * 64
        checked = 0
        channel_counts: Dict[str, int] = {ch: 0 for ch in VALID_COMPUTATION_CHANNELS}
        pattern_counts: Dict[str, int] = {}

        for line_idx, raw_line in enumerate(target.read_text(encoding="utf-8").splitlines()):
            if not raw_line.strip():
                continue
            try:
                rec = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                return False, {
                    "chain_valid": False,
                    "reason": f"JSON_DECODE_ERROR_LINE_{line_idx}: {exc}",
                    "entries_checked": checked,
                    "ledger_path": str(target),
                }

            if int(rec.get("seq_index", -1)) != checked:
                return False, {
                    "chain_valid": False,
                    "reason": f"SEQ_INDEX_MISMATCH_AT_{checked}",
                    "entries_checked": checked,
                    "ledger_path": str(target),
                }
            if int(rec.get("vital_max_hp", -1)) != VITAL_MAX_HP:
                return False, {
                    "chain_valid": False,
                    "reason": f"VITAL_MAX_HP_BREACH_AT_{checked}",
                    "entries_checked": checked,
                    "ledger_path": str(target),
                }
            if rec.get("prev_entry_sha256") != expected_prev:
                return False, {
                    "chain_valid": False,
                    "reason": f"PREV_HASH_MISMATCH_AT_{checked}",
                    "entries_checked": checked,
                    "ledger_path": str(target),
                }

            unsigned_record = {
                "seq_index": rec["seq_index"],
                "timestamp_ns": rec["timestamp_ns"],
                "channel": rec["channel"],
                "operation": rec["operation"],
                "stratum": rec["stratum"],
                "duration_ms": rec["duration_ms"],
                "epistemic_status": rec["epistemic_status"],
                "detected_patterns": rec.get("detected_patterns", []),
                "payload": rec["payload"],
                "prev_entry_sha256": rec["prev_entry_sha256"],
                "vital_max_hp": rec["vital_max_hp"],
            }
            recomputed = self._compute_entry_hash(unsigned_record)
            if rec.get("entry_sha256") != recomputed:
                return False, {
                    "chain_valid": False,
                    "reason": f"SHA256_TAMPER_DETECTED_AT_{checked}",
                    "entries_checked": checked,
                    "ledger_path": str(target),
                }

            ch = str(rec.get("channel", "unknown"))
            channel_counts[ch] = channel_counts.get(ch, 0) + 1
            for pat in rec.get("detected_patterns", []):
                pattern_counts[pat] = pattern_counts.get(pat, 0) + 1

            expected_prev = recomputed
            checked += 1

        return True, {
            "chain_valid": True,
            "entries_checked": checked,
            "head_sha256": expected_prev,
            "channel_counts": channel_counts,
            "pattern_counts": pattern_counts,
            "ledger_path": str(target),
        }

    def write_summary_snapshot(self) -> Dict[str, Any]:
        durations = sorted(e.duration_ms for e in self.entries)
        channel_counts: Dict[str, int] = {ch: 0 for ch in VALID_COMPUTATION_CHANNELS}
        pattern_counts: Dict[str, int] = {}
        for entry in self.entries:
            channel_counts[entry.channel] = channel_counts.get(entry.channel, 0) + 1
            for pat in entry.detected_patterns:
                pattern_counts[pat] = pattern_counts.get(pat, 0) + 1

        if durations:
            p95_idx = min(len(durations) - 1, max(0, int(0.95 * len(durations)) - 1))
            median_ms = round(statistics.median(durations), 4)
            p95_ms = round(durations[p95_idx], 4)
            total_ms = round(sum(durations), 4)
        else:
            median_ms = 0.0
            p95_ms = 0.0
            total_ms = 0.0

        summary = {
            "schema_version": "COMPUTATION_LOG_SUMMARY_V1",
            "vital_max_hp": VITAL_MAX_HP,
            "total_entries": len(self.entries),
            "head_sha256": self._head_sha256,
            "ledger_path": str(self.ledger_path),
            "channel_counts": channel_counts,
            "detected_reasoning_patterns": pattern_counts,
            "pattern_catalog": REASONING_PATTERN_CATALOG,
            "duration_stats_ms": {
                "total_ms": total_ms,
                "median_ms": median_ms,
                "p95_ms": p95_ms,
            },
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
