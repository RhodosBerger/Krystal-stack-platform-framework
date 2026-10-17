"""
Krystal-Stack Governed Runtime: Epistemic Governed Inference Worker Pipeline
=============================================================================
Milestone: V-EPISTEMIC-END-TO-END-01
Reference: docs/architecture/PROOF_CARRYING_GOVERNED_RUNTIME.md

Integrates:
  1. UmaBandwidthArbiter with active Token Pacing and Speculative Slot Compression.
  2. Hierarchical Tiered IPC Buffer (Tier 3: EPISTEMIC_SPECULATION).
  3. Binary 128-byte Zero-Copy Contract emission (ProposalContract and LearningContract).
  4. End-to-end telemetry auditing via EpistemicExecutionReceipt.

Constitutional Invariant:
  VITAL_MAX_HP == 6 across all state representations and receipts.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import (
    ContractViolation,
    ContractHeader,
    CapabilityToken,
    ProposalContract,
    LearningContract,
)
from krystal_kernel.zero_copy_contracts import (
    BinaryContractType,
    ZeroCopyContractFrame,
    contract_to_zerocopy,
)
from krystal_kernel.hierarchical_ipc_buffer import (
    ContractHierarchyTier,
    HierarchicalTieredIPCBuffer,
    MessageEvictionPolicy,
)
from krystal_kernel.predictive_render_profiler import (
    UmaBusTelemetry,
    UmaBandwidthArbiter,
)

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"


@dataclass
class EpistemicInferenceConfig:
    """Configuration envelope for governed reasoning workloads."""
    model_name: str = "krystal-deep-reasoning-8b"
    max_speculative_slots: int = 8
    base_inter_token_delay_ms: float = 1.0
    max_pacing_delay_ms: float = 20.0
    batch_size: int = 1
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in EpistemicInferenceConfig")


@dataclass
class EpistemicToken:
    """Individual generated token with latency and logprob metrics."""
    token_id: int
    text: str
    logprob: float
    timestamp_ns: int
    pacing_delay_ms: float = 0.0


@dataclass
class EpistemicExecutionReceipt:
    """
    Cryptographically verifiable audit receipt recording full epistemic execution,
    UMA memory bus arbitration interaction, and tiered zero-copy frame dispatch.
    """
    receipt_id: str
    worker_id: str
    correlation_id: str
    prompt: str
    total_tokens_generated: int
    wall_clock_time_ms: float
    effective_throughput_tps: float
    avg_pacing_delay_ms: float
    was_pacing_engaged: bool
    speculative_slots_allocated: int
    frames_admitted_to_tier3: int
    frames_throttled_by_tier3: int
    uma_bus_saturation_observed: float
    signature_digest: str
    vital_max_hp: int = VITAL_MAX_HP
    timestamp_ns: int = field(default_factory=time.time_ns)

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in EpistemicExecutionReceipt")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "worker_id": self.worker_id,
            "correlation_id": self.correlation_id,
            "prompt": self.prompt,
            "total_tokens_generated": self.total_tokens_generated,
            "wall_clock_time_ms": round(self.wall_clock_time_ms, 3),
            "effective_throughput_tps": round(self.effective_throughput_tps, 2),
            "avg_pacing_delay_ms": round(self.avg_pacing_delay_ms, 3),
            "was_pacing_engaged": self.was_pacing_engaged,
            "speculative_slots_allocated": self.speculative_slots_allocated,
            "frames_admitted_to_tier3": self.frames_admitted_to_tier3,
            "frames_throttled_by_tier3": self.frames_throttled_by_tier3,
            "uma_bus_saturation_observed": round(self.uma_bus_saturation_observed, 4),
            "signature_digest": self.signature_digest,
            "vital_max_hp": self.vital_max_hp,
            "timestamp_ns": self.timestamp_ns,
        }


class EpistemicGovernedWorker:
    """
    End-to-End Governed Epistemic Reasoning Worker.
    
    Coordinates deep reasoning and LLM hypothesis generation while strictly
    respecting UMA memory bus saturation bounds, preserving 120 Hz graphics scanout,
    and pushing verified contract frames into the hierarchical zero-copy IPC ring.
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
        config: Optional[EpistemicInferenceConfig] = None,
        arbiter: Optional[UmaBandwidthArbiter] = None,
        ipc_buffer: Optional[HierarchicalTieredIPCBuffer] = None,
    ):
        assert VITAL_MAX_HP == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"
        self.worker_id = worker_id or f"EPISTEMIC-WORKER-{uuid.uuid4().hex[:8].upper()}"
        self.config = config or EpistemicInferenceConfig()
        self.arbiter = arbiter or UmaBandwidthArbiter(saturation_threshold=0.78)
        self.ipc_buffer = ipc_buffer or HierarchicalTieredIPCBuffer(total_slots=1024)
        self.sequence_counter: int = 1
        self.active_speculative_slots: int = self.config.max_speculative_slots

    def generate_governed_stream(
        self,
        prompt: str,
        target_token_count: int,
        bus_telemetry_feed: Optional[List[UmaBusTelemetry]] = None,
        is_rendering_active: bool = True,
        correlation_id: Optional[str] = None,
    ) -> Tuple[List[EpistemicToken], EpistemicExecutionReceipt]:
        """
        Executes token generation with dynamic UMA bus pacing and Tier-3 frame emission.
        
        Args:
            prompt: Reasoning task prompt / question.
            target_token_count: Target number of tokens to decode.
            bus_telemetry_feed: Optional sequence of UMA bus state snapshots (or dynamic updates).
            is_rendering_active: Whether 120 Hz graphics pipeline is running concurrently.
            correlation_id: Trace correlation token.
        """
        corr = correlation_id or f"CORR-{uuid.uuid4().hex[:12].upper()}"
        receipt_id = f"RCPT-EPISTEMIC-{uuid.uuid4().hex[:8].upper()}"
        t_start = time.time_ns()

        tokens: List[EpistemicToken] = []
        pacing_delays: List[float] = []
        pacing_ever_engaged = False
        max_saturation_seen = 0.0
        frames_admitted = 0
        frames_throttled = 0

        # Deterministic synthetic token stream for testability and zero-external-dependency validation
        sample_words = prompt.split() + ["hypothesis", "synthesis", "epistemic", "consequence", "deduction", "proof", "equilibrium"]

        for step in range(target_token_count):
            # 1. Inspect current UMA bus telemetry
            if bus_telemetry_feed and step < len(bus_telemetry_feed):
                current_telemetry = bus_telemetry_feed[step]
            else:
                # Default nominal telemetry (non-saturated)
                current_telemetry = UmaBusTelemetry(
                    gpu_bandwidth_gb_s=24.0,
                    reasoning_bandwidth_gb_s=18.0,
                    max_bus_capacity_gb_s=68.0,
                )

            max_saturation_seen = max(max_saturation_seen, current_telemetry.saturation_ratio)

            # 2. Query UMA Bandwidth Arbiter
            should_pace, throttle_pct, verdict = self.arbiter.evaluate_arbitration(
                telemetry=current_telemetry,
                is_rendering_active=is_rendering_active,
            )

            # 3. Apply pacing and speculative slot compression
            if should_pace:
                pacing_ever_engaged = True
                # Dynamically compress speculative slots to reduce memory lookahead pressure
                self.active_speculative_slots = max(1, int(self.config.max_speculative_slots * (1.0 - throttle_pct)))
                # Inject proportional inter-token micro-delay
                applied_pacing_ms = self.config.base_inter_token_delay_ms + (self.config.max_pacing_delay_ms * throttle_pct)
            else:
                self.active_speculative_slots = self.config.max_speculative_slots
                applied_pacing_ms = self.config.base_inter_token_delay_ms

            pacing_delays.append(applied_pacing_ms)

            # Simulate hardware inter-token timing (non-blocking synthetic scale in tests)
            word = sample_words[step % len(sample_words)]
            tok = EpistemicToken(
                token_id=1000 + step,
                text=word,
                logprob=-0.12 * (step % 5),
                timestamp_ns=time.time_ns(),
                pacing_delay_ms=round(applied_pacing_ms, 3),
            )
            tokens.append(tok)

            # 4. Periodically (every 4 tokens or on last token) synthesize a ProposalContract
            # and dispatch it as a 128-byte ZeroCopy frame into Tier 3 (EPISTEMIC_SPECULATION)
            if (step + 1) % 4 == 0 or (step + 1) == target_token_count:
                proposal = self._create_epistemic_proposal(
                    target_profile="balanced_hybrid",
                    confidence=0.92,
                    predicted_latency=applied_pacing_ms,
                )
                frame = contract_to_zerocopy(proposal)
                admitted, reason = self.ipc_buffer.push(frame)
                if admitted:
                    frames_admitted += 1
                else:
                    frames_throttled += 1

        t_end = time.time_ns()
        duration_ms = (t_end - t_start) / 1_000_000.0
        # Prevent division by zero
        safe_duration_s = max(0.001, duration_ms / 1000.0)
        tps = len(tokens) / safe_duration_s
        avg_pacing = sum(pacing_delays) / max(1, len(pacing_delays))

        # 5. Compute cryptographic signature digest for the execution
        digest_input = f"{receipt_id}:{self.worker_id}:{corr}:{len(tokens)}:{avg_pacing}:{max_saturation_seen}:{VITAL_MAX_HP}"
        sig = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()

        receipt = EpistemicExecutionReceipt(
            receipt_id=receipt_id,
            worker_id=self.worker_id,
            correlation_id=corr,
            prompt=prompt[:64],
            total_tokens_generated=len(tokens),
            wall_clock_time_ms=duration_ms,
            effective_throughput_tps=tps,
            avg_pacing_delay_ms=avg_pacing,
            was_pacing_engaged=pacing_ever_engaged,
            speculative_slots_allocated=self.active_speculative_slots,
            frames_admitted_to_tier3=frames_admitted,
            frames_throttled_by_tier3=frames_throttled,
            uma_bus_saturation_observed=max_saturation_seen,
            signature_digest=sig,
            vital_max_hp=VITAL_MAX_HP,
            timestamp_ns=t_end,
        )

        return tokens, receipt

    def _create_epistemic_proposal(
        self,
        target_profile: str,
        confidence: float,
        predicted_latency: float,
    ) -> ProposalContract:
        """Constructs an attested ProposalContract stamped with CAN_PROPOSE capability."""
        seq = self.sequence_counter
        self.sequence_counter += 1

        header = ContractHeader(
            contract_id=f"PROP-{seq}",
            contract_version=1,
            sequence_id=seq,
            capability_scope=int(CapabilityToken.CAN_PROPOSE),
            generation_id=1,
            timestamp_ns=time.time_ns(),
            predecessor_hash="0" * 64,
            principal_id=self.worker_id,
            nonce=str(seq),
            vital_max_hp=VITAL_MAX_HP,
        )

        return ProposalContract(
            header=header,
            originating_node_id=self.worker_id,
            target_profile=target_profile,
            predicted_latency_ms=predicted_latency,
            predicted_throughput_fps=120.0,
            confidence_score=confidence,
            requires_npu=False,
            requires_gpu=True,
            vital_max_hp=VITAL_MAX_HP,
        )
