#!/usr/bin/env python3
"""
================================================================================
KRYSTAL STACK: POLYMORPHIC INGRESS ADAPTER & UNEXPECTED INFERENCE ROUTER
================================================================================
Module:       krystal_kernel/polymorphic_ingress_adapter.py
Architecture: Auto-Aligning 128B Zero-Copy Ingress Bridge & Multi-Endpoint Router
Invariants:   VITAL_MAX_HP == 6 | Proposal != Authority | Delta S == 0 on breach
Theory Graph: Preserves 56 Nodes (0 cycles, no Node 57, no T13)

Solves the fundamental trade-off of strict 128-byte C-ABI contracts:
  1. Accepts arbitrary, unexpected, or nested data structures (dicts, jagged
     lists, raw bytes, dataclasses, sensor streams, or corrupted NaN/Inf inputs).
  2. Stores variable-length payloads in a ring-bounded Side-Car Overflow Arena
     and binds their 32-byte SHA-256 digest into the 128B ZeroCopyContractFrame.
  3. Routes unexpected asynchronous inference bursts across KSL LLM Topologies
     (ADAPTIVE_IMPEDANCE_ROUTING, SPECULATIVE_SHADOW_CASCADE, PYRAMIDAL_TIERED_KV_STREAMING)
     based on instantaneous UMA memory controller pressure (mu_UMA < 0.78).
  4. Provides unified API endpoint dispatch across Kernel, Vulkan, Janet, and WSL2.
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from krystal_kernel.constants import (
    VITAL_MAX_HP,
    FRAME_SIZE_BYTES,
    CapabilityScope,
    ContractViolation,
)
from krystal_kernel.zero_copy_contracts import (
    ZeroCopyContractFrame,
    BinaryContractType,
    PROFILE_TO_ID,
    VERDICT_TO_ID,
    canonicalize_float32,
)
from krystal_kernel.ksl.llm_topology_schemas import (
    LlmInterconnectionTopology,
    TargetDeviceBlock,
    InterconnectionChannelType,
)


# ── 1. SIDE-CAR OVERFLOW ARENA (BOUNDED ZERO-LEAK STORAGE) ───────────────────

class SideCarOverflowArena:
    """
    Ring-bounded in-memory arena that stores variable-length unexpected data
    structures while the core SPSC ring transmits only the 128-byte aligned
    ZeroCopyContractFrame containing the 32-byte SHA-256 digest.
    """

    def __init__(self, max_slots: int = 256, vital_max_hp: int = VITAL_MAX_HP) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP == 6 violated in SideCarOverflowArena: {vital_max_hp}")
        if max_slots < 1:
            raise ValueError("max_slots must be >= 1")
        self.max_slots: int = max_slots
        self.vital_max_hp: int = vital_max_hp
        self._slots: "OrderedDict[bytes, bytes]" = OrderedDict()

    def store(self, serialized_bytes: bytes) -> Tuple[bytes, int]:
        """
        Stores serialized bytes in the bounded arena and returns:
        (sha256_digest_32b, byte_length).
        """
        digest_32b = hashlib.sha256(serialized_bytes).digest()
        if digest_32b in self._slots:
            self._slots.move_to_end(digest_32b)
        else:
            if len(self._slots) >= self.max_slots:
                self._slots.popitem(last=False)
            self._slots[digest_32b] = serialized_bytes
        return digest_32b, len(serialized_bytes)

    def retrieve(self, digest_32b: bytes) -> Optional[bytes]:
        """Retrieves the raw payload by its 32-byte SHA-256 digest."""
        return self._slots.get(digest_32b)

    @property
    def active_slot_count(self) -> int:
        return len(self._slots)


# ── 2. STRUCTURAL INTROSPECTION & POLYMORPHIC INGRESS RESULT ─────────────────

@dataclass(frozen=True)
class PolymorphicIngressReceipt:
    """
    Immutable receipt produced after normalizing an unexpected data structure
    into a 128-byte ZeroCopyContractFrame.
    """
    sequence: int
    detected_schema_kind: str
    extracted_latency_ms: float
    extracted_drs_scale: float
    uma_bus_pressure: float
    verdict_code: str
    target_profile: str
    state_delta_committed: bool
    arena_sha256_hex: str
    arena_payload_bytes: int
    frame_128b_raw: bytes
    frame_128b_hex: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP == 6 violated in PolymorphicIngressReceipt: {self.vital_max_hp}")
        if len(self.frame_128b_raw) != FRAME_SIZE_BYTES:
            raise ContractViolation(f"Frame size must be 128B, got {len(self.frame_128b_raw)}")


@dataclass(frozen=True)
class UnexpectedInferenceRouteDecision:
    """
    Routing decision when an asynchronous/unexpected inference burst arrives
    during active system execution.
    """
    burst_id: str
    selected_topology: LlmInterconnectionTopology
    primary_device: TargetDeviceBlock
    secondary_device: TargetDeviceBlock
    transport_channel: InterconnectionChannelType
    vulkan_drs_scale_factor: float
    speculative_draft_depth: int
    uma_bus_pressure: float
    fallback_triggered: bool
    frame_128b_hex: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP == 6 violated in UnexpectedInferenceRouteDecision: {self.vital_max_hp}")


# ── 3. POLYMORPHIC INGRESS ADAPTER ───────────────────────────────────────────

class PolymorphicIngressAdapter:
    """
    Multi-layered abstraction bridge that converts unexpected/unaligned external
    data structures into verified 128-byte ZeroCopyContractFrames while enforcing
    deterministic fallback (Delta S = 0) on anomalous or corrupted inputs.
    """

    LATENCY_KEYS = ("gpu_render_ms", "latency_ms", "render_ms", "duration_ms", "exec_ms", "tau_ms")
    PRESSURE_KEYS = ("uma_pressure", "mu_uma", "bus_pressure", "load_factor", "pressure", "memory_load")
    SCALE_KEYS = ("drs_scale_factor", "scale_factor", "resolution_scale", "drs")

    def __init__(
        self,
        session_key: bytes = b"KRYSTAL_SOVEREIGN_POLYMORPHIC_K1",
        arena_max_slots: int = 256,
        vital_max_hp: int = VITAL_MAX_HP,
    ) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP == 6 violated in PolymorphicIngressAdapter: {vital_max_hp}")
        self.session_key: bytes = session_key
        self.vital_max_hp: int = vital_max_hp
        self.arena = SideCarOverflowArena(max_slots=arena_max_slots, vital_max_hp=vital_max_hp)
        self._sequence_counter: int = 0

    def _serialize_unexpected_payload(self, payload: Any) -> Tuple[bytes, str]:
        """Deterministically serializes any unexpected structure into canonical bytes."""
        if isinstance(payload, bytes):
            return payload, "raw_bytes"
        if isinstance(payload, bytearray) or isinstance(payload, memoryview):
            return bytes(payload), "binary_buffer"
        if isinstance(payload, str):
            return payload.encode("utf-8", errors="replace"), "utf8_string"
        if is_dataclass(payload) and not isinstance(payload, type):
            data_dict = asdict(payload)
            return json.dumps(data_dict, sort_keys=True, default=str).encode("utf-8"), "dataclass_struct"
        if isinstance(payload, dict):
            return json.dumps(payload, sort_keys=True, default=str).encode("utf-8"), "nested_dict"
        if isinstance(payload, (list, tuple, set)):
            return json.dumps(list(payload), default=str).encode("utf-8"), "jagged_sequence"
        if hasattr(payload, "__dict__"):
            return json.dumps(vars(payload), sort_keys=True, default=str).encode("utf-8"), "custom_object"
        return repr(payload).encode("utf-8"), "primitive_scalar"

    def _extract_numeric_scalars(self, payload: Any) -> Tuple[float, float, bool]:
        """
        Recursively inspects an arbitrary structure to extract:
          (latency_ms, uma_pressure, has_corrupted_numbers)
        """
        found_latency: Optional[float] = None
        found_pressure: Optional[float] = None
        corrupted = False

        def _check_finite(v: Any) -> Optional[float]:
            nonlocal corrupted
            if isinstance(v, bool):
                return None
            if isinstance(v, (int, float)):
                f = float(v)
                if math.isnan(f) or math.isinf(f) or f < 0.0:
                    corrupted = True
                    return None
                return f
            return None

        def _walk(node: Any, depth: int = 0) -> None:
            nonlocal found_latency, found_pressure, corrupted
            if depth > 8:
                return
            if isinstance(node, dict):
                for k, v in node.items():
                    k_low = str(k).lower()
                    if any(lk in k_low for lk in self.LATENCY_KEYS) and found_latency is None:
                        val = _check_finite(v)
                        if val is not None:
                            found_latency = val
                    elif any(pk in k_low for pk in self.PRESSURE_KEYS) and found_pressure is None:
                        val = _check_finite(v)
                        if val is not None:
                            found_pressure = min(1.0, val)
                    elif isinstance(v, (int, float)) and not isinstance(v, bool):
                        _check_finite(v)
                    elif isinstance(v, (dict, list, tuple)):
                        _walk(v, depth + 1)
            elif isinstance(node, (list, tuple)):
                numeric_items: List[float] = []
                for item in node:
                    if isinstance(item, (dict, list, tuple)):
                        _walk(item, depth + 1)
                    else:
                        val = _check_finite(item)
                        if val is not None:
                            numeric_items.append(val)
                if numeric_items and found_latency is None:
                    found_latency = numeric_items[0]
                if len(numeric_items) >= 2 and found_pressure is None:
                    found_pressure = min(1.0, numeric_items[1])
            elif is_dataclass(node) and not isinstance(node, type):
                _walk(asdict(node), depth + 1)
            elif hasattr(node, "__dict__"):
                _walk(vars(node), depth + 1)
            elif isinstance(node, (int, float)) and not isinstance(node, bool):
                val = _check_finite(node)
                if val is not None and found_latency is None:
                    found_latency = val

        _walk(payload)
        return (
            found_latency if found_latency is not None else 1.0,
            found_pressure if found_pressure is not None else 0.45,
            corrupted,
        )

    def ingest_unexpected_structure(
        self,
        payload: Any,
        contract_type: BinaryContractType = BinaryContractType.EVIDENCE,
        nonce: int = 1,
    ) -> PolymorphicIngressReceipt:
        """
        Ingests any unexpected data structure, stores its body in the Side-Car
        Overflow Arena, normalizes metrics, and emits a signed 128B frame.
        If corrupted/non-finite numbers are detected, triggers deterministic
        safe fallback (Delta S = 0, AUTH_DEGRADED_SAFE_FALLBACK).
        """
        self._sequence_counter += 1
        seq = self._sequence_counter

        raw_bytes, schema_kind = self._serialize_unexpected_payload(payload)
        digest_32b, payload_len = self.arena.store(raw_bytes)
        latency_ms, uma_pressure, is_corrupted = self._extract_numeric_scalars(payload)

        if is_corrupted or uma_pressure >= 0.95:
            # Deterministic fallback: preserve state (Delta S = 0)
            verdict_code = "AUTH_DEGRADED_SAFE_FALLBACK"
            profile_name = "deterministic_safe"
            state_committed = False
            safe_latency = 0.0
            drs_scale = 0.50
        else:
            verdict_code = "AUTH_COMMIT_APPROVED"
            if uma_pressure >= 0.78:
                profile_name = "ultra_saver"
                drs_scale = 0.75
            elif uma_pressure < 0.40:
                profile_name = "max_burst"
                drs_scale = 1.00
            else:
                profile_name = "balanced_hybrid"
                drs_scale = 1.00
            state_committed = True
            safe_latency = canonicalize_float32(latency_ms)

        frame = ZeroCopyContractFrame(
            vital_hp=VITAL_MAX_HP,
            contract_type=int(contract_type),
            payload_size=payload_len,
            sequence=seq,
            generation=1,
            cap_scope=int(CapabilityScope.CAN_PROPOSE),
            timestamp_ns=time.time_ns(),
            nonce=nonce,
            predecessor_hash=digest_32b,
            profile_id=PROFILE_TO_ID[profile_name],
            verdict_id=VERDICT_TO_ID[verdict_code],
            mask=0x01 if state_committed else 0x00,
            gpu_render_ms=safe_latency,
            drs_scale_factor=canonicalize_float32(drs_scale),
        )
        packed_128b = frame.pack_authenticated(self.session_key)

        return PolymorphicIngressReceipt(
            sequence=seq,
            detected_schema_kind=schema_kind,
            extracted_latency_ms=safe_latency,
            extracted_drs_scale=drs_scale,
            uma_bus_pressure=uma_pressure,
            verdict_code=verdict_code,
            target_profile=profile_name,
            state_delta_committed=state_committed,
            arena_sha256_hex=digest_32b.hex(),
            arena_payload_bytes=payload_len,
            frame_128b_raw=packed_128b,
            frame_128b_hex=packed_128b.hex(),
            vital_max_hp=VITAL_MAX_HP,
        )

    def route_unexpected_inference_burst(
        self,
        burst_id: str,
        prompt_tokens: int,
        uma_bus_pressure: float,
        speculative_divergence_detected: bool = False,
    ) -> UnexpectedInferenceRouteDecision:
        """
        Routes an unexpected inference burst across our 7 LLM interconnection
        topologies based on instantaneous UMA bus pressure and speculative state.
        """
        receipt = self.ingest_unexpected_structure(
            {
                "burst_id": burst_id,
                "prompt_tokens": prompt_tokens,
                "uma_pressure": uma_bus_pressure,
                "speculative_divergence": speculative_divergence_detected,
            },
            contract_type=BinaryContractType.EXECUTION,
        )

        if not receipt.state_delta_committed or uma_bus_pressure >= 0.78:
            # High bus impedance -> Pyramidal KV-Streaming + SRAM fallback
            topology = LlmInterconnectionTopology.PYRAMIDAL_TIERED_KV_STREAMING
            primary = TargetDeviceBlock.SRAM_L1_L2_CACHE
            secondary = TargetDeviceBlock.GNA_COPROCESSOR_2_0
            channel = InterconnectionChannelType.SRAM_DMA_BURST
            drs_scale = 0.65
            draft_depth = 1
            fallback = True
        elif speculative_divergence_detected:
            # Speculative branch miss -> Instant rollback & Adaptive Impedance Routing
            topology = LlmInterconnectionTopology.ADAPTIVE_IMPEDANCE_ROUTING
            primary = TargetDeviceBlock.CPU_WILLOW_COVE_AVX512
            secondary = TargetDeviceBlock.GPU_IRIS_XE_DP4A
            channel = InterconnectionChannelType.ZERO_COPY_UMA_RING
            drs_scale = 0.75
            draft_depth = 2
            fallback = False
        elif uma_bus_pressure >= 0.55:
            # Moderate bus load -> Heterogeneous Asymmetric Pipeline
            topology = LlmInterconnectionTopology.HETEROGENEOUS_ASYMMETRIC_PIPELINE
            primary = TargetDeviceBlock.CPU_WILLOW_COVE_AVX512
            secondary = TargetDeviceBlock.GNA_COPROCESSOR_2_0
            channel = InterconnectionChannelType.ZERO_COPY_UMA_RING
            drs_scale = 0.80
            draft_depth = 4
            fallback = False
        else:
            # Low bus load -> Full Speculative Shadow Cascade (NPU draft + GPU verify)
            topology = LlmInterconnectionTopology.SPECULATIVE_SHADOW_CASCADE
            primary = TargetDeviceBlock.GPU_IRIS_XE_DP4A
            secondary = TargetDeviceBlock.GNA_COPROCESSOR_2_0
            channel = InterconnectionChannelType.SPECULATIVE_TOKEN_PIPE
            drs_scale = 1.00
            draft_depth = 6
            fallback = False

        return UnexpectedInferenceRouteDecision(
            burst_id=burst_id,
            selected_topology=topology,
            primary_device=primary,
            secondary_device=secondary,
            transport_channel=channel,
            vulkan_drs_scale_factor=drs_scale,
            speculative_draft_depth=draft_depth,
            uma_bus_pressure=uma_bus_pressure,
            fallback_triggered=fallback,
            frame_128b_hex=receipt.frame_128b_hex,
            vital_max_hp=VITAL_MAX_HP,
        )


# ── 4. MULTI-ENDPOINT API CONNECTOR ──────────────────────────────────────────

class PolymorphicApiEndpointConnector:
    """
    Unified API Endpoint Router connecting the Polymorphic Ingress Adapter
    to Krystal Stack's core REST/IPC endpoints:
      - POST /api/v1/ingress/polymorphic
      - POST /api/v1/inference/unexpected_burst
      - POST /api/janet/decode_binary
      - POST /api/wsl/hypervisor_bridge
      - GET  /api/v1/telemetry/observer_matrix
    """

    SUPPORTED_ENDPOINTS: Tuple[str, ...] = (
        "POST /api/v1/ingress/polymorphic",
        "POST /api/v1/inference/unexpected_burst",
        "POST /api/janet/decode_binary",
        "POST /api/wsl/hypervisor_bridge",
        "GET /api/v1/telemetry/observer_matrix",
        "POST /api/v1/ml/glyph_stroke_compile",
        "POST /api/v1/cache/bootstrap_preset",
        "POST /api/v1/linguistics/word_problem_validate",
        "POST /api/v1/linguistics/etymological_parse",
        "POST /api/v1/graph/inject_open_weights",
        "POST /api/v1/gamesa_grid/hex_dataset",
        "POST /api/v1/typography/phi_charset",
        "POST /api/v1/voice/geometric_glyph_synth",
        "POST /api/v1/acoustics/chromatic_hex_scale",
        "POST /api/v1/cv/lip_reading_glottal",
        "POST /api/v1/autocomplete/onomatopoeic_whisper",
        "POST /api/v1/provenance/reversible_trace",
        "POST /api/v1/evolution/retrospective_joule_audit",
        "POST /api/v1/perception/bezier_isa_morph",
        "POST /api/v1/mcp_ncp/control_plane",
        "POST /api/v1/runtime/hex_coherent_synthesis",
        "POST /api/v1/math/kernel_trigonometric_eval",
        "GET /api/v1/webapp/unified_knowledge_kernel",
        "POST /api/v1/webapp/unified_knowledge_kernel",
    )

    def __init__(self, adapter: Optional[PolymorphicIngressAdapter] = None) -> None:
        self.adapter = adapter or PolymorphicIngressAdapter()
        from krystal_kernel.visual_glyph_stroke_ml_compiler import (
            AdaptiveKerasSpeculativeCore,
            GridPrefillBootstrapEngine,
            LlmLinguisticWordProblemValidator,
            SlavicGermanicEtymologicalParser,
            OpenWeightLinguisticGraphInjector,
            Gamesa3DGridVulkanHexEncoder,
            GoldenRatioGlyphSetGenerator,
            GeometricGlyphBrowserVoiceEngine,
            ChromaticHexDecibelScaleEngine,
            RhetoricalLipReadingAndGlottalEngine,
            OnomatopoeicPythonAutocompleteModel,
            ReversibleSignalProvenanceFramework,
            RetrospectiveEvolutionAndJouleAmplitudeEngine,
            PerceptualQueryLibraryAndBezierIsaEngine,
            McpNcpUnifiedControlPlaneBridge,
            HexGeometricCoherentRuntimeAndSynthesisEngine,
            KernelGovernedTrigonometricHarmonicEngine,
        )
        self.speculative_core = AdaptiveKerasSpeculativeCore()
        self.bootstrap_engine = GridPrefillBootstrapEngine()
        self.word_validator = LlmLinguisticWordProblemValidator()
        self.etym_parser = SlavicGermanicEtymologicalParser()
        self.graph_injector = OpenWeightLinguisticGraphInjector()
        self.gamesa_hex_encoder = Gamesa3DGridVulkanHexEncoder()
        self.phi_generator = GoldenRatioGlyphSetGenerator()
        self.voice_engine = GeometricGlyphBrowserVoiceEngine()
        self.chromatic_engine = ChromaticHexDecibelScaleEngine()
        self.lip_glottal_engine = RhetoricalLipReadingAndGlottalEngine()
        self.python_whisperer = OnomatopoeicPythonAutocompleteModel()
        self.provenance_framework = ReversibleSignalProvenanceFramework()
        self.retro_joule_engine = RetrospectiveEvolutionAndJouleAmplitudeEngine()
        self.bezier_isa_engine = PerceptualQueryLibraryAndBezierIsaEngine()
        self.mcp_ncp_bridge = McpNcpUnifiedControlPlaneBridge()
        self.hex_coherent_runtime = HexGeometricCoherentRuntimeAndSynthesisEngine()
        self.trig_engine = KernelGovernedTrigonometricHarmonicEngine()

    def dispatch(self, method: str, path: str, payload: Any = None) -> Dict[str, Any]:
        """Dispatches an API request to the corresponding governed handler."""
        route_key = f"{method.upper().strip()} {path.strip()}"

        if route_key == "POST /api/v1/ingress/polymorphic":
            receipt = self.adapter.ingest_unexpected_structure(payload)
            return {
                "status_code": 200 if receipt.state_delta_committed else 202,
                "endpoint": route_key,
                "detected_schema_kind": receipt.detected_schema_kind,
                "verdict_code": receipt.verdict_code,
                "target_profile": receipt.target_profile,
                "state_delta_committed": receipt.state_delta_committed,
                "arena_sha256_hex": receipt.arena_sha256_hex,
                "frame_size_bytes": len(receipt.frame_128b_raw),
                "frame_128b_hex": receipt.frame_128b_hex,
                "vital_max_hp": receipt.vital_max_hp,
            }

        if route_key == "POST /api/v1/inference/unexpected_burst":
            data = payload if isinstance(payload, dict) else {"raw": payload}
            decision = self.adapter.route_unexpected_inference_burst(
                burst_id=str(data.get("burst_id", "burst-auto-01")),
                prompt_tokens=int(data.get("prompt_tokens", 256)),
                uma_bus_pressure=float(data.get("uma_pressure", 0.48)),
                speculative_divergence_detected=bool(data.get("speculative_divergence", False)),
            )
            return {
                "status_code": 200,
                "endpoint": route_key,
                "burst_id": decision.burst_id,
                "selected_topology": decision.selected_topology.value,
                "primary_device": decision.primary_device.value,
                "secondary_device": decision.secondary_device.value,
                "transport_channel": decision.transport_channel.value,
                "vulkan_drs_scale_factor": decision.vulkan_drs_scale_factor,
                "speculative_draft_depth": decision.speculative_draft_depth,
                "fallback_triggered": decision.fallback_triggered,
                "frame_128b_hex": decision.frame_128b_hex,
                "vital_max_hp": decision.vital_max_hp,
            }

        if route_key == "POST /api/janet/decode_binary":
            receipt = self.adapter.ingest_unexpected_structure(
                payload, contract_type=BinaryContractType.PROPOSAL
            )
            return {
                "status_code": 200,
                "endpoint": route_key,
                "ksyn_header_hex": "4b53594e",
                "arena_sha256_hex": receipt.arena_sha256_hex,
                "frame_128b_hex": receipt.frame_128b_hex,
                "vital_max_hp": receipt.vital_max_hp,
            }

        if route_key == "POST /api/wsl/hypervisor_bridge":
            receipt = self.adapter.ingest_unexpected_structure(
                payload, contract_type=BinaryContractType.STATE
            )
            return {
                "status_code": 200,
                "endpoint": route_key,
                "vsock_port": 19283,
                "transport": "AF_VSOCK_ZERO_COPY_128B",
                "domain_locks_isolated": True,
                "state_delta_committed": receipt.state_delta_committed,
                "frame_128b_hex": receipt.frame_128b_hex,
                "vital_max_hp": receipt.vital_max_hp,
            }

        if route_key == "GET /api/v1/telemetry/observer_matrix":
            return {
                "status_code": 200,
                "endpoint": route_key,
                "matrix_shape": [9, 10],
                "active_arena_slots": self.adapter.arena.active_slot_count,
                "uma_saturation_ceiling": 0.78,
                "vital_max_hp": VITAL_MAX_HP,
            }

        if route_key == "POST /api/v1/ml/glyph_stroke_compile":
            data = payload if isinstance(payload, dict) else {"text": str(payload or "")}
            text = str(data.get("text", "KRYSTAL STACK"))
            uma_pressure = float(data.get("uma_pressure", 0.42))
            branches = data.get(
                "candidate_branches",
                [
                    {"branch_id": "b_verified_01", "weight_confidence": 0.88, "entropy_reduction": 0.35},
                    {"branch_id": "b_divergent_02", "weight_confidence": 0.25, "entropy_reduction": 0.0},
                ],
            )
            branch_res = self.speculative_core.evaluate_and_prune_deductive_branches(
                operation_id=str(data.get("operation_id", "op-glyph-01")),
                text_payload=text,
                candidate_branches=branches,
                uma_bus_pressure=uma_pressure,
            )
            return {
                "status_code": 200,
                "endpoint": route_key,
                "tensor_semaphore": branch_res.tensor_semaphore,
                "allocated_weight_links": branch_res.allocated_weight_links,
                "surviving_branches": list(branch_res.surviving_branches),
                "pruned_branches": list(branch_res.pruned_branches),
                "state_delta_committed": branch_res.state_delta_committed,
                "frame_128b_hex": branch_res.frame_128b_hex,
                "vital_max_hp": branch_res.vital_max_hp,
            }

        if route_key == "POST /api/v1/cache/bootstrap_preset":
            data = payload if isinstance(payload, dict) else {}
            profile = self.bootstrap_engine.build_prefill_cache_preset(
                preset_id=str(data.get("preset_id", "PRESET_TACTICAL_ARENA_8K")),
                grid_cols=int(data.get("grid_cols", 16)),
                grid_rows=int(data.get("grid_rows", 16)),
            )
            return {
                "status_code": 200,
                "endpoint": route_key,
                "profile_id": profile.profile_id,
                "linked_scene_preset_id": profile.linked_scene_preset_id,
                "grid_dimensions": list(profile.grid_dimensions),
                "l2_sram_prefill_kb": profile.l2_sram_prefill_kb,
                "uma_ring_prefill_mb": profile.uma_ring_prefill_mb,
                "preloaded_photographic_clusters": profile.preloaded_photographic_clusters,
                "frame_128b_hex": profile.frame_128b_hex,
                "vital_max_hp": profile.vital_max_hp,
            }

        if route_key == "POST /api/v1/linguistics/word_problem_validate":
            data = payload if isinstance(payload, dict) else {}
            res = self.word_validator.validate_word_problem(
                problem_id=str(data.get("problem_id", "wp-01")),
                premise_text_sk=str(
                    data.get("premise_text_sk", "Písmená a znaky sú čiary, z ktorých je tvar zložený.")
                ),
                cognate_parallel_text=str(
                    data.get("cognate_parallel_text", "Glyph stroke and crystal shape form the grid.")
                ),
                expected_semantic_anchor=str(
                    data.get("expected_semantic_anchor", "MEANING_MORPHOLOGICAL_LINE_COMPOSITION")
                ),
            )
            res["status_code"] = 200
            res["endpoint"] = route_key
            return res

        if route_key == "POST /api/v1/linguistics/etymological_parse":
            data = payload if isinstance(payload, dict) else {}
            collocations = data.get(
                "collocations",
                ["zámok pamäte", "zámek paměti", "Speicherschloss", "jazyková vrstva"],
            )
            res = self.etym_parser.parse_etymological_collocations(collocations)
            res["status_code"] = 200
            res["endpoint"] = route_key
            return res

        if route_key == "POST /api/v1/graph/inject_open_weights":
            data = payload if isinstance(payload, dict) else {}
            synth = self.graph_injector.inject_weights_and_couple_reasoning(
                repository_id=str(data.get("repository_id", "XERJ_UNIVERSAL_MORPHOLOGY_V1")),
                signal_phrases=data.get("signal_phrases"),
                uma_bus_pressure=float(data.get("uma_pressure", 0.40)),
            )
            return {
                "status_code": 200,
                "endpoint": route_key,
                "repository_id": synth.repository_id,
                "preserved_theory_nodes": synth.preserved_theory_nodes,
                "inductive_signal_hypotheses": synth.inductive_signal_hypotheses,
                "deductive_verified_closures": synth.deductive_verified_closures,
                "coupled_reasoning_equilibrium": synth.coupled_reasoning_equilibrium,
                "state_delta_committed": synth.state_delta_committed,
                "frame_128b_hex": synth.frame_128b_hex,
                "vital_max_hp": synth.vital_max_hp,
            }

        if route_key == "POST /api/v1/gamesa_grid/hex_dataset":
            data = payload if isinstance(payload, dict) else {}
            ds = self.gamesa_hex_encoder.build_framework_ml_training_dataset(
                seed_corpus=data.get("seed_corpus"),
                uma_bus_pressure=float(data.get("uma_pressure", 0.42)),
            )
            ds["status_code"] = 200
            ds["endpoint"] = route_key
            return ds

        if route_key == "POST /api/v1/typography/phi_charset":
            data = payload if isinstance(payload, dict) else {}
            res = self.phi_generator.generate_phi_character_set(
                characters=str(data.get("characters", "KRYSTALČŠŽ"))
            )
            res["status_code"] = 200
            res["endpoint"] = route_key
            return res

        if route_key == "POST /api/v1/voice/geometric_glyph_synth":
            data = payload if isinstance(payload, dict) else {}
            res = self.voice_engine.synthesize_geometric_text_to_voice_signals(
                text=str(data.get("text", "Písmená sú symboly a znaky sú čiary.")),
                vocal_fold_f0_hz=data.get("vocal_fold_f0_hz"),
            )
            res["status_code"] = 200
            res["endpoint"] = route_key
            return res

        if route_key == "POST /api/v1/acoustics/chromatic_hex_scale":
            data = payload if isinstance(payload, dict) else {}
            res = self.chromatic_engine.build_hex_scales_and_melodic_phrasing(
                phrase_text=str(data.get("phrase_text", "Písmená sú symboly a znaky sú čiary.")),
                base_gain_db=float(data.get("base_gain_db", 0.0)),
            )
            res["status_code"] = 200
            res["endpoint"] = route_key
            return res

        if route_key == "POST /api/v1/cv/lip_reading_glottal":
            data = payload if isinstance(payload, dict) else {}
            res = self.lip_glottal_engine.analyze_lip_reading_and_glottal_tonus(
                spoken_text=str(data.get("spoken_text", "Kryštál rezonuje v tóne.")),
                base_f0_hz=float(data.get("base_f0_hz", 261.63)),
                rhetorical_preset=str(data.get("rhetorical_preset", "CICERONIAN_EXORDIUM_RISE")),
            )
            res["status_code"] = 200
            res["endpoint"] = route_key
            return res

        if route_key == "POST /api/v1/autocomplete/onomatopoeic_whisper":
            data = payload if isinstance(payload, dict) else {}
            res = self.python_whisperer.suggest_onomatopoeic_completions(
                prefix_token=str(data.get("prefix_token", "šum_")),
                composition_schema=str(data.get("composition_schema", "LYRIC_ACOUSTIC_CADENCE")),
            )
            res["status_code"] = 200
            res["endpoint"] = route_key
            return res

        if route_key == "POST /api/v1/provenance/reversible_trace":
            data = payload if isinstance(payload, dict) else {}
            res = self.provenance_framework.trace_and_reverse_signal_flow(
                signal_payload=str(data.get("signal_payload", "Písmená sú symboly a znaky sú čiary.")),
                hex_tone_id=str(data.get("hex_tone_id", "0xC2")),
                observer_perspective=str(data.get("observer_perspective", "ACOUSTIC_PHONOLOGICAL_GLOTTAL_VIEW")),
            )
            res["status_code"] = 200
            res["endpoint"] = route_key
            return res

        if route_key == "POST /api/v1/evolution/retrospective_joule_audit":
            data = payload if isinstance(payload, dict) else {}
            res = self.retro_joule_engine.evaluate_retrospective_and_joule_boost(
                baseline_latency_ms=float(data.get("baseline_latency_ms", 47.0)),
                optimized_latency_ms=float(data.get("optimized_latency_ms", 2.309)),
                rapl_power_envelope_watts=float(data.get("rapl_power_envelope_watts", 15.0)),
                current_mimic_tonus=float(data.get("current_mimic_tonus", 0.78)),
                acoustic_slope_db_per_sec=float(data.get("acoustic_slope_db_per_sec", 4.5)),
            )
            res["status_code"] = 200
            res["endpoint"] = route_key
            return res

        if route_key == "POST /api/v1/perception/bezier_isa_morph":
            data = payload if isinstance(payload, dict) else {}
            res = self.bezier_isa_engine.morph_text_clusters_with_bezier_and_isa(
                source_cluster=str(data.get("source_cluster", "zámok pamäte")),
                target_cluster=str(data.get("target_cluster", "Speicherschloss")),
                steps=int(data.get("steps", 5)),
            )
            res["status_code"] = 200
            res["endpoint"] = route_key
            return res

        if route_key == "POST /api/v1/mcp_ncp/control_plane":
            data = payload if isinstance(payload, dict) else {}
            res = self.mcp_ncp_bridge.orchestrate_mcp_ncp_control_plane(
                mcp_tool_name=str(data.get("mcp_tool_name", "krystal_glyph_bezier_isa_morph")),
                uma_bus_pressure=float(data.get("uma_bus_pressure", 0.36)),
                typography_mode=str(data.get("typography_mode", "SANS_SERIF_DIRECT_LINEAR")),
            )
            res["status_code"] = 200
            res["endpoint"] = route_key
            return res

        if route_key == "POST /api/v1/runtime/hex_coherent_synthesis":
            data = payload if isinstance(payload, dict) else {}
            res = self.hex_coherent_runtime.execute_coherent_hex_runtime_synthesis(
                coherent_text=str(
                    data.get("coherent_text", "Písmená sú symboly a znaky sú čiary v koherentnom celku.")
                ),
                typography_mode=str(data.get("typography_mode", "SANS_SERIF_DIRECT_LINEAR")),
                hex_tone_id=str(data.get("hex_tone_id", "0xC4")),
                uma_bus_pressure=float(data.get("uma_bus_pressure", 0.35)),
            )
            res["status_code"] = 200
            res["endpoint"] = route_key
            return res

        if route_key == "POST /api/v1/math/kernel_trigonometric_eval":
            data = payload if isinstance(payload, dict) else {}
            res = self.trig_engine.evaluate_kernel_trigonometric_state(
                angle_deg=float(data.get("angle_deg", 58.282525)),
                glottal_f0_hz=float(data.get("glottal_f0_hz", 261.63)),
                time_sec=float(data.get("time_sec", 0.002309)),
            )
            res["status_code"] = 200
            res["endpoint"] = route_key
            return res

        if route_key in (
            "GET /api/v1/webapp/unified_knowledge_kernel",
            "POST /api/v1/webapp/unified_knowledge_kernel",
        ):
            data = payload if isinstance(payload, dict) else {}
            text_in = str(
                data.get("text", "Písmená sú symboly a znaky sú čiary v koherentnom celku.")
            )
            angle_in = float(data.get("angle_deg", 58.282525))
            uma_in = float(data.get("uma_bus_pressure", 0.34))

            runtime_state = self.hex_coherent_runtime.execute_coherent_hex_runtime_synthesis(
                coherent_text=text_in,
                uma_bus_pressure=uma_in,
            )
            trig_state = self.trig_engine.evaluate_kernel_trigonometric_state(angle_deg=angle_in)
            bezier_state = self.bezier_isa_engine.morph_text_clusters_with_bezier_and_isa()
            mcp_ncp_state = self.mcp_ncp_bridge.orchestrate_mcp_ncp_control_plane(uma_bus_pressure=uma_in)
            joule_state = self.retro_joule_engine.evaluate_retrospective_and_joule_boost()

            return {
                "status_code": 200,
                "endpoint": route_key,
                "webapp_backend_status": "UNIFIED_KNOWLEDGE_KERNEL_ONLINE",
                "hex_coherent_runtime": runtime_state,
                "kernel_trigonometry_and_deductions": trig_state,
                "bezier_isa_and_typography_modes": bezier_state,
                "mcp_ncp_control_plane": mcp_ncp_state,
                "retrospective_joule_and_parameter_scaling": joule_state,
                "vital_max_hp": VITAL_MAX_HP,
            }

        return {
            "status_code": 404,
            "endpoint": route_key,
            "error": "Endpoint not registered in PolymorphicApiEndpointConnector",
            "vital_max_hp": VITAL_MAX_HP,
        }
