"""
KRYSTAL-STACK // TEST SUITE: AEP INFERENCE WORKLOAD & SLOVAK LLM GOVERNOR
=============================================================================
Verification Protocol: docs/architecture/INFERENCE_WORKLOAD.md
Classification: [TEST-VERIFIED]
Invariant: VITAL_MAX_HP == 6
"""

import pytest
from krystal_kernel.aep_engine import (
    AepTransactionalEngine,
    AepReceipt,
    AepVerdict,
    WorkloadType,
    ExecutionLane,
    VitalHpInvariantRule,
    VITAL_MAX_HP,
)
from krystal_kernel.slovak_llm_adapter import (
    SlovakLLMAdapter,
    ModelSpec,
    InferenceRequest,
    InferenceResponse,
)
from krystal_kernel.llm_inference_governor import (
    AepInferenceGovernor,
    GovernorMode,
    LLMInferenceState,
    transition_llm_inference,
)


class TestSlovakLLMAdapter:
    """Verifies sovereign adapter, license guards, and memory safety."""

    def test_approved_license_verification(self):
        """Apache-2.0 license is approved for sovereign deployment."""
        adapter = SlovakLLMAdapter(ModelSpec(license="Apache-2.0"))
        ok, msg = adapter.validate_license()
        assert ok is True
        assert "LICENSE_COMPLIANT" in msg

    def test_incompatible_license_rejection(self):
        """Proprietary or unrecognized license is strictly rejected."""
        adapter = SlovakLLMAdapter(ModelSpec(license="Proprietary-Closed-Source"))
        ok, msg = adapter.validate_license()
        assert ok is False
        assert "LICENSE_INCOMPATIBLE" in msg

        req = InferenceRequest(prompt="Test prompt")
        with pytest.raises(PermissionError) as exc_info:
            adapter.execute_inference(req)
        assert "LICENSE_INCOMPATIBLE" in str(exc_info.value)

    def test_memory_footprint_quantization_estimation(self):
        """Validates working set footprint calculations for various quantizations."""
        adapter = SlovakLLMAdapter()
        # 14.77B * 2.0 bytes + 512MB KV
        fp16_mb = adapter.estimate_memory_footprint_mb("FP16")
        assert 28000 < fp16_mb < 30000

        # Q6_K ~ 11.8 - 12.5 GB
        q6_mb = adapter.estimate_memory_footprint_mb("Q6_K")
        assert 11500 < q6_mb < 12500

        # Q4_K_M ~ 8.5 - 9.5 GB
        q4_mb = adapter.estimate_memory_footprint_mb("Q4_K_M")
        assert 8400 < q4_mb < 9500

    def test_memory_headroom_guard(self):
        """Enforces halt/exception when model footprint exceeds safe physical RAM headroom."""
        # Specifying unquantized FP16 which requires ~29GB (exceeds total system RAM 24GB)
        adapter = SlovakLLMAdapter(ModelSpec(quantization="FP16"))
        fits, avail_mb, req_mb = adapter.check_memory_headroom("FP16")
        assert fits is False
        assert req_mb > avail_mb * 0.85

        req = InferenceRequest(prompt="Test oversized load")
        with pytest.raises(MemoryError) as exc_info:
            adapter.execute_inference(req, enforce_headroom=True)
        assert "INSUFFICIENT_HARDWARE_RAM" in str(exc_info.value)

    def test_slovak_tokenization_and_diacritics(self):
        """Preserves accented Slovak characters and produces subword tokens."""
        adapter = SlovakLLMAdapter()
        text = "Epistemické jadro KrystalStack overuje invarianty bez straty dát."
        tokens = adapter.tokenize_slovak(text)
        assert len(tokens) >= 8
        assert any("Epis" in t for t in tokens)
        assert any("Krys" in t for t in tokens)

    def test_inference_timing_telemetry(self):
        """Measures microsecond-precision TTFT, TPOT, and throughput."""
        # Use Q2_K to ensure it fits safely in memory check during test
        adapter = SlovakLLMAdapter(ModelSpec(quantization="Q2_K"))
        req = InferenceRequest(prompt="Prehľad telemetrie procesora", max_new_tokens=20)
        resp = adapter.execute_inference(req, enforce_headroom=False)
        assert isinstance(resp, InferenceResponse)
        assert resp.ttft_us > 0
        assert resp.tpot_us > 0
        assert resp.total_duration_us >= resp.ttft_us
        assert resp.completion_tokens > 0
        assert resp.tokens_per_second > 0.0
        assert resp.vital_max_hp == VITAL_MAX_HP


class TestAepInferenceGovernor:
    """Verifies Governor policies, backpressure handling, and transactional commit."""

    def test_static_baseline_greedy_dispatch(self):
        """Static baseline executes with full requested tokens under normal pressure."""
        adapter = SlovakLLMAdapter(ModelSpec(parameter_count_b=3.0, quantization="Q4_K_M"))
        gov = AepInferenceGovernor(mode=GovernorMode.STATIC_BASELINE, adapter=adapter)
        committed, receipt, resp = gov.dispatch_inference(
            prompt="Analyzuj stav uzla",
            max_tokens=32,
            current_bus_pressure=0.20,
        )
        assert committed is True
        assert receipt.verdict == AepVerdict.COMMITTED
        assert receipt.workload == WorkloadType.INFERENCE
        assert receipt.inference_telemetry is not None
        assert gov.current_state.completed_requests == 1
        assert gov.current_state.vital_max_hp == VITAL_MAX_HP

    def test_adaptive_governor_moderate_pressure_pacing(self):
        """Under moderate bus pressure (0.50), adaptive governor paces tokens by 50%."""
        adapter = SlovakLLMAdapter(ModelSpec(parameter_count_b=3.0, quantization="Q4_K_M"))
        gov = AepInferenceGovernor(mode=GovernorMode.ADAPTIVE_GOVERNOR, adapter=adapter)
        committed, receipt, resp = gov.dispatch_inference(
            prompt="Vypočítaj maticu",
            max_tokens=40,
            current_bus_pressure=0.50,
        )
        assert committed is True
        assert receipt.verdict == AepVerdict.COMMITTED
        # Requested 40 tokens -> paced to 20
        assert resp.completion_tokens <= 20
        assert receipt.execution_lane == ExecutionLane.COMPUTE

    def test_adaptive_governor_high_pressure_cpu_fallback(self):
        """Under high bus pressure (0.75), governor compresses response and falls back to CPU."""
        adapter = SlovakLLMAdapter(ModelSpec(parameter_count_b=3.0, quantization="Q4_K_M"))
        gov = AepInferenceGovernor(mode=GovernorMode.ADAPTIVE_GOVERNOR, adapter=adapter)
        committed, receipt, resp = gov.dispatch_inference(
            prompt="Kritický alarm",
            max_tokens=60,
            target_device="GPU",
            current_bus_pressure=0.75,
        )
        assert committed is True
        assert receipt.verdict == AepVerdict.COMMITTED
        # CPU routing to protect Iris Xe UMA bandwidth for display scanout
        assert receipt.device == "CPU"
        assert receipt.execution_lane == ExecutionLane.BACKGROUND
        assert resp.completion_tokens <= 24

    def test_zero_dirty_state_mutation_on_critical_backpressure(self):
        """Under critical bus pressure (>0.85), request is rejected with Delta S == 0."""
        adapter = SlovakLLMAdapter(ModelSpec(parameter_count_b=3.0, quantization="Q4_K_M"))
        gov = AepInferenceGovernor(mode=GovernorMode.ADAPTIVE_GOVERNOR, adapter=adapter)
        initial_state_digest = gov.current_state.session_id
        initial_completed = gov.current_state.completed_requests

        committed, receipt, resp = gov.dispatch_inference(
            prompt="Pretečenie zbernice",
            max_tokens=32,
            current_bus_pressure=0.92,
        )
        # S' is discarded; Delta S == 0
        assert committed is False
        assert resp is None
        assert receipt.verdict == AepVerdict.DISCARDED_BACKPRESSURE_REJECTED
        assert receipt.state_before == receipt.state_after  # Invariant: St+1 == St
        assert gov.current_state.completed_requests == initial_completed
        assert gov.current_state.session_id == initial_state_digest

    def test_receipt_inference_telemetry_serialization(self):
        """Receipt correctly serializes inference dictionary and telemetry."""
        adapter = SlovakLLMAdapter(ModelSpec(parameter_count_b=3.0, quantization="Q4_K_M"))
        gov = AepInferenceGovernor(mode=GovernorMode.ADAPTIVE_GOVERNOR, adapter=adapter)
        committed, receipt, resp = gov.dispatch_inference(
            prompt="Generuj certifikát",
            max_tokens=16,
            current_bus_pressure=0.15,
        )
        assert committed is True
        receipt_dict = receipt.to_dict()
        assert "inference" in receipt_dict
        inf = receipt_dict["inference"]
        assert inf["model_id"] == "slovak-nlp/Qwen3-14B-sk"
        assert inf["quantization"] == "Q4_K_M"
        assert inf["license_compliant"] is True
        assert "ttft_us" in inf
        assert "tokens_per_second" in inf
        assert inf["vital_max_hp"] == 6
