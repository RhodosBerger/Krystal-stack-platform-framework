#!/usr/bin/env python3
"""
================================================================================
UNIT TESTS: RECURRENT VARIATIONS STREAM DECODER & VIRTUAL HOST EMULATOR
================================================================================
Module:     tests/test_sovereign_recurrent_stream_emulator.py
Author:     Antigravity Cognitive Invariants Group (2026)
Invariants: VITAL_MAX_HP == 6 | Proposer != Evaluator | Delta S == 0 on breach
================================================================================
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.sovereign_runtime import SubsystemNodeId
from krystal_kernel.sovereign_recurrent_stream_emulator import (
    VariationKind,
    ComputationLayerId,
    ActionGenre,
    StreamVariationSample,
    DecodedHarmonicStream,
    VirtualExecutionTrace,
    HistoryDrawerEntry,
    SyntheticInferenceProfile,
    CrossoverActionReceipt,
    RecurrentVariationsStreamDecoder,
    VirtualLayerHostEmulator,
    RecurrentPathwaysGovernorExtension,
    DependencyGraphHistoryDrawer,
    SyntheticInferencePatternScaler,
    ApiCrossoverActionEndpointManager,
    GLOBAL_RECURRENT_STREAM_DECODER,
    GLOBAL_VIRTUAL_HOST_EMULATOR,
    GLOBAL_RECURRENT_GOVERNOR_EXT,
    GLOBAL_HISTORY_DRAWER,
    GLOBAL_SYNTHETIC_PATTERN_SCALER,
    GLOBAL_CROSSOVER_ENDPOINT_MANAGER,
)


class TestRecurrentVariationsStreamDecoder(unittest.TestCase):
    """Verifies stream decoding, frequency extraction, and token quantization."""

    def setUp(self) -> None:
        self.decoder = RecurrentVariationsStreamDecoder(vital_max_hp=VITAL_MAX_HP)

    def test_empty_stream_decoding(self) -> None:
        """Empty sample list returns baseline stream without crashing."""
        decoded = self.decoder.decode_stream("STRM-EMPTY", "MEAS-00", [])
        self.assertEqual(decoded.fundamental_freq_hz, 0.0)
        self.assertEqual(decoded.stability_score, 1.0)
        self.assertEqual(decoded.quantized_tokens, [])

    def test_sample_decoding_and_token_quantization(self) -> None:
        """Stream samples are quantized into harmonic tokens with jitter metrics."""
        samples = [
            StreamVariationSample("S1", "MEAS-01", VariationKind.LATENCY_JITTER, ComputationLayerId.L2_FPGA_CHROMA, 1.2, 0.0, 15, 1_000_000_000),
            StreamVariationSample("S2", "MEAS-01", VariationKind.LATENCY_JITTER, ComputationLayerId.L2_FPGA_CHROMA, 1.4, 1.57, 25, 1_016_666_666),
            StreamVariationSample("S3", "MEAS-01", VariationKind.LATENCY_JITTER, ComputationLayerId.L2_FPGA_CHROMA, 1.3, 3.14, 18, 1_033_333_333),
        ]
        decoded = self.decoder.decode_stream("STRM-ACTIVE", "MEAS-01", samples)
        self.assertEqual(decoded.measurement_id, "MEAS-01")
        self.assertGreater(decoded.fundamental_freq_hz, 0.0)
        self.assertGreater(decoded.mean_jitter_ns, 0.0)
        self.assertEqual(len(decoded.quantized_tokens), 3)
        self.assertTrue(decoded.quantized_tokens[0].startswith("TOK_LAT_L2"))


class TestVirtualLayerHostEmulator(unittest.TestCase):
    """Verifies host virtualization of multi-layer dependency graphs."""

    def setUp(self) -> None:
        self.emulator = VirtualLayerHostEmulator(vital_max_hp=VITAL_MAX_HP)

    def test_successful_virtual_emulation(self) -> None:
        """Nominal conditions produce valid execution trace with Delta S == 1."""
        layers = [
            ComputationLayerId.L0_REGISTER_FILE,
            ComputationLayerId.L1_MICROCODE_SRAM,
            ComputationLayerId.L2_FPGA_CHROMA,
            ComputationLayerId.L3_DMA_RING
        ]
        trace = self.emulator.emulate_graph_execution(layers, current_uma_pressure=0.35, current_temp_c=45.0)
        self.assertTrue(trace.is_valid_transition)
        self.assertEqual(trace.state_mutation_delta_s, 1)
        self.assertGreater(trace.total_virtual_latency_us, 0.0)
        self.assertGreater(trace.total_virtual_energy_uj, 0.0)

    def test_rejection_on_uma_bus_saturation(self) -> None:
        """UMA pressure >= 0.78 must abort with Delta S == 0."""
        layers = [ComputationLayerId.L4_SHADOW_VBLANK]
        trace = self.emulator.emulate_graph_execution(layers, current_uma_pressure=0.82, current_temp_c=45.0)
        self.assertFalse(trace.is_valid_transition)
        self.assertEqual(trace.state_mutation_delta_s, 0)
        self.assertIn("UMA pressure", trace.rejection_reason)

    def test_rejection_on_thermal_limit(self) -> None:
        """Temperature >= 75°C must abort with Delta S == 0."""
        layers = [ComputationLayerId.L5_KSL_CORE]
        trace = self.emulator.emulate_graph_execution(layers, current_uma_pressure=0.30, current_temp_c=76.5)
        self.assertFalse(trace.is_valid_transition)
        self.assertEqual(trace.state_mutation_delta_s, 0)
        self.assertIn("Arrhenius limit", trace.rejection_reason)


class TestRecurrentPathwaysGovernorExtension(unittest.TestCase):
    """Verifies recurrent pathway tracking tied to Measurement IDs."""

    def setUp(self) -> None:
        self.gov = RecurrentPathwaysGovernorExtension(vital_max_hp=VITAL_MAX_HP)

    def test_pathway_registration_and_stress_adaptation(self) -> None:
        """Stress hits decrease multiplier; normal hits recover."""
        self.gov.register_pathway("MEAS-FPGA-MAILBOX", "FPGA_MAILBOX_DMA")
        m1 = self.gov.record_pathway_hit("MEAS-FPGA-MAILBOX", detected_stress=True)
        self.assertLess(m1, 1.0)

        # Recovery
        m2 = self.gov.record_pathway_hit("MEAS-FPGA-MAILBOX", detected_stress=False)
        self.assertGreater(m2, m1)


class TestDependencyGraphHistoryDrawer(unittest.TestCase):
    """Verifies snapshot archival and pending script validation."""

    def setUp(self) -> None:
        self.drawer = DependencyGraphHistoryDrawer(vital_max_hp=VITAL_MAX_HP)

    def test_graph_snapshot_archival(self) -> None:
        """Snapshots are archived with SHA256 hashes and timestamp."""
        nodes = ["N1", "N2", "N3"]
        edges = [("N1", "N2"), ("N2", "N3")]
        entry = self.drawer.archive_graph_snapshot("MEAS-SNAP-01", "v1.4.2", nodes, edges)
        self.assertEqual(len(self.drawer.drawer_archive), 1)
        self.assertEqual(entry.node_count, 3)
        self.assertEqual(entry.edge_count, 2)
        self.assertEqual(len(entry.snapshot_hash), 64)

    def test_pending_script_bare_assert_rejection(self) -> None:
        """Python scripts with bare assert are rejected."""
        bad_script = "def run():\n    assert 1 == 1\n"
        ok, reason = self.drawer.pass_pending_script(bad_script, language="python", declared_dependencies=[])
        self.assertFalse(ok)
        self.assertIn("Bare assert forbidden", reason)

    def test_pending_script_valid_pass(self) -> None:
        """Clean scripts pass validation."""
        good_script = "def run():\n    if vital_max_hp != 6: raise ContractViolation('breach')\n"
        ok, reason = self.drawer.pass_pending_script(good_script, language="python", declared_dependencies=["VITAL_MAX_HP"])
        self.assertTrue(ok)


class TestSyntheticInferencePatternScaler(unittest.TestCase):
    """Verifies synthetic inference scaling across models and backends."""

    def setUp(self) -> None:
        self.scaler = SyntheticInferencePatternScaler(vital_max_hp=VITAL_MAX_HP)

    def test_onnx_and_gpu_scaling_projection(self) -> None:
        """GPU/NPU backends yield lower energy per token and sublinear latency."""
        prof_cpu = self.scaler.project_pattern_scaling("phantom_net.onnx", "CPU_BASELINE", batch_size=4)
        prof_gpu = self.scaler.project_pattern_scaling("phantom_net.onnx", "IRIS_XE_GPU", batch_size=4)

        self.assertLess(prof_gpu.energy_joules_per_tok, prof_cpu.energy_joules_per_tok)
        self.assertGreater(prof_gpu.pattern_stability_index, 0.0)


class TestApiCrossoverActionEndpointManager(unittest.TestCase):
    """Verifies crossover endpoint routing and HMAC audit receipts."""

    def setUp(self) -> None:
        self.crossover = ApiCrossoverActionEndpointManager(vital_max_hp=VITAL_MAX_HP)

    def test_crossover_action_dispatch(self) -> None:
        """Dispatched actions are recorded with cryptographic HMAC-SHA256 signatures."""
        receipt = self.crossover.dispatch_crossover_action(
            genre=ActionGenre.HARDWARE_ACTUATION,
            endpoint_route="/api/sovereign/crossover/dispatch_dma",
            source=SubsystemNodeId.WSL_BRIDGE,
            target=SubsystemNodeId.CHRONO_SOC,
            action_payload={"bytes": 1024, "ring": "DMA_SPSC_0"}
        )
        self.assertEqual(len(self.crossover.action_log), 1)
        self.assertEqual(receipt.genre, ActionGenre.HARDWARE_ACTUATION)
        self.assertEqual(len(receipt.hmac_digest), 64)
        self.assertEqual(receipt.vital_max_hp, VITAL_MAX_HP)


if __name__ == "__main__":
    unittest.main()
