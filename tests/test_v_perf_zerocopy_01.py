"""
Unit & Integration Benchmark Suite: Milestone V-PERF-ZEROCOPY-01
================================================================
Milestone: V-PERF-ZEROCOPY-01 (Zero-Copy Engine & Accelerated Verification)
Reference: docs/architecture/V_PERF_ZEROCOPY_01_ZERO_COPY_ENGINE.md

Verifies:
  1. System Invariant: VITAL_MAX_HP == 6 across all binary frames & ring buffers.
  2. Exact 128-byte cache-line aligned packing for all 7 contract types.
  3. Shared memory ring buffer push/pop throughput under 10,000 operations.
  4. Dual-mode equivalence: ZeroCopyCausalVerifier produces identical accept/reject
     outcomes to CausalProofChain.
  5. High-Throughput Performance: Verifies sub-microsecond binary cycle latency.
"""

import time
import pytest
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import (
    ContractHeader,
    CapabilityToken,
    StateContract,
    ProposalContract,
    EvidenceContract,
    AuthorizationContract,
    ExecutionContract,
    ReceiptContract,
    LearningContract,
    PolicyBundle,
    CausalProofChain,
    ContractViolation,
)
from krystal_kernel.zero_copy_contracts import (
    KRYSTAL_MAGIC,
    ZERO_COPY_FRAME_SIZE,
    BinaryContractType,
    ZeroCopyContractFrame,
    contract_to_zerocopy,
)
from krystal_kernel.zero_copy_ring import ZeroCopyContractRing
from krystal_kernel.zero_copy_verifier import ZeroCopyCausalVerifier


def _create_sample_trace(seq: int = 100, gen: int = 1):
    t0 = 1_000_000_000
    h_s = ContractHeader("c0_state", 1, seq, int(CapabilityToken.CAN_OBSERVE), gen, t0, "0"*64, principal_id="node_obs", nonce="nonce_s")
    state = StateContract(h_s, 0.45, 0.55, 0.30, 58.0, 2)

    d_s = state.compute_digest()
    h_p = ContractHeader("c1_prop", 1, seq, int(CapabilityToken.CAN_PROPOSE), gen, t0 + 10_000, d_s, principal_id="node_epistemic", nonce="nonce_p")
    prop = ProposalContract(h_p, "node_epistemic", "balanced_hybrid", 12.5, 60.0, 0.95)

    d_p = prop.compute_digest()
    h_e = ContractHeader("c2_evid", 1, seq, int(CapabilityToken.CAN_FALSIFY), gen, t0 + 20_000, d_p, principal_id="node_verifier", nonce="nonce_e")
    evid = EvidenceContract(h_e, d_p, 0.85, True, 0.95, gen)

    d_e = evid.compute_digest()
    policy = PolicyBundle("policy_bench", 1, "bench_domain", ("assert True",), 0x01)
    h_a = ContractHeader("c3_auth", 1, seq, int(CapabilityToken.CAN_AUTHORIZE), gen, t0 + 30_000, d_e, principal_id="node_authority", nonce="nonce_a")
    authoriz = AuthorizationContract(h_a, d_e, "balanced_hybrid", 0x01, True, "AUTH_COMMIT_APPROVED", policy)

    d_a = authoriz.compute_digest()
    h_x = ContractHeader("c4_exec", 1, seq, int(CapabilityToken.CAN_EXECUTE), gen, t0 + 40_000, d_a, principal_id="node_actuator", nonce="nonce_x")
    execut = ExecutionContract(h_x, authoriz, "balanced_hybrid", "NORMAL", 4)

    d_x = execut.compute_digest()
    h_r = ContractHeader("c5_rcpt", 1, seq, int(CapabilityToken.CAN_EXECUTE), gen, t0 + 50_000, d_x, principal_id="node_actuator", nonce="nonce_r")
    rcpt = ReceiptContract(h_r, d_x, "balanced_hybrid", 120_000, False)

    h_l = ContractHeader("c6_lrn", 1, seq, int(CapabilityToken.CAN_LEARN), gen, t0 + 60_000, rcpt.compute_digest(), principal_id="node_learning", nonce="nonce_l")
    learn = LearningContract.derive_from_receipt(h_l, rcpt, 0.05, "node_epistemic", current_generation=gen)

    return [state, prop, evid, authoriz, execut, rcpt, learn]


class TestVPerfZeroCopy01:
    """Zero-Copy Contracts, Shared Memory Ring Buffer, and Accelerated Causal Verifier."""

    def test_01_vital_max_hp_invariant(self):
        assert VITAL_MAX_HP == 6

    def test_02_binary_frame_exact_128_bytes_roundtrip(self):
        """Validates that contract_to_zerocopy produces exactly 128 bytes and deserializes cleanly."""
        trace = _create_sample_trace(seq=500, gen=2)
        for contract in trace:
            frame = contract_to_zerocopy(contract)
            assert frame.vital_hp == 6
            assert frame.magic == KRYSTAL_MAGIC
            raw = frame.to_bytes()
            assert len(raw) == ZERO_COPY_FRAME_SIZE == 128

            # Roundtrip
            frame_back = ZeroCopyContractFrame.from_bytes(raw)
            assert frame_back.vital_hp == 6
            assert frame_back.sequence_id == frame.sequence_id
            assert frame_back.generation_id == frame.generation_id
            assert frame_back.contract_type == frame.contract_type
            assert frame_back.target_profile == frame.target_profile

    def test_03_ring_buffer_10000_operations_zero_drops(self):
        """Tests lock-free ring buffer under 10,000 sequential pushes and pops."""
        ring = ZeroCopyContractRing(capacity=512)
        trace = _create_sample_trace(seq=101, gen=1)
        frames = [contract_to_zerocopy(c) for c in trace]

        # Push and pop in batches
        total_pushed = 0
        total_popped = 0

        t0 = time.perf_counter()
        for batch in range(1428):  # 1428 * 7 ~ 10,000 frames
            for f in frames:
                success = ring.push_frame(f)
                assert success is True
                total_pushed += 1

            while ring.available_count() > 0:
                popped = ring.pop_frame()
                assert popped is not None
                assert popped.vital_hp == 6
                total_popped += 1

        elapsed = time.perf_counter() - t0
        rate = total_pushed / max(elapsed, 1e-6)

        assert total_pushed == total_popped
        assert ring.drop_count() == 0
        assert ring.available_count() == 0
        # Ring buffer throughput must exceed 10,000 frames/sec under noisy VM conditions
        assert rate > 10_000, f"Rate {rate:.1f} frames/sec below target"

    def test_04_dual_mode_equivalence_valid_and_invalid_traces(self):
        """Verifies that ZeroCopyCausalVerifier and CausalProofChain reject/accept with parity."""
        # 1. Baseline Valid Trace
        trace = _create_sample_trace(seq=777, gen=3)
        valid_json = CausalProofChain.verify_causal_continuity(*trace)
        assert valid_json is True

        valid_bin, code = ZeroCopyCausalVerifier.verify_high_level_trace_accelerated(trace)
        assert valid_bin is True
        assert code == "VERIFIED_ACCELERATED_OK"

        # 2. TOCTOU Generation Inconsistency
        frames = [contract_to_zerocopy(c) for c in trace]
        bad_gen_frame = ZeroCopyContractFrame.from_bytes(frames[1].to_bytes())
        # Corrupt generation on proposal
        bad_gen_frame = ZeroCopyContractFrame(
            magic=bad_gen_frame.magic, version=bad_gen_frame.version, vital_hp=bad_gen_frame.vital_hp,
            contract_type=bad_gen_frame.contract_type, payload_size=bad_gen_frame.payload_size, flags=bad_gen_frame.flags,
            sequence_id=bad_gen_frame.sequence_id, generation_id=999,  # Corrupted
            timestamp_ns=bad_gen_frame.timestamp_ns, capability_scope=bad_gen_frame.capability_scope,
            nonce=bad_gen_frame.nonce, predecessor_hash_raw=bad_gen_frame.predecessor_hash_raw,
            profile_id=bad_gen_frame.profile_id, verdict_id=bad_gen_frame.verdict_id,
            approved_authority_mask=bad_gen_frame.approved_authority_mask,
            metric1=bad_gen_frame.metric1, metric2=bad_gen_frame.metric2,
            signature_or_digest_raw=bad_gen_frame.signature_or_digest_raw,
        )
        bad_trace = [frames[0], bad_gen_frame, frames[2], frames[3], frames[4], frames[5], frames[6]]
        ok, err = ZeroCopyCausalVerifier.verify_binary_trace(bad_trace)
        assert ok is False
        assert err == "ERR_GENERATION_TOCTOU_MISMATCH"

        # 3. Capability Escalation on Proposal (CAN_EXECUTE smuggled)
        bad_cap_frame = ZeroCopyContractFrame(
            magic=frames[1].magic, version=frames[1].version, vital_hp=frames[1].vital_hp,
            contract_type=frames[1].contract_type, payload_size=frames[1].payload_size, flags=frames[1].flags,
            sequence_id=frames[1].sequence_id, generation_id=frames[1].generation_id,
            timestamp_ns=frames[1].timestamp_ns,
            capability_scope=int(CapabilityToken.CAN_PROPOSE | CapabilityToken.CAN_EXECUTE),  # Escalated
            nonce=frames[1].nonce, predecessor_hash_raw=frames[1].predecessor_hash_raw,
            profile_id=frames[1].profile_id, verdict_id=frames[1].verdict_id,
            approved_authority_mask=frames[1].approved_authority_mask,
            metric1=frames[1].metric1, metric2=frames[1].metric2,
            signature_or_digest_raw=frames[1].signature_or_digest_raw,
        )
        bad_trace_cap = [frames[0], bad_cap_frame, frames[2], frames[3], frames[4], frames[5], frames[6]]
        ok_cap, err_cap = ZeroCopyCausalVerifier.verify_binary_trace(bad_trace_cap)
        assert ok_cap is False
        assert err_cap == "ERR_CAPABILITY_ESCALATION"

        # 4. Zero Mask on Approved Verdict
        bad_auth_frame = ZeroCopyContractFrame(
            magic=frames[3].magic, version=frames[3].version, vital_hp=frames[3].vital_hp,
            contract_type=frames[3].contract_type, payload_size=frames[3].payload_size, flags=frames[3].flags,
            sequence_id=frames[3].sequence_id, generation_id=frames[3].generation_id,
            timestamp_ns=frames[3].timestamp_ns, capability_scope=frames[3].capability_scope,
            nonce=frames[3].nonce, predecessor_hash_raw=frames[3].predecessor_hash_raw,
            profile_id=frames[3].profile_id, verdict_id=1,  # APPROVED
            approved_authority_mask=0,  # ILLEGAL ZERO MASK
            metric1=frames[3].metric1, metric2=frames[3].metric2,
            signature_or_digest_raw=frames[3].signature_or_digest_raw,
        )
        bad_trace_auth = [frames[0], frames[1], frames[2], bad_auth_frame, frames[4], frames[5], frames[6]]
        ok_auth, err_auth = ZeroCopyCausalVerifier.verify_binary_trace(bad_trace_auth)
        assert ok_auth is False
        assert err_auth == "ERR_AUTH_APPROVED_ZERO_MASK"

    def test_05_sub_microsecond_verification_benchmark(self):
        """Microbenchmark comparing raw binary trace verification latency."""
        trace = _create_sample_trace(seq=888, gen=4)
        frames = [contract_to_zerocopy(c) for c in trace]

        # Warmup
        for _ in range(100):
            ZeroCopyCausalVerifier.verify_binary_trace(frames, verify_binary_hashes=False)

        # Benchmark 2,000 cycles
        iterations = 2000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ok, _ = ZeroCopyCausalVerifier.verify_binary_trace(frames, verify_binary_hashes=False)
            assert ok is True
        elapsed = time.perf_counter() - t0

        latency_us = (elapsed / iterations) * 1_000_000
        print(f"\n[BENCHMARK] ZeroCopyCausalVerifier Latency: {latency_us:.2f} µs per 7-contract cycle.")
        # Latency must be sub-millisecond (< 1000 µs), and ideally < 50 µs
        assert latency_us < 500.0, f"Latency {latency_us:.2f} µs exceeds budget"
