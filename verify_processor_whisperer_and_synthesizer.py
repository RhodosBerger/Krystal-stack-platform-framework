#!/usr/bin/env python3
"""
VERIFICATION TEST SUITE: PROCESSOR WHISPERER & JANET SCRIPT SYNTHESIZER
=======================================================================
Verifies:
  1. Janet Script Synthesizer & Hexadecimal Bytecode Stream.
  2. GF(2) Binary Matrix Parity & Self-Healing Syndrome Engine.
  3. Instruction Set Meta Governor (PGO ISA recommendation).
  4. Processor Whisperer In-Memory Fast Ring & Asynchronous SSD Persistence.
"""

import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from krystal_kernel import (
    EngineOpcode,
    JanetScriptEngineSynthesizer,
    BinaryMatrixSelfHealingLog,
    HardwareEventOrigin,
    InstructionSetMetaGovernor,
    RecommendedISA,
    ProcessorInstructionWhisperer,
    VITAL_MAX_HP
)

def test_janet_script_synthesizer():
    print("[TEST 1] Testing Janet Script Synthesizer & Hexadecimal Bytecode...")
    synth = JanetScriptEngineSynthesizer()
    script_obj = synth.synthesize_script(scene_name="TestScene", seed=101)

    assert "(vital/assert-hp :expected 6)" in script_obj.script_source, "Invariant HP=6 must be present"
    assert "(sdf/chalice" in script_obj.script_source, "SDF chalice must be synthesized"
    assert "(urban/extrude-spire" in script_obj.script_source, "Urban spire must be synthesized"
    assert script_obj.bytecode_bytes.startswith(b"KSYN"), "Bytecode must start with KSYN magic header"
    assert len(script_obj.bytecode_bytes) >= 64, f"Bytecode must be at least 64 bytes, got {len(script_obj.bytecode_bytes)}"
    assert "KSYN" in script_obj.hex_dump, "Hex dump must contain KSYN ASCII representation"
    print(f"  -> Synthesizer OK: {script_obj.function_count} S-expressions, {len(script_obj.bytecode_bytes)} B bytecode, valid hex dump.")

def test_binary_matrix_self_healing():
    print("[TEST 2] Testing GF(2) Binary Matrix Self-Healing & Syndrome Attribution...")
    log_engine = BinaryMatrixSelfHealingLog()

    # 1. Clean event
    e_clean = log_engine.log_hardware_event(
        pc=0x100, is_current_switch=False, is_l1_write=True, is_l3_write=False, is_dram_spill=False
    )
    assert e_clean.syndrome == 0, "Clean event must have zero syndrome"
    assert e_clean.healed is False, "Clean event does not need healing"
    assert e_clean.origin == HardwareEventOrigin.L1_CACHE_WRITE, "Origin must be L1"

    # 2. Corrupted event (simulated bit flip on position 1..7)
    for flip_pos in range(1, 8):
        e_noisy = log_engine.log_hardware_event(
            pc=0x200 + flip_pos, is_current_switch=False, is_l1_write=False, is_l3_write=True, is_dram_spill=False,
            simulated_bit_flip_pos=flip_pos
        )
        assert e_noisy.healed is True, f"Bit flip at pos {flip_pos} must be healed"
        assert e_noisy.origin == HardwareEventOrigin.L3_CACHE_WRITE, f"Must recover L3 origin after flip at {flip_pos}"

    # 3. Current switch event
    e_cs = log_engine.log_hardware_event(
        pc=0x300, is_current_switch=True, is_l1_write=False, is_l3_write=False, is_dram_spill=False
    )
    assert e_cs.origin == HardwareEventOrigin.CURRENT_SWITCH, "Must attribute Current Switch origin"
    print(f"  -> Binary matrix OK: 7/7 single-bit flip anomalies repaired, total healed={log_engine.total_healed_events}.")

def test_instruction_set_governor():
    print("[TEST 3] Testing Instruction Set Meta Governor...")
    
    # Case A: Mostly L1 writes
    log_a = BinaryMatrixSelfHealingLog()
    for i in range(20):
        log_a.log_hardware_event(pc=i, is_current_switch=False, is_l1_write=True, is_l3_write=False, is_dram_spill=False)
    gov_a = InstructionSetMetaGovernor(log_a)
    rec_a = gov_a.evaluate_optimal_instruction_set()
    assert rec_a.recommended_isa == RecommendedISA.AVX512_FMA_UNROLLED
    assert rec_a.unroll_factor == 8

    # Case B: Heavy L3 writebacks
    log_b = BinaryMatrixSelfHealingLog()
    for i in range(20):
        log_b.log_hardware_event(pc=i, is_current_switch=False, is_l1_write=False, is_l3_write=True, is_dram_spill=False)
    gov_b = InstructionSetMetaGovernor(log_b)
    rec_b = gov_b.evaluate_optimal_instruction_set()
    assert rec_b.recommended_isa == RecommendedISA.STREAMING_NT_STORES
    assert rec_b.prefetch_distance_bytes == 128

    # Case C: High current switching
    log_c = BinaryMatrixSelfHealingLog()
    for i in range(20):
        log_c.log_hardware_event(pc=i, is_current_switch=True, is_l1_write=False, is_l3_write=False, is_dram_spill=False)
    gov_c = InstructionSetMetaGovernor(log_c)
    rec_c = gov_c.evaluate_optimal_instruction_set()
    assert rec_c.recommended_isa == RecommendedISA.SCALAR_COMPACT_SSE4
    print("  -> ISA Meta Governor OK: Dynamic tier selection (AVX-512 FMA, Streaming NT Stores, Scalar SSE4) verified.")

def test_processor_whisperer_and_ssd_flush():
    print("[TEST 4] Testing Processor Whisperer & Asynchronous SSD Flush...")
    test_ssd_file = "logs/test_verification_whisperer.hexlog"
    if os.path.exists(test_ssd_file):
        os.remove(test_ssd_file)

    whisperer = ProcessorInstructionWhisperer(ssd_log_filepath=test_ssd_file, ring_capacity=1024)

    # 1. Fast in-memory instruction hint
    t0 = time.perf_counter_ns()
    hint = whisperer.whisper_instruction_hint(pc=0x4000)
    elapsed_ns = time.perf_counter_ns() - t0
    assert "isa_hint" in hint
    assert elapsed_ns < 500_000, f"In-memory hint must be sub-millisecond, took {elapsed_ns} ns"

    # 2. Record hardware events
    for pc in range(0x1000, 0x1020):
        whisperer.record_hardware_pulse(
            pc=pc,
            is_current_switch=(pc % 5 == 0),
            is_l1_write=(pc % 2 == 0),
            is_l3_write=(pc % 3 == 0),
            is_dram_spill=False,
            simulate_anomaly=(pc % 7 == 0)
        )

    # Allow async SSD worker to flush
    time.sleep(0.3)
    whisperer.stop()

    assert os.path.exists(test_ssd_file), "SSD log file must be created on disk"
    with open(test_ssd_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "PC:0x1000" in content, "Flushed content must contain recorded PCs"
    assert "HEALED:True" in content, "Flushed content must contain self-healed entries"
    file_size = os.path.getsize(test_ssd_file)
    print(f"  -> Processor Whisperer OK: In-memory query took {elapsed_ns/1000:.1f}µs, SSD log flushed ({file_size} bytes).")

if __name__ == "__main__":
    print("=================================================================")
    print("KRYSTAL-STACK: VERIFYING PROCESSOR WHISPERER & JANET SYNTHESIZER")
    print("=================================================================")
    test_janet_script_synthesizer()
    test_binary_matrix_self_healing()
    test_instruction_set_governor()
    test_processor_whisperer_and_ssd_flush()
    print("=================================================================")
    print("ALL TESTS PASSED WITH 100% INTEGRITY!")
    print("=================================================================")
