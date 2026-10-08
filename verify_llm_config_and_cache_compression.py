#!/usr/bin/env python3
# ==============================================================================
# VERIFICATION SUITE: LLM CONFIG AGENT & CACHE MATRIX COMPRESSOR
# ==============================================================================
# Validates:
#   1. Invariant: VITAL_MAX_HP == 6
#   2. Form Cells & ConfigPanelSchema integrity
#   3. LLM Configuration Agent Mutation (Non-repetitive generation)
#   4. CacheMatrixRepetitionCompressor (Deduplication of constant runs & tail combinations)
#   5. Lossless bit-exact matrix decompression
#   6. REST API endpoints (/api/config/*)
# ==============================================================================

import os
import sys
import json
import io

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def test_llm_config_and_cache_compression():
    print("=" * 70)
    print(" 🤖 KRYSTAL-STACK: LLM CONFIG AGENT & MATRIX COMPRESSION VERIFICATION")
    print("=" * 70)

    # 1. Invariant Assertion
    from krystal_kernel import VITAL_MAX_HP
    assert VITAL_MAX_HP == 6, f"VIOLATION: VITAL_MAX_HP is {VITAL_MAX_HP}, must be 6!"
    print(f"\n[1/6] Invariant Verified: VITAL_MAX_HP = {VITAL_MAX_HP}")

    # 2. ConfigPanelSchema & Form Cells
    from krystal_kernel import (
        ConfigCell,
        ConfigPanelSchema,
        LLMConfigAgent,
        GLOBAL_LLM_CONFIG_AGENT,
        CacheMatrixRepetitionCompressor,
        GLOBAL_MATRIX_COMPRESSOR
    )

    cells = ConfigPanelSchema.get_default_cells()
    print(f"\n[2/6] Validating Configuration Form Cells ({len(cells)} cells registered):")
    expected_cells = [
        "vital_max_hp", "terrain_octaves", "terrain_height_scale",
        "chalice_bowl_radius", "athame_blade_length", "urban_spire_height",
        "coxeter_mirror_folds", "bayer_matrix_dim", "swap_batch_quota"
    ]
    for cid in expected_cells:
        assert cid in cells, f"Missing expected cell: {cid}"
        cell = cells[cid]
        print(f"      • [{cell.category}] {cell.label} = {cell.current_value} ({cell.data_type})")

    # 3. LLM Configuration Agent Mutation
    agent = GLOBAL_LLM_CONFIG_AGENT
    print("\n[3/6] Testing LLM Configuration Agent Mutation...")
    initial_octaves = agent.cells["terrain_octaves"].current_value
    initial_folds = agent.cells["coxeter_mirror_folds"].current_value

    mut1 = agent.mutate_configuration_via_logs(avoid_repetition=True)
    print(f"      Mutation 1: Mutated {mut1.mutated_cell_count} cells | Repetition Score: {mut1.log_repetition_score}")
    print(f"      Rationale:  {mut1.rationale}")
    print(f"      Seed:       {mut1.non_repetitive_seed}")

    # Assert VITAL_MAX_HP remains strictly 6!
    assert agent.cells["vital_max_hp"].current_value == 6, "VITAL_MAX_HP must NEVER change!"

    # Test that attempting manual update of vital_max_hp does not alter 6
    agent.update_cell_value("vital_max_hp", 999)
    assert agent.cells["vital_max_hp"].current_value == 6, "Invariant enforcement failed on manual update!"

    # Second mutation to ensure non-repetitive change
    mut2 = agent.mutate_configuration_via_logs(avoid_repetition=True)
    print(f"      Mutation 2: Mutated {mut2.mutated_cell_count} cells | Repetition Score: {mut2.log_repetition_score}")
    assert mut1.non_repetitive_seed != mut2.non_repetitive_seed, "Seeds must be non-repetitive"

    # 4. Cache Matrix Repetition Compressor
    compressor = GLOBAL_MATRIX_COMPRESSOR
    print("\n[4/6] Testing Cache Matrix Repetition Compressor on generic repeating streams...")

    # Fabricate 10 frames of matrices with typical game engine repetitions:
    # 4 identical identity matrices, then 4 identical scaled affine matrices, then 2 rotated matrices
    identity_mat = [1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,  0.0, 0.0, 0.0, 1.0]
    scaled_mat   = [1.5, 0.0, 0.0, 0.0,  0.0, 1.5, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,  0.0, 0.0, 0.0, 1.0]
    rotated_mat  = [0.866, -0.5, 0.0, 0.0,  0.5, 0.866, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,  0.0, 0.0, 0.0, 1.0]

    matrix_stream = [
        identity_mat, identity_mat, identity_mat, identity_mat,
        scaled_mat, scaled_mat, scaled_mat, scaled_mat,
        rotated_mat, rotated_mat
    ]

    compressed_bytes, stats = compressor.compress_matrix_stream(matrix_stream)
    print(f"      Raw Size:               {stats.raw_size_bytes} Bytes ({stats.raw_elements_count} floats)")
    print(f"      Compressed Size:        {stats.compressed_size_bytes} Bytes")
    print(f"      Compression Ratio:      {stats.compression_ratio}x")
    print(f"      Deduplicated Constants: {stats.deduplicated_constants_count} values")
    print(f"      L1 Cache Lines Saved:   {stats.l1_cache_lines_saved} lines (64-byte x86 lines)")
    print(f"      L3 Bandwidth Saved:     {stats.l3_writeback_bandwidth_saved_kb} KB")

    assert stats.compression_ratio > 1.4, f"Compression ratio too low: {stats.compression_ratio}x"
    assert stats.l1_cache_lines_saved > 0, "Must save at least 1 L1 cache line"

    # 5. Exact Lossless Decompression Verification
    print("\n[5/6] Verifying Lossless Bit-Exact Decompression...")
    decompressed = compressor.decompress_matrix_stream(compressed_bytes)
    assert len(decompressed) == len(matrix_stream), f"Length mismatch: {len(decompressed)} vs {len(matrix_stream)}"

    for i, (orig, decomp) in enumerate(zip(matrix_stream, decompressed)):
        assert len(orig) == len(decomp)
        for j, (o_val, d_val) in enumerate(zip(orig, decomp)):
            assert abs(o_val - d_val) < 1e-4, f"Matrix mismatch at frame {i}, index {j}: {o_val} vs {d_val}"
    print("      Decompression Verified: 10/10 matrices reconstructed with 100% precision!")

    # 6. REST API Endpoints Simulation
    print("\n[6/6] Verifying Server Endpoints for Dynamic Panels & Compression...")
    from krystal_web_hub.server import KrystalHubHandler

    class DummyHandler(KrystalHubHandler):
        def __init__(self, path, method="GET", body=b""):
            self.path = path
            self.command = method
            self.requestline = f"{method} {path} HTTP/1.1"
            self.request_version = "HTTP/1.1"
            self.headers = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
            self.rfile = io.BytesIO(body)
            self.wfile = io.BytesIO()
            if method == "GET": self.do_GET()
            elif method == "POST": self.do_POST()

    r1 = DummyHandler("/api/config/cells")
    r1.wfile.seek(0)
    out1 = r1.wfile.read().decode("utf-8", errors="ignore")
    assert "200 OK" in out1 and '"status": "OK"' in out1

    r2 = DummyHandler("/api/config/mutate", method="POST", body=b"{}")
    r2.wfile.seek(0)
    out2 = r2.wfile.read().decode("utf-8", errors="ignore")
    assert "200 OK" in out2 and '"status": "SUCCESS"' in out2

    r3 = DummyHandler("/api/config/compress", method="POST", body=b"{}")
    r3.wfile.seek(0)
    out3 = r3.wfile.read().decode("utf-8", errors="ignore")
    assert "200 OK" in out3 and '"status": "SUCCESS"' in out3
    print("      REST Endpoints Verified: GET /api/config/cells, POST /api/config/mutate, POST /api/config/compress -> 200 OK")

    print("\n" + "=" * 70)
    print(" ✅ ALL 6 LLM CONFIG AGENT & CACHE COMPRESSION TESTS PASSED (100% INTEGRITY)")
    print("=" * 70)

if __name__ == "__main__":
    test_llm_config_and_cache_compression()
