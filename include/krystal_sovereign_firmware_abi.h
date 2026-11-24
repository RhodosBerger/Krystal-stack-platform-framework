/*
 * Krystal Stack Platform Framework — Sovereign Rust Firmware, Retroactive Validator,
 * Stream-CUDA Extension & Kisak-Gallium Rust Driver C11 ABI Contract
 *
 * Copyright (c) 2026 Dušan Kopecký (@RhodosBerger). All Rights Reserved.
 * Authorized Partner Addendum: Ivan Novikov (xerj.org).
 *
 * Constitutional Invariants:
 *   - VITAL_MAX_HP == 6 (\neg Valid => \Delta S = 0)
 *   - UMA_SATURATION_CEILING_MILLI == 780 (0.78)
 *   - Zero external crate bloat (#![no_std]), L1i-cache resident binary
 */

#ifndef KRYSTAL_SOVEREIGN_FIRMWARE_ABI_H
#define KRYSTAL_SOVEREIGN_FIRMWARE_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KSFW_ABI_VERSION               0x00010200u
#define KSFW_MAGIC_BANK                0x4B534657u /* "KSFW" */
#define KSFW_MAGIC_GALLIUM_PIPE        0x4B47414Cu /* "KGAL" */
#define KSFW_MAGIC_STREAM_CUDA         0x4B534355u /* "KSCU" */
#define KSFW_VITAL_MAX_HP              6u
#define KSFW_UMA_CEILING_MILLI         780u

/* 128-Byte Transactional Firmware Bank State (Bank Alpha / Bank Beta) */
typedef struct
#if defined(__GNUC__) || defined(__clang__)
__attribute__((aligned(128)))
#endif
KsfwBankState128B {
    uint32_t magic;                  /* 0..4:   0x4B534657 ("KSFW") */
    uint32_t svn_epoch;              /* 4..8:   Monotonic anti-rollback epoch */
    uint8_t  bank_id;                /* 8..9:   0 = Alpha, 1 = Beta */
    uint8_t  vital_max_hp;           /* 9..10:  Must equal 6 */
    uint8_t  thermal_band;           /* 10..11: THRM_01..05 (1..5) */
    uint8_t  epp_hint;               /* 11..12: Energy Performance Preference (0..255) */
    uint16_t rapl_pl1_watts_x10;     /* 12..14: Package Power Limit PL1 (deci-Watts) */
    uint16_t uncore_min_mhz;         /* 14..16: Ring/Uncore floor MHz */
    uint16_t uma_filtered_milli;     /* 16..18: TH_03 [0.25,0.50,0.25] filtered UMA */
    uint16_t uma_shadow_price_milli; /* 18..20: TH_09 Contactless Solder shadow price */
    uint16_t allowed_xe_eus;         /* 20..22: Active Iris Xe EUs out of 96 */
    uint16_t allowed_stream_procs;   /* 22..24: Active FP32 Stream Processors out of 768 */
    uint32_t crypto_isa_mask;        /* 24..28: Bitmask: SHA-NI|VAES|AVX512_IFMA|MLKEM_NTT */
    uint32_t kubc_ops_executed;      /* 28..32: Executed 64-bit KUBC VLIW instructions */
    uint64_t kubc_register_acc;      /* 32..40: Deterministic KUBC accumulator */
    uint64_t rollback_noop_count;    /* 40..48: Constitutional rollbacks (\Delta S = 0) */
    uint64_t state_fnv1a_digest;     /* 48..56: Integrity attestation digest */
    uint64_t last_tsc_cycles;        /* 56..64: Hardware RDTSC execution delta */
    uint8_t  reserved_pad[64];       /* 64..128: Pad to exact 128B (2 cache lines) */
} KsfwBankState128B;

/* 64-Byte Retroactive Feedback Validator -> Generator & Aggregator Descriptor */
typedef struct
#if defined(__GNUC__) || defined(__clang__)
__attribute__((aligned(64)))
#endif
KsfwRetroValidatorReceipt64B {
    uint32_t receipt_magic;          /* 0..4:   0x5256414Cu ("RVAL") */
    uint8_t  proposer_plane_id;      /* 4..5:   Must differ from verifier_plane_id */
    uint8_t  verifier_plane_id;      /* 5..6:   Independent Verifier ID */
    uint8_t  vital_max_hp;           /* 6..7:   Must equal 6 */
    uint8_t  emission_authorized;    /* 7..8:   1 = Generator+Aggregator exposed, 0 = Quarantined */
    uint32_t generator_tile_bytes;   /* 8..12:  Emitted zero-copy micro-batch tile size */
    uint32_t generator_simd_lanes;   /* 12..16: Emitted SIMD width (8/16/32/64) */
    uint32_t aggregator_tree_fanin;  /* 16..20: Lock-free reduction fan-in factor */
    uint32_t sanitized_denormals;    /* 20..24: TH_01 FTZ/DAZ flushed subnormals */
    uint32_t measured_gain_permil;   /* 24..28: Verified throughput ratio (1000 = 1.00x) */
    uint32_t uma_util_milli;         /* 28..32: Bus utilization during validation */
    uint64_t aggregated_checksum;    /* 32..40: Deterministic tree-reduction output bits */
    uint64_t receipt_digest;         /* 40..48: FNV-1a cryptographic receipt */
    uint8_t  reserved_tail[16];      /* 48..64: Exact 64B cache line */
} KsfwRetroValidatorReceipt64B;

/* 128-Byte Stream-CUDA + Kisak-Gallium Unified Pipe & FPS Booster Descriptor */
typedef struct
#if defined(__GNUC__) || defined(__clang__)
__attribute__((aligned(128)))
#endif
KsfwGalliumStreamCuda128B {
    uint32_t magic;                  /* 0..4:   0x4B47414Cu ("KGAL") */
    uint16_t stream_id;              /* 4..6:   CUDA-equivalent stream queue */
    uint8_t  simd_subgroup_width;    /* 6..7:   8, 16, or 32 (Warp/Subgroup) */
    uint8_t  vital_max_hp;           /* 7..8:   Must equal 6 */
    uint32_t grid_blocks_xyz[3];     /* 8..20:  CUDA gridDim (x,y,z) */
    uint32_t thread_block_xyz[3];    /* 20..32: CUDA blockDim (x,y,z) */
    uint32_t slm_shared_bytes;       /* 32..36: Subslice SLM (<= 65536 B = __shared__) */
    uint32_t gallium_driver_mask;    /* 36..40: IRIS|ANV|ZINK|RUSTICL|LLVMPIPE|D3D12 */
    uint64_t cso_pipeline_hash;      /* 40..48: Zero-heap Gallium Constant State Object hash */
    uint16_t uma_util_milli;         /* 48..50: Current UMA load (milli) */
    uint16_t occupancy_permil;       /* 50..52: Active HW threads occupancy (0..1000) */
    uint16_t active_stream_procs;    /* 52..54: Allocated FP32 Stream Processors (0..768) */
    uint16_t glthread_batch_cmds;    /* 54..56: Batched u_threaded_context commands */
    uint32_t base_render_fps_x10;    /* 56..60: Native render FPS * 10 */
    uint32_t boosted_output_fps_x10; /* 60..64: Temporal Stream-Synthesized FPS * 10 */
    uint32_t synth_frame_us;         /* 64..68: Shadow-phase frame synthesis latency (us) */
    uint32_t chronometric_shadow_ok; /* 68..72: 1 if Scanout * Compute == 0 */
    uint64_t pipe_screen_digest;     /* 72..80: Attested Kisak-Gallium pipe_screen hash */
    uint8_t  reserved_pad[48];       /* 80..128: Pad to exact 128B */
} KsfwGalliumStreamCuda128B;

uint32_t ksfw_abi_version(void);
int32_t  ksfw_init_dual_banks(KsfwBankState128B* bank_a, KsfwBankState128B* bank_b);
int32_t  ksfw_transactional_firmware_step(
    KsfwBankState128B* active_bank,
    KsfwBankState128B* shadow_bank,
    uint16_t raw_uma_prev_milli,
    uint16_t raw_uma_curr_milli,
    uint16_t raw_uma_next_milli,
    uint16_t temp_deci_c,
    uint8_t  requested_vital_hp,
    const uint64_t* kubc_words,
    uint32_t kubc_word_count
);
int32_t  ksfw_validate_and_expose_generator_aggregator(
    KsfwRetroValidatorReceipt64B* receipt_out,
    const float* raw_input_f32,
    float* sanitized_tiled_out_f32,
    uint32_t element_count,
    uint8_t proposer_id,
    uint8_t verifier_id,
    uint8_t vital_hp,
    uint16_t uma_util_milli,
    uint32_t candidate_gain_permil
);
int32_t  ksfw_dispatch_gallium_stream_cuda_and_fps_boost(
    KsfwGalliumStreamCuda128B* desc,
    uint32_t base_frame_time_us,
    uint32_t motion_vectors_count,
    uint8_t  dwell_stable_frames
);
int32_t  ksfw_apply_vulkan_engine_preset(
    KsfwGalliumStreamCuda128B* desc,
    uint8_t  engine_preset_id,
    uint32_t base_frame_time_us,
    uint16_t uma_util_milli,
    uint8_t  dwell_stable_frames
);
int32_t  ksfw_fedora_nvk_dlss_and_vram_governor(
    KsfwGalliumStreamCuda128B* desc,
    uint8_t  dlss_quality_mode,
    uint32_t total_system_ram_mb,
    uint32_t requested_llm_vram_mb,
    uint16_t uma_util_milli,
    uint32_t* granted_vram_mb_out
);

#ifdef __cplusplus
}
#endif

#endif /* KRYSTAL_SOVEREIGN_FIRMWARE_ABI_H */
