/**
 * ==============================================================================
 * KRYSTAL-STACK: BYTECODE POWER ALTERNATION & NEURAL SUPER-SAMPLING C/C++ BRIDGE
 * ==============================================================================
 * File: include/krystal_bytecode_nss_bridge.h
 * Description: Low-level C99/C++ definitions, structs, and inline math helpers
 *              for Bytecode Dependency Tracking, Phase-Staggered Current Pacing,
 *              and K-NSS Open Neural Super-Sampling on Intel 11th Gen Iris Xe.
 *
 * System Invariant: KRYSTAL_VITAL_MAX_HP = 6
 * Author: Dušan Kopecký & Krystal Architecture Council (2026)
 * ==============================================================================
 */

#ifndef KRYSTAL_BYTECODE_NSS_BRIDGE_H
#define KRYSTAL_BYTECODE_NSS_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>

#define KRYSTAL_VITAL_MAX_HP 6

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Physical Silicon Domains on Intel 11th Gen Tiger Lake
 */
typedef enum {
    KRYSTAL_DOMAIN_CPU_WILLOW_COVE  = 0,
    KRYSTAL_DOMAIN_CPU_AVX512_VNNI  = 1,
    KRYSTAL_DOMAIN_GPU_IRIS_XE_EUS  = 2,
    KRYSTAL_DOMAIN_GPU_DP4A_TENSOR  = 3,
    KRYSTAL_DOMAIN_UMA_MEMORY_FABRIC= 4,
    KRYSTAL_DOMAIN_TEXTURE_SAMPLER  = 5
} KrystalSiliconDomain;

/**
 * Bytecode Opcode Primitives
 */
typedef enum {
    KRYSTAL_OP_LOAD_UMA_TENSOR      = 0x10,
    KRYSTAL_OP_CPU_VECTOR_FILTER    = 0x20,
    KRYSTAL_OP_GPU_RAYMARCH_SDF     = 0x30,
    KRYSTAL_OP_GPU_DP4A_SUPERRES    = 0x40,
    KRYSTAL_OP_APPLY_AFFINE_XFORM   = 0x50,
    KRYSTAL_OP_VSYNC_TIME_BARRIER   = 0x60,
    KRYSTAL_OP_PRIORITY_INTR        = 0x70
} KrystalBytecodeOpcode;

/**
 * Open-Source K-NSS Quality Profiles
 */
typedef enum {
    KRYSTAL_NSS_ULTRA_PERFORMANCE   = 0, /* 3.0x scale */
    KRYSTAL_NSS_PERFORMANCE         = 1, /* 2.0x scale (e.g. 540p -> 1080p) */
    KRYSTAL_NSS_BALANCED            = 2, /* 1.7x scale */
    KRYSTAL_NSS_QUALITY             = 3, /* 1.5x scale */
    KRYSTAL_NSS_ULTRA_QUALITY       = 4  /* 1.3x scale */
} KrystalNssQualityProfile;

/**
 * Bytecode Instruction Descriptor
 */
typedef struct {
    uint32_t instruction_id;
    KrystalBytecodeOpcode opcode;
    KrystalSiliconDomain domain;
    float nominal_current_amperes;
    uint32_t duration_cycles;
    uint32_t priority_level; /* 1=Normal, 2=Elevated, 3=Critical Immediate */
} KrystalBytecodeInstruction;

/**
 * Micro-Sliced Step Execution Descriptor
 */
typedef struct {
    uint32_t slice_index;
    float total_current_amperes;
    float voltage_droop_risk_pct;
    float time_budget_us;
    bool priority_action_injected;
    uint32_t vital_max_hp; /* 6 */
} KrystalMicroSliceDescriptor;

/**
 * K-NSS Resolution Scaling Descriptor
 */
typedef struct {
    uint32_t render_width;
    uint32_t render_height;
    uint32_t target_width;
    uint32_t target_height;
    float scale_factor;
    float pixels_saved_pct;
} KrystalNssResolutionDescriptor;

/**
 * K-NSS Benchmark & Execution Result
 */
typedef struct {
    KrystalNssQualityProfile profile;
    float native_frame_time_ms;
    float knss_frame_time_ms;
    float effective_fps_native;
    float effective_fps_knss;
    float speedup_multiplier;
    float latency_saved_ms;
    uint64_t dp4a_tensor_cycles;
    uint32_t vital_max_hp;
} KrystalNssBenchmarkResult;

/**
 * Calculates scale factor for a given quality profile
 */
static inline float krystal_nss_get_scale_factor(KrystalNssQualityProfile profile) {
    switch (profile) {
        case KRYSTAL_NSS_ULTRA_PERFORMANCE: return 3.0f;
        case KRYSTAL_NSS_PERFORMANCE:       return 2.0f;
        case KRYSTAL_NSS_BALANCED:          return 1.7f;
        case KRYSTAL_NSS_QUALITY:           return 1.5f;
        case KRYSTAL_NSS_ULTRA_QUALITY:     return 1.3f;
        default:                            return 2.0f;
    }
}

/**
 * Checks if current alternation prevents voltage droop
 */
static inline bool krystal_is_current_within_vrm_safe_envelope(float peak_current_a) {
    return (peak_current_a <= 38.0f);
}

#ifdef __cplusplus
}
#endif

#endif /* KRYSTAL_BYTECODE_NSS_BRIDGE_H */
