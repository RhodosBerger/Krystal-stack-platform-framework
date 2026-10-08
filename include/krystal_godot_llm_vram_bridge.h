/**
 * ==============================================================================
 * KRYSTAL-STACK: GODOT, VRAM UNLOCKER & SPECULATIVE K-ISA C/C++ BRIDGE
 * ==============================================================================
 * File: include/krystal_godot_llm_vram_bridge.h
 * Description: Low-level C99/C++ definitions and inline math models for
 *              Windows VRAM Aperture Unlocking, Local Quantized LLM Inference,
 *              K-ISA Speculative Instruction Set, and Godot 4.x Viewport Pacing.
 *
 * System Invariant: KRYSTAL_VITAL_MAX_HP = 6
 * Author: Dušan Kopecký & Krystal Architecture Council (2026)
 * ==============================================================================
 */

#ifndef KRYSTAL_GODOT_LLM_VRAM_BRIDGE_H
#define KRYSTAL_GODOT_LLM_VRAM_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#define KRYSTAL_VITAL_MAX_HP 6

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Unlocked VRAM Aperture Tiers on Intel Iris Xe
 */
typedef enum {
    KRYSTAL_VRAM_TIER_CLAMPED_128MB = 0,
    KRYSTAL_VRAM_TIER_UNLOCKED_2GB   = 1,
    KRYSTAL_VRAM_TIER_UNLOCKED_4GB   = 2,
    KRYSTAL_VRAM_TIER_UNLOCKED_8GB   = 3
} KrystalVramApertureTier;

/**
 * Supported Quantization Primitives
 */
typedef enum {
    KRYSTAL_QUANT_INT4_GGUF_AWQ     = 0,
    KRYSTAL_QUANT_INT8_VNNI_DP4A    = 1,
    KRYSTAL_QUANT_FP16_HALF         = 2,
    KRYSTAL_QUANT_FP32_NATIVE       = 3
} KrystalQuantizationFormat;

/**
 * K-ISA Speculative Micro-Opcodes
 */
typedef enum {
    KRYSTAL_ISA_SPEC_PREFETCH_UMA    = 0xA1,
    KRYSTAL_ISA_SPEC_INTERP_FRAME   = 0xA2,
    KRYSTAL_ISA_SAFE_VOLT_CLAMP     = 0xA3,
    KRYSTAL_ISA_FALLBACK_REVERT     = 0xA4,
    KRYSTAL_ISA_FUSE_INT8_DP4A      = 0xA5,
    KRYSTAL_ISA_VERIFY_INVARIANT_HP = 0xA6
} KrystalKIsaOpcode;

/**
 * Local LLM Token Benchmark Descriptor
 */
typedef struct {
    KrystalVramApertureTier aperture_tier;
    KrystalQuantizationFormat quantization;
    float allocated_vram_gb;
    float effective_bandwidth_gbps;
    float tokens_per_second;
    float time_to_first_token_ms;
    float token_speedup_vs_clamped;
    uint32_t ring_bus_pcie_stalls_per_sec;
    float junction_temp_c;
    float voltage_core_v;
    float projected_lifespan_years;
    uint32_t vital_max_hp;
} KrystalLlmTokenBenchmarkDescriptor;

/**
 * Optimal Process Plan Descriptor
 */
typedef struct {
    float target_tokens_sec;
    KrystalVramApertureTier recommended_aperture;
    KrystalQuantizationFormat recommended_quant;
    float balanced_vcore_v;
    float balanced_vgt_v;
    uint32_t target_frequency_mhz;
    float thermal_envelope_c;
    float chip_lifespan_index;
    bool safeguard_active;
    uint32_t vital_max_hp;
} KrystalOptimalPlanDescriptor;

/**
 * Arrhenius Electromigration Chip Lifespan Estimator (JEDEC JESD85)
 * Calibrated for Intel 10nm SuperFin (Cobalt interconnects, Ea = 0.5 eV, nominal 75°C @ 1.05V)
 */
static inline float krystal_calculate_arrhenius_lifespan(float voltage_v, float temp_c) {
    const float nominal_volts = 1.05f;
    const float nominal_temp_k = 348.15f; // 75°C in Kelvin
    float current_temp_k = temp_c + 273.15f;

    float v_clamp = voltage_v > 0.6f ? voltage_v : 0.6f;
    float v_stress = (nominal_volts / v_clamp) * (nominal_volts / v_clamp);
    float t_accel = expf((0.5f / 8.617e-5f) * ((1.0f / current_temp_k) - (1.0f / nominal_temp_k)));

    float projected = 10.0f * v_stress * t_accel;
    if (projected < 1.5f) return 1.5f;
    if (projected > 25.0f) return 25.0f;
    return projected;
}

/**
 * Resolves Token Speedup vs Clamped 128MB Aperture
 */
static inline float krystal_resolve_token_speedup(KrystalVramApertureTier tier) {
    switch (tier) {
        case KRYSTAL_VRAM_TIER_UNLOCKED_8GB: return 5.79f;
        case KRYSTAL_VRAM_TIER_UNLOCKED_4GB: return 5.33f;
        case KRYSTAL_VRAM_TIER_UNLOCKED_2GB: return 3.71f;
        case KRYSTAL_VRAM_TIER_CLAMPED_128MB:
        default: return 1.0f;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* KRYSTAL_GODOT_LLM_VRAM_BRIDGE_H */
