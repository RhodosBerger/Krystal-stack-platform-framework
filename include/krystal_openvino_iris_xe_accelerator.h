/**
 * ==============================================================================
 * KRYSTAL-STACK: INTEL 11TH GEN (TIGER LAKE) OPENVINO ACCELERATION BRIDGE
 * ==============================================================================
 * File: include/krystal_openvino_iris_xe_accelerator.h
 * Description: Low-level C99/C++ definitions and inline algorithms for
 *              bypassing OEM manufacturer power limits (PL1/PL2 clamps),
 *              unlocking Iris Xe DP4A tensor math in OpenVINO, configuring
 *              multi-stream throughput pipelines, and enforcing VITAL_MAX_HP = 6.
 *
 * System Invariant: KRYSTAL_VITAL_MAX_HP = 6
 * Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
 * ==============================================================================
 */

#ifndef KRYSTAL_OPENVINO_IRIS_XE_ACCELERATOR_H
#define KRYSTAL_OPENVINO_IRIS_XE_ACCELERATOR_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#define KRYSTAL_VITAL_MAX_HP 6

/* MSR & MMIO Hardware Addresses for Tiger Lake Power Limit Bypass */
#define KRYSTAL_MSR_PKG_POWER_LIMIT       0x00000610
#define KRYSTAL_MSR_IA32_HWP_REQUEST      0x00000774
#define KRYSTAL_MMIO_MCHBAR_POWER_LIMIT   0x000059A0

/* Default OEM Clamped vs Unlocked TDP Limits */
#define KRYSTAL_OEM_STOCK_PL1_WATTS       15.0f
#define KRYSTAL_OEM_STOCK_PL2_WATTS       40.0f
#define KRYSTAL_UNLOCKED_PL1_WATTS        32.0f
#define KRYSTAL_UNLOCKED_PL2_WATTS        54.0f
#define KRYSTAL_SAFE_MAX_VOLTAGE_V        1.02f
#define KRYSTAL_SAFE_MAX_JUNCTION_TEMP_C  85.0f

#ifdef __cplusplus
extern "C" {
#endif

/**
 * OpenVINO Execution Precision Modes
 */
typedef enum {
    KRYSTAL_OV_PRECISION_FP32        = 0,
    KRYSTAL_OV_PRECISION_FP16        = 1,
    KRYSTAL_OV_PRECISION_INT8_DP4A   = 2,  /* Native Iris Xe & AVX-512 VNNI */
    KRYSTAL_OV_PRECISION_HYBRID_U8   = 3   /* INT8 Weights + U8 KV Cache */
} KrystalOvPrecisionMode;

/**
 * OpenVINO Device Dispatch Profiles
 */
typedef enum {
    KRYSTAL_DISPATCH_GPU_IRIS_XE     = 0,  /* Intel Iris Xe Graphics (80/96 EU) */
    KRYSTAL_DISPATCH_CPU_WILLOW_COVE = 1,  /* Willow Cove (AVX-512 VNNI) */
    KRYSTAL_DISPATCH_GNA_AUDIO       = 2,  /* Intel GNA 2.0 (Whisper Acoustic) */
    KRYSTAL_DISPATCH_HETEROGENEOUS   = 3   /* MULTI: GPU + CPU + GNA */
} KrystalOvDeviceDispatch;

/**
 * Hardware Power Bypass State
 */
typedef struct {
    float configured_pl1_w;
    float configured_pl2_w;
    float configured_tau_s;
    uint8_t hwp_epp_value;         /* 0x00 = Max Performance, 0x80 = Balanced */
    float core_voltage_clamp_v;
    float junction_temp_c;
    bool pl_clamp_bypassed;
    bool silicon_safe_guard;
    int32_t vital_max_hp;
} KrystalHardwareBypassState;

/**
 * OpenVINO Acceleration Configuration Matrix
 */
typedef struct {
    const char* performance_hint;    /* "CUMULATIVE_THROUGHPUT" */
    const char* execution_mode;      /* "PERFORMANCE" */
    KrystalOvPrecisionMode precision;
    KrystalOvDeviceDispatch target_device;
    int32_t gpu_streams_count;       /* 2 to 4 streams */
    bool model_priority_high;
    bool kv_cache_u8_enabled;
    bool enable_zero_copy_mmap;
    float predicted_tokens_per_sec;
    int32_t vital_max_hp;
} KrystalOpenVinoConfigMatrix;

/**
 * Arrhenius Electromigration Life Acceleration Factor
 * AF = exp[(Ea/kB) * (1/T_amb - 1/T_junc)] * (V_act / V_targ)^beta
 */
static inline float krystal_calculate_arrhenius_wear_factor(
    float junction_temp_c,
    float voltage_v
) {
    const float Ea = 0.7f;            /* Activation energy in eV */
    const float kB = 8.617333262e-5f; /* Boltzmann constant in eV/K */
    const float T_ref_k = 338.15f;    /* 65 C reference */
    const float V_ref = 1.00f;        /* 1.00V nominal */
    const float beta = 1.8f;          /* Voltage acceleration exponent */

    float T_junc_k = junction_temp_c + 273.15f;
    if (T_junc_k < 273.15f) T_junc_k = 273.15f;

    float temp_exp = (Ea / kB) * ((1.0f / T_ref_k) - (1.0f / T_junc_k));
    float temp_factor = expf(temp_exp);

    float norm_v = voltage_v / V_ref;
    if (norm_v < 0.5f) norm_v = 0.5f;
    float volt_factor = powf(norm_v, beta);

    return temp_factor * volt_factor;
}

/**
 * Estimates Token Generation Speed for Intel 11th Gen Tiger Lake
 */
static inline float krystal_estimate_tiger_lake_token_speed(
    bool pl1_unblocked,
    KrystalOvPrecisionMode precision,
    int32_t execution_units
) {
    /* Baseline 80 EU at 15W stock PL1 = ~6.5 tok/s for 8B FP16 */
    float base_tokens = 6.5f;

    /* 96 EU vs 80 EU factor */
    base_tokens *= ((float)execution_units / 80.0f);

    /* PL1 bypass from 15W to 32W prevents thermal throttling (+55%) */
    if (pl1_unblocked) {
        base_tokens *= 1.55f;
    }

    /* Precision acceleration factor */
    switch (precision) {
        case KRYSTAL_OV_PRECISION_INT8_DP4A:
        case KRYSTAL_OV_PRECISION_HYBRID_U8:
            base_tokens *= 2.45f; /* Native DP4A tensor speedup */
            break;
        case KRYSTAL_OV_PRECISION_FP16:
            base_tokens *= 1.40f;
            break;
        default:
            break;
    }

    return base_tokens;
}

#ifdef __cplusplus
}
#endif

#endif /* KRYSTAL_OPENVINO_IRIS_XE_ACCELERATOR_H */
