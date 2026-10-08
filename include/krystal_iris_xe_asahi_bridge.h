// ==============================================================================
// KRYSTAL-STACK: INTEL IRIS XE & ASAHI POWER GOVERNOR BRIDGE
// File: include/krystal_iris_xe_asahi_bridge.h
// Description: Cross-platform C/C++ shared definitions for Intel Iris Xe (96 EUs)
//              Unified Memory Architecture (UMA) quotients and Asahi Linux DVFS.
// System Invariant: VITAL_MAX_HP = 6
// ==============================================================================

#ifndef KRYSTAL_IRIS_XE_ASAHI_BRIDGE_H
#define KRYSTAL_IRIS_XE_ASAHI_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KRYSTAL_VITAL_MAX_HP 6

/* ========================================================================== */
/* Asahi Linux Inspired Power States & Thermal Limits                         */
/* ========================================================================== */
#define KRYSTAL_THERMAL_SAFE_TARGET_C     78.0
#define KRYSTAL_THERMAL_FUSE_REGULATE_C   95.0
#define KRYSTAL_THERMAL_HARD_TRIP_C      100.0

typedef enum KrystalAsahiPState {
    KRYSTAL_ASAHI_P0_QUIESCENT    = 0, /* 0.70V, 800 MHz CPU, 300 MHz GPU */
    KRYSTAL_ASAHI_P1_EFFICIENCY   = 1, /* 0.80V, 1600 MHz CPU, 600 MHz GPU */
    KRYSTAL_ASAHI_P2_BALANCED     = 2, /* 0.92V, 2800 MHz CPU, 1000 MHz GPU */
    KRYSTAL_ASAHI_P3_BURST_ACCEL  = 3, /* 1.12V, 4200 MHz CPU, 1350 MHz GPU (Iris Xe peak) */
    KRYSTAL_ASAHI_P4_THERMAL_GUARD= 4  /* 0.88V, stable floor without frequency cliff */
} KrystalAsahiPState;

/* ========================================================================== */
/* Intel Iris Xe Unified Memory Architecture (UMA) Quotients                  */
/* ========================================================================== */
typedef enum KrystalUmaQuotientStep {
    KRYSTAL_UMA_Q32_MB  = 32,
    KRYSTAL_UMA_Q64_MB  = 64,
    KRYSTAL_UMA_Q128_MB = 128,
    KRYSTAL_UMA_Q256_MB = 256,
    KRYSTAL_UMA_Q512_MB = 512
} KrystalUmaQuotientStep;

/* ========================================================================== */
/* Unified Memory Buffer Descriptor                                           */
/* ========================================================================== */
typedef struct KrystalUnifiedBufferDescriptor {
    uint64_t physical_address;
    void*    host_virtual_ptr;
    uint64_t size_bytes;
    uint32_t quotient_tier_mb;
    bool     is_host_coherent;
    bool     is_zero_copy;
    uint32_t vital_max_hp; /* Always 6 */
} KrystalUnifiedBufferDescriptor;

/* ========================================================================== */
/* VSync Frame Pacing & Self-Healing Telemetry Status                         */
/* ========================================================================== */
typedef struct KrystalPacingStatus {
    uint64_t           frame_index;
    double             frame_time_ms;
    double             vsync_budget_ms; /* 8.333ms for 120Hz, 16.667ms for 60Hz */
    bool               vsync_locked;
    double             thermal_junction_temp_c;
    double             thermal_fuse_headroom_c;
    KrystalAsahiPState active_p_state;
    uint32_t           active_uma_quotient_mb;
    bool               transient_grace_active;
    double             grace_window_remaining_sec;
    uint32_t           vital_max_hp; /* Always 6 */
} KrystalPacingStatus;

/* ========================================================================== */
/* Inline Mathematical Helper Functions                                       */
/* ========================================================================== */

/**
 * Calculates whether the thermal fuse is preserved while allowing maximum HW acceleration.
 */
static inline bool krystal_is_thermal_fuse_safe(double junction_temp_c) {
    return (junction_temp_c < KRYSTAL_THERMAL_HARD_TRIP_C);
}

/**
 * Resolves the next UMA memory quotient based on frame pacing slack.
 */
static inline uint32_t krystal_resolve_uma_quotient(double frame_time_ms, double vsync_budget_ms) {
    if (frame_time_ms > (vsync_budget_ms * 0.88)) {
        return KRYSTAL_UMA_Q256_MB; // Boost to 256MB to widen bandwidth
    } else if (frame_time_ms > vsync_budget_ms) {
        return KRYSTAL_UMA_Q512_MB; // Max 512MB unified tier
    } else if (frame_time_ms < (vsync_budget_ms * 0.50)) {
        return KRYSTAL_UMA_Q64_MB;  // Energy saving tier
    }
    return KRYSTAL_UMA_Q128_MB;     // Nominal 128MB tier
}

#ifdef __cplusplus
}
#endif

#endif /* KRYSTAL_IRIS_XE_ASAHI_BRIDGE_H */
