// ==============================================================================
// KRYSTAL-STACK: UNIVERSAL KERNEL TELEMETRY & SCHEDULER ABSTRACTIONS
// File: include/krystal_kernel_telemetry_hook.h
// Description: Cross-platform C/C++ universal definitions of kernel process
//              abstractions, scheduler states, and context-switch telemetry.
// Targets: Windows NT Kernel, Linux (CFS/EEVDF), Android Linux, Bare-metal
// System Invariant: VITAL_MAX_HP = 6
// ==============================================================================

#ifndef KRYSTAL_KERNEL_TELEMETRY_HOOK_H
#define KRYSTAL_KERNEL_TELEMETRY_HOOK_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KRYSTAL_VITAL_MAX_HP 6

/* ========================================================================== */
/* Universal Process and Thread Execution States                              */
/* ========================================================================== */
typedef enum KrystalThreadState {
    KRYSTAL_THREAD_NEW        = 0,
    KRYSTAL_THREAD_READY      = 1,
    KRYSTAL_THREAD_RUNNING    = 2,
    KRYSTAL_THREAD_BLOCKED    = 3,
    KRYSTAL_THREAD_THRASHING  = 4,  /* Pathological state: Rapid involuntary preemption */
    KRYSTAL_THREAD_TERMINATED = 5
} KrystalThreadState;

typedef enum KrystalIntegrityStatus {
    KRYSTAL_STATUS_OPTIMAL               = 0, /* CS rate within baseline bounds */
    KRYSTAL_STATUS_NOMINAL               = 1, /* Mild thread migration */
    KRYSTAL_STATUS_THRASHING_WARNING     = 2, /* Context switches > 2.2x baseline */
    KRYSTAL_STATUS_CRITICAL_INTERFERENCE = 3  /* Severe context switch storm (>3.5x) */
} KrystalIntegrityStatus;

/* ========================================================================== */
/* Universal Kernel Thread Control Block (TCB)                                */
/* ========================================================================== */
typedef struct KrystalThreadControlBlock {
    uint32_t           thread_id;
    uint32_t           process_id;
    KrystalThreadState state;
    uint32_t           priority_class;
    uint64_t           assigned_cpu_affinity_mask;
    uint64_t           total_context_switches;
    uint64_t           involuntary_preemptions;
    uint64_t           cpu_time_user_ns;
    uint64_t           cpu_time_kernel_ns;
    uint64_t           last_quantum_slice_ns;
    double             thrashing_penalty_factor;
} KrystalThreadControlBlock;

/* ========================================================================== */
/* Universal Kernel Telemetry Snapshot                                        */
/* ========================================================================== */
typedef struct KrystalKernelTelemetrySnapshot {
    uint64_t               timestamp_ns;
    double                 cpu_utilization_pct;
    double                 context_switches_per_sec;
    double                 system_calls_per_sec;
    double                 cs_to_syscall_ratio;
    double                 expected_baseline_cs;
    double                 thrashing_index;
    double                 integrity_score;     /* 1.0 (Optimal) to 0.0 (Collapse) */
    KrystalIntegrityStatus status;
    bool                   anomaly_detected;
    uint32_t               vital_max_hp;        /* Invariant: 6 */
} KrystalKernelTelemetrySnapshot;

/* ========================================================================== */
/* Logical Kernel Operations & Algorithms                                     */
/* ========================================================================== */

/**
 * Calculates the Thread Thrashing Index (T_thrash).
 * Formula: T_thrash = (CS_actual / CS_baseline) * (CPU_Load / 50.0)
 */
static inline double krystal_compute_thrashing_index(
    double actual_cs_rate,
    double baseline_cs_rate,
    double cpu_utilization_pct
) {
    if (baseline_cs_rate <= 0.0) baseline_cs_rate = 6500.0;
    double load_weight = (cpu_utilization_pct <= 0.0) ? 0.2 : (cpu_utilization_pct / 50.0);
    return (actual_cs_rate / baseline_cs_rate) * load_weight;
}

/**
 * Computes Processor Integrity Score from Thrashing Index.
 */
static inline double krystal_compute_integrity_score(double thrashing_index) {
    if (thrashing_index <= 1.2) {
        return 1.0;
    } else if (thrashing_index <= 2.2) {
        double score = 1.0 - (thrashing_index - 1.2) * 0.25;
        return (score < 0.70) ? 0.70 : score;
    } else if (thrashing_index <= 3.5) {
        double score = 0.70 - (thrashing_index - 2.2) * 0.23;
        return (score < 0.40) ? 0.40 : score;
    } else {
        double score = 0.40 - (thrashing_index - 3.5) * 0.10;
        return (score < 0.05) ? 0.05 : score;
    }
}

/**
 * Calculates dynamically adjusted time quantum for thread execution to damp thrashing.
 * Q_adj = Q_base * (1.0 / (1.0 + T_thrash^1.5))
 */
static inline uint64_t krystal_adjust_time_quantum(
    uint64_t base_quantum_ns,
    double thrashing_index
) {
    if (thrashing_index <= 1.2) return base_quantum_ns;
    double damping = 1.0 / (1.0 + (thrashing_index - 1.2) * 1.5);
    uint64_t adjusted = (uint64_t)(base_quantum_ns * damping);
    return (adjusted < 100000ULL) ? 100000ULL : adjusted; /* Minimum 100 microseconds */
}

/* ========================================================================== */
/* Platform-Specific Telemetry Capture Hooks                                  */
/* ========================================================================== */

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

/**
 * Reads exact total context switches from Windows NT Executive via ntdll.
 */
static inline uint64_t krystal_win32_query_context_switches(void) {
    typedef LONG(NTAPI* PFN_NtQuerySystemInformation)(ULONG, PVOID, ULONG, PULONG);
    HMODULE hNtdll = GetModuleHandleA("ntdll.dll");
    if (!hNtdll) return 0;

    PFN_NtQuerySystemInformation pfn = 
        (PFN_NtQuerySystemInformation)GetProcAddress(hNtdll, "NtQuerySystemInformation");
    if (!pfn) return 0;

    uint8_t buffer[512] = {0};
    ULONG ret_len = 0;
    // SystemPerformanceInformation = 2
    if (pfn(2, buffer, sizeof(buffer), &ret_len) == 0) {
        // Offset of ContextSwitches in SYSTEM_PERFORMANCE_INFORMATION is 288 on x64
        uint32_t* cs_ptr = (uint32_t*)(buffer + 288);
        return (uint64_t)(*cs_ptr);
    }
    return 0;
}
#endif

#ifdef __cplusplus
}
#endif

#endif /* KRYSTAL_KERNEL_TELEMETRY_HOOK_H */
