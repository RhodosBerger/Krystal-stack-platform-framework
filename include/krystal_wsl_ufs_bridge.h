/**
 * ==============================================================================
 * KRYSTAL-STACK: WSL2 COPROCESSOR & UFS LOG TRIAGE C/C++ BRIDGE
 * ==============================================================================
 * File: include/krystal_wsl_ufs_bridge.h
 * Description: Low-level C/C++ definitions and inline utility functions
 *              bridging Windows NTSTATUS signals with Linux/WSL2 POSIX signals,
 *              UFS (Unix File System) inode health checking, and automated
 *              sysadmin coprocessor triage.
 *
 * System Invariant: KRYSTAL_VITAL_MAX_HP = 6
 * Author: Dušan Kopecký & Krystal Architecture Council (2026)
 * ==============================================================================
 */

#ifndef KRYSTAL_WSL_UFS_BRIDGE_H
#define KRYSTAL_WSL_UFS_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#define KRYSTAL_VITAL_MAX_HP 6

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Categorized Process Failure Signatures
 */
typedef enum {
    KRYSTAL_FAIL_HEALTHY_NOMINAL        = 0,
    KRYSTAL_FAIL_SEGFAULT_ACCESS_VIOL   = 1, /* 0xC0000005 / SIGSEGV */
    KRYSTAL_FAIL_OOM_EXHAUSTION         = 2, /* 0xC0000017 / ENOMEM */
    KRYSTAL_FAIL_UNINTERRUPTIBLE_IO_HANG= 3, /* D-state / ETIMEDOUT */
    KRYSTAL_FAIL_ZOMBIE_LEAK            = 4, /* Z-state / ECHILD */
    KRYSTAL_FAIL_THERMAL_PROXIMITY      = 5, /* Temp > 95C / Trip risk */
    KRYSTAL_FAIL_MUTEX_DEADLOCK         = 6  /* 0xC0000194 / EDEADLK */
} KrystalWslFailureType;

/**
 * Automated System Administrator Remediation Actions
 */
typedef enum {
    KRYSTAL_ADMIN_CONTINUE_STEADY       = 0,
    KRYSTAL_ADMIN_QUARANTINE_AND_DUMP   = 1,
    KRYSTAL_ADMIN_ESCALATE_UMA_512MB    = 2,
    KRYSTAL_ADMIN_RECLAIM_INODES        = 3,
    KRYSTAL_ADMIN_RESET_RING_BUFFER     = 4,
    KRYSTAL_ADMIN_REAP_ZOMBIES_CLEAN_IPC= 5,
    KRYSTAL_ADMIN_ASAHI_BALANCED_VOLT   = 6
} KrystalAdminRemediationAction;

/**
 * Process Diagnostic Descriptor
 */
typedef struct {
    uint32_t target_pid;
    char process_name[64];
    uint32_t raw_status_code;           /* e.g., 0xC0000005 */
    KrystalWslFailureType failure_type;
    uint32_t confidence_pct;
    KrystalAdminRemediationAction prescribed_action;
    bool human_admin_required;
    uint32_t vital_max_hp;              /* Must be 6 */
} KrystalWslProcessDiagnostic;

/**
 * UFS (Unix File System) Volume Health Descriptor
 */
typedef struct {
    uint64_t total_blocks;
    uint64_t free_blocks;
    uint32_t total_inodes;
    uint32_t used_inodes;
    uint32_t stale_sockets_purged;
    uint32_t block_corruption_score_pct;
    bool journal_synchronized;
    uint32_t vital_max_hp;
} KrystalUfsVolumeHealth;

/**
 * Translates Win32 NTSTATUS code to POSIX/Linux Signal & Failure Type
 */
static inline KrystalWslFailureType krystal_translate_ntstatus_to_failure(uint32_t ntstatus) {
    switch (ntstatus) {
        case 0xC0000005: /* STATUS_ACCESS_VIOLATION */
            return KRYSTAL_FAIL_SEGFAULT_ACCESS_VIOL;
        case 0xC0000017: /* STATUS_NO_MEMORY */
            return KRYSTAL_FAIL_OOM_EXHAUSTION;
        case 0xC0000194: /* STATUS_POSSIBLE_DEADLOCK */
            return KRYSTAL_FAIL_MUTEX_DEADLOCK;
        case 0xC00000B5: /* STATUS_IO_TIMEOUT */
            return KRYSTAL_FAIL_UNINTERRUPTIBLE_IO_HANG;
        case 0x00000000: /* STATUS_SUCCESS */
        default:
            return KRYSTAL_FAIL_HEALTHY_NOMINAL;
    }
}

/**
 * Maps failure type to automated administrative remediation action
 */
static inline KrystalAdminRemediationAction krystal_resolve_admin_remediation(KrystalWslFailureType fail_type) {
    switch (fail_type) {
        case KRYSTAL_FAIL_SEGFAULT_ACCESS_VIOL:
            return KRYSTAL_ADMIN_QUARANTINE_AND_DUMP;
        case KRYSTAL_FAIL_OOM_EXHAUSTION:
            return KRYSTAL_ADMIN_ESCALATE_UMA_512MB;
        case KRYSTAL_FAIL_UNINTERRUPTIBLE_IO_HANG:
        case KRYSTAL_FAIL_MUTEX_DEADLOCK:
            return KRYSTAL_ADMIN_RESET_RING_BUFFER;
        case KRYSTAL_FAIL_ZOMBIE_LEAK:
            return KRYSTAL_ADMIN_REAP_ZOMBIES_CLEAN_IPC;
        case KRYSTAL_FAIL_THERMAL_PROXIMITY:
            return KRYSTAL_ADMIN_ASAHI_BALANCED_VOLT;
        case KRYSTAL_FAIL_HEALTHY_NOMINAL:
        default:
            return KRYSTAL_ADMIN_CONTINUE_STEADY;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* KRYSTAL_WSL_UFS_BRIDGE_H */
