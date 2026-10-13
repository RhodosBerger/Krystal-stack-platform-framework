/* ============================================================================
 * KRYSTAL-STACK: CANONICAL 16-BIT STATUS ABI HEADER
 * Generated automatically from krystal_kernel.verification.semantic_protocol
 * Constitutional Invariant: VITAL_MAX_HP == 6
 * ============================================================================ */

#ifndef KRYSTAL_STATUS_ABI_H
#define KRYSTAL_STATUS_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint16_t krystal_status_t;

#define KRYSTAL_VITAL_MAX_HP 6

/* Status Code Family Masks */
#define KRYSTAL_FAMILY_MASK   0xF000
#define KRYSTAL_SUBKIND_MASK  0x0F00
#define KRYSTAL_VALUE_MASK    0x00FF

#define KRYSTAL_GET_FAMILY(status)  (((status) & KRYSTAL_FAMILY_MASK) >> 12)
#define KRYSTAL_GET_SUBKIND(status) (((status) & KRYSTAL_SUBKIND_MASK) >> 8)
#define KRYSTAL_GET_VALUE(status)   ((status) & KRYSTAL_VALUE_MASK)

/* Status Code Constants */

/* 0x1--- : Epistemic Evidence & Truth Lattice (What the system knows). */
#define KRYSTAL_EPI_UNKNOWN                  0x1001
#define KRYSTAL_EPI_OBSERVED                 0x1002
#define KRYSTAL_EPI_DERIVED                  0x1003
#define KRYSTAL_EPI_PREDICTED                0x1004
#define KRYSTAL_EPI_CALIBRATED               0x1005
#define KRYSTAL_EPI_VALIDATED                0x1006
#define KRYSTAL_EPI_FALSIFIED                0x1007
#define KRYSTAL_EPI_CONTRADICTORY            0x1008
#define KRYSTAL_EPI_QUARANTINED              0x1009

/* 0x2--- : Temporal Lease & Freshness Dynamics (How long it is valid). */
#define KRYSTAL_TMP_FRESH                    0x2001
#define KRYSTAL_TMP_LEASED                   0x2002
#define KRYSTAL_TMP_STALE                    0x2003
#define KRYSTAL_TMP_EXPIRED                  0x2004
#define KRYSTAL_TMP_MONOTONIC_LATCH          0x2005
#define KRYSTAL_TMP_BISTABLE_HYSTERETIC      0x2006
#define KRYSTAL_TMP_SHADOW_HYPOTHETICAL      0x2007

/* 0x3--- : Authoritative Governance Verdicts (What authority permits). */
#define KRYSTAL_AUTH_COMMIT_APPROVED         0x3001
#define KRYSTAL_AUTH_DEFERRED                0x3002
#define KRYSTAL_AUTH_QUARANTINED             0x3003
#define KRYSTAL_AUTH_VETOED                  0x3004
#define KRYSTAL_AUTH_SHADOW_ONLY             0x3005

/* 0x41-- : Binary Safety Filter (Sub-family under Invariant/Safety). */
#define KRYSTAL_SAFE_PASS                    0x4101
#define KRYSTAL_SAFE_BREACH                  0x4102

/* 0x40-- : Constitutional Invariants & Vitality (VITAL_MAX_HP == 6). */
#define KRYSTAL_INV_DEPLETED_ZERO            0x4000
#define KRYSTAL_INV_CRITICAL_LOCK            0x4001
#define KRYSTAL_INV_DEGRADED_WARN            0x4003
#define KRYSTAL_INV_NOMINAL_6HP              0x4006
#define KRYSTAL_INV_BREACH_TRIPPED           0x400E

/* 0x5--- : Actuation & Physical Execution Plane. */
#define KRYSTAL_ACT_IDLE                     0x5001
#define KRYSTAL_ACT_DISPATCHED               0x5002
#define KRYSTAL_ACT_EXECUTED                 0x5003
#define KRYSTAL_ACT_REVERTED                 0x5004

/* 0x6--- : Transactional Lifecycle (2-Phase Commit & Rollback). */
#define KRYSTAL_TXN_IDLE                     0x6001
#define KRYSTAL_TXN_PREPARED                 0x6002
#define KRYSTAL_TXN_VERIFIED                 0x6003
#define KRYSTAL_TXN_COMMITTED                0x6004
#define KRYSTAL_TXN_ABORTED                  0x6005
#define KRYSTAL_TXN_COMPENSATING             0x6006
#define KRYSTAL_TXN_ROLLED_BACK              0x6007

/* 0x7--- : Physical Raw Silicon Observations. */
#define KRYSTAL_OBS_RAW_SAMPLE               0x7001
#define KRYSTAL_OBS_CALIBRATED_SAMPLE        0x7002
#define KRYSTAL_OBS_DRIFT_DETECTED           0x7003

/* 0x8--- : Mathematically Derived Signals. */
#define KRYSTAL_DER_COMPUTED_DELTA           0x8001
#define KRYSTAL_DER_PRESSURE_SCORE           0x8002
#define KRYSTAL_DER_THERMAL_HEADROOM         0x8003
#define KRYSTAL_DER_INTERFERENCE_INDEX       0x8004

/* 0x9--- : Probabilistic Model Forecasts. */
#define KRYSTAL_PRED_POINT_ESTIMATE          0x9001
#define KRYSTAL_PRED_CONFIDENCE_BOUND        0x9002
#define KRYSTAL_PRED_BREACH_WARNING          0x9003
#define KRYSTAL_PRED_COUNTERFACTUAL_DELTA    0x9004

/* 0xA--- : Native Memory Fabric & Lifetimes. */
#define KRYSTAL_MEM_ARENA_ALLOC              0xA001
#define KRYSTAL_MEM_ZERO_COPY_VIEW           0xA002
#define KRYSTAL_MEM_QUARANTINED              0xA003

/* 0xB--- : Physical Silicon Hardware Execution Status. */
#define KRYSTAL_HW_NOMINAL                   0xB001
#define KRYSTAL_HW_THROTTLED_THERMAL         0xB002
#define KRYSTAL_HW_RAPL_CLAMPED              0xB003

/* 0xC--- : Systemic Paradoxes & Pathological Resonance States. */
#define KRYSTAL_PDX_DORMANT                  0xC001
#define KRYSTAL_PDX_RESONATING               0xC002
#define KRYSTAL_PDX_QUARANTINED              0xC003
#define KRYSTAL_PDX_AUTO_HEALED              0xC004

/* 0xD--- : Merkle Hash Trees & Cryptographic DAG Integrity. */
#define KRYSTAL_MRK_ROOT_VALID               0xD001
#define KRYSTAL_MRK_LEAF_STALE               0xD002
#define KRYSTAL_MRK_HASH_MISMATCH            0xD003

/* 0xE--- : Dynamic Contextual Policies. */
#define KRYSTAL_POL_ACTIVE                   0xE001
#define KRYSTAL_POL_CALIBRATING              0xE002
#define KRYSTAL_POL_SUPERSEDED               0xE003
#define KRYSTAL_POL_REJECTED                 0xE004
#define KRYSTAL_POL_EXPERIMENTAL             0xE005

#ifdef __cplusplus
}
#endif

#endif /* KRYSTAL_STATUS_ABI_H */
