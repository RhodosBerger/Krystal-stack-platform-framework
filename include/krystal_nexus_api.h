/*
 * KRYSTAL-NEXUS COEXISTENCE C-ABI INTERFACE
 * ============================================================================
 * Zero-copy shared memory frame protocol for external graphics & AI engines.
 * Milestone: V-ZEROCOPY-MEMORY-MODEL-02 (Cross-ISA Publication Semantics)
 * Platform invariant: KRYSTAL_VITAL_MAX_HP == 6
 */

#ifndef KRYSTAL_NEXUS_API_H
#define KRYSTAL_NEXUS_API_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#include <stdatomic.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define KRYSTAL_MAGIC             0x4B525953  /* "KRYS" */
#define KRYSTAL_PROTOCOL_VERSION  2
#define KRYSTAL_FRAME_SIZE_BYTES  128
#define KRYSTAL_VITAL_MAX_HP      6
#define KRYSTAL_RING_HEADER_BYTES 128

/*
 * 128-byte cache-line aligned zero-copy contract frame.
 * Occupies exactly two 64-byte CPU cache lines (CL0 = 0..63, CL1 = 64..127).
 * Natural scalar alignment ensures zero padding across x86_64, ARM64, and RISC-V.
 */
typedef struct _Alignas(64) {
    /* CL0: 0..63 (Naturally aligned scalars) */
    uint32_t magic;                /* 0..3   */
    uint32_t version;              /* 4..7   */
    uint16_t vital_hp;             /* 8..9   (Invariant == 6) */
    uint16_t contract_type;        /* 10..11 */
    uint32_t payload_size;         /* 12..15 */
    uint32_t flags;                /* 16..19 */
    uint32_t sequence;             /* 20..23 */
    uint32_t generation;           /* 24..27 */
    uint32_t cap_scope;            /* 28..31 */
    uint64_t timestamp_ns;         /* 32..39 (8B aligned at 32 % 8 == 0) */
    uint64_t nonce;                /* 40..47 (8B aligned at 40 % 8 == 0) */
    uint8_t  predecessor_hash[32]; /* 48..79 (Spans CL0 [48..63] & CL1 [64..79]) */

    /* CL1: 64..127 */
    uint16_t profile_id;           /* 80..81 */
    uint16_t verdict_id;           /* 82..83 */
    uint32_t mask;                 /* 84..87 */
    float    gpu_render_ms;        /* 88..91 */
    float    drs_scale_factor;     /* 92..95 */
    uint8_t  frame_hmac[32];       /* 96..127 (HMAC-SHA256 over bytes 0..95) */
} KrystalZeroCopyFrame;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(KrystalZeroCopyFrame) == 128, "Frame size must be exactly 128B (2 cache lines)");
_Static_assert(_Alignof(KrystalZeroCopyFrame) == 64, "Frame must be 64B cache-line aligned");
_Static_assert(offsetof(KrystalZeroCopyFrame, timestamp_ns) == 32, "timestamp_ns offset drift");
_Static_assert(offsetof(KrystalZeroCopyFrame, nonce) == 40, "nonce offset drift");
_Static_assert(offsetof(KrystalZeroCopyFrame, profile_id) == 80, "CL1 payload offset drift");
_Static_assert(offsetof(KrystalZeroCopyFrame, frame_hmac) == 96, "HMAC digest offset drift");
#endif

/*
 * 128-byte cache-line isolated SPSC ring buffer header.
 * Eliminates false-sharing: Producer exclusively writes to CL0, Consumer exclusively writes to CL1.
 */
typedef struct _Alignas(64) {
    /* CL0: Producer cache line (bytes 0..63) */
    uint32_t magic;          /* 0..3: 0x4B525953 */
    uint32_t vital_hp;       /* 4..7: Must equal 6 */
    uint32_t capacity;       /* 8..11: Number of 128B slots */
    uint32_t slot_size;      /* 12..15: 128 bytes */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    _Atomic uint32_t head;   /* 16..19: Producer atomic head pointer */
#else
    volatile uint32_t head;
#endif
    uint8_t  _pad0[44];      /* 20..63: Padding to complete CL0 */

    /* CL1: Consumer cache line (bytes 64..127) */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    _Atomic uint32_t tail;   /* 64..67: Consumer atomic tail pointer */
    uint8_t  _pad1[4];       /* 68..71 */
    _Atomic uint64_t drop_count; /* 72..79: Dropped / rejected frame counter */
#else
    volatile uint32_t tail;
    uint8_t  _pad1[4];
    volatile uint64_t drop_count;
#endif
    uint8_t  _reserved[48];  /* 80..127: Padding to complete CL1 */
} KrystalRingHeader;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(KrystalRingHeader) == 128, "Ring header must be exactly 128B (2 cache lines)");
_Static_assert(_Alignof(KrystalRingHeader) == 64, "Ring header must be 64B cache-line aligned");
_Static_assert(offsetof(KrystalRingHeader, head) == 16, "Head pointer must be in CL0 at offset 16");
_Static_assert(offsetof(KrystalRingHeader, tail) == 64, "Tail pointer must be in CL1 at offset 64");
_Static_assert(offsetof(KrystalRingHeader, drop_count) == 72, "Drop counter must be in CL1 at offset 72");
#endif

/*
 * Slot address calculation helper:
 * Guarantees Addr(Slot_n) % 64 == 0 and Addr(Slot_n) % 128 == 0 when base is aligned.
 */
static inline KrystalZeroCopyFrame* krystal_ring_get_slot(void* ring_base, uint32_t slot_idx, uint32_t capacity) {
    uint8_t* base_ptr = (uint8_t*)ring_base;
    uint8_t* slots_start = base_ptr + KRYSTAL_RING_HEADER_BYTES;
    uint32_t wrapped_idx = slot_idx % capacity;
    return (KrystalZeroCopyFrame*)(slots_start + (wrapped_idx * KRYSTAL_FRAME_SIZE_BYTES));
}

#ifdef __cplusplus
}
#endif

#endif /* KRYSTAL_NEXUS_API_H */
