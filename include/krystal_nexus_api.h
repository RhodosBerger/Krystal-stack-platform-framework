/*
 * KRYSTAL-NEXUS COEXISTENCE C-ABI INTERFACE
 * ============================================================================
 * Zero-copy shared memory frame protocol for external graphics & AI engines.
 * Milestone: V-ZEROCOPY-MEMORY-MODEL-03 (Native Atomic Publication & ABI Compile Proof)
 * Platform invariant: KRYSTAL_VITAL_MAX_HP == 6
 */

#ifndef KRYSTAL_NEXUS_API_H
#define KRYSTAL_NEXUS_API_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#include <stdatomic.h>
#endif

#ifdef __cplusplus
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cstring>

/* C++ compatibility aliases for C11 keywords */
#ifndef _Alignas
#define _Alignas(n) alignas(n)
#endif
#ifndef _Alignof
#define _Alignof(t) alignof(t)
#endif
#ifndef _Static_assert
#define _Static_assert(cond, msg) static_assert(cond, msg)
#endif

#define KRYSTAL_LOAD_RELAXED(a)         ((a).load(std::memory_order_relaxed))
#define KRYSTAL_LOAD_ACQUIRE(a)         ((a).load(std::memory_order_acquire))
#define KRYSTAL_STORE_RELEASE(a, v)     ((a).store((v), std::memory_order_release))
#define KRYSTAL_FETCH_ADD_RELAXED(a, v) ((a).fetch_add((v), std::memory_order_relaxed))
#else
#define KRYSTAL_LOAD_RELAXED(a)         atomic_load_explicit(&(a), memory_order_relaxed)
#define KRYSTAL_LOAD_ACQUIRE(a)         atomic_load_explicit(&(a), memory_order_acquire)
#define KRYSTAL_STORE_RELEASE(a, v)     atomic_store_explicit(&(a), (v), memory_order_release)
#define KRYSTAL_FETCH_ADD_RELAXED(a, v) atomic_fetch_add_explicit(&(a), (v), memory_order_relaxed)
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define KRYSTAL_API_INLINE static inline __attribute__((unused))
#else
#define KRYSTAL_API_INLINE static inline
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
 *
 * C11 syntax: _Alignas(64) on the first member elevates the struct alignment to 64 bytes
 * without non-standard struct attributes.
 */
typedef struct {
    /* CL0: 0..63 (Naturally aligned scalars with 64B base alignment) */
    _Alignas(64) uint32_t magic;   /* 0..3   */
    uint32_t version;              /* 4..7   */
    uint16_t vital_hp;             /* 8..9   (Invariant == 6) */
    uint16_t contract_type;        /* 10..11 */
    uint32_t payload_size;         /* 12..15 (CL1 Application Payload = 16 bytes) */
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

#if (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L) || defined(__cplusplus)
_Static_assert(sizeof(KrystalZeroCopyFrame) == 128, "Frame size must be exactly 128B (2 cache lines)");
_Static_assert(_Alignof(KrystalZeroCopyFrame) == 64, "Frame must be 64B cache-line aligned");
_Static_assert(offsetof(KrystalZeroCopyFrame, timestamp_ns) == 32, "timestamp_ns offset drift");
_Static_assert(offsetof(KrystalZeroCopyFrame, nonce) == 40, "nonce offset drift");
_Static_assert(offsetof(KrystalZeroCopyFrame, profile_id) == 80, "CL1 payload offset drift");
_Static_assert(offsetof(KrystalZeroCopyFrame, frame_hmac) == 96, "HMAC digest offset drift");
#endif

/*
 * 128-byte cache-line isolated SPSC ring buffer header.
 * Eliminates false-sharing:
 * - Producer exclusively writes to CL0: head (offset 16) and drop_count (offset 24).
 * - Consumer exclusively writes to CL1: tail (offset 64).
 * - 64-bit uint64 atomics prevent 32-bit rollover after 4 billion frames.
 */
typedef struct {
    /* CL0: Producer cache line (bytes 0..63) - Exclusively written by Producer */
    _Alignas(64) uint32_t magic;   /* 0..3: 0x4B525953 */
    uint32_t vital_hp;             /* 4..7: Must equal 6 */
    uint32_t capacity;             /* 8..11: Number of 128B slots */
    uint32_t slot_size;            /* 12..15: 128 bytes */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    _Atomic uint64_t head;         /* 16..23: Producer atomic head pointer */
    _Atomic uint64_t drop_count;   /* 24..31: Producer drop counter (on Producer CL0!) */
#elif defined(__cplusplus)
    std::atomic<uint64_t> head;    /* 16..23: Producer atomic head pointer */
    std::atomic<uint64_t> drop_count; /* 24..31: Producer drop counter (on Producer CL0!) */
#else
    uint64_t head;
    uint64_t drop_count;
#endif
    uint8_t  _pad0[32];            /* 32..63: Padding to complete CL0 */

    /* CL1: Consumer cache line (bytes 64..127) - Exclusively written by Consumer */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    _Atomic uint64_t tail;         /* 64..71: Consumer atomic tail pointer (CL1 offset 64) */
#elif defined(__cplusplus)
    std::atomic<uint64_t> tail;    /* 64..71: Consumer atomic tail pointer (CL1 offset 64) */
#else
    uint64_t tail;
#endif
    uint8_t  _reserved[56];        /* 72..127: Padding to complete CL1 */
} KrystalRingHeader;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(KrystalRingHeader) == 128, "Ring header must be exactly 128B (2 cache lines)");
_Static_assert(_Alignof(KrystalRingHeader) == 64, "Ring header must be 64B cache-line aligned");
_Static_assert(offsetof(KrystalRingHeader, head) == 16, "Head pointer must be in CL0 at offset 16");
_Static_assert(offsetof(KrystalRingHeader, drop_count) == 24, "Drop counter must be in CL0 at offset 24");
_Static_assert(offsetof(KrystalRingHeader, tail) == 64, "Tail pointer must be in CL1 at offset 64");
#elif defined(__cplusplus)
static_assert(sizeof(KrystalRingHeader) == 128, "Ring header must be exactly 128B (2 cache lines)");
static_assert(alignof(KrystalRingHeader) == 64, "Ring header must be 64B cache-line aligned");
#endif

/*
 * Slot address calculation helper:
 * Precondition: RingBaseAddress % 128 == 0 (128-byte aligned virtual base address).
 * Because RingHeader is 128 bytes and each slot is 128 bytes:
 *   Addr(Slot_n) = RingBaseAddress + 128 + (wrapped_idx * 128)
 *   Addr(Slot_n) % 128 == (RingBaseAddress % 128) == 0.
 * Guarantees Addr(Slot_n) % 64 == 0 and Addr(Slot_n) % 128 == 0 for all slots.
 */
KRYSTAL_API_INLINE KrystalZeroCopyFrame* krystal_ring_get_slot(void* ring_base, uint64_t slot_idx, uint32_t capacity) {
    uint8_t* base_ptr = (uint8_t*)ring_base;
    uint8_t* slots_start = base_ptr + KRYSTAL_RING_HEADER_BYTES;
    uint64_t wrapped_idx = slot_idx % capacity;
    return (KrystalZeroCopyFrame*)(slots_start + (wrapped_idx * KRYSTAL_FRAME_SIZE_BYTES));
}

#if (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L) || defined(__cplusplus)
/*
 * Native C11/C++20 SPSC Publication Protocol (memory_order_release).
 * Guarantees all writes to CL0 and CL1 of slot_n happen-before consumer observes head increment.
 * ARM64: STLR instruction
 * RISC-V: fence rw,w
 * x86_64: MOV with compiler barrier
 */
KRYSTAL_API_INLINE bool krystal_ring_publish(KrystalRingHeader* hdr, void* ring_base, const KrystalZeroCopyFrame* frame) {
    uint64_t head = KRYSTAL_LOAD_RELAXED(hdr->head);
    uint64_t tail = KRYSTAL_LOAD_ACQUIRE(hdr->tail);

    if ((head - tail) >= hdr->capacity) {
        /* Ring full: record drop in producer's own CL0 */
        KRYSTAL_FETCH_ADD_RELAXED(hdr->drop_count, 1);
        return false;
    }

    KrystalZeroCopyFrame* slot = krystal_ring_get_slot(ring_base, head, hdr->capacity);
    /* Direct 128-byte memcpy into slot (CL0 + CL1) */
    memcpy(slot, frame, sizeof(KrystalZeroCopyFrame));

    /* Release publication store */
    KRYSTAL_STORE_RELEASE(hdr->head, head + 1);
    return true;
}

/*
 * Native C11/C++20 SPSC Consumption Protocol (memory_order_acquire).
 * Guarantees read of head happens-before reading slot content, preventing speculative reads of old cache lines.
 * ARM64: LDAR instruction
 * RISC-V: fence r,rw
 * x86_64: MOV with compiler barrier
 */
KRYSTAL_API_INLINE bool krystal_ring_consume(KrystalRingHeader* hdr, void* ring_base, KrystalZeroCopyFrame* out_frame) {
    uint64_t head = KRYSTAL_LOAD_ACQUIRE(hdr->head);
    uint64_t tail = KRYSTAL_LOAD_RELAXED(hdr->tail);

    if (head == tail) {
        return false; /* Ring empty */
    }

    KrystalZeroCopyFrame* slot = krystal_ring_get_slot(ring_base, tail, hdr->capacity);
    memcpy(out_frame, slot, sizeof(KrystalZeroCopyFrame));

    /* Release tail store so producer can reclaim slot */
    KRYSTAL_STORE_RELEASE(hdr->tail, tail + 1);
    return true;
}
#endif

#ifdef __cplusplus
}
#endif

#endif /* KRYSTAL_NEXUS_API_H */
