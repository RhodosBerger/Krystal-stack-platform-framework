/**
 * ==============================================================================
 * KRYSTAL-STACK: NATIVE PULSE SEQUENCE LOGGER C-ABI (MODULE H-07 / E-07)
 * ==============================================================================
 * Hot-Path Interface: Zero-allocation, lock-free, zero-syscall, no JSON.
 * Constitutional Invariant: VITAL_MAX_HP == 6
 * Hardware Target: Intel Tiger Lake / Iris Xe (UMA Zero-Copy Pipeline)
 * ==============================================================================
 */

#ifndef KRYSTAL_PULSE_ABI_H
#define KRYSTAL_PULSE_ABI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VITAL_MAX_HP 6
#define PULSE_RING_DEFAULT_CAPACITY 65536
#define PULSE_BATCH_SIZE 16

/**
 * PulseFrame: Exactly 8 bytes, 8-byte aligned.
 * Layout:
 *   [0..3] uint32_t delta_ticks: Raw TSC delta ticks from epoch (no float division in hot-path)
 *   [4]    uint8_t  event_code:  Hexadecimal Telemetry Event Code (0x10..0xFF)
 *   [5]    uint8_t  invariant:   Constitutional invariant marker (MUST be 6)
 *   [6..7] uint16_t payload:     Quantized 16-bit measurement payload
 */
#if defined(_MSC_VER)
__declspec(align(8))
#endif
typedef struct 
#if defined(__GNUC__) || defined(__clang__)
__attribute__((aligned(8)))
#endif
{
    uint32_t delta_ticks;
    uint8_t  event_code;
    uint8_t  invariant;
    uint16_t payload;
} PulseFrame;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(PulseFrame) == 8, "PulseFrame size must be exactly 8 bytes");
#endif

/**
 * SPSC PulseRing Buffer
 * Cacheline-separated producer and consumer counters to eliminate false sharing.
 */
typedef struct {
    // Producer Cacheline (Hot-Path Write)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((aligned(64)))
#elif defined(_MSC_VER)
    __declspec(align(64))
#endif
    uint64_t write_index;

    uint64_t epoch_ticks;
    uint32_t capacity_mask; // capacity - 1 (capacity must be a power of 2)
    uint32_t batch_count;   // Count of frames currently in batch_buffer (0..15)
    PulseFrame batch_buffer[PULSE_BATCH_SIZE];

    // Consumer Cacheline (Cold-Path Read)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((aligned(64)))
#elif defined(_MSC_VER)
    __declspec(align(64))
#endif
    uint64_t read_index;

    // Buffer storage pointer or start of frames
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((aligned(64)))
#elif defined(_MSC_VER)
    __declspec(align(64))
#endif
    PulseFrame* frames;
} PulseRing;

/**
 * Initialize a PulseRing instance with a pre-allocated array of frames.
 * @param ring Pointer to PulseRing structure
 * @param buffer Pointer to contiguous array of PulseFrame (must hold capacity elements)
 * @param capacity Number of frames (must be a power of 2, e.g. 65536)
 * @return 0 on success, -1 on invalid parameter
 */
int pulse_ring_init(PulseRing* ring, PulseFrame* buffer, uint32_t capacity);

/**
 * Allocate and initialize a PulseRing with heap-allocated frames.
 * @param capacity Capacity of the ring (power of 2)
 * @return Pointer to initialized PulseRing or NULL
 */
PulseRing* pulse_ring_create(uint32_t capacity);

/**
 * Free a dynamically created PulseRing.
 */
void pulse_ring_destroy(PulseRing* ring);

/**
 * Read the current hardware Timestamp Counter (TSC) without syscall.
 */
uint64_t pulse_read_tsc(void);

/**
 * Calibrate the invariant TSC frequency in Hz outside the hot path.
 * On modern Intel processors, invariant TSC runs at a fixed nominal rate.
 */
uint64_t pulse_calibrate_tsc_freq_hz(void);

/**
 * HOT-PATH H1: Ordinary aligned 64-bit store with acquire/release semantics.
 */
void pulse_emit_h1_ordinary(PulseRing* ring, uint8_t event_code, uint16_t payload);

/**
 * HOT-PATH H2: Non-temporal store (MOVNTI) bypassing CPU cache hierarchy.
 */
void pulse_emit_h2_nontemporal(PulseRing* ring, uint8_t event_code, uint16_t payload);

/**
 * HOT-PATH H3: 16-frame batch buffer emission with single release publication barrier.
 */
void pulse_emit_h3_batch16(PulseRing* ring, uint8_t event_code, uint16_t payload);

/**
 * Flush any remaining uncommitted frames in the 16-frame batch buffer.
 */
void pulse_flush_batch(PulseRing* ring);

/**
 * COLD-PATH READ: Reads one frame from the SPSC ring.
 * @param ring Pointer to PulseRing
 * @param out_frame Pointer where the frame will be copied
 * @return 1 if a frame was read, 0 if the ring is empty
 */
int pulse_ring_read(PulseRing* ring, PulseFrame* out_frame);

/**
 * COLD-PATH BATCH READ: Reads up to max_count frames into out_frames array.
 * @return Number of frames actually read
 */
uint32_t pulse_ring_read_batch(PulseRing* ring, PulseFrame* out_frames, uint32_t max_count);

/**
 * Number of unread frames currently in the ring.
 */
uint32_t pulse_ring_available(const PulseRing* ring);

/**
 * Statistics structure for native microsecond/nanosecond benchmarking.
 */
typedef struct {
    double mean_ns;
    double p50_ns;
    double p95_ns;
    double p99_ns;
    double min_ns;
    double max_ns;
    double variance;
    uint32_t iterations;
    uint32_t sample_count;
} PulseBenchmarkStats;

/**
 * Benchmark strategy directly in native C (eliminates language boundary observer artifact).
 * @param strategy 1 for H1 (ordinary), 2 for H2 (non-temporal), 3 for H3 (batch16)
 */
int pulse_benchmark_native(PulseRing* ring, int strategy, uint32_t iterations, PulseBenchmarkStats* out_stats);

#ifdef __cplusplus
}
#endif

#endif /* KRYSTAL_PULSE_ABI_H */

