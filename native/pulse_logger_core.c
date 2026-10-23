/**
 * ==============================================================================
 * KRYSTAL-STACK: NATIVE PULSE SEQUENCE LOGGER CORE (MODULE H-07 / E-07)
 * ==============================================================================
 * Hot-Path Implementation: Zero-allocation, lockless, zero-syscall, no JSON.
 * High-performance C11 with x86_64 intrinsics and atomic release semantics.
 * ==============================================================================
 */

#include "krystal_pulse_abi.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#endif

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

static uint64_t s_cached_tsc_freq_hz = 0;

/**
 * Fast inline hardware TSC counter reading (sub-nanosecond, no call overhead).
 */
static inline __attribute__((always_inline)) uint64_t pulse_read_tsc_fast(void) {
#if defined(__x86_64__) || defined(_M_X64)
    #if defined(__GNUC__) || defined(__clang__)
    return __builtin_ia32_rdtsc();
    #elif defined(_MSC_VER)
    return __rdtsc();
    #else
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
    #endif
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

/**
 * Read hardware TSC counter with sub-nanosecond hardware resolution (C ABI export).
 */
uint64_t pulse_read_tsc(void) {
    return pulse_read_tsc_fast();
}

/**
 * Calibrate the invariant TSC frequency in Hz outside of the hot path.
 * Runs once at startup using a 50 ms sample against the OS monotonic clock.
 */
uint64_t pulse_calibrate_tsc_freq_hz(void) {
    if (s_cached_tsc_freq_hz > 0) {
        return s_cached_tsc_freq_hz;
    }

#if defined(_WIN32)
    LARGE_INTEGER qpc_freq, qpc_start, qpc_end;
    QueryPerformanceFrequency(&qpc_freq);
    QueryPerformanceCounter(&qpc_start);
    uint64_t tsc_start = pulse_read_tsc_fast();

    // Sleep 50ms
    Sleep(50);

    QueryPerformanceCounter(&qpc_end);
    uint64_t tsc_end = pulse_read_tsc_fast();

    double elapsed_sec = (double)(qpc_end.QuadPart - qpc_start.QuadPart) / (double)qpc_freq.QuadPart;
    if (elapsed_sec > 0.0) {
        s_cached_tsc_freq_hz = (uint64_t)((double)(tsc_end - tsc_start) / elapsed_sec);
    } else {
        s_cached_tsc_freq_hz = 2400000000ULL; // Fallback 2.4 GHz
    }
#else
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    uint64_t tsc_start = pulse_read_tsc_fast();

    // Sleep ~50ms
    struct timespec sleep_time = {0, 50000000}; // 50 ms
    nanosleep(&sleep_time, NULL);

    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    uint64_t tsc_end = pulse_read_tsc_fast();

    double elapsed_sec = (double)(ts_end.tv_sec - ts_start.tv_sec) + 
                         (double)(ts_end.tv_nsec - ts_start.tv_nsec) * 1e-9;
    if (elapsed_sec > 0.0) {
        s_cached_tsc_freq_hz = (uint64_t)((double)(tsc_end - tsc_start) / elapsed_sec);
    } else {
        s_cached_tsc_freq_hz = 2400000000ULL; // Fallback 2.4 GHz
    }
#endif

    return s_cached_tsc_freq_hz;
}

/**
 * Initialize pre-allocated SPSC PulseRing.
 */
int pulse_ring_init(PulseRing* ring, PulseFrame* buffer, uint32_t capacity) {
    if (!ring || !buffer || capacity == 0) {
        return -1;
    }
    // Capacity must be power of 2
    if ((capacity & (capacity - 1)) != 0) {
        return -1;
    }

    memset(ring, 0, sizeof(PulseRing));
    ring->frames = buffer;
    ring->capacity_mask = capacity - 1;
    ring->epoch_ticks = pulse_read_tsc_fast();
    ring->write_index = 0;
    ring->read_index = 0;
    ring->batch_count = 0;

    return 0;
}

/**
 * Create heap-allocated PulseRing.
 */
PulseRing* pulse_ring_create(uint32_t capacity) {
    if (capacity == 0 || (capacity & (capacity - 1)) != 0) {
        capacity = PULSE_RING_DEFAULT_CAPACITY;
    }

    PulseRing* ring = (PulseRing*)calloc(1, sizeof(PulseRing));
    if (!ring) return NULL;

    PulseFrame* buffer = (PulseFrame*)calloc(capacity, sizeof(PulseFrame));
    if (!buffer) {
        free(ring);
        return NULL;
    }

    if (pulse_ring_init(ring, buffer, capacity) != 0) {
        free(buffer);
        free(ring);
        return NULL;
    }

    return ring;
}

/**
 * Destroy heap-allocated PulseRing.
 */
void pulse_ring_destroy(PulseRing* ring) {
    if (ring) {
        if (ring->frames) {
            free(ring->frames);
            ring->frames = NULL;
        }
        free(ring);
    }
}

/**
 * HOT-PATH H1: Ordinary aligned 64-bit store with acquire/release semantics.
 */
void pulse_emit_h1_ordinary(PulseRing* ring, uint8_t event_code, uint16_t payload) {
    uint64_t ticks = pulse_read_tsc_fast();
    uint32_t delta = (uint32_t)(ticks - ring->epoch_ticks);
    uint64_t w = ring->write_index;
    uint32_t idx = (uint32_t)(w & ring->capacity_mask);

    // Single 64-bit packed value: [payload: 16][invariant 0x06: 8][code: 8][delta_ticks: 32]
    uint64_t frame_val = ((uint64_t)payload << 48) |
                         ((uint64_t)VITAL_MAX_HP << 40) |
                         ((uint64_t)event_code << 32) |
                         (uint64_t)delta;

    // Aligned 64-bit store into ring
    *(uint64_t*)&ring->frames[idx] = frame_val;

    // Release fence: publisher commits write_index
#if defined(__GNUC__) || defined(__clang__)
    __atomic_store_n(&ring->write_index, w + 1, __ATOMIC_RELEASE);
#else
    ring->write_index = w + 1;
#endif
}

/**
 * HOT-PATH H2: Non-temporal store (MOVNTI) bypassing CPU cache hierarchy.
 */
void pulse_emit_h2_nontemporal(PulseRing* ring, uint8_t event_code, uint16_t payload) {
    uint64_t ticks = pulse_read_tsc_fast();
    uint32_t delta = (uint32_t)(ticks - ring->epoch_ticks);
    uint64_t w = ring->write_index;
    uint32_t idx = (uint32_t)(w & ring->capacity_mask);

    uint64_t frame_val = ((uint64_t)payload << 48) |
                         ((uint64_t)VITAL_MAX_HP << 40) |
                         ((uint64_t)event_code << 32) |
                         (uint64_t)delta;

#if defined(__x86_64__) || defined(_M_X64)
    #if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__(
        "movnti %1, %0"
        : "=m"(*(uint64_t*)&ring->frames[idx])
        : "r"(frame_val)
        : "memory"
    );
    #elif defined(_MSC_VER)
    _mm_stream_si64((long long int*)&ring->frames[idx], (long long int)frame_val);
    #else
    *(uint64_t*)&ring->frames[idx] = frame_val;
    #endif
#else
    *(uint64_t*)&ring->frames[idx] = frame_val;
#endif

#if defined(__GNUC__) || defined(__clang__)
    __atomic_store_n(&ring->write_index, w + 1, __ATOMIC_RELEASE);
#else
    ring->write_index = w + 1;
#endif
}

/**
 * HOT-PATH H3: 16-frame batch buffer emission with single release publication barrier.
 */
void pulse_emit_h3_batch16(PulseRing* ring, uint8_t event_code, uint16_t payload) {
    uint64_t ticks = pulse_read_tsc_fast();
    uint32_t delta = (uint32_t)(ticks - ring->epoch_ticks);
    uint32_t cnt = ring->batch_count;

    uint64_t frame_val = ((uint64_t)payload << 48) |
                         ((uint64_t)VITAL_MAX_HP << 40) |
                         ((uint64_t)event_code << 32) |
                         (uint64_t)delta;

    *(uint64_t*)&ring->batch_buffer[cnt] = frame_val;
    cnt++;

#if defined(__GNUC__) || defined(__clang__)
    if (__builtin_expect(cnt == PULSE_BATCH_SIZE, 0)) {
#else
    if (cnt == PULSE_BATCH_SIZE) {
#endif
        uint64_t w = ring->write_index;
        uint32_t mask = ring->capacity_mask;

        // Copy 16 frames sequentially (128 bytes total)
        for (uint32_t i = 0; i < PULSE_BATCH_SIZE; i++) {
            *(uint64_t*)&ring->frames[(w + i) & mask] = *(uint64_t*)&ring->batch_buffer[i];
        }

        // Single atomic release publication barrier for all 16 frames!
#if defined(__GNUC__) || defined(__clang__)
        __atomic_store_n(&ring->write_index, w + PULSE_BATCH_SIZE, __ATOMIC_RELEASE);
#else
        ring->write_index = w + PULSE_BATCH_SIZE;
#endif
        ring->batch_count = 0;
    } else {
        ring->batch_count = cnt;
    }
}

/**
 * Flush any pending uncommitted frames in the 16-frame batch buffer.
 */
void pulse_flush_batch(PulseRing* ring) {
    if (!ring || ring->batch_count == 0) return;

    uint32_t cnt = ring->batch_count;
    uint64_t w = ring->write_index;
    uint32_t mask = ring->capacity_mask;

    for (uint32_t i = 0; i < cnt; i++) {
        *(uint64_t*)&ring->frames[(w + i) & mask] = *(uint64_t*)&ring->batch_buffer[i];
    }

#if defined(__GNUC__) || defined(__clang__)
    __atomic_store_n(&ring->write_index, w + cnt, __ATOMIC_RELEASE);
#else
    ring->write_index = w + cnt;
#endif
    ring->batch_count = 0;
}

/**
 * COLD-PATH READ: Reads one frame from the SPSC ring.
 */
int pulse_ring_read(PulseRing* ring, PulseFrame* out_frame) {
    if (!ring || !out_frame) return 0;

    uint64_t r = ring->read_index;
#if defined(__GNUC__) || defined(__clang__)
    uint64_t w = __atomic_load_n(&ring->write_index, __ATOMIC_ACQUIRE);
#else
    uint64_t w = ring->write_index;
#endif

    if (r >= w) {
        return 0; // Empty
    }

    uint32_t idx = (uint32_t)(r & ring->capacity_mask);
    *out_frame = ring->frames[idx];

#if defined(__GNUC__) || defined(__clang__)
    __atomic_store_n(&ring->read_index, r + 1, __ATOMIC_RELEASE);
#else
    ring->read_index = r + 1;
#endif

    return 1;
}

/**
 * COLD-PATH BATCH READ: Reads up to max_count frames into out_frames array.
 */
uint32_t pulse_ring_read_batch(PulseRing* ring, PulseFrame* out_frames, uint32_t max_count) {
    if (!ring || !out_frames || max_count == 0) return 0;

    uint64_t r = ring->read_index;
#if defined(__GNUC__) || defined(__clang__)
    uint64_t w = __atomic_load_n(&ring->write_index, __ATOMIC_ACQUIRE);
#else
    uint64_t w = ring->write_index;
#endif

    if (r >= w) {
        return 0;
    }

    uint64_t available = w - r;
    uint32_t to_read = (uint32_t)(available < (uint64_t)max_count ? available : (uint64_t)max_count);
    uint32_t mask = ring->capacity_mask;

    for (uint32_t i = 0; i < to_read; i++) {
        out_frames[i] = ring->frames[(r + i) & mask];
    }

#if defined(__GNUC__) || defined(__clang__)
    __atomic_store_n(&ring->read_index, r + to_read, __ATOMIC_RELEASE);
#else
    ring->read_index = r + to_read;
#endif

    return to_read;
}

/**
 * Number of unread frames currently in the ring.
 */
uint32_t pulse_ring_available(const PulseRing* ring) {
    if (!ring) return 0;
#if defined(__GNUC__) || defined(__clang__)
    uint64_t w = __atomic_load_n(&ring->write_index, __ATOMIC_ACQUIRE);
    uint64_t r = __atomic_load_n(&ring->read_index, __ATOMIC_ACQUIRE);
#else
    uint64_t w = ring->write_index;
    uint64_t r = ring->read_index;
#endif
    return (uint32_t)(w > r ? w - r : 0);
}

static int compare_doubles(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/**
 * Benchmark strategy directly in native C (eliminates language boundary observer artifact).
 */
int pulse_benchmark_native(PulseRing* ring, int strategy, uint32_t iterations, PulseBenchmarkStats* out_stats) {
    if (!ring || !out_stats || iterations == 0) return -1;

    uint64_t tsc_freq = pulse_calibrate_tsc_freq_hz();
    if (tsc_freq == 0) tsc_freq = 2400000000ULL;
    double ns_per_tick = 1e9 / (double)tsc_freq;

    // Warmup
    for (uint32_t i = 0; i < 5000; i++) {
        pulse_emit_h1_ordinary(ring, 0x10, (uint16_t)i);
    }
    PulseFrame dummy;
    while (pulse_ring_read(ring, &dummy));

    const uint32_t chunk_size = 64;
    uint32_t num_chunks = iterations / chunk_size;
    if (num_chunks == 0) num_chunks = 1;
    if (num_chunks > 100000) num_chunks = 100000;

    double* samples = (double*)malloc(num_chunks * sizeof(double));
    if (!samples) return -1;

    double sum_ns = 0.0;
    uint32_t valid_chunks = 0;

    for (uint32_t c = 0; c < num_chunks; c++) {
        uint64_t t0 = pulse_read_tsc_fast();
        if (strategy == 1) {
            for (uint32_t i = 0; i < chunk_size; i++) {
                pulse_emit_h1_ordinary(ring, 0x10, (uint16_t)i);
            }
        } else if (strategy == 2) {
            for (uint32_t i = 0; i < chunk_size; i++) {
                pulse_emit_h2_nontemporal(ring, 0x10, (uint16_t)i);
            }
        } else {
            for (uint32_t i = 0; i < chunk_size; i++) {
                pulse_emit_h3_batch16(ring, 0x10, (uint16_t)i);
            }
        }
        uint64_t t1 = pulse_read_tsc_fast();
        double elapsed_ns = (double)(t1 - t0) * ns_per_tick;
        double per_event_ns = elapsed_ns / (double)chunk_size;
        samples[valid_chunks++] = per_event_ns;
        sum_ns += per_event_ns;

        // Drain ring periodically to prevent overflow (ring capacity is 65536)
        if (pulse_ring_available(ring) >= 32768) {
            while (pulse_ring_read(ring, &dummy));
        }
    }

    if (strategy == 3) {
        pulse_flush_batch(ring);
    }
    while (pulse_ring_read(ring, &dummy));

    qsort(samples, valid_chunks, sizeof(double), compare_doubles);

    double mean = sum_ns / valid_chunks;
    double var_sum = 0.0;
    for (uint32_t i = 0; i < valid_chunks; i++) {
        double diff = samples[i] - mean;
        var_sum += diff * diff;
    }

    out_stats->mean_ns = mean;
    out_stats->p50_ns = samples[(uint32_t)(valid_chunks * 0.50)];
    out_stats->p95_ns = samples[(uint32_t)(valid_chunks * 0.95)];
    out_stats->p99_ns = samples[(uint32_t)(valid_chunks * 0.99)];
    out_stats->min_ns = samples[0];
    out_stats->max_ns = samples[valid_chunks - 1];
    out_stats->variance = var_sum / valid_chunks;
    out_stats->iterations = iterations;
    out_stats->sample_count = valid_chunks;

    free(samples);
    return 0;
}

