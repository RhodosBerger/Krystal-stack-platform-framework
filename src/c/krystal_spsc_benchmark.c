/*
 * KRYSTAL-NEXUS SPSC BENCHMARK HARNESS
 * ============================================================================
 * Measures hardware cycle overhead and nanosecond latency for 128-byte
 * KrystalZeroCopyFrame serialization and Acquire/Release SPSC handoff.
 *
 * Supported ISAs: x86_64 (RDTSCP), ARM64 (CNTVCT_EL0), RISC-V (RDCYCLE).
 * Platform Invariant: KRYSTAL_VITAL_MAX_HP == 6
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <inttypes.h>

#include "../../include/krystal_nexus_api.h"

#define BENCHMARK_CAPACITY   2048
#define BENCHMARK_WARMUP     10000
#define BENCHMARK_OPS        500000

/*
 * Portable high-resolution cycle counter
 */
static inline uint64_t read_hw_cycles(void) {
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t aux;
    return __builtin_ia32_rdtscp(&aux);
#elif defined(__aarch64__)
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
#elif defined(__riscv)
    uint64_t val;
    __asm__ volatile("rdcycle %0" : "=r"(val));
    return val;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/*
 * Self-contained SHA-256 for zero external dependencies
 */
static inline uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

static void sha256_block_transform(uint32_t state[8], const uint8_t data[64]) {
    static const uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)data[i*4] << 24) | ((uint32_t)data[i*4+1] << 16) |
               ((uint32_t)data[i*4+2] << 8) | ((uint32_t)data[i*4+3]);
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + K[i] + w[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

static void fast_sha256_96b(const uint8_t input[96], uint8_t output[32]) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    /* Block 1 (64 bytes: 0..63) */
    sha256_block_transform(state, input);

    /* Block 2 (32 bytes data + padding: 64..95 + 0x80 + zeroes + length 768 bits) */
    uint8_t block2[64];
    memcpy(block2, input + 64, 32);
    block2[32] = 0x80;
    memset(block2 + 33, 0, 23);
    /* Length in bits = 96 * 8 = 768 = 0x300 */
    block2[56] = 0; block2[57] = 0; block2[58] = 0; block2[59] = 0;
    block2[60] = 0; block2[61] = 0; block2[62] = 0x03; block2[63] = 0x00;
    sha256_block_transform(state, block2);

    for (int i = 0; i < 8; i++) {
        output[i*4]   = (uint8_t)(state[i] >> 24);
        output[i*4+1] = (uint8_t)(state[i] >> 16);
        output[i*4+2] = (uint8_t)(state[i] >> 8);
        output[i*4+3] = (uint8_t)(state[i]);
    }
}

/*
 * Multi-threaded benchmark context
 */
typedef struct {
    void* ring_base;
    uint32_t capacity;
    uint32_t num_operations;
    _Atomic bool ready;
    uint64_t total_push_cycles;
    uint64_t total_pop_cycles;
    uint64_t torn_reads_detected;
} SPSCBenchmarkContext;

static void* producer_thread(void* arg) {
    SPSCBenchmarkContext* ctx = (SPSCBenchmarkContext*)arg;
    KrystalRingHeader* hdr = (KrystalRingHeader*)ctx->ring_base;

    while (!atomic_load_explicit(&ctx->ready, memory_order_acquire)) {
        /* Wait for start signal */
    }

    uint64_t start_cycles = read_hw_cycles();
    for (uint32_t i = 0; i < ctx->num_operations; i++) {
        uint32_t head = atomic_load_explicit(&hdr->head, memory_order_relaxed);
        uint32_t tail = atomic_load_explicit(&hdr->tail, memory_order_acquire);

        while ((head - tail) >= ctx->capacity) {
            /* Ring is full: spin / backpressure */
            tail = atomic_load_explicit(&hdr->tail, memory_order_acquire);
        }

        KrystalZeroCopyFrame* slot = krystal_ring_get_slot(ctx->ring_base, head, ctx->capacity);

        /* Populate 96-byte prefix */
        slot->magic = KRYSTAL_MAGIC;
        slot->version = KRYSTAL_PROTOCOL_VERSION;
        slot->vital_hp = KRYSTAL_VITAL_MAX_HP;
        slot->contract_type = 1;
        slot->payload_size = 16;
        slot->flags = 0x0001;
        slot->sequence = i;
        slot->generation = 1;
        slot->cap_scope = 0x0003;
        slot->timestamp_ns = 1792181862000ULL + i;
        slot->nonce = (uint64_t)i * 2ULL;

        /* Write split predecessor_hash (spans CL0 & CL1) */
        memset(slot->predecessor_hash, (uint8_t)(i & 0xFF), 32);

        slot->profile_id = 1;
        slot->verdict_id = 2;
        slot->mask = 0xF;
        slot->gpu_render_ms = 8.33f;
        slot->drs_scale_factor = 1.0f;

        /* Compute 96-byte digest */
        fast_sha256_96b((const uint8_t*)slot, slot->frame_hmac);

        /* Release publication: STLR / fence ensures CL0 + CL1 are visible before head increments */
        atomic_store_explicit(&hdr->head, head + 1, memory_order_release);
    }
    ctx->total_push_cycles = read_hw_cycles() - start_cycles;
    return NULL;
}

static void* consumer_thread(void* arg) {
    SPSCBenchmarkContext* ctx = (SPSCBenchmarkContext*)arg;
    KrystalRingHeader* hdr = (KrystalRingHeader*)ctx->ring_base;

    while (!atomic_load_explicit(&ctx->ready, memory_order_acquire)) {
        /* Wait for start signal */
    }

    uint64_t start_cycles = read_hw_cycles();
    uint32_t received = 0;
    uint8_t digest_check[32];

    while (received < ctx->num_operations) {
        uint32_t head = atomic_load_explicit(&hdr->head, memory_order_acquire);
        uint32_t tail = atomic_load_explicit(&hdr->tail, memory_order_relaxed);

        if (head == tail) {
            /* Ring is empty: spin */
            continue;
        }

        KrystalZeroCopyFrame* slot = krystal_ring_get_slot(ctx->ring_base, tail, ctx->capacity);

        /* Verify invariant vital_hp == 6 */
        if (slot->vital_hp != KRYSTAL_VITAL_MAX_HP) {
            ctx->torn_reads_detected++;
        }

        /* Check HMAC/digest for torn read detection */
        fast_sha256_96b((const uint8_t*)slot, digest_check);
        if (memcmp(slot->frame_hmac, digest_check, 32) != 0) {
            ctx->torn_reads_detected++;
        }

        /* Consume slot and advance tail with release */
        atomic_store_explicit(&hdr->tail, tail + 1, memory_order_release);
        received++;
    }
    ctx->total_pop_cycles = read_hw_cycles() - start_cycles;
    return NULL;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    printf("====================================================================\n");
    printf("KRYSTAL-NEXUS SPSC HARDWARE BENCHMARK (V-ZEROCOPY-MEMORY-MODEL-02)\n");
    printf("====================================================================\n");

    /* 1. Structural Invariant Verification */
    printf("[1/4] Verifying structural C11 ABI contracts...\n");
    if (sizeof(KrystalZeroCopyFrame) != 128) {
        fprintf(stderr, "FAIL: sizeof(KrystalZeroCopyFrame) != 128 (got %zu)\n", sizeof(KrystalZeroCopyFrame));
        return 1;
    }
    if (_Alignof(KrystalZeroCopyFrame) != 64) {
        fprintf(stderr, "FAIL: _Alignof(KrystalZeroCopyFrame) != 64 (got %zu)\n", _Alignof(KrystalZeroCopyFrame));
        return 1;
    }
    if (sizeof(KrystalRingHeader) != 128) {
        fprintf(stderr, "FAIL: sizeof(KrystalRingHeader) != 128 (got %zu)\n", sizeof(KrystalRingHeader));
        return 1;
    }
    if (_Alignof(KrystalRingHeader) != 64) {
        fprintf(stderr, "FAIL: _Alignof(KrystalRingHeader) != 64 (got %zu)\n", _Alignof(KrystalRingHeader));
        return 1;
    }
    printf("      -> Struct sizes and 64B cache line alignments: VERIFIED\n");

    /* 2. Buffer Allocation (Page-aligned at 4096B) */
    printf("[2/4] Allocating 4096-byte aligned SPSC shared memory region...\n");
    size_t ring_total_bytes = KRYSTAL_RING_HEADER_BYTES + (BENCHMARK_CAPACITY * KRYSTAL_FRAME_SIZE_BYTES);
    void* ring_mem = NULL;
#if defined(_WIN32)
    ring_mem = _aligned_malloc(ring_total_bytes, 4096);
#else
    if (posix_memalign(&ring_mem, 4096, ring_total_bytes) != 0) {
        fprintf(stderr, "FAIL: posix_memalign failed\n");
        return 1;
    }
#endif
    memset(ring_mem, 0, ring_total_bytes);

    KrystalRingHeader* hdr = (KrystalRingHeader*)ring_mem;
    hdr->magic = KRYSTAL_MAGIC;
    hdr->vital_hp = KRYSTAL_VITAL_MAX_HP;
    hdr->capacity = BENCHMARK_CAPACITY;
    hdr->slot_size = KRYSTAL_FRAME_SIZE_BYTES;
    atomic_init(&hdr->head, 0);
    atomic_init(&hdr->tail, 0);
    atomic_init(&hdr->drop_count, 0);

    /* Verify slot alignment across all slots */
    for (uint32_t i = 0; i < BENCHMARK_CAPACITY; i++) {
        KrystalZeroCopyFrame* slot = krystal_ring_get_slot(ring_mem, i, BENCHMARK_CAPACITY);
        uintptr_t addr = (uintptr_t)slot;
        if (addr % 64 != 0 || addr % 128 != 0) {
            fprintf(stderr, "FAIL: Slot %u unaligned: %p\n", i, (void*)slot);
            return 1;
        }
    }
    printf("      -> 2048 slots verified with Addr %% 64 == 0 and Addr %% 128 == 0\n");

    /* 3. Single-Thread Serial Write Throughput */
    printf("[3/4] Benchmarking single-thread frame serialization & SHA-256 (100,000 ops)...\n");
    KrystalZeroCopyFrame sample_frame;
    sample_frame.magic = KRYSTAL_MAGIC;
    sample_frame.version = KRYSTAL_PROTOCOL_VERSION;
    sample_frame.vital_hp = KRYSTAL_VITAL_MAX_HP;
    sample_frame.contract_type = 1;
    sample_frame.payload_size = 16;
    sample_frame.flags = 0;
    sample_frame.sequence = 42;
    sample_frame.generation = 1;
    sample_frame.cap_scope = 1;
    sample_frame.timestamp_ns = 12345678ULL;
    sample_frame.nonce = 9999ULL;
    memset(sample_frame.predecessor_hash, 0xAB, 32);
    sample_frame.profile_id = 1;
    sample_frame.verdict_id = 2;
    sample_frame.mask = 0xF;
    sample_frame.gpu_render_ms = 8.33f;
    sample_frame.drs_scale_factor = 1.0f;

    uint64_t t0 = get_time_ns();
    uint64_t c0 = read_hw_cycles();
    for (int i = 0; i < 100000; i++) {
        sample_frame.sequence = (uint32_t)i;
        fast_sha256_96b((const uint8_t*)&sample_frame, sample_frame.frame_hmac);
    }
    uint64_t c_elapsed = read_hw_cycles() - c0;
    uint64_t ns_elapsed = get_time_ns() - t0;
    double cycles_per_op = (double)c_elapsed / 100000.0;
    double ns_per_op = (double)ns_elapsed / 100000.0;
    printf("      -> Serial Write & 96B SHA-256: %.2f cycles/frame (%.2f ns/frame)\n", cycles_per_op, ns_per_op);

    /* 4. Multi-Threaded SPSC Concurrent Benchmark */
    printf("[4/4] Benchmarking concurrent Multi-Threaded SPSC Handoff (%u ops)...\n", BENCHMARK_OPS);
    SPSCBenchmarkContext bench_ctx;
    bench_ctx.ring_base = ring_mem;
    bench_ctx.capacity = BENCHMARK_CAPACITY;
    bench_ctx.num_operations = BENCHMARK_OPS;
    atomic_init(&bench_ctx.ready, false);
    bench_ctx.total_push_cycles = 0;
    bench_ctx.total_pop_cycles = 0;
    bench_ctx.torn_reads_detected = 0;

    pthread_t p_tid, c_tid;
    pthread_create(&c_tid, NULL, consumer_thread, &bench_ctx);
    pthread_create(&p_tid, NULL, producer_thread, &bench_ctx);

    /* Signal both threads to start */
    uint64_t bench_start_ns = get_time_ns();
    atomic_store_explicit(&bench_ctx.ready, true, memory_order_release);

    pthread_join(p_tid, NULL);
    pthread_join(c_tid, NULL);
    uint64_t bench_total_ns = get_time_ns() - bench_start_ns;

    double total_sec = (double)bench_total_ns / 1000000000.0;
    double throughput = (double)BENCHMARK_OPS / (total_sec > 0 ? total_sec : 1e-9);
    double ns_per_handoff = (double)bench_total_ns / (double)BENCHMARK_OPS;

    printf("\n======================= BENCHMARK RESULTS =======================\n");
    printf("Total Operations:         %u frames (128 bytes each)\n", BENCHMARK_OPS);
    printf("Elapsed Time:             %.3f ms\n", (double)bench_total_ns / 1000000.0);
    printf("Throughput:               %.2f frames/second (%.2f MB/s)\n", throughput, (throughput * 128.0) / (1024.0 * 1024.0));
    printf("Average Handoff Latency:  %.2f ns/frame\n", ns_per_handoff);
    printf("Torn Reads Detected:      %" PRIu64 "\n", bench_ctx.torn_reads_detected);
    printf("Drop Count:               %" PRIu64 "\n", atomic_load_explicit(&hdr->drop_count, memory_order_relaxed));
    printf("Status:                   %s\n", (bench_ctx.torn_reads_detected == 0 ? "PASSED [HARDWARE-MEASURED]" : "FAILED"));
    printf("=================================================================\n");

#if defined(_WIN32)
    _aligned_free(ring_mem);
#else
    free(ring_mem);
#endif
    return (bench_ctx.torn_reads_detected == 0) ? 0 : 1;
}
