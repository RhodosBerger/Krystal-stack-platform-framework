/*
 * KRYSTAL-NEXUS C11 ABI SMOKE TEST TRANSLATION UNIT
 * ============================================================================
 * Exercises krystal_nexus_api.h under strict compiler flags:
 *   -std=c11 -Wall -Wextra -Werror -pedantic
 * Exercises layout, static asserts, publish, and consume.
 * Platform invariant: KRYSTAL_VITAL_MAX_HP == 6
 */

#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "krystal_nexus_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

int main(void) {
    /* 1. Invariant check */
    _Static_assert(KRYSTAL_VITAL_MAX_HP == 6, "Platform invariant breach");

    /* 2. Instantiate and zero-initialize contract structures */
    KrystalZeroCopyFrame frame;
    memset(&frame, 0, sizeof(frame));
    frame.magic = KRYSTAL_MAGIC;
    frame.version = KRYSTAL_PROTOCOL_VERSION;
    frame.vital_hp = KRYSTAL_VITAL_MAX_HP;
    frame.contract_type = 1;
    frame.payload_size = 16;
    frame.gpu_render_ms = 8.33f;
    frame.drs_scale_factor = 1.0f;

    KrystalRingHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = KRYSTAL_MAGIC;
    hdr.vital_hp = KRYSTAL_VITAL_MAX_HP;
    hdr.capacity = 16;
    hdr.slot_size = KRYSTAL_FRAME_SIZE_BYTES;

    /* 3. Allocate 128B-aligned ring memory (RingBaseAddress % 128 == 0) */
    void* ring_mem = NULL;
    size_t total_ring_bytes = KRYSTAL_RING_HEADER_BYTES + ((size_t)hdr.capacity * KRYSTAL_FRAME_SIZE_BYTES);

#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    if (posix_memalign(&ring_mem, 128, total_ring_bytes) != 0) {
        return 1;
    }
#else
    ring_mem = malloc(total_ring_bytes + 128);
    if (!ring_mem) return 1;
    uintptr_t addr = (uintptr_t)ring_mem;
    ring_mem = (void*)((addr + 127) & ~(uintptr_t)127);
#endif

    /* 4. Exercise and verify type contracts of publish and consume */
    bool pub_ok = krystal_ring_publish(&hdr, ring_mem, &frame);
    if (!pub_ok) {
        return 2;
    }

    KrystalZeroCopyFrame out_frame;
    memset(&out_frame, 0, sizeof(out_frame));
    bool con_ok = krystal_ring_consume(&hdr, ring_mem, &out_frame);
    if (!con_ok) {
        return 3;
    }

    if (out_frame.vital_hp != 6 || out_frame.magic != KRYSTAL_MAGIC) {
        return 4;
    }

    return 0;
}
