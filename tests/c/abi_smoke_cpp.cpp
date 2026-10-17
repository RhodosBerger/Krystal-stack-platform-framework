/*
 * KRYSTAL-NEXUS C++20 ABI SMOKE TEST TRANSLATION UNIT
 * ============================================================================
 * Verifies that krystal_nexus_api.h compiles cleanly as modern C++ (-std=c++20)
 * for Unreal Engine 5, Unity native plugins, and modern C++ game engines.
 * Platform invariant: KRYSTAL_VITAL_MAX_HP == 6
 */

#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "krystal_nexus_api.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdint>

int main() {
    static_assert(KRYSTAL_VITAL_MAX_HP == 6, "Platform invariant breach");
    static_assert(sizeof(KrystalZeroCopyFrame) == 128, "Frame must be 128B in C++");
    static_assert(alignof(KrystalZeroCopyFrame) == 64, "Frame must be 64B aligned in C++");
    static_assert(sizeof(KrystalRingHeader) == 128, "RingHeader must be 128B in C++");
    static_assert(alignof(KrystalRingHeader) == 64, "RingHeader must be 64B aligned in C++");

    KrystalZeroCopyFrame frame{};
    frame.magic = KRYSTAL_MAGIC;
    frame.version = KRYSTAL_PROTOCOL_VERSION;
    frame.vital_hp = KRYSTAL_VITAL_MAX_HP;
    frame.contract_type = 1;
    frame.payload_size = 16;
    frame.gpu_render_ms = 8.33f;
    frame.drs_scale_factor = 1.0f;

    KrystalRingHeader hdr{};
    hdr.magic = KRYSTAL_MAGIC;
    hdr.vital_hp = KRYSTAL_VITAL_MAX_HP;
    hdr.capacity = 16;
    hdr.slot_size = KRYSTAL_FRAME_SIZE_BYTES;

    size_t total_ring_bytes = KRYSTAL_RING_HEADER_BYTES + (static_cast<size_t>(hdr.capacity) * KRYSTAL_FRAME_SIZE_BYTES);
    void* ring_mem = nullptr;

#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    if (posix_memalign(&ring_mem, 128, total_ring_bytes) != 0) {
        return 1;
    }
#else
    ring_mem = std::malloc(total_ring_bytes + 128);
    if (!ring_mem) return 1;
    uintptr_t addr = reinterpret_cast<uintptr_t>(ring_mem);
    ring_mem = reinterpret_cast<void*>((addr + 127) & ~static_cast<uintptr_t>(127));
#endif

    bool pub_ok = krystal_ring_publish(&hdr, ring_mem, &frame);
    if (!pub_ok) {
        return 2;
    }

    KrystalZeroCopyFrame out_frame{};
    bool con_ok = krystal_ring_consume(&hdr, ring_mem, &out_frame);
    if (!con_ok) {
        return 3;
    }

    if (out_frame.vital_hp != 6 || out_frame.magic != KRYSTAL_MAGIC) {
        return 4;
    }

    return 0;
}
