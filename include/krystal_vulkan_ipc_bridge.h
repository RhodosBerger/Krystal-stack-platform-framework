/**
 * ============================================================================
 * KRYSTAL-STACK NEXTGEN: VULKAN IPC ACCELERATION C-ABI BRIDGE
 * ============================================================================
 * Standard ANSI C / C++ exportable header.
 * Allows any non-hardware application (Node.js, Godot GDScript/C++, Python,
 * Electron, Rust, Go, or high-level scripting tools) to invoke hardware-
 * accelerated Vulkan compute kernels and SIMD tensor processing via zero-copy
 * shared memory and 20-byte packed binary packets.
 *
 * System Invariant: VITAL_MAX_HP = 6.
 *
 * Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
 * ============================================================================
 */

#ifndef KRYSTAL_VULKAN_IPC_BRIDGE_H
#define KRYSTAL_VULKAN_IPC_BRIDGE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef KRYSTAL_BUILD_DLL
    #define KRYSTAL_API __declspec(dllexport)
  #else
    #define KRYSTAL_API __declspec(dllimport)
  #endif
#else
  #define KRYSTAL_API __attribute__((visibility("default")))
#endif

#define KRYSTAL_MAGIC_BYTES      0x4B525953  /* "KRYS" */
#define KRYSTAL_PROTOCOL_VERSION 2
#define KRYSTAL_VITAL_MAX_HP     6

/* Operation Opcodes */
typedef enum KrystalIPCOpcode {
    KRYSTAL_OP_NOOP             = 0x00,
    KRYSTAL_OP_TENSOR_GEMM      = 0x01,  /* Hardware Matrix Multiply (INT8/FP16/FP32) */
    KRYSTAL_OP_TOKEN_INTERN     = 0x02,  /* SIMD Symbol Interning & Token Pooling */
    KRYSTAL_OP_AABB_CULL        = 0x03,  /* Subgroup SIMD16 Bounding Box Culling */
    KRYSTAL_OP_MATRIX_COMPRESS  = 0x04,  /* L1-Cache Matrix Repetition Compressor */
    KRYSTAL_OP_RAYMARCH_FRAME   = 0x05,  /* Vulkan Host-Visible Screen SDF Raymarching */
    KRYSTAL_OP_MAX              = 0x06
} KrystalIPCOpcode;

/* 20-byte Strictly Packed Binary IPC Header */
#pragma pack(push, 1)
typedef struct KrystalIPCPacketHeader {
    uint32_t magic;         /* 0x4B525953 ("KRYS") */
    uint32_t version;       /* Protocol version (2) */
    uint16_t vital_hp;      /* Invariant check: must be 6 */
    uint16_t opcode;        /* KrystalIPCOpcode */
    uint32_t payload_size;  /* Payload length in bytes */
    uint32_t reserved;      /* Alignment / future flags (0) */
} KrystalIPCPacketHeader;
#pragma pack(pop)

/* Hardware Bridge Telemetry Snapshot */
typedef struct KrystalIPCTelemetry {
    uint32_t vital_max_hp;
    uint32_t vulkan_device_detected;
    char     device_name[128];
    uint32_t compute_queue_family;
    uint64_t total_dispatches;
    double   last_dispatch_latency_us;
    double   ringbuffer_utilization_pct;
    double   shm_bandwidth_gb_s;
} KrystalIPCTelemetry;

/**
 * Initializes the Vulkan IPC Acceleration Bridge and maps the Named Shared Memory.
 * @param shm_ring_name   Name of shared memory ring (e.g. "krystal_vulkan_shm_ring")
 * @param ring_size_kb    Size of ringbuffer in Kilobytes (e.g. 1024 KB = 1 MB)
 * @return 0 on success, negative error code on failure.
 */
KRYSTAL_API int krystal_ipc_init(const char* shm_ring_name, uint32_t ring_size_kb);

/**
 * Dispatches an accelerated operation to the Vulkan Compute / SIMD backend.
 * @param opcode    KrystalIPCOpcode (e.g. KRYSTAL_OP_TENSOR_GEMM)
 * @param in_data   Pointer to input parameters/tensors
 * @param in_size   Size of input data in bytes
 * @param out_data  Caller-allocated buffer for results
 * @param out_size  In: buffer capacity, Out: actual result bytes written
 * @return 0 on success, negative error code on failure.
 */
KRYSTAL_API int krystal_ipc_dispatch(
    uint32_t opcode,
    const void* in_data,
    uint32_t in_size,
    void* out_data,
    uint32_t* out_size
);

/**
 * Queries real-time hardware telemetry and IPC throughput.
 */
KRYSTAL_API int krystal_ipc_query_telemetry(KrystalIPCTelemetry* out_telemetry);

/**
 * Closes the bridge and releases shared memory mapping.
 */
KRYSTAL_API void krystal_ipc_close(void);

#ifdef __cplusplus
}
#endif

#endif /* KRYSTAL_VULKAN_IPC_BRIDGE_H */
