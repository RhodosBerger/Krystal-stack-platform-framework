/*
 * Krystal Stack — C11 Hardware Instruction Set Diagrams & Vulkan/WebGL/Intel Arc
 * Rendering Pipeline Analyzer (FXAA -> SSAA -> TSAA -> GTAO -> Hardware Ray-Tracing)
 *
 * Compiles cleanly under:
 *   gcc / clang / zig cc -std=c11 -O3 -Wall -Wextra -Werror -Iinclude src/c/krystal_isa_vulkan_webgl_diagrams.c
 *
 * Constitutional Invariant: VITAL_MAX_HP == 6
 */

#include <stdint.h>
#include <stdio.h>

#define KRYSTAL_C11_VITAL_MAX_HP 6

typedef struct IsaPipelineStageRow {
    const char* process_stage;
    const char* primary_x86_isa;
    const char* gpu_arc_xe_isa;
    const char* webgl_vulkan_lib;
    uint32_t    register_bits;
    uint32_t    vital_hp_guard;
} IsaPipelineStageRow;

static const IsaPipelineStageRow KRYSTAL_ISA_TABLE[] = {
    {
        "1. Zero-Copy Ring & Dual-Bank Lock",
        "LOCK CMPXCHG + TZCNT + RDTSC",
        "VK_KHR_buffer_device_address (128B)",
        "Vulkan 1.4 Native / C11 SPSC",
        128u,
        KRYSTAL_C11_VITAL_MAX_HP
    },
    {
        "2. Subnormal Purge & Matrix Prefetch",
        "AVX-512F + FMA + PREFETCHT0 (64B)",
        "Xe Subgroup SIMD16/32 ShuffleXor",
        "OpenVINO USM + Mesa Gallium NIR",
        512u,
        KRYSTAL_C11_VITAL_MAX_HP
    },
    {
        "3. 2B Gemma INT8/FP16 & Crypto Attest",
        "AVX-512 VNNI (VPDPBUSD) + SHA-NI + VAES",
        "Intel Arc XMX DPAS / Iris Xe DP4A",
        "OpenVINO GPU + WebGPU / WASM SIMD128",
        512u,
        KRYSTAL_C11_VITAL_MAX_HP
    },
    {
        "4. AA & AO Ladder (FXAA -> TSAA + GTAO)",
        "AVX2 Gather + Non-Temporal MOVNTI",
        "Xe Sampler Feedback + 64KB Subslice SLM",
        "ANGLE (WebGL2->Vulkan) / wgpu-native",
        256u,
        KRYSTAL_C11_VITAL_MAX_HP
    },
    {
        "5. Final Stage Predictive Ray-Tracing",
        "AVX-512 BVH4 Wide Box-Ray Intersect",
        "Intel Arc RTU (VK_KHR_ray_query + XMX)",
        "Vulkan Ray-Query / Babylon WebGPU",
        512u,
        KRYSTAL_C11_VITAL_MAX_HP
    }
};

_Static_assert(KRYSTAL_C11_VITAL_MAX_HP == 6, "Constitutional VITAL_MAX_HP == 6 violated");

int main(void) {
    const size_t count = sizeof(KRYSTAL_ISA_TABLE) / sizeof(KRYSTAL_ISA_TABLE[0]);
    printf("+-----------------------------------------+-----------------------------------------+-----------------------------------------+--------------------------------------+\n");
    printf("| PROCES / RENDER STUPEN (VITAL_HP == 6)  | CPU INSTRUKCNA SADA (WILLOW COVE)       | GPU INSTRUKCNA SADA (IRIS XE / ARC)     | VULKAN / WEBGL AKCELERACNA KNIZNICA  |\n");
    printf("+-----------------------------------------+-----------------------------------------+-----------------------------------------+--------------------------------------+\n");
    for (size_t i = 0; i < count; ++i) {
        if (KRYSTAL_ISA_TABLE[i].vital_hp_guard != 6u) {
            return 6;
        }
        printf(
            "| %-39s | %-39s | %-39s | %-36s |\n",
            KRYSTAL_ISA_TABLE[i].process_stage,
            KRYSTAL_ISA_TABLE[i].primary_x86_isa,
            KRYSTAL_ISA_TABLE[i].gpu_arc_xe_isa,
            KRYSTAL_ISA_TABLE[i].webgl_vulkan_lib
        );
    }
    printf("+-----------------------------------------+-----------------------------------------+-----------------------------------------+--------------------------------------+\n");
    return 0;
}
