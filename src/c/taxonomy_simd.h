#ifndef TAXONOMY_SIMD_H
#define TAXONOMY_SIMD_H

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
  #define TAXONOMY_API __declspec(dllexport)
#else
  #define TAXONOMY_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * CrystalSec 3.0 Phase 5.3C / 5.3D - Native CPU SIMD Taxonomy Kernels
 *
 * Canonical Numerical Contract:
 * - INT8 Canonical Semantics:
 *     input A: uint8_t activations
 *     input B: int8_t weights
 *     accumulator: int32_t
 *     operation: sum(A[i] * B[i]) - pure dot product accumulation
 *     post-op activation: explicit and separate (e.g. ReLU max(0, acc))
 * - FP32_VECTOR_MATH: Horner form cubic polynomial y = ((c3*x + c2)*x + c1)*x + c0
 * - MEMORY_BOUND_STREAM: strided access out[i] = buffer[i * stride] * 1.001f
 * - BRANCH_HEAVY_CONTROL: genuine scalar branch tree (testing branch predictor)
 * - MIXED_COMPUTE_MEMORY: paired dot-product and stream traversal
 */

typedef struct {
    uint32_t sse42;
    uint32_t avx;
    uint32_t avx2;
    uint32_t fma;
    uint32_t avx512f;
    uint32_t avx512bw;
    uint32_t avx512vl;
    uint32_t avx512_vnni;
    uint32_t os_avx_state;
    uint32_t os_avx512_state;
} CpuNativeCapabilities;

typedef struct {
    uint32_t selected_path;     /* 1=SCALAR, 2=AVX2, 3=AVX512, 4=VNNI */
    uint32_t status_code;       /* 0=SUCCESS, non-zero=ERROR */
    uint64_t cycles_elapsed;
    double elapsed_ms;
} TaxonomyExecutionInfo;

#define TAXONOMY_PATH_AUTO     0
#define TAXONOMY_PATH_SCALAR   1
#define TAXONOMY_PATH_AVX2     2
#define TAXONOMY_PATH_AVX512   3
#define TAXONOMY_PATH_VNNI     4

#define TAXONOMY_STATUS_OK                 0
#define TAXONOMY_STATUS_INVALID_ARG       -1
#define TAXONOMY_STATUS_UNSUPPORTED_ISA   -2
#define TAXONOMY_STATUS_EXECUTION_FAILED  -3

TAXONOMY_API int taxonomy_get_capabilities(CpuNativeCapabilities* out_caps);

/* Pure canonical dot-product reduction: acc = sum(activations[i] * weights[i]) */
TAXONOMY_API int taxonomy_int8_dot_reduction(
    const uint8_t* activations,
    const int8_t* weights,
    size_t n,
    int32_t* out_accumulator,
    int apply_relu_post_op,
    int requested_path,
    TaxonomyExecutionInfo* info
);

/* Vectorized INT8 product: out[i] = (int32_t)activations[i] * (int32_t)weights[i] */
TAXONOMY_API int taxonomy_int8_dot(
    const uint8_t* activations,
    const int8_t* weights,
    float* out,
    size_t n,
    int requested_path,
    TaxonomyExecutionInfo* info
);

TAXONOMY_API int taxonomy_fp32_poly(
    const float* inputs,
    const float* coeffs,
    float* out,
    size_t n,
    int requested_path,
    TaxonomyExecutionInfo* info
);

TAXONOMY_API int taxonomy_memory_stream(
    const float* buffer,
    size_t total_elements,
    size_t stride,
    float* out,
    size_t out_count,
    int requested_path,
    TaxonomyExecutionInfo* info
);

TAXONOMY_API int taxonomy_branch_control(
    const int32_t* keys,
    float* out,
    size_t n,
    int requested_path,
    TaxonomyExecutionInfo* info
);

TAXONOMY_API int taxonomy_mixed_compute_memory(
    const uint8_t* activations,
    const int8_t* weights,
    float* out,
    size_t n,
    int requested_path,
    TaxonomyExecutionInfo* info
);

TAXONOMY_API int taxonomy_last_execution_info(TaxonomyExecutionInfo* out_info);

#ifdef __cplusplus
}
#endif

#endif /* TAXONOMY_SIMD_H */
