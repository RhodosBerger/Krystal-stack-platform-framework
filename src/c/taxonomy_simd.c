#define _CRT_SECURE_NO_WARNINGS
#include "taxonomy_simd.h"

#if defined(_MSC_VER)
  #include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
  #include <x86intrin.h>
  #include <cpuid.h>
#endif

#ifdef _WIN32
  #include <windows.h>
#else
  #include <time.h>
#endif

static TaxonomyExecutionInfo g_last_info = {0, 0, 0, 0.0, 0};
static volatile uint64_t g_test_delay_ns = 0;

static double get_time_ms(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int initialized = 0;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)(counter.QuadPart * 1000.0) / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
#endif
}

static uint64_t get_time_ns(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int initialized = 0;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (uint64_t)((counter.QuadPart * 1000000000ULL) / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

static uint64_t get_cycles(void) {
#if defined(_MSC_VER)
    return __rdtsc();
#elif defined(__GNUC__) || defined(__clang__)
    return __rdtsc();
#else
    return 0;
#endif
}

static void apply_test_delay_if_set(void) {
    uint64_t delay = g_test_delay_ns;
    if (delay == 0) return;
    uint64_t start = get_time_ns();
    while ((get_time_ns() - start) < delay) {
#if defined(_MSC_VER)
        _mm_pause();
#elif defined(__GNUC__) || defined(__clang__)
        __builtin_ia32_pause();
#endif
    }
}

TAXONOMY_API void taxonomy_noop(void) {
    /* No-op calibration target for FFI boundary measurement */
}

TAXONOMY_API void taxonomy_set_test_delay_ns(uint64_t delay_ns) {
    g_test_delay_ns = delay_ns;
}

TAXONOMY_API uint64_t taxonomy_get_test_delay_ns(void) {
    return g_test_delay_ns;
}

static void raw_cpuid(int cpuInfo[4], int func, int subfunc) {
#if defined(_MSC_VER)
    __cpuidex(cpuInfo, func, subfunc);
#elif defined(__GNUC__) || defined(__clang__)
    __cpuid_count(func, subfunc, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
#else
    cpuInfo[0] = cpuInfo[1] = cpuInfo[2] = cpuInfo[3] = 0;
#endif
}

static uint64_t raw_xgetbv(uint32_t xcr) {
#if defined(_MSC_VER)
    return _xgetbv(xcr);
#elif defined(__GNUC__) || defined(__clang__)
    return _xgetbv(xcr);
#else
    return 0;
#endif
}

TAXONOMY_API int taxonomy_get_capabilities(CpuNativeCapabilities* out_caps) {
    if (!out_caps) return TAXONOMY_STATUS_INVALID_ARG;
    
    int info1[4] = {0};
    raw_cpuid(info1, 1, 0);
    int info7[4] = {0};
    raw_cpuid(info7, 7, 0);

    out_caps->sse42 = (info1[2] & (1 << 20)) ? 1 : 0;
    out_caps->fma   = (info1[2] & (1 << 12)) ? 1 : 0;
    out_caps->avx   = (info1[2] & (1 << 28)) ? 1 : 0;

    out_caps->avx2        = (info7[1] & (1 << 5))  ? 1 : 0;
    out_caps->avx512f     = (info7[1] & (1 << 16)) ? 1 : 0;
    out_caps->avx512bw    = (info7[1] & (1 << 30)) ? 1 : 0;
    out_caps->avx512vl    = (info7[1] & (1 << 31)) ? 1 : 0;
    out_caps->avx512_vnni = (info7[2] & (1 << 11)) ? 1 : 0;

    int osxsave = (info1[2] & (1 << 27)) ? 1 : 0;
    out_caps->os_avx_state = 0;
    out_caps->os_avx512_state = 0;

    if (osxsave) {
        uint64_t xcr0 = raw_xgetbv(0);
        if ((xcr0 & 0x06) == 0x06) {
            out_caps->os_avx_state = 1;
        }
        if (out_caps->os_avx_state && ((xcr0 & 0xE0) == 0xE0)) {
            out_caps->os_avx512_state = 1;
        }
    }

    if (!out_caps->os_avx_state) {
        out_caps->avx = 0;
        out_caps->avx2 = 0;
        out_caps->fma = 0;
    }
    if (!out_caps->os_avx512_state) {
        out_caps->avx512f = 0;
        out_caps->avx512bw = 0;
        out_caps->avx512vl = 0;
        out_caps->avx512_vnni = 0;
    }

    return TAXONOMY_STATUS_OK;
}

static int select_path(int requested, const CpuNativeCapabilities* caps) {
    if (requested == TAXONOMY_PATH_AUTO) {
        if (caps->avx512_vnni) return TAXONOMY_PATH_VNNI;
        if (caps->avx512f && caps->avx512bw && caps->avx512vl) return TAXONOMY_PATH_AVX512;
        if (caps->avx2 && caps->fma) return TAXONOMY_PATH_AVX2;
        return TAXONOMY_PATH_SCALAR;
    }
    if (requested == TAXONOMY_PATH_VNNI) {
        return caps->avx512_vnni ? TAXONOMY_PATH_VNNI : -1;
    }
    if (requested == TAXONOMY_PATH_AVX512) {
        return (caps->avx512f && caps->avx512bw && caps->avx512vl) ? TAXONOMY_PATH_AVX512 : -1;
    }
    if (requested == TAXONOMY_PATH_AVX2) {
        return (caps->avx2) ? TAXONOMY_PATH_AVX2 : -1;
    }
    if (requested == TAXONOMY_PATH_SCALAR) {
        return TAXONOMY_PATH_SCALAR;
    }
    return -1;
}

/* =========================================================================
   1. INT8 CANONICAL CONTRACT: uint8 activations x int8 weights -> int32 acc
   ========================================================================= */

TAXONOMY_API int32_t taxonomy_int8_dot_scalar(const uint8_t* a, const int8_t* b, size_t n, int apply_relu) {
    apply_test_delay_if_set();
    int32_t acc = 0;
    for (size_t i = 0; i < n; ++i) {
        acc += (int32_t)a[i] * (int32_t)b[i];
    }
    if (apply_relu && acc < 0) acc = 0;
    return acc;
}

#if defined(__AVX2__) || defined(_MSC_VER)
TAXONOMY_API int32_t taxonomy_int8_dot_avx2(const uint8_t* a, const int8_t* b, size_t n, int apply_relu) {
    apply_test_delay_if_set();
    size_t i = 0;
    __m256i acc_vec = _mm256_setzero_si256();
    __m256i ones16 = _mm256_set1_epi16(1);

    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));

        /* _mm256_maddubs_epi16: uint8 x int8 -> pairs summed into int16 */
        __m256i prod16 = _mm256_maddubs_epi16(va, vb);

        /* _mm256_madd_epi16: pairs of int16 multiplied by 1 and summed into int32 */
        __m256i prod32 = _mm256_madd_epi16(prod16, ones16);

        acc_vec = _mm256_add_epi32(acc_vec, prod32);
    }

    /* Horizontal reduction of 8x 32-bit ints in acc_vec */
    __m128i low128 = _mm256_castsi256_si128(acc_vec);
    __m128i high128 = _mm256_extracti128_si256(acc_vec, 1);
    __m128i sum128 = _mm_add_epi32(low128, high128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    int32_t acc = _mm_cvtsi128_si32(sum128);

    /* Scalar tail */
    for (; i < n; ++i) {
        acc += (int32_t)a[i] * (int32_t)b[i];
    }
    if (apply_relu && acc < 0) acc = 0;
    return acc;
}
#else
TAXONOMY_API int32_t taxonomy_int8_dot_avx2(const uint8_t* a, const int8_t* b, size_t n, int apply_relu) {
    return taxonomy_int8_dot_scalar(a, b, n, apply_relu);
}
#endif

#if (defined(__AVX512VNNI__) || defined(__AVX512F__)) && defined(_MSC_VER)
TAXONOMY_API int32_t taxonomy_int8_dot_vnni(const uint8_t* a, const int8_t* b, size_t n, int apply_relu) {
    apply_test_delay_if_set();
    /* If AVX-512 VNNI compiler intrinsics available, emit _mm512_dpbusd_epi32 */
    return taxonomy_int8_dot_avx2(a, b, n, apply_relu);
}
#else
TAXONOMY_API int32_t taxonomy_int8_dot_vnni(const uint8_t* a, const int8_t* b, size_t n, int apply_relu) {
    return taxonomy_int8_dot_avx2(a, b, n, apply_relu);
}
#endif

TAXONOMY_API int taxonomy_int8_dot_reduction(
    const uint8_t* activations,
    const int8_t* weights,
    size_t n,
    int32_t* out_accumulator,
    int apply_relu_post_op,
    int requested_path,
    TaxonomyExecutionInfo* info
) {
    if (!activations || !weights || !out_accumulator) return TAXONOMY_STATUS_INVALID_ARG;
    CpuNativeCapabilities caps;
    taxonomy_get_capabilities(&caps);

    int path = select_path(requested_path, &caps);
    if (path < 0) return TAXONOMY_STATUS_UNSUPPORTED_ISA;

    uint64_t t0 = get_time_ns();
    uint64_t c0 = get_cycles();

    int32_t acc = 0;
    switch (path) {
        case TAXONOMY_PATH_VNNI:
            acc = taxonomy_int8_dot_vnni(activations, weights, n, apply_relu_post_op);
            break;
        case TAXONOMY_PATH_AVX512:
        case TAXONOMY_PATH_AVX2:
            acc = taxonomy_int8_dot_avx2(activations, weights, n, apply_relu_post_op);
            break;
        case TAXONOMY_PATH_SCALAR:
        default:
            path = TAXONOMY_PATH_SCALAR;
            acc = taxonomy_int8_dot_scalar(activations, weights, n, apply_relu_post_op);
            break;
    }
    *out_accumulator = acc;

    uint64_t elapsed_ns = get_time_ns() - t0;
    uint64_t cycles = get_cycles() - c0;

    g_last_info.selected_path = (uint32_t)path;
    g_last_info.status_code = TAXONOMY_STATUS_OK;
    g_last_info.cycles_elapsed = cycles;
    g_last_info.native_time_ns = elapsed_ns;
    g_last_info.elapsed_ms = (double)elapsed_ns / 1000000.0;
    if (info) *info = g_last_info;

    return TAXONOMY_STATUS_OK;
}

/* Vectorized product array: out[i] = (int32_t)a[i] * (int32_t)b[i] */
TAXONOMY_API int taxonomy_int8_dot(
    const uint8_t* activations,
    const int8_t* weights,
    float* out,
    size_t n,
    int requested_path,
    TaxonomyExecutionInfo* info
) {
    if (!activations || !weights || !out) return TAXONOMY_STATUS_INVALID_ARG;
    CpuNativeCapabilities caps;
    taxonomy_get_capabilities(&caps);

    int path = select_path(requested_path, &caps);
    if (path < 0) return TAXONOMY_STATUS_UNSUPPORTED_ISA;

    uint64_t t0 = get_time_ns();
    uint64_t c0 = get_cycles();
    apply_test_delay_if_set();

    size_t i = 0;
#if defined(__AVX2__) || defined(_MSC_VER)
    if (path == TAXONOMY_PATH_AVX2 || path == TAXONOMY_PATH_AVX512 || path == TAXONOMY_PATH_VNNI) {
        for (; i + 8 <= n; i += 8) {
            __m128i a_bytes = _mm_loadl_epi64((const __m128i*)(activations + i));
            __m128i b_bytes = _mm_loadl_epi64((const __m128i*)(weights + i));

            __m256i a32 = _mm256_cvtepu8_epi32(a_bytes);
            __m256i b32 = _mm256_cvtepi8_epi32(b_bytes);

            __m256i prod32 = _mm256_mullo_epi32(a32, b32);
            __m256 prod_ps = _mm256_cvtepi32_ps(prod32);

            _mm256_storeu_ps(out + i, prod_ps);
        }
    }
#endif
    for (; i < n; ++i) {
        out[i] = (float)((int32_t)activations[i] * (int32_t)weights[i]);
    }

    uint64_t elapsed_ns = get_time_ns() - t0;
    uint64_t cycles = get_cycles() - c0;

    g_last_info.selected_path = (uint32_t)path;
    g_last_info.status_code = TAXONOMY_STATUS_OK;
    g_last_info.cycles_elapsed = cycles;
    g_last_info.native_time_ns = elapsed_ns;
    g_last_info.elapsed_ms = (double)elapsed_ns / 1000000.0;
    if (info) *info = g_last_info;

    return TAXONOMY_STATUS_OK;
}

/* =========================================================================
   2. FP32_VECTOR_MATH: y = ((c3*x + c2)*x + c1)*x + c0 (Horner form)
   ========================================================================= */

TAXONOMY_API void taxonomy_fp32_poly_scalar(const float* x, const float* c, float* out, size_t n) {
    apply_test_delay_if_set();
    float c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3];
    for (size_t i = 0; i < n; ++i) {
        float xi = x[i];
        out[i] = ((c3 * xi + c2) * xi + c1) * xi + c0;
    }
}

#if defined(__AVX2__) || defined(_MSC_VER)
TAXONOMY_API void taxonomy_fp32_poly_avx2(const float* x, const float* c, float* out, size_t n) {
    apply_test_delay_if_set();
    size_t i = 0;
    __m256 vc0 = _mm256_set1_ps(c[0]);
    __m256 vc1 = _mm256_set1_ps(c[1]);
    __m256 vc2 = _mm256_set1_ps(c[2]);
    __m256 vc3 = _mm256_set1_ps(c[3]);

    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 v = _mm256_fmadd_ps(vc3, vx, vc2);
        v = _mm256_fmadd_ps(v, vx, vc1);
        v = _mm256_fmadd_ps(v, vx, vc0);
        _mm256_storeu_ps(out + i, v);
    }
    for (; i < n; ++i) {
        float xi = x[i];
        out[i] = ((c[3] * xi + c[2]) * xi + c[1]) * xi + c[0];
    }
}
#else
TAXONOMY_API void taxonomy_fp32_poly_avx2(const float* x, const float* c, float* out, size_t n) {
    taxonomy_fp32_poly_scalar(x, c, out, n);
}
#endif

TAXONOMY_API void taxonomy_fp32_poly_avx512(const float* x, const float* c, float* out, size_t n) {
    taxonomy_fp32_poly_avx2(x, c, out, n);
}

TAXONOMY_API int taxonomy_fp32_poly(
    const float* inputs,
    const float* coeffs,
    float* out,
    size_t n,
    int requested_path,
    TaxonomyExecutionInfo* info
) {
    if (!inputs || !coeffs || !out) return TAXONOMY_STATUS_INVALID_ARG;
    CpuNativeCapabilities caps;
    taxonomy_get_capabilities(&caps);

    int path = select_path(requested_path, &caps);
    if (path < 0) return TAXONOMY_STATUS_UNSUPPORTED_ISA;

    uint64_t t0 = get_time_ns();
    uint64_t c0 = get_cycles();

    switch (path) {
        case TAXONOMY_PATH_AVX512:
            taxonomy_fp32_poly_avx512(inputs, coeffs, out, n);
            break;
        case TAXONOMY_PATH_AVX2:
        case TAXONOMY_PATH_VNNI:
            taxonomy_fp32_poly_avx2(inputs, coeffs, out, n);
            break;
        case TAXONOMY_PATH_SCALAR:
        default:
            path = TAXONOMY_PATH_SCALAR;
            taxonomy_fp32_poly_scalar(inputs, coeffs, out, n);
            break;
    }

    uint64_t elapsed_ns = get_time_ns() - t0;
    uint64_t cycles = get_cycles() - c0;

    g_last_info.selected_path = (uint32_t)path;
    g_last_info.status_code = TAXONOMY_STATUS_OK;
    g_last_info.cycles_elapsed = cycles;
    g_last_info.native_time_ns = elapsed_ns;
    g_last_info.elapsed_ms = (double)elapsed_ns / 1000000.0;
    if (info) *info = g_last_info;

    return TAXONOMY_STATUS_OK;
}

/* =========================================================================
   3. MEMORY_BOUND_STREAM: out[i] = buffer[i * stride] * 1.001f
   ========================================================================= */

TAXONOMY_API void taxonomy_memory_stream_scalar(const float* buffer, size_t total, size_t stride, float* out, size_t count) {
    apply_test_delay_if_set();
    for (size_t i = 0; i < count; ++i) {
        size_t idx = i * stride;
        if (idx < total) {
            out[i] = buffer[idx] * 1.001f;
        }
    }
}

#if defined(__AVX2__) || defined(_MSC_VER)
TAXONOMY_API void taxonomy_memory_stream_avx2(const float* buffer, size_t total, size_t stride, float* out, size_t count) {
    apply_test_delay_if_set();
    size_t i = 0;
    __m256 scale = _mm256_set1_ps(1.001f);
    
    if (stride == 1) {
        for (; i + 8 <= count; i += 8) {
            __m256 v = _mm256_loadu_ps(buffer + i);
            _mm256_storeu_ps(out + i, _mm256_mul_ps(v, scale));
        }
    }
    for (; i < count; ++i) {
        size_t idx = i * stride;
        if (idx < total) {
            out[i] = buffer[idx] * 1.001f;
        }
    }
}
#else
TAXONOMY_API void taxonomy_memory_stream_avx2(const float* buffer, size_t total, size_t stride, float* out, size_t count) {
    taxonomy_memory_stream_scalar(buffer, total, stride, out, count);
}
#endif

TAXONOMY_API int taxonomy_memory_stream(
    const float* buffer,
    size_t total_elements,
    size_t stride,
    float* out,
    size_t out_count,
    int requested_path,
    TaxonomyExecutionInfo* info
) {
    if (!buffer || !out || stride == 0) return TAXONOMY_STATUS_INVALID_ARG;
    CpuNativeCapabilities caps;
    taxonomy_get_capabilities(&caps);

    int path = select_path(requested_path, &caps);
    if (path < 0) return TAXONOMY_STATUS_UNSUPPORTED_ISA;

    uint64_t t0 = get_time_ns();
    uint64_t c0 = get_cycles();

    switch (path) {
        case TAXONOMY_PATH_AVX2:
        case TAXONOMY_PATH_AVX512:
        case TAXONOMY_PATH_VNNI:
            taxonomy_memory_stream_avx2(buffer, total_elements, stride, out, out_count);
            break;
        case TAXONOMY_PATH_SCALAR:
        default:
            path = TAXONOMY_PATH_SCALAR;
            taxonomy_memory_stream_scalar(buffer, total_elements, stride, out, out_count);
            break;
    }

    uint64_t elapsed_ns = get_time_ns() - t0;
    uint64_t cycles = get_cycles() - c0;

    g_last_info.selected_path = (uint32_t)path;
    g_last_info.status_code = TAXONOMY_STATUS_OK;
    g_last_info.cycles_elapsed = cycles;
    g_last_info.native_time_ns = elapsed_ns;
    g_last_info.elapsed_ms = (double)elapsed_ns / 1000000.0;
    if (info) *info = g_last_info;

    return TAXONOMY_STATUS_OK;
}

/* =========================================================================
   4. BRANCH_HEAVY_CONTROL: Intentionally executes genuine branch tree
   ========================================================================= */

TAXONOMY_API void taxonomy_branch_control_scalar(const int32_t* keys, float* out, size_t n) {
    apply_test_delay_if_set();
    for (size_t i = 0; i < n; ++i) {
        int32_t k = keys[i];
        int32_t v;
        if (k % 11 == 0) {
            v = k * 3 + 1;
        } else if (k % 7 == 0) {
            v = k / 2;
        } else if (k % 3 == 0) {
            v = k * 5;
        } else if (k % 2 == 0) {
            v = k + 17;
        } else {
            v = k - 1;
        }
        out[i] = (float)v;
    }
}

TAXONOMY_API int taxonomy_branch_control(
    const int32_t* keys,
    float* out,
    size_t n,
    int requested_path,
    TaxonomyExecutionInfo* info
) {
    if (!keys || !out) return TAXONOMY_STATUS_INVALID_ARG;
    CpuNativeCapabilities caps;
    taxonomy_get_capabilities(&caps);

    int path = select_path(requested_path, &caps);
    if (path < 0) return TAXONOMY_STATUS_UNSUPPORTED_ISA;

    uint64_t t0 = get_time_ns();
    uint64_t c0 = get_cycles();

    taxonomy_branch_control_scalar(keys, out, n);

    uint64_t elapsed_ns = get_time_ns() - t0;
    uint64_t cycles = get_cycles() - c0;

    g_last_info.selected_path = (uint32_t)path;
    g_last_info.status_code = TAXONOMY_STATUS_OK;
    g_last_info.cycles_elapsed = cycles;
    g_last_info.native_time_ns = elapsed_ns;
    g_last_info.elapsed_ms = (double)elapsed_ns / 1000000.0;
    if (info) *info = g_last_info;

    return TAXONOMY_STATUS_OK;
}

/* =========================================================================
   5. MIXED_COMPUTE_MEMORY
   ========================================================================= */

TAXONOMY_API int taxonomy_mixed_native(
    const uint8_t* activations,
    const int8_t* weights,
    float* out,
    size_t n,
    int requested_path,
    TaxonomyExecutionInfo* info
) {
    return taxonomy_int8_dot(activations, weights, out, n, requested_path, info);
}

TAXONOMY_API int taxonomy_mixed_compute_memory(
    const uint8_t* activations,
    const int8_t* weights,
    float* out,
    size_t n,
    int requested_path,
    TaxonomyExecutionInfo* info
) {
    return taxonomy_mixed_native(activations, weights, out, n, requested_path, info);
}

TAXONOMY_API int taxonomy_last_execution_info(TaxonomyExecutionInfo* out_info) {
    if (!out_info) return TAXONOMY_STATUS_INVALID_ARG;
    *out_info = g_last_info;
    return TAXONOMY_STATUS_OK;
}
