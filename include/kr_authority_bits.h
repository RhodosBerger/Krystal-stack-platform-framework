/* AUTOMATICALLY GENERATED FROM specs/authority_profile_bits_v1.yaml. DO NOT EDIT DIRECTLY. */
#ifndef KR_AUTHORITY_BITS_H
#define KR_AUTHORITY_BITS_H

#include <stdint.h>

typedef uint64_t kr_authority_mask;

enum kr_profile_bit {
    KR_PROFILE_NONE = 0ULL,
    KR_PROFILE_ULTRA_SAVER = (1ULL << 0), /* 0x0000000000000001 */
    KR_PROFILE_SAVER_NPU = (1ULL << 1), /* 0x0000000000000002 */
    KR_PROFILE_BALANCED_NPU = (1ULL << 2), /* 0x0000000000000004 */
    KR_PROFILE_BALANCED_HYBRID = (1ULL << 3), /* 0x0000000000000008 */
    KR_PROFILE_PERFORMANCE_GPU = (1ULL << 4), /* 0x0000000000000010 */
    KR_PROFILE_MAX_BURST = (1ULL << 5), /* 0x0000000000000020 */
};

typedef struct {
    kr_authority_mask requested;
    kr_authority_mask thermal_safe;
    kr_authority_mask battery_safe;
    kr_authority_mask capability_supported;
    kr_authority_mask evidence_supported;
    kr_authority_mask policy_allowed;
} kr_authority_input;

static inline kr_authority_mask kr_reduce_authority(const kr_authority_input *in) {
    return in->requested
         & in->thermal_safe
         & in->battery_safe
         & in->capability_supported
         & in->evidence_supported
         & in->policy_allowed;
}

#endif /* KR_AUTHORITY_BITS_H */
