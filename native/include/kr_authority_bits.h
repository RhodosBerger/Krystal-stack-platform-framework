#ifndef KRYSTAL_AUTHORITY_BITS_H
#define KRYSTAL_AUTHORITY_BITS_H

#include <stdint.h>

typedef uint64_t kr_authority_mask;

enum kr_profile_bit {
    KR_PROFILE_ULTRA_SAVER      = 1ull << 0,
    KR_PROFILE_SAVER_NPU        = 1ull << 1,
    KR_PROFILE_BALANCED_NPU     = 1ull << 2,
    KR_PROFILE_BALANCED_HYBRID  = 1ull << 3,
    KR_PROFILE_PERFORMANCE_GPU  = 1ull << 4,
    KR_PROFILE_MAX_BURST        = 1ull << 5
};

#define KR_PROFILE_MASK_ALL ((kr_authority_mask)0x3full)

static inline kr_authority_mask kr_reduce_authority(
    kr_authority_mask requested,
    kr_authority_mask thermal_safe,
    kr_authority_mask battery_safe,
    kr_authority_mask capability_supported,
    kr_authority_mask evidence_supported,
    kr_authority_mask policy_allowed)
{
    return requested
         & thermal_safe
         & battery_safe
         & capability_supported
         & evidence_supported
         & policy_allowed;
}

#endif /* KRYSTAL_AUTHORITY_BITS_H */
