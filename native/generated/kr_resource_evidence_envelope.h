/* GENERATED from native/spec/binary_layouts.yaml - DO NOT EDIT */
#ifndef KR_RESOURCE_EVIDENCE_ENVELOPE_H
#define KR_RESOURCE_EVIDENCE_ENVELOPE_H
#include <stdint.h>
#include <stddef.h>
#include <assert.h>

typedef enum {
    KR_EPISTEMIC_KIND_OBSERVED = 0,
    KR_EPISTEMIC_KIND_DERIVED = 1,
    KR_EPISTEMIC_KIND_PREDICTED = 2,
    KR_EPISTEMIC_KIND_VALIDATED = 3,
    KR_EPISTEMIC_KIND_COUNTERFACTUAL = 4,
    KR_EPISTEMIC_KIND_QUARANTINED = 5,
} kr_epistemic_kind_t;

typedef enum {
    KR_TEMPORAL_MODE_MONOTONIC_LATCH = 0,
    KR_TEMPORAL_MODE_EPOCH_LEASED = 1,
    KR_TEMPORAL_MODE_BISTABLE_HYSTERETIC = 2,
    KR_TEMPORAL_MODE_SHADOW_HYPOTHETICAL = 3,
} kr_temporal_mode_t;

typedef enum {
    KR_OWNERSHIP_OWNED = 0,
    KR_OWNERSHIP_BORROWED = 1,
    KR_OWNERSHIP_VIEW = 2,
    KR_OWNERSHIP_PINNED = 3,
    KR_OWNERSHIP_SHARED_RING = 4,
} kr_ownership_t;

typedef struct _Alignas(64) kr_resource_evidence_envelope {
    uint8_t magic[4];
    uint16_t abi_version;
    uint16_t flags;
    uint64_t state_version;
    uint64_t observed_ns;
    uint64_t lease_ttl_ns;
    double value;
    double uncertainty;
    uint32_t domain_id;
    uint8_t epistemic_kind;
    uint8_t temporal_mode;
    uint8_t ownership;
    uint8_t reserved0;
    uint32_t crc32;
    uint32_t reserved1;
} kr_resource_evidence_envelope_t;

static_assert(sizeof(kr_resource_evidence_envelope_t) == 64, "kr_resource_evidence_envelope size");
static_assert(_Alignof(kr_resource_evidence_envelope_t) == 64, "kr_resource_evidence_envelope align");
static_assert(offsetof(kr_resource_evidence_envelope_t, magic) == 0, "kr_resource_evidence_envelope.magic offset");
static_assert(offsetof(kr_resource_evidence_envelope_t, abi_version) == 4, "kr_resource_evidence_envelope.abi_version offset");
static_assert(offsetof(kr_resource_evidence_envelope_t, flags) == 6, "kr_resource_evidence_envelope.flags offset");
static_assert(offsetof(kr_resource_evidence_envelope_t, state_version) == 8, "kr_resource_evidence_envelope.state_version offset");
static_assert(offsetof(kr_resource_evidence_envelope_t, observed_ns) == 16, "kr_resource_evidence_envelope.observed_ns offset");
static_assert(offsetof(kr_resource_evidence_envelope_t, lease_ttl_ns) == 24, "kr_resource_evidence_envelope.lease_ttl_ns offset");
static_assert(offsetof(kr_resource_evidence_envelope_t, value) == 32, "kr_resource_evidence_envelope.value offset");
static_assert(offsetof(kr_resource_evidence_envelope_t, uncertainty) == 40, "kr_resource_evidence_envelope.uncertainty offset");
static_assert(offsetof(kr_resource_evidence_envelope_t, domain_id) == 48, "kr_resource_evidence_envelope.domain_id offset");
static_assert(offsetof(kr_resource_evidence_envelope_t, epistemic_kind) == 52, "kr_resource_evidence_envelope.epistemic_kind offset");
static_assert(offsetof(kr_resource_evidence_envelope_t, temporal_mode) == 53, "kr_resource_evidence_envelope.temporal_mode offset");
static_assert(offsetof(kr_resource_evidence_envelope_t, ownership) == 54, "kr_resource_evidence_envelope.ownership offset");
static_assert(offsetof(kr_resource_evidence_envelope_t, reserved0) == 55, "kr_resource_evidence_envelope.reserved0 offset");
static_assert(offsetof(kr_resource_evidence_envelope_t, crc32) == 56, "kr_resource_evidence_envelope.crc32 offset");
static_assert(offsetof(kr_resource_evidence_envelope_t, reserved1) == 60, "kr_resource_evidence_envelope.reserved1 offset");

#endif /* KR_RESOURCE_EVIDENCE_ENVELOPE_H */
