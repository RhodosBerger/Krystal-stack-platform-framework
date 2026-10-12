#pragma once
/**
 * ==============================================================================
 * KRYSTAL-STACK: C++ HARDWARE PROBE & PHYSICAL EVIDENCE ENGINE
 * ==============================================================================
 * Module: cpp/hardware_probe/hardware_probe.hpp
 * Description:
 *     Low-latency, zero-allocation hardware probing interface for silicon telemetry:
 *       - CPU Package Power (Intel RAPL / AMD Energy status)
 *       - Die Temperature (hwmon / ACPI coretemp)
 *       - Clock domain binding & monotonic timestamping
 *       - Zero OS mutation guarantee (Sprint A read-only constraint)
 * ==============================================================================
 */

#include <string>
#include <vector>
#include <chrono>
#include <cstdint>
#include <span>
#include <memory>
#include <unordered_map>

namespace krystal::epistemic {

constexpr uint32_t VITAL_MAX_HP = 6;

enum class ValidityState {
    FRESH,
    STALE,
    EXPIRED,
    CONTRADICTORY,
    UNKNOWN
};

enum class EvidenceKind {
    HARDWARE_MEASURED,
    RUNTIME_MEASURED,
    INTEGRATION_VERIFIED,
    DERIVED,
    INFERRED
};

struct StateVersionCpp {
    uint32_t major{1};
    uint32_t minor{0};
    uint64_t sequence{1};
    double wall_timestamp_utc{0.0};
    std::string clock_domain_id{"host_clock_0"};
    std::string causal_parent{""};
};

template <typename T>
struct EpistemicValueCpp {
    T value;
    std::string provenance;
    EvidenceKind evidence_kind{EvidenceKind::HARDWARE_MEASURED};
    double uncertainty{0.02};
    uint64_t valid_from_ns{0};
    uint64_t valid_to_ns{0};
    std::string domain;
    StateVersionCpp state_version;
    std::string source_adapter{"CppHardwareProbe"};
    std::string clock_domain_id{"host_clock_0"};
    double calibration_error{0.01};

    ValidityState get_validity_state(uint64_t now_ns) const {
        if (now_ns < valid_from_ns) return ValidityState::UNKNOWN;
        if (now_ns > valid_to_ns) return ValidityState::EXPIRED;
        uint64_t stale_limit = static_cast<uint64_t>((valid_to_ns - valid_from_ns) * 0.7);
        if ((now_ns - valid_from_ns) > stale_limit) return ValidityState::STALE;
        return ValidityState::FRESH;
    }
};

struct HardwareTelemetrySample {
    double temperature_celsius{0.0};
    double package_power_watts{0.0};
    uint64_t tsc_cycles{0};
    uint64_t sample_monotonic_ns{0};
};

class HardwareProbeEngine {
public:
    HardwareProbeEngine(std::string host_fingerprint = "node-alpha-01", std::string clock_domain = "host_clock_0");
    ~HardwareProbeEngine() = default;

    // Zero-mutation observation
    HardwareTelemetrySample sample_silicon_telemetry();

    // Builds typed EpistemicValue container
    EpistemicValueCpp<double> create_thermal_epistemic_value(double temp_celsius, const StateVersionCpp& sv);

    // Asserts read-only invariant (must be 0 mutations)
    void assert_zero_mutation() const;
    uint32_t get_mutation_count() const { return mutation_count_; }

    // Renders machine-readable JSON matching fixtures/epistemic_value_v1.json
    std::string serialize_epistemic_thermal_json(const EpistemicValueCpp<double>& ev) const;

private:
    std::string host_fingerprint_;
    std::string clock_domain_id_;
    mutable uint32_t mutation_count_{0};
    bool is_read_only_{true};
};

} // namespace krystal::epistemic
