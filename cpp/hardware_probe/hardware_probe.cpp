/**
 * ==============================================================================
 * KRYSTAL-STACK: C++ HARDWARE PROBE IMPLEMENTATION
 * ==============================================================================
 */

#include "hardware_probe.hpp"
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace krystal::epistemic {

HardwareProbeEngine::HardwareProbeEngine(std::string host_fingerprint, std::string clock_domain)
    : host_fingerprint_(std::move(host_fingerprint)),
      clock_domain_id_(std::move(clock_domain)) {}

HardwareTelemetrySample HardwareProbeEngine::sample_silicon_telemetry() {
    auto now = std::chrono::steady_clock::now();
    uint64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

    // Emulated physical registers (in production, reads /sys/class/hwmon or MSR 0x611)
    HardwareTelemetrySample sample;
    sample.temperature_celsius = 52.4;
    sample.package_power_watts = 18.2;
    sample.tsc_cycles = now_ns * 3; // Approx 3 GHz TSC
    sample.sample_monotonic_ns = now_ns;

    return sample;
}

EpistemicValueCpp<double> HardwareProbeEngine::create_thermal_epistemic_value(
    double temp_celsius,
    const StateVersionCpp& sv
) {
    auto now = std::chrono::steady_clock::now();
    uint64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    uint64_t ttl_ns = 2'000'000'000ULL; // 2 seconds

    EpistemicValueCpp<double> ev;
    ev.value = temp_celsius;
    ev.provenance = "CppHardwareProbe:hwmon:coretemp";
    ev.evidence_kind = EvidenceKind::HARDWARE_MEASURED;
    ev.uncertainty = 0.02;
    ev.valid_from_ns = now_ns;
    ev.valid_to_ns = now_ns + ttl_ns;
    ev.domain = "hardware.thermal";
    ev.state_version = sv;
    ev.source_adapter = "CppHardwareProbe";
    ev.clock_domain_id = clock_domain_id_;
    ev.calibration_error = 0.015;

    return ev;
}

void HardwareProbeEngine::assert_zero_mutation() const {
    if (mutation_count_ > 0) {
        throw std::runtime_error("Zero-mutation breach: hardware probe executed OS write operations!");
    }
}

std::string HardwareProbeEngine::serialize_epistemic_thermal_json(const EpistemicValueCpp<double>& ev) const {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"value\": " << std::fixed << std::setprecision(1) << ev.value << ",\n";
    ss << "  \"provenance\": \"" << ev.provenance << "\",\n";
    ss << "  \"evidence_kind\": \"HARDWARE_MEASURED\",\n";
    ss << "  \"uncertainty\": " << std::fixed << std::setprecision(2) << ev.uncertainty << ",\n";
    ss << "  \"valid_from_ns\": " << ev.valid_from_ns << ",\n";
    ss << "  \"valid_to_ns\": " << ev.valid_to_ns << ",\n";
    ss << "  \"domain\": \"" << ev.domain << "\",\n";
    ss << "  \"state_version\": {\n";
    ss << "    \"major\": " << ev.state_version.major << ",\n";
    ss << "    \"minor\": " << ev.state_version.minor << ",\n";
    ss << "    \"sequence\": " << ev.state_version.sequence << ",\n";
    ss << "    \"wall_timestamp_utc\": " << std::fixed << std::setprecision(1) << ev.state_version.wall_timestamp_utc << ",\n";
    ss << "    \"clock_domain_id\": \"" << ev.state_version.clock_domain_id << "\",\n";
    ss << "    \"causal_parent\": \"" << ev.state_version.causal_parent << "\"\n";
    ss << "  },\n";
    ss << "  \"source_adapter\": \"" << ev.source_adapter << "\",\n";
    ss << "  \"clock_domain_id\": \"" << ev.clock_domain_id << "\",\n";
    ss << "  \"calibration_error\": " << std::fixed << std::setprecision(3) << ev.calibration_error << "\n";
    ss << "}";
    return ss.str();
}

} // namespace krystal::epistemic
